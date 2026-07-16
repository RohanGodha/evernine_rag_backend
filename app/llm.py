"""LLM enrichment via Groq structured output.

Why this design (defensible in the walkthrough):
  * We use Groq's json_schema structured output in STRICT mode (constrained
    decoding) -> the model is token-constrained to our schema, so it can never
    emit an invalid channel/objective or a non-JSON blob. This is the brief's
    "function/tool calling or structured/JSON output -- not free-text parsing".
  * We do NOT use a ReAct / multi-step agent loop. Per-row enrichment is a
    single deterministic extraction task; a loop would add latency, cost and
    non-determinism for no benefit. One constrained call per row is the right
    tool. (A ReAct-style loop would only make sense for the stretch /insights
    aggregation endpoint.)

Failure handling: retry transient errors, then fall back to a deterministic
rule-based enrichment so one bad model response never sinks the batch.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from groq import Groq
from pydantic import ValidationError

from .config import settings
from .schemas import (
    EnrichmentResult, PartialEnrichment, Channel, Objective, CleanedMetrics,
)
from .cleaning import CleanOutcome

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a marketing-analytics normalizer for a D2C skincare brand. "
    "For a single ad campaign you receive its raw channel label, name, "
    "description and (possibly incomplete) performance metrics. Return ONLY the "
    "structured fields defined by the schema.\n\n"
    "Rules:\n"
    "1. normalized_channel: map the raw channel onto the canonical set. "
    "fb / Meta / Facebook Ads / meta ads / IG / Instagram -> Meta. "
    "google / adwords / branded or non-brand search -> Google Search. "
    "Performance Max / PMax -> Google PMax. YouTube / yt -> YouTube. "
    "Klaviyo email / email -> Email. sms -> SMS. influencer / creator flat-fee "
    "-> Influencer. Anything real but unmappable -> Other.\n"
    "2. inferred_objective: infer from name + description one of "
    "awareness, consideration, conversion, retention. Winback/loyalty/VIP/"
    "lapsed -> retention. Prospecting/acquisition/branded+non-brand search/"
    "retargeting-to-buy -> conversion. Reels/UGC/consideration -> consideration. "
    "Reach/bumper/brand video with no direct response -> awareness.\n"
    "3. health_score (0-100): judge campaign health from efficiency signals "
    "(ROAS = revenue/spend, CTR, CVR) AND data completeness. Missing tracking "
    "or no conversions/revenue should lower the score. Give a one-line rationale."
)


def _build_user_prompt(c: CleanOutcome, currency: Optional[str]) -> str:
    m: CleanedMetrics = c.metrics
    return (
        f"raw_channel: {c.raw_channel!r}\n"
        f"name: {c.name!r}\n"
        f"description: {c.description!r}\n"
        f"currency: {currency!r}\n"
        f"spend: {m.spend}\n"
        f"impressions: {m.impressions}\n"
        f"clicks: {m.clicks}\n"
        f"conversions: {m.conversions}\n"
        f"revenue: {m.revenue}\n"
        f"data_flags: {c.flags}\n"
    )


PARTIAL_SYSTEM_PROMPT = (
    "You are a marketing-analytics normalizer for a D2C skincare brand. The "
    "campaign's channel has ALREADY been resolved and is given to you. Return "
    "ONLY the structured fields defined by the schema.\n\n"
    "Rules:\n"
    "1. inferred_objective: infer from name + description one of "
    "awareness, consideration, conversion, retention. Winback/loyalty/VIP/"
    "lapsed -> retention. Prospecting/acquisition/branded+non-brand search/"
    "retargeting-to-buy -> conversion. Reels/UGC/consideration -> consideration. "
    "Reach/bumper/brand video with no direct response -> awareness.\n"
    "2. health_score (0-100): judge campaign health from efficiency signals "
    "(ROAS = revenue/spend, CTR, CVR) AND data completeness. Missing tracking "
    "or no conversions/revenue should lower the score. Give a one-line rationale."
)


def _json_schema() -> dict:
    """Groq strict mode needs all fields required + additionalProperties:false.
    Pydantic v2 emits exactly that for our model (extra='forbid')."""
    return {
        "name": "campaign_enrichment",
        "strict": True,
        "schema": EnrichmentResult.model_json_schema(),
    }


def _partial_json_schema() -> dict:
    return {
        "name": "campaign_enrichment_partial",
        "strict": True,
        "schema": PartialEnrichment.model_json_schema(),
    }


class LLMClient:
    def __init__(self) -> None:
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )
        return self._client

    def enrich(self, c: CleanOutcome, currency: Optional[str]) -> tuple[EnrichmentResult, str, str]:
        """Return (result, source, channel_source).

        Deterministic-first for channel: a known variant is resolved by the
        static lookup table (channel_source='static') and the LLM only enriches
        objective + score. Unknown variants fall through to the LLM classifying
        the channel too (channel_source='llm').

        source is 'llm' or 'fallback' (the latter when the LLM fails entirely).
        Never raises: a total LLM failure degrades to a rule-based result.
        """
        static_channel = resolve_channel_static(c.raw_channel, c.name)
        last_err: Optional[Exception] = None

        for attempt in range(settings.llm_max_retries + 1):
            try:
                if static_channel is not None:
                    partial = self._call_llm_partial(c, currency, static_channel)
                    result = EnrichmentResult(
                        normalized_channel=static_channel,
                        inferred_objective=partial.inferred_objective,
                        health_score=partial.health_score,
                        health_rationale=partial.health_rationale,
                    )
                    return result, "llm", "static"
                result = self._call_llm_full(c, currency)
                return result, "llm", "llm"
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                logger.warning("LLM output invalid (attempt %d): %s", attempt + 1, e)
            except Exception as e:  # network / rate-limit / API error
                last_err = e
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)

        logger.error("LLM enrichment failed after retries, using fallback: %s", last_err)
        # Even in fallback, prefer the deterministically-resolved channel.
        channel_source = "static" if static_channel is not None else "fallback"
        return fallback_enrich(c, static_channel), "fallback", channel_source

    def _call_llm_full(self, c: CleanOutcome, currency: Optional[str]) -> EnrichmentResult:
        resp = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(c, currency)},
            ],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": _json_schema()},
        )
        content = resp.choices[0].message.content or "{}"
        return EnrichmentResult.model_validate_json(content)

    def _call_llm_partial(
        self, c: CleanOutcome, currency: Optional[str], channel: Channel,
    ) -> PartialEnrichment:
        user = f"resolved_channel: {channel.value}\n" + _build_user_prompt(c, currency)
        resp = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": PARTIAL_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": _partial_json_schema()},
        )
        content = resp.choices[0].message.content or "{}"
        return PartialEnrichment.model_validate_json(content)


# --- Static channel lookup (deterministic-first) ---

# Exact, known abbreviations/variants -> canonical channel. This is the fast,
# deterministic path: for a known variant we skip asking the LLM to classify
# the channel at all. `resolve_channel_static` returns None on a miss, and only
# then does the LLM classify the channel.
_STATIC_CHANNEL_MAP = {
    "fb": Channel.META,
    "meta": Channel.META,
    "meta ads": Channel.META,
    "facebook": Channel.META,
    "facebook ads": Channel.META,
    "ig": Channel.META,
    "instagram": Channel.META,
    "google": Channel.GOOGLE_SEARCH,
    "google ads": Channel.GOOGLE_SEARCH,
    "gads": Channel.GOOGLE_SEARCH,
    "adwords": Channel.GOOGLE_SEARCH,
    "youtube": Channel.YOUTUBE,
    "yt": Channel.YOUTUBE,
    "email": Channel.EMAIL,
    "klaviyo": Channel.EMAIL,
    "klaviyo email": Channel.EMAIL,
    "sms": Channel.SMS,
    "influencer": Channel.INFLUENCER,
}


def resolve_channel_static(raw_channel: Optional[str], name: Optional[str]) -> Optional[Channel]:
    """Deterministic channel resolution for known variants. Returns None if the
    variant is unknown (so the LLM should classify it).

    Special case: 'Google Ads' + a Performance Max name -> Google PMax, which a
    flat table can't express, so we detect PMax from the name here.
    """
    key = (raw_channel or "").strip().lower()
    text = f"{raw_channel or ''} {name or ''}".lower()
    if "pmax" in text or "performance max" in text:
        return Channel.GOOGLE_PMAX
    return _STATIC_CHANNEL_MAP.get(key)


# --- Deterministic fallback (also reused by the eval/self-check) ---

# Keys are matched on WORD BOUNDARIES so short tokens (ig, fb, yt, sms) don't
# match inside unrelated words (e.g. "ig" in "pigeon"). Order matters: PMax is
# checked before generic Google, Meta before Google, etc.
_CHANNEL_RULES = [
    (("pmax", "performance max"), Channel.GOOGLE_PMAX),
    (("fb", "meta", "facebook", "ig", "instagram"), Channel.META),
    (("google", "adwords", "search"), Channel.GOOGLE_SEARCH),
    (("youtube", "yt"), Channel.YOUTUBE),
    (("email", "klaviyo"), Channel.EMAIL),
    (("sms",), Channel.SMS),
    (("influencer", "creator"), Channel.INFLUENCER),
]


def _has_token(hay: str, token: str) -> bool:
    """Word-boundary match; handles multi-word tokens like 'performance max'."""
    return re.search(rf"\b{re.escape(token)}\b", hay) is not None


def fallback_channel(raw_channel: Optional[str], name: Optional[str]) -> Channel:
    hay = f"{raw_channel or ''} {name or ''}".lower()
    for keys, ch in _CHANNEL_RULES:
        if any(_has_token(hay, k) for k in keys):
            return ch
    return Channel.OTHER


def fallback_objective(name: Optional[str], description: Optional[str]) -> Objective:
    hay = f"{name or ''} {description or ''}".lower()
    if any(k in hay for k in ("winback", "lapsed", "loyalty", "vip", "retention", "early access")):
        return Objective.RETENTION
    if any(k in hay for k in ("awareness", "reach", "bumper", "brand awareness")):
        return Objective.AWARENESS
    if any(k in hay for k in ("ugc", "reels", "consideration", "creator")):
        return Objective.CONSIDERATION
    return Objective.CONVERSION


def fallback_enrich(c: CleanOutcome, channel: Optional[Channel] = None) -> EnrichmentResult:
    """Rule-based enrichment used when the LLM is unavailable/misbehaving.

    If a channel was already resolved (e.g. by the static table), it's reused;
    otherwise it's derived from the keyword rules."""
    m = c.metrics
    if channel is None:
        channel = fallback_channel(c.raw_channel, c.name)
    objective = fallback_objective(c.name, c.description)

    # Simple heuristic health score from ROAS + data completeness.
    score = 50
    if m.spend and m.revenue:
        roas = m.revenue / m.spend if m.spend else 0
        if roas >= 8:
            score = 90
        elif roas >= 4:
            score = 75
        elif roas >= 1.5:
            score = 60
        else:
            score = 35
    if m.conversions is None and m.revenue is None:
        score = min(score, 25)  # no tracking => low confidence in value
    score = max(0, min(100, score))

    return EnrichmentResult(
        normalized_channel=channel,
        inferred_objective=objective,
        health_score=score,
        health_rationale="Rule-based fallback (LLM unavailable): scored from ROAS and data completeness.",
    )


llm_client = LLMClient()
