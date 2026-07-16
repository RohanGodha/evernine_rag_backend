"""Deterministic cleaning & validation.

Design principle: do the *mechanical* work here (parsing, junk rejection,
duplicate detection, quality flags) and leave only *judgment* (channel
normalization, objective inference, health scoring) to the LLM.

This is defensible: there is no single "correct" cleaning, so we make the
rules explicit and record every decision as a flag or a rejection reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, List

from .schemas import RawCampaign, CleanedMetrics


# Tokens that mean "no real value" when they appear where a number should be.
_JUNK_NUMERIC = {"n/a", "na", "none", "null", "lots", "many", "-", "", "?"}

# Signals that a whole row is a test/draft/placeholder and should be rejected.
_JUNK_ROW_MARKERS = ("do not use", "draft row", "asdf test", "test test ignore")


@dataclass
class CleanOutcome:
    """Result of cleaning one raw row."""
    ok: bool                              # False => reject (skip enrichment)
    reason: Optional[str] = None          # why rejected
    campaign_id: Optional[str] = None
    name: Optional[str] = None
    raw_channel: Optional[str] = None
    description: Optional[str] = None
    metrics: CleanedMetrics = field(default_factory=CleanedMetrics)
    flags: List[str] = field(default_factory=list)


def _parse_number(value: Any) -> tuple[Optional[float], Optional[str]]:
    """Return (number, flag). flag is set when we had to coerce or reject.
    Returns (None, None) for a legitimately-missing (null) value."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, "non_numeric_value"
    if isinstance(value, (int, float)):
        return float(value), None

    s = str(value).strip()
    if s.lower() in _JUNK_NUMERIC:
        return None, "non_numeric_value"

    # Strip thousands separators, currency symbols, spaces: "42,500" -> 42500
    cleaned = re.sub(r"[,\s₹$]", "", s)
    try:
        return float(cleaned), "coerced_from_string"
    except ValueError:
        return None, "non_numeric_value"


def _as_int(value: Optional[float]) -> Optional[int]:
    return int(value) if value is not None else None


def clean_row(raw: RawCampaign, currency: Optional[str]) -> CleanOutcome:
    """Clean and validate a single raw campaign row."""
    flags: List[str] = []

    cid = (raw.id or "").strip() or None
    name = (raw.name or "").strip() or None
    raw_channel = (raw.channel or "").strip() or None
    description = (raw.description or "").strip() or None

    # --- Hard validation: reject rows that can't be meaningfully enriched ---
    if not cid:
        return CleanOutcome(ok=False, reason="missing_id", name=name)

    blob = " ".join(filter(None, [name, description])).lower()
    if any(marker in blob for marker in _JUNK_ROW_MARKERS):
        return CleanOutcome(
            ok=False, reason="junk_or_test_row", campaign_id=cid, name=name,
            raw_channel=raw_channel, description=description,
        )

    if not raw_channel or raw_channel in {"?", "??", "???"}:
        # No usable channel AND typically paired with junk metrics -> reject.
        return CleanOutcome(
            ok=False, reason="unusable_channel", campaign_id=cid, name=name,
            raw_channel=raw_channel, description=description,
        )

    # --- Numeric coercion (soft: flag, don't reject) ---
    spend, f = _parse_number(raw.spend)
    if f == "coerced_from_string":
        flags.append("spend_coerced_from_string")
    elif f == "non_numeric_value":
        flags.append("spend_invalid")

    impressions_f, imp_flag = _parse_number(raw.impressions)
    if imp_flag == "non_numeric_value":
        flags.append("impressions_invalid")
    clicks_f, clk_flag = _parse_number(raw.clicks)
    if clk_flag == "non_numeric_value":
        flags.append("clicks_invalid")
    conversions_f, cnv_flag = _parse_number(raw.conversions)
    if cnv_flag == "non_numeric_value":
        flags.append("conversions_invalid")
    revenue, rev_flag = _parse_number(raw.revenue)
    if rev_flag == "coerced_from_string":
        flags.append("revenue_coerced_from_string")
    elif rev_flag == "non_numeric_value":
        flags.append("revenue_invalid")

    # --- Negative-value sanitisation (flag + null out impossible negatives) ---
    if spend is not None and spend < 0:
        flags.append("negative_spend_nulled")
        spend = None
    if clicks_f is not None and clicks_f < 0:
        flags.append("negative_clicks_nulled")
        clicks_f = None

    # --- Missing-data flags (informative, not fatal) ---
    if impressions_f is None and "impressions_invalid" not in flags:
        flags.append("impressions_missing")
    if spend is None and "spend_invalid" not in flags and "negative_spend_nulled" not in flags:
        flags.append("spend_missing")
    if name is None:
        flags.append("name_blank")

    metrics = CleanedMetrics(
        spend=spend,
        impressions=_as_int(impressions_f),
        clicks=_as_int(clicks_f),
        conversions=_as_int(conversions_f),
        revenue=revenue,
    )

    return CleanOutcome(
        ok=True,
        campaign_id=cid,
        name=name,
        raw_channel=raw_channel,
        description=description,
        metrics=metrics,
        flags=flags,
    )
