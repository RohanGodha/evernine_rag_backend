"""Tests for the rule-based fallback enrichment (used when the LLM misbehaves)."""
from app.cleaning import clean_row
from app.schemas import RawCampaign, Channel, Objective, CleanedMetrics
from app.llm import fallback_channel, fallback_objective, fallback_enrich
from app.cleaning import CleanOutcome


def test_channel_mapping_variants():
    assert fallback_channel("fb", "x") == Channel.META
    assert fallback_channel("Meta", "x") == Channel.META
    assert fallback_channel("Facebook Ads", "x") == Channel.META
    assert fallback_channel("meta ads", "x") == Channel.META
    assert fallback_channel("IG", "x") == Channel.META
    assert fallback_channel("google", "Search — Branded") == Channel.GOOGLE_SEARCH
    assert fallback_channel("adwords", "x") == Channel.GOOGLE_SEARCH
    assert fallback_channel("Google Ads", "Performance Max — full catalog") == Channel.GOOGLE_PMAX
    assert fallback_channel("yt", "x") == Channel.YOUTUBE
    assert fallback_channel("Klaviyo email", "x") == Channel.EMAIL
    assert fallback_channel("sms", "x") == Channel.SMS
    assert fallback_channel("influencer", "x") == Channel.INFLUENCER
    assert fallback_channel("carrier pigeon", "x") == Channel.OTHER


def test_objective_inference():
    assert fallback_objective("Winback — lapsed 90d", "no purchase") == Objective.RETENTION
    assert fallback_objective("VIP early access", "loyalty tier") == Objective.RETENTION
    assert fallback_objective("Brand Awareness — Hero", "reach") == Objective.AWARENESS
    assert fallback_objective("IG Reels — UGC", "consideration") == Objective.CONSIDERATION
    assert fallback_objective("Prospecting LAL 1%", "conversion optimized") == Objective.CONVERSION


def test_fallback_score_high_for_strong_roas():
    outcome = CleanOutcome(
        ok=True, campaign_id="c", name="Search — Branded", raw_channel="google",
        description="high intent",
        metrics=CleanedMetrics(spend=28000, revenue=1965000, conversions=1310),
    )
    result = fallback_enrich(outcome)
    assert result.normalized_channel == Channel.GOOGLE_SEARCH
    assert result.health_score >= 75


def test_fallback_score_low_when_no_tracking():
    outcome = CleanOutcome(
        ok=True, campaign_id="c", name="Influencer flat-fee", raw_channel="influencer",
        description="no pixel/tracking",
        metrics=CleanedMetrics(spend=350000, revenue=None, conversions=None),
    )
    result = fallback_enrich(outcome)
    assert result.normalized_channel == Channel.INFLUENCER
    assert result.health_score <= 25
    assert result.enrichment_source if hasattr(result, "enrichment_source") else True
