"""Tests for the deterministic cleaning / validation layer.

These cover the specific traps in the mock data (data/campaigns_raw.json):
string numbers, junk rows, negatives, nulls, blank names, bad channels.
"""
from app.cleaning import clean_row
from app.schemas import RawCampaign


def _clean(**kw):
    return clean_row(RawCampaign(**kw), currency="INR")


def test_comma_string_spend_is_coerced():
    out = _clean(id="cmp_002", name="Retarget", channel="Meta",
                 description="dpa", spend="42,500")
    assert out.ok
    assert out.metrics.spend == 42500.0
    assert "spend_coerced_from_string" in out.flags


def test_zero_spend_is_valid_not_missing():
    out = _clean(id="cmp_006", name="Winback", channel="Klaviyo email",
                 description="flow", spend=0)
    assert out.ok
    assert out.metrics.spend == 0.0
    assert "spend_missing" not in out.flags


def test_junk_test_row_is_rejected():
    out = _clean(id="cmp_013", name="Test — DO NOT USE (draft row)",
                 channel="???", description="asdf test test ignore",
                 spend="N/A", impressions="lots", clicks=-5)
    assert not out.ok
    assert out.reason == "junk_or_test_row"


def test_unusable_channel_rejected():
    out = _clean(id="x", name="real name", channel="???", description="d")
    assert not out.ok
    assert out.reason == "unusable_channel"


def test_missing_id_rejected():
    out = _clean(id="", name="n", channel="fb", description="d")
    assert not out.ok
    assert out.reason == "missing_id"


def test_negative_spend_nulled_and_flagged():
    out = _clean(id="cmp_014", name="Retention VIP", channel="email",
                 description="loyalty", spend=-1200)
    assert out.ok
    assert out.metrics.spend is None
    assert "negative_spend_nulled" in out.flags


def test_null_impressions_flagged_missing():
    out = _clean(id="cmp_009", name="SMS nudge", channel="sms",
                 description="flow", spend=9500, impressions=None)
    assert out.ok
    assert out.metrics.impressions is None
    assert "impressions_missing" in out.flags


def test_blank_name_flagged_but_kept():
    out = _clean(id="cmp_011", name="", channel="yt",
                 description="Bumper ads, reach buy.", spend=40000)
    assert out.ok
    assert out.name is None
    assert "name_blank" in out.flags


def test_non_numeric_value_flagged_invalid():
    out = _clean(id="z", name="n", channel="google", description="d",
                 conversions="many")
    assert out.ok
    assert out.metrics.conversions is None
    assert "conversions_invalid" in out.flags
