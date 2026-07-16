"""Pydantic schemas.

Three layers:
  1. RawCampaign      - loose, tolerant parsing of the messy input.
  2. EnrichmentResult - the STRICT structured output we ask the LLM to return.
  3. Enriched/response models - what we store and serve.

The enums below define the *canonical* vocabulary. The LLM is constrained
to these values via JSON-schema structured output, so it can never invent a
channel or objective outside the set we control.
"""
from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict


# --- Canonical vocabularies (the "clean" set the LLM must map onto) ---

class Channel(str, Enum):
    META = "Meta"            # fb, Meta, Facebook Ads, meta ads, IG, Instagram
    GOOGLE_SEARCH = "Google Search"   # google, adwords (branded/non-brand search)
    GOOGLE_PMAX = "Google PMax"       # Performance Max
    YOUTUBE = "YouTube"      # YouTube, yt
    EMAIL = "Email"          # Klaviyo email, email
    SMS = "SMS"              # sms
    INFLUENCER = "Influencer"  # influencer, flat-fee creator
    OTHER = "Other"          # anything real but unmapped


class Objective(str, Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    CONVERSION = "conversion"
    RETENTION = "retention"


# --- Layer 1: raw input (deliberately permissive) ---

class RawCampaign(BaseModel):
    """A single raw row exactly as exported. Everything optional / Any because
    the source is intentionally dirty (strings where numbers expected, nulls,
    'N/A', etc.). We do NOT validate hard here; the cleaning layer does."""
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    name: Optional[str] = None
    channel: Optional[str] = None
    description: Optional[str] = None
    spend: Any = None
    impressions: Any = None
    clicks: Any = None
    conversions: Any = None
    revenue: Any = None


class IngestPayload(BaseModel):
    """Accepts either the full export object {brand, campaigns:[...]} or a bare
    list of campaigns."""
    brand: Optional[str] = None
    currency: Optional[str] = None
    note: Optional[str] = None
    campaigns: List[RawCampaign] = Field(default_factory=list)


# --- Layer 2: STRICT LLM structured output ---

class EnrichmentResult(BaseModel):
    """Exactly what the LLM returns per campaign. All fields required so we can
    run Groq strict mode (constrained decoding). No free-text parsing."""
    model_config = ConfigDict(extra="forbid")

    normalized_channel: Channel
    inferred_objective: Objective
    health_score: int = Field(ge=0, le=100)
    health_rationale: str = Field(min_length=1, max_length=280)


# --- Layer 3: persisted / served models ---

class CleanedMetrics(BaseModel):
    """Deterministically cleaned numeric fields (may still be partial)."""
    spend: Optional[float] = None
    impressions: Optional[int] = None
    clicks: Optional[int] = None
    conversions: Optional[int] = None
    revenue: Optional[float] = None


class EnrichedCampaign(BaseModel):
    id: str
    name: Optional[str]
    raw_channel: Optional[str]
    description: Optional[str]
    currency: Optional[str]

    spend: Optional[float]
    impressions: Optional[int]
    clicks: Optional[int]
    conversions: Optional[int]
    revenue: Optional[float]

    normalized_channel: str
    inferred_objective: str
    health_score: int
    health_rationale: str

    enrichment_source: str  # "llm" | "fallback"
    data_flags: List[str]   # non-fatal quality warnings

    model_config = ConfigDict(from_attributes=True)


# --- Ingest reporting (per-row success/failure summary) ---

class RowResult(BaseModel):
    id: Optional[str]
    status: str            # "ingested" | "skipped" | "failed" | "duplicate"
    enrichment_source: Optional[str] = None  # llm | fallback
    reason: Optional[str] = None
    flags: List[str] = Field(default_factory=list)


class IngestSummary(BaseModel):
    total: int
    ingested: int
    skipped: int
    failed: int
    duplicates: int
    results: List[RowResult]
