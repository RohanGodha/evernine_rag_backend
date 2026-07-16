"""FastAPI app: ingestion + retrieval endpoints."""
from __future__ import annotations

from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import init_db, get_session, EnrichedCampaign as CampaignORM
from .schemas import IngestPayload, IngestSummary, EnrichedCampaign as CampaignOut
from .enrichment import ingest_campaigns

app = FastAPI(
    title="Aster & Oak — Campaign Enrichment API",
    version="1.0.0",
    description="Ingests messy marketing campaigns, enriches them with an LLM "
                "(structured output), stores in Postgres, and serves them back.",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/campaigns/ingest", response_model=IngestSummary)
def ingest(payload: IngestPayload, session: Session = Depends(get_session)) -> IngestSummary:
    """Read raw campaigns, enrich each, persist, and return a per-row summary.
    A single bad row or model response is isolated and never sinks the batch."""
    if not payload.campaigns:
        raise HTTPException(status_code=400, detail="No campaigns provided.")
    return ingest_campaigns(payload, session)


@app.get("/campaigns", response_model=List[CampaignOut])
def list_campaigns(
    session: Session = Depends(get_session),
    channel: Optional[str] = Query(None, description="Filter by normalized_channel"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum health_score"),
    objective: Optional[str] = Query(None, description="Filter by inferred_objective"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> List[CampaignOut]:
    """Retrieve enriched campaigns with at least one filter available."""
    stmt = select(CampaignORM)
    if channel:
        stmt = stmt.where(CampaignORM.normalized_channel == channel)
    if min_score is not None:
        stmt = stmt.where(CampaignORM.health_score >= min_score)
    if objective:
        stmt = stmt.where(CampaignORM.inferred_objective == objective)
    stmt = stmt.order_by(CampaignORM.health_score.desc()).limit(limit).offset(offset)
    return list(session.scalars(stmt).all())


@app.get("/campaigns/{campaign_id}", response_model=CampaignOut)
def get_campaign(campaign_id: str, session: Session = Depends(get_session)) -> CampaignOut:
    obj = session.get(CampaignORM, campaign_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found.")
    return obj
