"""Ingestion orchestration.

Pipeline per row (fully isolated - one failure never sinks the batch):
    clean -> dedupe (keep first) -> LLM enrich (retry/fallback) -> upsert

Records a per-row outcome for the response AND persists it to the audit tables.
Idempotent: re-ingesting the same file updates rows in place (PK = business id)
and reports repeats as duplicates rather than creating new rows.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from .schemas import (
    IngestPayload, IngestSummary, RowResult, RawCampaign,
)
from .cleaning import clean_row
from .llm import llm_client, fallback_channel
from .db import EnrichedCampaign, IngestRun, IngestRowResult
from . import rag

logger = logging.getLogger(__name__)


def ingest_campaigns(payload: IngestPayload, session: Session) -> IngestSummary:
    run = IngestRun(total=len(payload.campaigns))
    session.add(run)
    session.flush()  # get run.id

    results: list[RowResult] = []
    seen_ids: set[str] = set()

    for raw in payload.campaigns:
        row = _process_row(raw, payload.currency, seen_ids, session, run.id)
        results.append(row)

    run.ingested = sum(1 for r in results if r.status == "ingested")
    run.skipped = sum(1 for r in results if r.status == "skipped")
    run.failed = sum(1 for r in results if r.status == "failed")
    run.duplicates = sum(1 for r in results if r.status == "duplicate")
    session.commit()

    return IngestSummary(
        total=run.total,
        ingested=run.ingested,
        skipped=run.skipped,
        failed=run.failed,
        duplicates=run.duplicates,
        results=results,
    )


def _record(session: Session, run_id: int, row: RowResult) -> None:
    session.add(IngestRowResult(
        run_id=run_id,
        campaign_id=row.id,
        status=row.status,
        enrichment_source=row.enrichment_source,
        reason=row.reason,
        flags=row.flags,
    ))


def _process_row(
    raw: RawCampaign,
    currency: Optional[str],
    seen_ids: set[str],
    session: Session,
    run_id: int,
) -> RowResult:
    """Process one raw row with full isolation. Any unexpected error is caught
    and turned into a 'failed' outcome rather than aborting the batch."""
    try:
        outcome = clean_row(raw, currency)

        if not outcome.ok:
            row = RowResult(
                id=outcome.campaign_id, status="skipped",
                reason=outcome.reason, flags=outcome.flags,
            )
            _record(session, run_id, row)
            return row

        cid = outcome.campaign_id  # guaranteed non-null when ok

        # Dedupe within this batch: keep first, flag repeats.
        if cid in seen_ids:
            row = RowResult(
                id=cid, status="duplicate",
                reason="duplicate_id_in_batch", flags=outcome.flags,
            )
            _record(session, run_id, row)
            return row
        seen_ids.add(cid)

        # Enrich (never raises: degrades to fallback).
        result, source = llm_client.enrich(outcome, currency)

        # Self-check / eval: cross-check the LLM's channel against the
        # deterministic rule. Disagreement is informational (flag, don't block).
        flags = list(outcome.flags)
        det_channel = fallback_channel(outcome.raw_channel, outcome.name)
        if source == "llm" and result.normalized_channel != det_channel:
            flags.append(
                f"channel_selfcheck_mismatch(llm={result.normalized_channel.value},"
                f"rule={det_channel.value})"
            )

        # Embed for semantic search (best-effort; null vector => SQL fallback).
        embedding = rag.compute_embedding(
            outcome.name, outcome.description,
            result.normalized_channel.value, result.inferred_objective.value,
        )
        if embedding is None:
            flags.append("embedding_unavailable")

        # Idempotent upsert by business id.
        existing = session.get(EnrichedCampaign, cid)
        payload = dict(
            name=outcome.name,
            raw_channel=outcome.raw_channel,
            description=outcome.description,
            currency=currency,
            spend=outcome.metrics.spend,
            impressions=outcome.metrics.impressions,
            clicks=outcome.metrics.clicks,
            conversions=outcome.metrics.conversions,
            revenue=outcome.metrics.revenue,
            normalized_channel=result.normalized_channel.value,
            inferred_objective=result.inferred_objective.value,
            health_score=result.health_score,
            health_rationale=result.health_rationale,
            enrichment_source=source,
            data_flags=flags,
        )
        # embedding is a separate optional column (only present when pgvector
        # is available); set it via attribute so ORM stays valid either way.
        if existing:
            for k, v in payload.items():
                setattr(existing, k, v)
            if embedding is not None and hasattr(existing, "embedding"):
                existing.embedding = embedding
        else:
            obj = EnrichedCampaign(id=cid, **payload)
            if embedding is not None and hasattr(obj, "embedding"):
                obj.embedding = embedding
            session.add(obj)

        row = RowResult(
            id=cid, status="ingested",
            enrichment_source=source, flags=flags,
        )
        _record(session, run_id, row)
        return row

    except Exception as e:  # last-resort isolation
        logger.exception("Unexpected error processing row: %s", e)
        row = RowResult(
            id=getattr(raw, "id", None), status="failed",
            reason=f"unexpected_error: {type(e).__name__}", flags=[],
        )
        _record(session, run_id, row)
        return row
