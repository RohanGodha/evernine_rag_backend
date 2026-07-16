"""Database engine, session, and ORM models.

Schema (3 tables):
  enriched_campaigns   - one row per successfully enriched campaign (dedup by id)
  ingest_runs          - one row per POST /campaigns/ingest call (audit)
  ingest_row_results   - per-row outcome for every raw row in a run (audit trail)

Indexes on normalized_channel and health_score back the required GET filters.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    create_engine, String, Integer, Float, Text, DateTime, ForeignKey, Index, text,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker,
)
from sqlalchemy.dialects.postgresql import JSONB

from .config import settings

logger = logging.getLogger(__name__)

# Embedding dimensionality of all-MiniLM-L6-v2. Kept here so ORM + embeddings
# module agree on a single source of truth.
EMBEDDING_DIM = 384

# pgvector's SQLAlchemy type is optional at runtime; guard the import so the app
# still boots (in SQL-only fallback mode) if the package isn't installed.
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except Exception:  # pragma: no cover - import environment dependent
    Vector = None  # type: ignore
    PGVECTOR_AVAILABLE = False


engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EnrichedCampaign(Base):
    __tablename__ = "enriched_campaigns"

    # Business id from the source is the natural PK -> gives idempotent upsert.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    raw_channel: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    spend: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    impressions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clicks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    conversions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # LLM-enriched fields (constrained to canonical vocab by the schema).
    normalized_channel: Mapped[str] = mapped_column(String(64), nullable=False)
    inferred_objective: Mapped[str] = mapped_column(String(32), nullable=False)
    health_score: Mapped[int] = mapped_column(Integer, nullable=False)
    health_rationale: Mapped[str] = mapped_column(Text, nullable=False)

    enrichment_source: Mapped[str] = mapped_column(String(16), nullable=False)  # llm|fallback
    data_flags: Mapped[list] = mapped_column(JSONB, default=list)

    # Semantic-search vector. Nullable: null => embeddings were unavailable for
    # this row, and search transparently degrades to SQL keyword matching.
    if PGVECTOR_AVAILABLE:
        embedding: Mapped[Optional[list]] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("ix_campaign_channel", "normalized_channel"),
        Index("ix_campaign_health", "health_score"),
    )


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    total: Mapped[int] = mapped_column(Integer, default=0)
    ingested: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, default=0)

    rows: Mapped[list["IngestRowResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class IngestRowResult(Base):
    __tablename__ = "ingest_row_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id", ondelete="CASCADE"))
    campaign_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # ingested|skipped|failed|duplicate
    enrichment_source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    flags: Mapped[list] = mapped_column(JSONB, default=list)

    run: Mapped["IngestRun"] = relationship(back_populates="rows")


def init_db() -> None:
    """Create tables if they don't exist (fine for this exercise; a real
    service would use Alembic migrations).

    Enables the pgvector extension first (best-effort) so the embedding column
    and its index can be created. If the extension isn't available the app
    still runs in SQL-only fallback mode.
    """
    if PGVECTOR_AVAILABLE:
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:  # pragma: no cover
            logger.warning("Could not enable pgvector extension: %s", e)

    Base.metadata.create_all(bind=engine)

    # Cosine-distance index for semantic search. Best-effort; seq scan is fine
    # at this data volume if the index can't be created.
    if PGVECTOR_AVAILABLE:
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_campaign_embedding "
                    "ON enriched_campaigns USING hnsw (embedding vector_cosine_ops)"
                ))
        except Exception as e:  # pragma: no cover
            logger.warning("Could not create vector index: %s", e)


def get_session():
    """FastAPI dependency yielding a DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
