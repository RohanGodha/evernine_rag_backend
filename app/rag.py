"""Vector RAG: semantic search + portfolio insights.

Both features degrade gracefully:
  * search()   -> pgvector cosine; falls back to SQL ILIKE keyword match.
  * insights() -> retrieve context + SQL aggregates -> LLM; falls back to a
                  deterministic aggregate summary if the LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .config import settings
from .db import EnrichedCampaign, PGVECTOR_AVAILABLE
from . import embeddings
from .llm import llm_client

logger = logging.getLogger(__name__)


# --- Embedding upsert (called during ingest) ---

def compute_embedding(
    name: Optional[str], description: Optional[str],
    normalized_channel: str, inferred_objective: str,
) -> Optional[List[float]]:
    """Compute an embedding for a campaign, or None if unavailable."""
    if not PGVECTOR_AVAILABLE or not embeddings.is_available():
        return None
    text = embeddings.build_embedding_text(
        name, description, normalized_channel, inferred_objective
    )
    return embeddings.encode(text)


# --- Semantic search ---

def search(session: Session, query: str, k: int = 5) -> tuple[List[EnrichedCampaign], str]:
    """Return (campaigns, mode). mode is 'vector' or 'keyword'."""
    if PGVECTOR_AVAILABLE and embeddings.is_available():
        qvec = embeddings.encode(query)
        if qvec is not None:
            try:
                stmt = (
                    select(EnrichedCampaign)
                    .where(EnrichedCampaign.embedding.isnot(None))
                    .order_by(EnrichedCampaign.embedding.cosine_distance(qvec))
                    .limit(k)
                )
                rows = list(session.scalars(stmt).all())
                if rows:
                    return rows, "vector"
            except Exception as e:  # pragma: no cover
                logger.warning("Vector search failed, using keyword fallback: %s", e)

    # Fallback: SQL keyword match over name/description.
    like = f"%{query}%"
    stmt = (
        select(EnrichedCampaign)
        .where(
            EnrichedCampaign.name.ilike(like)
            | EnrichedCampaign.description.ilike(like)
            | EnrichedCampaign.normalized_channel.ilike(like)
        )
        .order_by(EnrichedCampaign.health_score.desc())
        .limit(k)
    )
    return list(session.scalars(stmt).all()), "keyword"


# --- Portfolio insights ---

INSIGHTS_SYSTEM = (
    "You are a marketing portfolio analyst for a D2C skincare brand. Given "
    "aggregate campaign metrics and a few example campaigns, produce 2-3 sharp, "
    "specific portfolio-level observations (e.g. where budget is being wasted, "
    "which channels/objectives over- or under-perform). Be concrete and cite "
    "numbers. Return only the structured fields."
)


def _aggregate(session: Session) -> dict:
    """Deterministic portfolio aggregates from the DB."""
    rows = list(session.scalars(select(EnrichedCampaign)).all())
    by_channel: dict[str, dict] = {}
    total_spend = total_revenue = 0.0
    for c in rows:
        spend = c.spend or 0.0
        revenue = c.revenue or 0.0
        total_spend += spend
        total_revenue += revenue
        b = by_channel.setdefault(
            c.normalized_channel, {"spend": 0.0, "revenue": 0.0, "count": 0, "score_sum": 0}
        )
        b["spend"] += spend
        b["revenue"] += revenue
        b["count"] += 1
        b["score_sum"] += c.health_score
    for ch, b in by_channel.items():
        b["roas"] = round(b["revenue"] / b["spend"], 2) if b["spend"] else None
        b["avg_health"] = round(b["score_sum"] / b["count"], 1) if b["count"] else None
    return {
        "campaign_count": len(rows),
        "total_spend": total_spend,
        "total_revenue": total_revenue,
        "portfolio_roas": round(total_revenue / total_spend, 2) if total_spend else None,
        "by_channel": by_channel,
    }


def _deterministic_observations(agg: dict) -> List[str]:
    """Fallback observations computed without the LLM."""
    obs: List[str] = []
    channels = agg["by_channel"]
    scored = [(ch, b) for ch, b in channels.items() if b.get("roas") is not None]
    if scored:
        worst = min(scored, key=lambda x: x[1]["roas"])
        best = max(scored, key=lambda x: x[1]["roas"])
        obs.append(
            f"{worst[0]} has the lowest ROAS ({worst[1]['roas']}) on "
            f"{worst[1]['spend']:.0f} spend — likely wasted budget."
        )
        obs.append(
            f"{best[0]} is the most efficient channel (ROAS {best[1]['roas']})."
        )
    if agg["portfolio_roas"] is not None:
        obs.append(f"Portfolio ROAS is {agg['portfolio_roas']} across {agg['campaign_count']} campaigns.")
    return obs[:3] or ["Not enough data for portfolio observations."]


def insights(session: Session) -> dict:
    """Return portfolio insights. Uses RAG context + LLM; falls back to a
    deterministic aggregate summary."""
    agg = _aggregate(session)

    # Optional retrieval context: a few representative campaigns.
    context, mode = search(session, "budget efficiency and performance", k=5)
    context_lines = [
        f"- {c.id} [{c.normalized_channel}/{c.inferred_objective}] "
        f"spend={c.spend} revenue={c.revenue} health={c.health_score}"
        for c in context
    ]

    if not settings.groq_api_key:
        return {
            "source": "deterministic",
            "retrieval_mode": mode,
            "aggregates": agg,
            "observations": _deterministic_observations(agg),
        }

    prompt = (
        f"Aggregates (JSON):\n{json.dumps(agg, indent=2)}\n\n"
        f"Example campaigns (retrieved, mode={mode}):\n" + "\n".join(context_lines)
    )
    try:
        resp = llm_client.client.chat.completions.create(
            model=settings.groq_model,
            temperature=0,
            messages=[
                {"role": "system", "content": INSIGHTS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "portfolio_insights",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "observations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 3,
                            }
                        },
                        "required": ["observations"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "source": "llm",
            "retrieval_mode": mode,
            "aggregates": agg,
            "observations": data.get("observations", []),
        }
    except Exception as e:
        logger.warning("Insights LLM failed, using deterministic fallback: %s", e)
        return {
            "source": "deterministic",
            "retrieval_mode": mode,
            "aggregates": agg,
            "observations": _deterministic_observations(agg),
        }
