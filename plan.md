# Plan — Architecture & Design Decisions

## 1. High-level flow

```
                         POST /campaigns/ingest
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │  For each raw row (fully isolated):                │
        │    cleaning.clean_row  ──► reject? → skipped        │
        │         │ ok                                        │
        │    dedupe (keep first) ─► seen? → duplicate         │
        │         │                                           │
        │    channel: static lookup? ── hit → LLM(objective+score)  │
        │                            └ miss → LLM(channel+obj+score)│  ← structured output
        │         │  (LLM fails → rule-based fallback_enrich)        │
        │         │                                           │
        │    self-check (LLM channel vs deterministic) → flag │
        │         │                                           │
        │    [RAG off by default: no embedding computed]      │
        │         │                                           │
        │    db upsert (enriched_campaigns)                   │
        │         │                                           │
        │    record RowResult → ingest_row_results            │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
                    IngestSummary (per-row status)

  GET /campaigns            → SQL filters (channel / min_score / objective)
  GET /campaigns/{id}       → by PK, 404 if missing
  GET /campaigns/search?q=  → SQL ILIKE keyword  (pgvector cosine if RAG on)
  GET /campaigns/insights   → SQL aggregates → structured LLM (deterministic fallback)
```

## 2. Module map

| Module | Responsibility |
|--------|----------------|
| `app/config.py` | Env-driven settings (Groq key/model/base_url, DB URL, retries) |
| `app/schemas.py` | Canonical enums, strict `EnrichmentResult`, ingest/response models |
| `app/cleaning.py` | Deterministic parse / reject / flag (no LLM) |
| `app/llm.py` | Groq strict structured output + retry→rule-based fallback |
| `app/embeddings.py` | Lazy local MiniLM loader (guarded); text→vector |
| `app/rag.py` | Vector upsert, semantic search, insights (retrieve+generate) |
| `app/enrichment.py` | Orchestration: clean→dedupe→enrich→selfcheck→embed→persist |
| `app/db.py` | Engine/session + ORM (campaigns, ingest_runs, ingest_row_results) |
| `app/main.py` | FastAPI routes |
| `seed.py` | POST `data/campaigns_raw.json` to a running API |

## 3. Enrichment pattern — why NOT ReAct

Per-row enrichment is a **deterministic single-shot extraction**, not open-ended
reasoning/tool-use. A ReAct loop (reason→act→observe) would add latency, cost and
non-determinism for zero benefit. Instead: **one constrained JSON-schema call per
row** (`temperature=0`), validated by Pydantic, with retry→fallback→skip. ReAct-
style multi-step only appears (loosely) in `/insights`, and even there a single
structured call over retrieved context suffices.

Deterministic pre-cleaning does the mechanical work; the LLM only makes the
**judgment calls** (channel mapping, objective inference, health score).

## 4. RAG approach — comparison & decision (RAG DELIBERATELY OFF)

| Approach | Fit (14 rows) | Deps | Verdict |
|----------|---------------|------|---------|
| **SQL-aggregate insights + SQL keyword search** | High | none | **CHOSEN (default)** |
| pgvector + local embeddings (MiniLM) | Low — no retrieval need | torch, sentence-transformers, pgvector | kept behind `RAG_ENABLED` flag, off |
| External embeddings API (OpenAI/Cohere) | Low | extra API key | rejected — only a scoped Groq key is guaranteed |
| Groq embeddings | — | — | impossible — Groq has no embedding model |

**Decision:** RAG is the wrong tool here. This is **enrichment over
self-contained rows**, not Q&A over external knowledge — all information needed
to infer channel/objective/score is already in the input, so there is nothing to
retrieve. RAG would add latency + heavy deps for zero quality gain. It's kept
behind a flag purely to show it was evaluated.

- Default insights: SQL aggregates (spend/revenue/ROAS by channel & objective) →
  one strict structured LLM call → 2–3 observations; deterministic fallback.
- Default search: SQL `ILIKE` over name/description/channel.
- Flag path (`RAG_ENABLED=true`): pgvector cosine + retrieved context, embeddings
  from local MiniLM (384-dim) computed at ingest.

## 5. Data model

```
enriched_campaigns
  id (PK, business id)              name, raw_channel, description, currency
  spend, impressions, clicks, conversions, revenue   (cleaned, nullable)
  normalized_channel, inferred_objective             (canonical, indexed)
  health_score (indexed), health_rationale
  enrichment_source (llm|fallback), data_flags (jsonb)
  embedding vector(384)             (nullable; null ⇒ embeddings unavailable)
  created_at, updated_at

ingest_runs
  id (PK), created_at, total, ingested, skipped, failed, duplicates

ingest_row_results
  id (PK), run_id (FK), campaign_id, status, enrichment_source, reason, flags(jsonb)
```

Indexes: `normalized_channel`, `health_score` (back the required filters);
vector index on `embedding`.

## 6. Failure matrix (fallback for every stage)

| Stage | Failure | Detection | Fallback | Recorded as |
|-------|---------|-----------|----------|-------------|
| Cleaning | junk/test row, bad id/channel | rules | reject | `skipped` + reason |
| Cleaning | string/negative/null numbers | parse | coerce/null + flag | `ingested` + flags |
| Dedupe | repeated id in batch | set check | keep first | `duplicate` |
| LLM | invalid JSON / schema / API error | exception | retry → `fallback_enrich` | `ingested`, source=`fallback` |
| Self-check | LLM≠deterministic channel | comparison | keep LLM, flag mismatch | flag |
| Embeddings | model/pgvector unavailable | import guard | skip embed, null vector | flag + search degrades |
| DB | row-level error | try/except | isolate row | `failed` + reason |
| Search | no embeddings | availability flag | SQL `ILIKE` | — |
| Insights | LLM down | exception | deterministic aggregate summary | — |

## 7. `/insights` sequence

1. Aggregate from DB (spend/revenue/ROAS by channel & objective, count, avg score).
2. Retrieve top-k campaigns by a portfolio query embedding (context) — optional.
3. Build prompt with aggregates (+ context) → strict structured LLM call (2–3 observations).
4. Validate; on failure return deterministic aggregate summary.
