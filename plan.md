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
        │    llm.enrich  ──(retry)──► fallback_enrich         │  ← structured output
        │         │                                           │
        │    self-check (channel vs deterministic) → flag     │
        │         │                                           │
        │    embeddings.encode ──(guard)──► skip+flag         │  ← local MiniLM
        │         │                                           │
        │    db upsert (enriched_campaigns + vector)          │
        │         │                                           │
        │    record RowResult → ingest_row_results            │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
                    IngestSummary (per-row status)

  GET /campaigns            → SQL filters (channel / min_score / objective)
  GET /campaigns/{id}       → by PK, 404 if missing
  GET /campaigns/search?q=  → embed q → pgvector cosine  (fallback: SQL ILIKE)
  GET /campaigns/insights   → RAG: top-k + aggregates → LLM (fallback: SQL only)
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

## 4. RAG approach — comparison & decision

| Approach | Fit (14 rows) | Deps | Verdict |
|----------|---------------|------|---------|
| SQL-aggregate only | High | none | **fallback path** |
| **pgvector + local embeddings (MiniLM)** | Medium (demonstrative) | torch, sentence-transformers, pgvector | **CHOSEN** |
| External embeddings API (OpenAI/Cohere) | Medium | extra API key | rejected — only a scoped Groq key is guaranteed |
| Groq embeddings | — | — | impossible — Groq has no embedding model |

**Rationale:** the repo is a RAG backend and the exercise rewards demonstrating
retrieval-augmented generation. Local embeddings keep it **self-contained** (no
extra key), and a **graceful SQL fallback** keeps it robust for the live demo.
Honest caveat (ownership talking point): at 14 rows the semantic gain over SQL is
small; the value is demonstrating the pattern correctly and knowing when it's
overkill.

- Embedding text = `f"{name}. {description}. channel={normalized_channel} objective={inferred_objective}"`.
- Distance = cosine (`vector_cosine_ops`); index = ivfflat/HNSW (small N → seq scan is fine, index added for correctness).
- Embeddings computed **at ingest time**, after enrichment, stored on the campaign row.

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
