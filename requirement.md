# Requirements — Aster & Oak Campaign Enrichment

**Context:** Evernine Brands · Hive practical exercise (Full Stack + AI Engineer).
Ingest a messy marketing-campaign export for the fictional D2C skincare brand
*Aster & Oak*, enrich each campaign with an LLM using **structured output**,
persist to **PostgreSQL**, and serve it back over a **FastAPI** backend.
No frontend. LLM access is a **scoped Groq key** (OpenAI-compatible API).

Mock input: `data/campaigns_raw.json` (copied from `campaigns_raw.json`).

Every capability below lists an explicit **fallback** so a single bad row or a
misbehaving model never sinks the batch.

---

## Core requirements (must-have)

| # | Requirement | Where | Fallback |
|---|-------------|-------|----------|
| 1 | **Ingestion** — `POST /campaigns/ingest` reads raw campaigns, processes each | `main.py`, `enrichment.py` | Per-row isolation; any unexpected error → row marked `failed`, batch continues |
| 2 | **LLM enrichment (structured output)** — normalized_channel, inferred_objective, health_score(0–100)+rationale via JSON-schema (not free-text) | `llm.py`, `schemas.py` | `temperature=0` strict `json_schema`; on invalid/parse/API error: retry ×N → deterministic `fallback_enrich` (rule-based channel/objective + ROAS heuristic score) |
| 3 | **Persistence** — store enriched campaigns in Postgres, self-designed schema | `db.py` | `init_db()` autocreates tables (+ `vector` extension guarded) |
| 4 | **Retrieval** — `GET /campaigns` with ≥1 filter (channel / min_score), `GET /campaigns/{id}` | `main.py` | Missing id → HTTP 404; empty filters → returns all (paginated) |
| 5 | **Failure handling** — validate inputs, handle model failures (retry/fallback/skip), record which rows failed and why; per-row success/failure summary | `cleaning.py`, `enrichment.py`, `db.py` | Outcomes persisted to `ingest_runs` + `ingest_row_results`; statuses: ingested / skipped / failed / duplicate |

### Cleaning / reject / flag rules (no single "correct" cleaning — decisions are explicit)

| Case (from mock data) | Decision |
|-----------------------|----------|
| `spend: "42,500"` (cmp_002) | Coerce → 42500, flag `spend_coerced_from_string` |
| `spend: 0` (cmp_006 email) | Valid zero (owned channel), **not** missing |
| `spend: -1200` (cmp_014), `clicks: -5` (cmp_013) | Null the impossible negative, flag `negative_*_nulled` |
| `impressions: null` (cmp_009, cmp_012) | Keep row, flag `impressions_missing` |
| blank `name` (cmp_011) | Keep, flag `name_blank`; LLM infers objective from description |
| `"DO NOT USE (draft row)"`, `spend "N/A"`, `impressions "lots"` (cmp_013) | **Reject** → `skipped`, reason `junk_or_test_row` |
| channel `"???"` (cmp_013) | Unusable channel → reject |
| duplicate `id` cmp_008 | Keep first, second → `duplicate`, reason `duplicate_id_in_batch` |
| all metrics null (cmp_012 influencer) | Keep, flags; LLM/fallback scores low health (no tracking) |

---

## Stretch requirements (if time)

| Requirement | Where | Fallback |
|-------------|-------|----------|
| **Portfolio insights** — `GET /campaigns/insights` aggregates from DB + asks LLM for 2–3 observations | `rag.py`, `main.py` | RAG context (pgvector top-k) + SQL aggregates → LLM; if embeddings unavailable → SQL aggregates only; if LLM down → deterministic aggregate summary |
| **Semantic search** — `GET /campaigns/search?q=` over campaigns (vector RAG) | `rag.py`, `embeddings.py`, `main.py` | Embed query → pgvector cosine; if embeddings/pgvector unavailable → SQL `ILIKE` keyword match |
| **Lightweight eval / self-check** — verify one enriched field against a deterministic rule; flag disagreements | `enrichment.py` | Cross-check LLM `normalized_channel` vs deterministic `fallback_channel`; disagreement → flag `channel_selfcheck_mismatch` |
| **Idempotent re-ingest** — running twice creates no duplicates | `enrichment.py`, `db.py` | PK = business `id` upsert + in-batch dedupe |
| **Tests** — around parsing / validation logic | `tests/` | pytest: cleaning, fallback, RAG degradation |

---

## RAG constraint note

Groq offers **no text-embedding model** (chat/reasoning + speech only). Therefore
vector RAG uses **local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim)** for
embeddings while Groq handles generation. `sentence-transformers`/`pgvector` are
imported behind a guard: if unavailable at runtime the system degrades to SQL
keyword search and SQL-aggregate insights, and the app still runs fully.

---

## Non-functional

- **Secrets** only via env (`.env` gitignored; `.env.example` provided). Never hardcoded.
- **Determinism** where possible (`temperature=0`, deterministic cleaning/fallback).
- **Per-row isolation** — the batch is never aborted by one row.
- **Idempotency** — safe to re-ingest.
- **Auditability** — every row's outcome + reason + flags persisted.

---

## Assessment mapping (how each requirement earns marks)

- **AI/LLM judgment** → req 2, self-check, insights (schema + prompt + structured output + graceful failure).
- **Backend fundamentals** → req 1,3,4 + data model + status codes + error handling.
- **Engineering craft** → module boundaries, no secrets, tests, incremental commits, README.
- **Ownership & communication** → this doc + `plan.md` + code comments = defensible choices.
