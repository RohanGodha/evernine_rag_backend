# Tasks to Build

Milestone checklist. Each task notes its **fallback** and current **status**.
Legend: [x] done · [~] in progress · [ ] pending

---

## M0 — Scaffold  [x]
- [x] Repo structure (`app/`, `tests/`, `data/`)
- [x] `requirements.txt`, `.env.example`, `.gitignore`
- [x] `docker-compose.yml` (Postgres)
- [x] Git init + remote `RohanGodha/evernine_rag_backend`, branch `main`, GitHub connected

## M1 — Schemas  [x]
- [x] Canonical `Channel` / `Objective` enums
- [x] Strict `EnrichmentResult` (extra=forbid → Groq strict mode)
- [x] `RawCampaign` / `IngestPayload` (permissive input)
- [x] `RowResult` / `IngestSummary` (per-row reporting)

## M2 — Cleaning  [x]
- [x] Number parsing (`"42,500"`, currency, junk tokens)
- [x] Junk/test row rejection; unusable channel rejection
- [x] Negative-value nulling + flags; missing/blank flags
- **Fallback:** soft issues flagged (not fatal); only unrecoverable rows rejected

## M3 — Persistence  [~]
- [x] Engine/session, ORM models, autocreate `init_db()`
- [x] Indexes on `normalized_channel`, `health_score`
- [ ] Add `embedding vector(384)` column
- [ ] Enable `vector` extension in `init_db()` (guarded)
- [ ] Vector index (cosine)
- **Fallback:** if `vector` extension missing → column/index skipped, app still runs

## M4 — LLM enrichment  [x]
- [x] Groq client (OpenAI-compatible base_url)
- [x] Strict `json_schema` structured output, `temperature=0`
- [x] Deterministic-first channel: static lookup → LLM partial (objective+score);
      miss → LLM full (channel+objective+score)
- [x] Retry → rule-based `fallback_enrich` (channel/objective/ROAS score)
- **Fallback:** never raises; degrades to deterministic enrichment

## M5 — Orchestration  [x]
- [x] clean → dedupe (keep first) → enrich → upsert
- [x] Per-row isolation (unexpected error → `failed`)
- [x] Persist run + row outcomes
- [ ] Insert self-check + embedding steps (M7/M10)

## M6 — API core  [x]
- [x] `POST /campaigns/ingest` (per-row summary)
- [x] `GET /campaigns` (channel / min_score / objective filters)
- [x] `GET /campaigns/{id}` (404 if missing)
- [x] `GET /health`

## M7 — Embeddings (behind flag)  [x]
- [x] `app/embeddings.py`: lazy-load `all-MiniLM-L6-v2` (384-dim), import guard
- [x] Gated behind `RAG_ENABLED` (off by default — not needed for enrichment)

## M8 — RAG (behind flag) / insights  [x]
- [x] `GET /campaigns/insights`: SQL aggregates → strict LLM; **fallback:** deterministic aggregate summary (NO retrieval by default)
- [x] `search(q, k)`: SQL `ILIKE` keyword by default; pgvector cosine only if `RAG_ENABLED`
- [x] Decision recorded: RAG deliberately off (enrichment ≠ Q&A over external knowledge)

## M9 — API stretch  [x]
- [x] `GET /campaigns/search?q=&k=` (keyword)
- [x] `GET /campaigns/insights`

## M10 — Self-check / eval  [x]
- [x] When LLM classifies channel, compare vs deterministic `fallback_channel`
- [x] Flag `channel_selfcheck_mismatch` / `channel_resolved_static`, surfaced in ingest flags
- **Fallback:** disagreement is informational, never blocks ingest

## M11 — Tests  [ ]
- [ ] Cleaning: comma-number, junk reject, negative null, dedupe
- [ ] Fallback enrichment: channel/objective/score heuristics
- [ ] RAG degradation: embeddings-unavailable → SQL path

## M12 — README + verify  [ ]
- [ ] `README.md`: how to run, design choices, **AI & prompt choices**, RAG rationale, "with more time"
- [ ] `docker-compose up` → run API → `python seed.py` → exercise endpoints
- [ ] Final incremental commits pushed to GitHub
