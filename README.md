# Aster & Oak — Campaign Enrichment API

A vertical slice for the Evernine · Hive exercise: ingest a messy marketing
campaign export, enrich each campaign with an LLM using **structured output**,
persist to **PostgreSQL**, and serve it back over **FastAPI**. Includes an
optional **vector-RAG** layer (semantic search + portfolio insights). No frontend.

- **Backend:** FastAPI
- **LLM:** Groq (OpenAI-compatible API), strict JSON-schema structured output
- **DB:** PostgreSQL (+ pgvector for semantic search)
- **Embeddings:** local `sentence-transformers` (Groq has no embedding model)

See [`requirement.md`](requirement.md), [`plan.md`](plan.md), and
[`tasks2build.md`](tasks2build.md) for full traceability and design notes.

---

## How to run

### 1. Prerequisites
- Docker (for Postgres + pgvector), Python 3.11.

### 2. Configure
```bash
cp .env.example .env
# edit .env: set GROQ_API_KEY (scoped key). Defaults work for local Postgres.
```

### 3. Start Postgres
```bash
docker compose up -d          # pgvector/pgvector:pg16 on localhost:5432
```

### 4. Install + run the API
```bash
python -m venv .venv
.venv\Scripts\activate         # Windows  (source .venv/bin/activate on *nix)
pip install -r requirements.txt
uvicorn app.main:app --reload  # http://localhost:8000  (Swagger at /docs)
```

### 5. Ingest the sample data + explore
```bash
python seed.py                                   # POSTs data/campaigns_raw.json
curl "http://localhost:8000/campaigns?min_score=70"
curl "http://localhost:8000/campaigns?channel=Meta"
curl "http://localhost:8000/campaigns/cmp_001"
curl "http://localhost:8000/campaigns/search?q=retargeting%20cart"
curl "http://localhost:8000/campaigns/insights"
```

### Tests
```bash
pytest            # cleaning, fallback enrichment, RAG degradation
```

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/campaigns/ingest` | Ingest raw campaigns; returns per-row success/failure summary |
| GET | `/campaigns?channel=&min_score=&objective=` | List enriched campaigns (filters) |
| GET | `/campaigns/{id}` | One campaign (404 if missing) |
| GET | `/campaigns/search?q=&k=` | Semantic search (vector; SQL fallback) |
| GET | `/campaigns/insights` | Portfolio observations (RAG + LLM; deterministic fallback) |
| GET | `/health` | Liveness |

---

## Key design choices

- **Deterministic clean, then LLM judgment.** Mechanical fixes (parse `"42,500"`,
  reject junk/test rows, dedupe ids, null impossible negatives) happen in
  `cleaning.py`. The LLM only makes *judgment* calls: channel normalization,
  objective inference, health score. There's no single "correct" cleaning, so
  every reject has a `reason` and every soft fix a `flag`, all persisted.
- **Per-row isolation.** One bad row or bad model response never sinks the batch.
  Each row resolves to `ingested` / `skipped` / `failed` / `duplicate`, recorded
  in `ingest_runs` + `ingest_row_results`.
- **Idempotent re-ingest.** PK = the source's business `id`; re-ingesting upserts
  in place and reports in-batch repeats as `duplicate` (see cmp_008).
- **Schema-designed persistence.** Canonical enums back the required filters,
  with indexes on `normalized_channel` and `health_score`.

## AI & prompt choices

- **What I ask the model:** for each cleaned campaign, return four fields —
  `normalized_channel` (mapped onto a canonical enum), `inferred_objective`
  (awareness/consideration/conversion/retention), `health_score` (0–100), and a
  one-line `health_rationale`.
- **How the output is structured (and why):** Groq **strict `json_schema`**
  structured output (constrained decoding, `temperature=0`). The model is
  token-constrained to a Pydantic-derived schema, so it *cannot* emit an invalid
  channel/objective or non-JSON — this is the brief's "structured output, not
  free-text parsing". The canonical vocabulary lives in *my* enums, not the
  model's imagination.
- **Deterministic-first channel resolution:** a static lookup table
  (`resolve_channel_static`) maps known variants (fb, IG, GAds, adwords, yt …)
  to the canonical channel deterministically. On a **hit**, the channel is fixed
  and the LLM only enriches `inferred_objective` + `health_score` (partial
  schema) — consistent, cheaper, and the LLM can't misclassify a known channel.
  On a **miss** (unseen variant), the LLM classifies the channel too (full
  schema). Either way, the brief's "use the LLM for each valid campaign" holds,
  since objective + score always require the model. Each row records
  `channel_resolved_static` or a `channel_selfcheck_mismatch` flag.
- **Why not a ReAct/agent loop:** per-row enrichment is a deterministic
  single-shot extraction. A reason→act→observe loop would add latency, cost and
  non-determinism for no benefit. One constrained call per row is the right tool.
  (Agentic looping would only make sense for aggregation, i.e. `/insights`.)
- **How I handle model failure:** retry transient errors, then fall back to a
  deterministic rule-based enrichment (`fallback_enrich`: keyword channel map,
  objective heuristics, ROAS + data-completeness score). The row still lands,
  tagged `enrichment_source="fallback"`. A **self-check** also cross-checks the
  LLM's channel against the deterministic rule and flags disagreements
  (`channel_selfcheck_mismatch`).

## RAG choice (deliberately NOT used)

**I chose not to use RAG for this task**, and this is a considered decision:

- This is an **enrichment** task, not question-answering over external knowledge.
  Everything needed to infer channel, objective, and health score is **already
  in the campaign row** — there's nothing to retrieve.
- RAG earns its keep when the model needs company-specific documents/knowledge
  it doesn't have. Not the case here.
- For ~14 self-contained rows, pgvector + embeddings would add latency, a heavy
  `torch` dependency, and complexity for **zero quality gain** — over-engineering.

The vector-RAG layer (`app/embeddings.py`, `app/rag.py`, pgvector column) is kept
in the repo **behind a flag (`RAG_ENABLED=false`, off by default)** to show it was
explored — not because the task needs it. With RAG off:

- `GET /campaigns/search` is plain SQL keyword (`ILIKE`) search.
- `GET /campaigns/insights` is **SQL aggregation → one structured LLM call**, with
  a deterministic aggregate fallback. No retrieval.

Enabling RAG (`RAG_ENABLED=true` + `pip install -r requirements-rag.txt`) switches
search to pgvector cosine and adds retrieved context to insights — available to
demonstrate, but intentionally not the default.

## With more time

- Alembic migrations instead of `create_all`.
- Async LLM calls with bounded concurrency for larger batches.
- Confidence scores + human-review queue for low-confidence enrichments.
- Richer eval: cross-check objective/score against rules, track agreement rate.
- Dockerize the API itself for a one-command demo.
