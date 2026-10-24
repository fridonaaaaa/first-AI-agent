# DECISIONS.md

Each entry: what was decided, why, and which alternatives were rejected.

---

## 2026-10-24 — Initial dependencies

**Decision:** Added the following packages to pyproject.toml:
- `fastapi>=0.115` + `uvicorn[standard]>=0.32` — web framework required by architecture
- `pydantic>=2.9` + `pydantic-settings>=2.6` — typed settings and request/response schemas
- `structlog>=24.4` — structured JSON logging in production, readable dev renderer in development
- `psycopg2-binary>=2.9` — PostgreSQL sync driver (async driver asyncpg to be considered in Phase 2)
- `pgvector>=0.3` — pgvector Python client for ANN search
- `python-dotenv>=1.0` — .env loading in non-Docker local runs
- `httpx>=0.27` — async HTTP client + FastAPI TestClient
- `datasets>=3.0` — HuggingFace Datasets library for HotpotQA download in build_subset.py

**Alternatives rejected:**
- `flask` — FastAPI chosen because async support and Pydantic schemas are required
- `asyncpg` — deferred; psycopg2 simpler for initial skeleton; will revisit if async DB calls are needed
- `SQLAlchemy` — not added; raw psycopg2 is sufficient and avoids ORM abstraction the owner must explain

**Changed by owner:**

---

## 2026-10-24 — Branch name

**Decision:** Default branch named `main`.

**Changed by owner:**

---

## 2026-10-24 — DB schema design

**Decision:** Five tables: `documents`, `chunks`, `bm25_postings`, `bm25_stats`, `bm25_corpus`.
- `bm25_postings` stores (term, chunk_id, tf). BM25 scoring runs in Python over these rows — no ts_rank, no tsvector.
- `bm25_corpus` is a singleton row tracking total_docs, avg_doc_len, and the embedding model name (guards against model mismatch).
- `dup_group` on chunks links near-duplicates without deleting them (PLAN.md requirement).
- Vector dimension hard-coded to 384 in schema comment; actual column created with chosen dimension in Phase 2 migration after model is confirmed.

**Alternatives rejected:**
- Separate tables per chunking strategy — rejected; strategy column on chunks is simpler and allows cross-strategy queries.
- Elasticsearch for BM25 — rejected; assignment requires own implementation.

**Changed by owner:**

---

## 2026-10-24 — LLM provider: Gemini

**Decision:** `LLM_PROVIDER=gemini`. Owner confirmed Gemini as the free LLM provider.

**UNVERIFIED items (owner must check before Phase 3):**
- Current free-tier model name (check ai.google.dev — likely `gemini-1.5-flash` or `gemini-2.0-flash`).
- Current free-tier rate limits (RPM / TPD). Paste the source URL here once checked.
- Tool-calling (function calling) support on the free model — required for Phase 3 tool loop.

**Alternatives rejected:** Groq (owner chose Gemini).

**Changed by owner:**

---

## 2026-10-24 — Corpus size (Spike C: storage math)

**Measured:** `scripts/build_subset.py` with seed=42 produced **50,003 unique paragraphs** from 5,420 of 7,405 validation questions. Train split untouched.

**Storage estimate (Spike C):**

Assumptions: ~2 chunks per paragraph on average (short HotpotQA paragraphs, ~150–250 tokens). Estimated 100,000 chunks total.

| Component | 384-dim | 768-dim |
|---|---|---|
| Vectors (100k x dims x 4 bytes) | 153 MB | 307 MB |
| Text (100k chunks x ~300 bytes avg) | 30 MB | 30 MB |
| BM25 postings (100k x ~30 terms x 28 bytes) | 84 MB | 84 MB |
| Stats + metadata + indexes | ~20 MB | ~20 MB |
| **Total** | **~287 MB** | **~441 MB** |

**Free DB storage caps — UNVERIFIED. Owner must check before Phase 2:**
- Neon: neon.tech/pricing
- Supabase: supabase.com/pricing

**Recommendation:** Use 384-dim embedding model. ~287 MB fits under a 512 MB free cap with headroom. 768-dim (~441 MB) is too tight.

**Changed by owner:**

---

## 2026-10-24 — DB host: Supabase

**Decision:** Use Supabase free tier for Postgres + pgvector.

**Owner must verify before Phase 2:** Current free storage cap at supabase.com/pricing. Estimated DB size ~287 MB (384-dim). Confirm it fits.

**Connection string format** (goes in `.env` only, never committed):
`DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`

Find it in: Supabase dashboard → Project Settings → Database → Connection string → URI.

**Alternatives rejected:** Neon (owner chose Supabase).

**Changed by owner:**
