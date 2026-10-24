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
