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

---

## 2026-10-26 — Coverage reranker: dup_penalty scaling

**Decision:** Scale `dup_penalty` by `(1 − λ)` and zero it when `n_queries = 0`.

**Why:** The dup_penalty exists solely to protect coverage diversity — it prevents
selecting a near-duplicate when a different chunk could cover a new sub-query.
When λ = 1.0 (pure relevance mode) or no sub-queries are given, there is no
coverage goal, so the penalty would incorrectly demote near-duplicates in
pure-relevance ranking.

**Alternatives rejected:** Applying a fixed penalty always — broke `test_lambda_1_is_pure_relevance`
and `test_no_sub_queries_preserves_rrf_order`.

**Changed by owner:**

---

## 2026-10-26 — Coverage reranker: tie-breaking by (score, coverage_gain)

**Decision:** Use tuple comparison `(score, gain) > (best_score, best_gain)` in the
greedy loop so equal-score candidates are broken by coverage_gain.

**Why:** At default λ = 0.5, the second-hop chunk C (rrf=0.3, covers new sq2) tied
with chunk B (rrf=0.8, covers nothing new) at combined score 0.40. Without
tie-breaking, B won by list position, defeating the purpose of the reranker.
With the tie-break, C wins because gain=0.5 > gain=0.

**Alternatives rejected:** Lowering default λ below 0.5 — changes the semantics for
all queries, not just the edge case. Adding a tiny ε to gain — more obscure than a
clean tuple comparison.

**Changed by owner:**

---

## 2026-10-26 — Phase 2 retrieval modules complete

**Decision:** Retrieval pipeline is:
  1. `embeddings.py` — lazy-loaded SentenceTransformer singleton, returns `list[float]`
  2. `vector_store.py` — pgvector `<=>` cosine distance, returns similarity = 1 − distance
  3. `hybrid.py` — wires dense + BM25 + RRF + coverage rerank into one `hybrid_search()` call

**Why these boundaries:** Each module has one job. `hybrid.py` owns zero algorithm
logic — it only calls the four specialist modules and converts between their types.
This makes each piece testable and explainable in isolation.

**Alternatives rejected:** Single "retriever" class with all logic — violates single-purpose
rule and makes interview explanation harder.

**Changed by owner:**

---

## 2026-10-26 — Phase 6+7 documentation and Docker

**Decision:** Dockerfile changed from `pip install -e ".[dev]"` to `pip install ".[dev]"`.

**Why:** Editable installs require the source tree to be present and on `sys.path` as a
live symlink. In a Docker container there is no development workflow — the code is copied in
and never edited. A non-editable install is smaller, faster, and correct.

**Alternatives rejected:** Keeping `-e` — it works but is semantically wrong for a container.

---

## 2026-10-26 — Render chosen as deployment target

**Decision:** Render free Web Service (Docker runtime) as primary deployment target for Phase 7.

**Why:** Render supports Docker natively on its free tier, has environment variable injection
via dashboard (secrets never in image), auto-deploys on push, and provides a public HTTPS URL.
Hugging Face Spaces Docker was also viable but has a 16 GB storage limit that the embedding
model and corpus might approach.

**Alternatives rejected:** Railway (free tier removed), Fly.io (requires credit card for free tier).

**Changed by owner:**

---

## 2026-10-28 — Per-chat session isolation + source document names

**Decision:** Added three UX features:

1. **Source document names in "წყაროები"**: Added `doc_id` and `title` fields to `CandidateChunk`; joined `chunks` with `documents` in `_fetch_chunk_rows`; `SourceInfo` now carries `title`; UI shows the document name as the source header.

2. **Per-chat document isolation**: Added `session_id text` column to `documents` (applied via idempotent migration in `app/core/migrations.py` on startup). Documents uploaded in a chat are tagged with the chat's UUID. Search filters `(session_id IS NULL OR session_id = ?)` so each chat sees the global corpus plus only its own uploads.

3. **Full session management in UI**: Replaced the Q&A history array with proper session objects `{id, label, messages, files, activeDocId}`, each with a UUID. "ახალი საუბარი" creates a new session. Sessions are persisted to localStorage under `ai_sessions_v2`. Switching sessions swaps the chat view and the files list.

**Why `session_id` on `documents` rather than `chunks`:** Avoids denormalization; the join with the small `documents` table is cheap. Dedup scoped per-session so two chats can independently upload the same file.

**Alternatives rejected:** Storing `session_id` on chunks (redundant, harder to migrate); server-side session store (no server state needed — client UUID is sufficient).

**Changed by owner:**
