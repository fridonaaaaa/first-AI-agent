# AI_USAGE.md

Log of what Claude Code generated each session. Owner fills in "Changed by owner:" lines.

---

## Session 2026-10-24

**Generated:**
- Full repo skeleton: folder structure, `__init__.py` files, `.gitignore`, `.env.example`
- `pyproject.toml` with initial dependencies
- `Dockerfile` (slim Python 3.12, non-root user, healthcheck)
- `docker-compose.yml` (app + pgvector/pgvector:pg16 DB)
- `app/core/settings.py` — typed Pydantic settings reading all env vars
- `app/core/errors.py` — typed exception hierarchy
- `app/core/logging.py` — structlog configuration (dev/prod renderer)
- `app/main.py` — FastAPI app with lifespan hook
- `app/api/routes/health.py` — GET /health endpoint
- `tests/test_health.py` — health check test
- `scripts/db/schema.sql` — full Postgres schema (documents, chunks, bm25_postings, bm25_stats, bm25_corpus)
- `scripts/build_subset.py` — seeded HotpotQA subset builder writing data/manifest.json
- `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/AI_USAGE.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `docs/EVAL_REPORT.md` — doc skeletons
- Spike script stubs: `scripts/spikes/spike_embeddings.py`, `scripts/spikes/spike_llm.py`

**Changed by owner:**

---

## Session 2026-10-25 (Phase 1 — Ingestion)

**Generated:**
- `app/ingestion/parsers.py` — Markdown/TXT, HTML (BeautifulSoup), PDF (pdfminer.six), JSONL, HotpotQA format parsers
- `app/ingestion/chunkers.py` — `fixed_token` and `sentence_window` chunking strategies
- `app/ingestion/tokeniser.py` — shared BM25 tokeniser (lowercase, strip punctuation, no stopwords)
- `app/ingestion/dedupe.py` — SimHash 64-bit near-duplicate detection (char-4-grams, Hamming ≤5); 148 lines [owner-defended]
- `app/ingestion/registry.py` — `add_document`, `update_document`, `delete_document` with BM25 stats consistency
- `scripts/load_hotpotqa.py` — loads HotpotQA validation split, seeds=42, 50 003 unique paragraphs
- `tests/test_parsers.py`, `tests/test_chunkers.py`, `tests/test_dedupe.py`, `tests/test_tokeniser.py`, `tests/test_registry.py`
- `docs/explain/dedupe.md` — algorithm walkthrough + 5 interview Q&A

**Changed by owner:**

---

## Session 2026-10-25 (Phase 2 — Retrieval)

**Generated:**
- `app/retrieval/embeddings.py` — lazy SentenceTransformer singleton, `encode()`, `encode_one()`
- `app/retrieval/vector_store.py` — `search_dense()` using pgvector cosine (`<=>`)
- `app/retrieval/bm25.py` — Okapi BM25, normalised TF stored in DB, IDF from `bm25_stats`; 147 lines [owner-defended]
- `app/retrieval/fusion.py` — Reciprocal Rank Fusion: score = Σ 1/(k + rank_i); 47 lines [owner-defended]
- `app/retrieval/rerank.py` — coverage-aware MMR reranker: λ·rel + (1-λ)·gain − dup_penalty; 136 lines [owner-defended]
- `app/retrieval/hybrid.py` — wires dense + BM25 + RRF + reranker into `hybrid_search()`
- `scripts/embed_corpus.py` — bulk embedding job (batch 64, resume-safe, --dry-run flag)
- `tests/test_embeddings.py`, `tests/test_bm25.py`, `tests/test_fusion.py`, `tests/test_rerank.py`, `tests/test_hybrid.py`
- `docs/explain/bm25.md`, `docs/explain/fusion.md`, `docs/explain/rerank.py.md`

**Changed by owner:**

---

## Session 2026-10-25 (Phase 3 — RAG Pipeline)

**Generated:**
- `app/rag/llm/base.py` — `LLMClient` ABC, `LLMResponse`, `complete_json()` (strips markdown fences)
- `app/rag/llm/mock.py` — `MockLLMClient` with scriptable responses dict, `calls` log
- `app/rag/llm/gemini.py` — Gemini REST client, 4-retry exponential backoff (2s base), SHA-256 disk cache
- `app/rag/llm/factory.py` — `get_llm_client()` from `LLM_PROVIDER` env var
- `app/rag/query_understanding.py` — Georgian detection, translate → rewrite → decompose (≤4 sub-queries)
- `app/rag/citation.py` — `extract_citations()`, `verify_citations()` (strips IDs not in retrieved set)
- `app/rag/conflict.py` — dup-group + numeric-pair detection, LLM structured verdict (agree/conflict/unrelated)
- `app/rag/abstention.py` — gate 1 (RRF score threshold), gate 2 (LLM sufficiency check)
- `app/rag/pipeline.py` — `run_pipeline()` orchestrating all 8 steps, `PipelineResult`
- `tests/test_query_understanding.py`, `tests/test_citation.py`, `tests/test_conflict.py`, `tests/test_abstention.py`, `tests/test_pipeline.py`

**Changed by owner:**

---

## Session 2026-10-25 (Phase 4 — API + UI)

**Generated:**
- `app/api/auth.py` — `require_admin()` using `hmac.compare_digest` (timing-attack safe)
- `app/api/middleware.py` — per-IP sliding-window rate limiter + global daily cap
- `app/api/routes/ask.py` — POST /ask with rate limiting and graceful error handling
- `app/api/routes/documents.py` — GET/POST/PUT/DELETE /documents, 1 MB text size limit
- `app/api/routes/health.py` — GET /health (updated with DB connectivity check)
- `app/main.py` (updated) — registers all routers, mounts `app/ui/` as static files
- `app/ui/index.html` — Georgian single-page UI (ask panel + admin panel)
- `app/ui/i18n.js` — all Georgian strings in one `I18N` object, `escHtml()` XSS guard
- `tests/test_auth.py`, `tests/test_middleware.py`, `tests/test_routes_ask.py`, `tests/test_routes_documents.py`

**Changed by owner:**

---

## Session 2026-10-26 (Phase 5 — Evaluation)

**Generated:**
- `eval/questions/dev.jsonl` — 15 DEV questions (simple, multi_hop, comparison, unanswerable, georgian)
- `eval/questions/test.jsonl` — 25 TEST questions (same categories, never tuned on)
- `eval/configs/c1_dense_only.json` — dense-only retrieval config
- `eval/configs/c2_hybrid_rrf.json` — Dense + BM25 + RRF config
- `eval/configs/c3_hybrid_rerank.json` — C2 + coverage reranker config
- `eval/configs/c4_full_pipeline.json` — full pipeline with query decomposition
- `scripts/eval_retrieval.py` — Recall@K and MRR (no LLM needed)
- `scripts/eval_rag.py` — Answer F1, EM, citation precision, abstention accuracy; saves result JSON
- `scripts/make_report.py` — generates `docs/EVAL_REPORT.md` from result JSON files
- `tests/test_eval.py` — 20 unit tests for metric helpers + question/config format validation

**Changed by owner:**

---

## Session 2026-10-26 (Phase 6 + Phase 7 — Docs + Deploy)

**Generated:**
- `README.md` — fully rewritten in Georgian with architecture diagram, usage, API table, eval instructions
- `docs/ARCHITECTURE.md` — full layer diagram, DB schema, 8-step pipeline, owner-defended module table, security section
- `docs/DEPLOYMENT.md` — complete deployment guide (Render free tier, Supabase, env vars, verification steps)
- `docs/EVAL_REPORT.md` — template with table structure (numbers filled by `scripts/make_report.py` after live runs)
- `docs/INTERVIEW_PREP.md` — interview Q&A drills for all 4 owner-defended modules + pipeline walkthrough
- `.dockerignore` — excludes .env, .venv, data/, *.jsonl, docs/, eval/results/
- `pyproject.toml` (updated) — added `sentence-transformers>=3.0`, `numpy>=1.26`
- `scripts/embed_corpus.py` — bulk embedding script (created as Phase 2 gap was identified in Phase 6 audit)

**Changed by owner:**

## Session 2026-10-27 — Pipeline fixes + UI redesign

### What AI generated
- `gemini.py`: corrected model to `gemini-3.1-flash-lite` (tested via models list API)
- `bm25.py`: added `_fetch_all_terms()` method; updated `search_index()` to use 2 batched
  queries for DB path instead of 2N separate round-trips per term
- `vector_store.py`, `hybrid.py`, `embed_corpus.py`: fixed `chunk_id` → `id` column name
- `db.py`: `_connect()` retry loop (4 attempts, 1.5s delay) for DNS failures
- `query_understanding.py`: try/except fallbacks on all 3 LLM calls
- `registry.py`, `load_hotpotqa.py`: batch inserts via execute_values()
- `index.html`: full UI redesign (indigo header, cards, score pills, error states)
- `i18n.js`: added timeoutMsg string
- `.env`: ABSTENTION_THRESHOLD=0.005 (calibrated to actual RRF score range)

Changed by owner:

---

## Session 2026-10-28

**Generated:**
- `app/retrieval/rerank.py`: added `doc_id` and `title` fields to `CandidateChunk`
- `app/retrieval/hybrid.py`: updated `_fetch_chunk_rows` to JOIN with `documents`; added `session_id` param to `hybrid_search`
- `app/retrieval/vector_store.py`: added `session_id` filter branch to `search_dense`
- `app/retrieval/bm25.py`: added `session_id` param to `_get_postings`, `_fetch_all_terms`, `search_index`, `search`
- `app/rag/pipeline.py`: threaded `session_id` through `run_pipeline` → `hybrid_search`
- `app/api/routes/ask.py`: added `session_id` to `AskRequest`; added `title` to `SourceInfo`
- `app/api/routes/documents.py`: added `session_id` to `UploadRequest` and file upload form field
- `app/ingestion/registry.py`: scoped dedup and INSERT to `session_id`
- `app/core/migrations.py`: new file — idempotent `ALTER TABLE` for `session_id` column
- `app/main.py`: calls `apply_migrations` at startup
- `app/ui/index.html`: full session management rewrite — per-session messages, files, UUID sessions, localStorage persistence
- `tests/test_api.py`, `tests/test_rag.py`: updated mocks to accept new `session_id` kwarg

Changed by owner:

---

## Session 2026-10-28 (Phase 2 bug fixes + Phase 4 eval questions + Phase 5 eval runs)

**Generated / modified:**
- `app/retrieval/embeddings.py`: switched to `gemini-embedding-001` (was `text-embedding-004`); fixed URL to `v1beta`; fixed payload to camelCase `outputDimensionality`; fixed API key loading via `get_settings()` not `os.getenv()`
- `app/retrieval/rerank.py`: added `cosine_sim` field to `CandidateChunk`
- `app/retrieval/hybrid.py`: populated `cosine_sim` from dense search scores
- `app/retrieval/vector_store.py`: added dimension guard before query
- `app/rag/abstention.py`: rewrote gate to use cosine similarity threshold (replaces dead RRF gate)
- `app/rag/pipeline.py`: threaded cosine gate through pipeline
- `app/core/settings.py`: added `abstention_threshold` field (default 0.25)
- `app/api/routes/documents.py`: added `_require_write_access()`, session-scoped uploads without admin token, `.jsonl` support, removed `.docx`
- `app/ingestion/registry.py`: fixed column order bug in `list_documents`, added `DocumentNotFoundError` raises
- `app/ui/index.html`: fixed `accept` attribute to match allowed extensions
- `scripts/embed_corpus.py`: added `--sleep` flag, 60s recovery retry
- `Dockerfile`: added `libgomp1` for fastembed ONNX Runtime
- `eval/questions/dev.jsonl`: 40 real HotpotQA multi-hop questions (full rewrite)
- `eval/questions/test.jsonl`: 20 real HotpotQA multi-hop questions (full rewrite)
- `scripts/eval_retrieval.py`: fixed SQL column names (`c.id`, `d.id`), skip abstain questions
- `scripts/eval_rag.py`: fixed field names (`gold_answer`, `expected_behavior`, `category`)
- `tests/test_rag.py`: updated for cosine_sim abstention logic
- `tests/test_api.py`: updated `test_upload_document_missing_token` to expect 401
- `tests/test_eval.py`: updated to check `category` not `type`
- `docs/STATUS.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `docs/EVAL_REPORT.md`: full update
- `docs/DECISIONS.md`: appended 3 new decisions

Changed by owner:


---

## Session 2026-10-28 (continued) — Step 2: embedding model mismatch guard

Generated:
- `app/retrieval/embeddings.py`: added `get_stored_model_name(conn)` function
- `app/retrieval/hybrid.py`: added model mismatch check in `_run()` with BM25-only fallback;
  imported `get_logger`, `get_stored_model_name`
- `app/rag/pipeline.py`: added `retrieval_mode` field to `PipelineResult`; added
  `_check_mode()` helper; imported `get_db`, `get_stored_model_name`
- `app/api/routes/ask.py`: added `retrieval_mode` field to `AskResponse`
- `tests/test_embeddings.py`: 3 new tests for `get_stored_model_name`
- `tests/test_rag.py`: 1 new test `test_pipeline_retrieval_mode_bm25_only`
- `docs/STATUS.md`: updated (134 tests, Step 2 done, next steps listed)
- `docs/DECISIONS.md`: appended mismatch guard decision

Changed by owner:

---

## Session 2026-10-29 (Final hardening pass)

**Generated:**
- `app/rag/tools.py` — tool schemas (Gemini functionDeclarations format) and executors for
  `search`, `get_document`, `compare_claims`; `execute_tool()` dispatcher with session isolation
- `app/rag/llm/base.py` — added `complete_with_tools()` default implementation
- `app/rag/llm/gemini.py` — implemented `complete_with_tools()` using Gemini function-calling API;
  added `_messages_to_gemini()` and `_parse_tool_response()` helpers
- `app/rag/llm/mock.py` — added `complete_with_tools()` with scripted tool-call support for tests
- `app/rag/pipeline.py` — added `tool_trace` field to `PipelineResult`; replaced plain `llm.complete()`
  answer-generation with bounded tool loop using `complete_with_tools()`
- `app/api/routes/ask.py` — added `tool_trace` field to `AskResponse`
- `tests/test_tools.py` — 20 new tests: schema validation, execute_tool dispatching,
  MockLLMClient tool scripting, pipeline integration (tool_trace populated/empty),
  iteration cap, session isolation
- `eval/questions/test.jsonl` — added 10 questions (test-031 to test-040) to reach 40 total
- `docs/EVAL_REPORT.md` — updated to reflect 63 dev + 40 test, added chunking comparison section,
  error analysis, and metric limitations
- `docs/STATUS.md` — updated with current state of all steps
- `FINAL_CHECKLIST.md` — requirement-to-evidence matrix for all assignment requirements
- Appended to `docs/DECISIONS.md`: tool-calling architecture decision + test-split expansion

**Changed by owner:**
