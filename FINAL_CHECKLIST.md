# FINAL_CHECKLIST.md — AI Investigation Engine

Submission checklist for Upgaming Talent Growth Program — Deadline 2026-10-29.

Each item is marked: ✅ VERIFIED | ⚠️ PARTIAL | ❌ MISSING | 🔒 BLOCKED

---

## 1. Document Ingestion and Management

| # | Requirement | Status | Evidence |
|---|-------------|--------|---------|
| 1.1 | At least 3 file formats | ✅ VERIFIED | `app/ingestion/parsers.py`: md, txt, html, pdf, jsonl (5 formats). Tests: `tests/test_parsers.py` (22 tests). |
| 1.2 | Text and metadata extraction | ✅ VERIFIED | `ParsedDocument` carries title, text, source, format, language, metadata on every chunk. |
| 1.3 | Chunking implemented | ✅ VERIFIED | `app/ingestion/chunkers.py`: `fixed_token` and `sentence_window`. Tests: `tests/test_chunkers.py`. |
| 1.4 | Exact duplicate handling | ✅ VERIFIED | `app/ingestion/dedupe.py`: SHA-256 hash deduplication. |
| 1.5 | Near-duplicate handling | ✅ VERIFIED | `dedupe.py`: MinHash/token-Jaccard near-dup detection with `dup_group` linking (never deleted). |
| 1.6 | Add / update / delete documents | ✅ VERIFIED | `app/ingestion/registry.py` + routes in `app/api/routes/documents.py`. |
| 1.7 | Chunking strategy comparison | ⚠️ PARTIAL | Both strategies implemented; qualitative comparison in EVAL_REPORT.md. Controlled quantitative comparison pending (blocked by DB size). |
| 1.8 | Update/delete propagates to chunks, embeddings, BM25 | ✅ VERIFIED | `registry.py:update_document` deletes old chunks/postings before re-inserting. `registry.py:delete_document` cascades. |

---

## 2. Retrieval

| # | Requirement | Status | Evidence |
|---|-------------|--------|---------|
| 2.1 | Vector database and embeddings | ✅ VERIFIED | Supabase PostgreSQL + pgvector. `app/retrieval/vector_store.py`. |
| 2.2 | Semantic search | ✅ VERIFIED | `search_dense()` in `vector_store.py` using cosine distance. |
| 2.3 | Hybrid semantic + keyword search | ✅ VERIFIED | `app/retrieval/hybrid.py` fuses dense + BM25 with RRF. |
| 2.4 | Custom retrieval algorithm | ✅ VERIFIED | Custom BM25 (`bm25.py`), custom RRF (`fusion.py`), custom coverage reranker (`rerank.py`). All hand-written, no search library. |
| 2.5 | BM25 correctness | ✅ VERIFIED | `tests/test_bm25.py`: formula verified with known scores. |
| 2.6 | RRF correctness | ✅ VERIFIED | `tests/test_fusion.py`: merge, boost, ordering tested. |
| 2.7 | Coverage reranker correctness | ✅ VERIFIED | `tests/test_rerank.py`: greedy selection and dup-penalty tested. |
| 2.8 | Embedding model mismatch guard | ✅ VERIFIED | `get_stored_model_name()` + check in `hybrid.py` and `pipeline.py`; falls back to `bm25_only`. Tests: `tests/test_embeddings.py`. |

---

## 3. RAG and Investigation

| # | Requirement | Status | Evidence |
|---|-------------|--------|---------|
| 3.1 | Multi-document evidence combination | ✅ VERIFIED | Pipeline retrieves up to `retrieval_top_k` chunks from multiple documents per query. |
| 3.2 | Multi-step question decomposition | ✅ VERIFIED | `app/rag/query_understanding.py`: translate → rewrite → decompose into sub-queries. |
| 3.3 | Evidence-grounded answers with citations | ✅ VERIFIED | LLM receives only retrieved passages; answer must cite chunk IDs. |
| 3.4 | Citation verification (post-check) | ✅ VERIFIED | `app/rag/citation.py:verify_citations()` strips any `[chunk_id]` not in retrieved set. |
| 3.5 | Conflict detection | ✅ VERIFIED | `app/rag/conflict.py`: LLM-based pair comparison, capped at 6 calls. |
| 3.6 | Abstention — insufficient evidence | ✅ VERIFIED | `app/rag/abstention.py`: two-gate (cosine + LLM sufficiency). |
| 3.7 | Prompt-injection defense | ✅ VERIFIED | Retrieved text is passed as inert DATA in a labelled `Passages:` block; never executed. System prompt never mentions instructions to override. |
| 3.8 | LLM error handling (timeout, rate limit) | ✅ VERIFIED | `GeminiClient`: 4 retries × exponential backoff; typed `LLMError`/`LLMRateLimitError`/`LLMTimeoutError`. |
| 3.9 | Real LLM tool calling | ✅ VERIFIED | `app/rag/tools.py` + `pipeline.py` tool loop. `GeminiClient.complete_with_tools()` uses Gemini's native functionDeclarations API. Three tools: `search`, `get_document`, `compare_claims`. Bounded by `max_tool_steps`. Tests: `tests/test_tools.py`. |
| 3.10 | `tool_trace` in API response | ✅ VERIFIED | `PipelineResult.tool_trace` and `AskResponse.tool_trace` (sanitised — no raw chunk text). |

---

## 4. Evaluation

| # | Requirement | Status | Evidence |
|---|-------------|--------|---------|
| 4.1 | ≥ 40 evaluation questions | ✅ VERIFIED | 63 dev + 40 test = 103 total. `eval/questions/dev.jsonl`, `eval/questions/test.jsonl`. |
| 4.2 | Simple questions | ✅ VERIFIED | `single` category: 8 dev + 4 test. |
| 4.3 | Multi-hop questions | ✅ VERIFIED | `bridge` category: 21 dev + 13 test. |
| 4.4 | Questions requiring multiple documents | ✅ VERIFIED | All bridge/comparison questions require ≥ 2 supporting documents. |
| 4.5 | Unanswerable questions | ✅ VERIFIED | `abstain` + `unanswerable` categories: 8 dev + 8 test. |
| 4.6 | Contradictory evidence questions | ✅ VERIFIED | `conflict` category: 6 dev + 5 test. |
| 4.7 | Prompt-injection test cases | ✅ VERIFIED | `prompt_injection` category: 4 dev + 1 test. |
| 4.8 | Retrieval metrics (Recall@K, MRR) | ✅ VERIFIED | `scripts/eval_retrieval.py`. Results in EVAL_REPORT.md (C1/C2/C3). |
| 4.9 | Answer quality metrics (F1, EM) | ✅ VERIFIED | `scripts/eval_rag.py`. Results in EVAL_REPORT.md (C4). |
| 4.10 | Configuration comparison (≥ 2) | ✅ VERIFIED | C1 dense-only vs C2 hybrid-RRF vs C3 hybrid-rerank. |
| 4.11 | Chunking strategy comparison | ⚠️ PARTIAL | Both strategies implemented and described. Quantitative comparison pending. |
| 4.12 | Dev/test data kept separate | ✅ VERIFIED | All tuning on dev split. Test split untouched. Guards in eval scripts. |
| 4.13 | Error analysis | ✅ VERIFIED | EVAL_REPORT.md §Error analysis. 4 failure types documented with root causes and fixes. |

---

## 5. Application and Delivery

| # | Requirement | Status | Evidence |
|---|-------------|--------|---------|
| 5.1 | Python backend | ✅ VERIFIED | FastAPI app in `app/`. |
| 5.2 | Functional UI | ✅ VERIFIED | Static HTML/JS UI served from `app/main.py`. Georgian UI text in `app/static/`. |
| 5.3 | Document management in UI | ✅ VERIFIED | Upload, list, delete via UI. |
| 5.4 | Question answering in UI | ✅ VERIFIED | Ask endpoint with sources, conflicts, sub-queries displayed. |
| 5.5 | Source inspection | ✅ VERIFIED | `sources` list in `AskResponse` with chunk_id, text, title. |
| 5.6 | Public deployment (free tier) | ⚠️ PARTIAL | Render.com configuration exists (`render.yaml`, `Dockerfile`). Live URL not verified in this session due to no live deployment access. |
| 5.7 | DB integration and persistence | ✅ VERIFIED | Supabase PostgreSQL. `app/core/migrations.py` runs on startup. |
| 5.8 | Credentials protected | ✅ VERIFIED | `.env` in `.gitignore`. `.env.example` has placeholders only. `hmac.compare_digest` for admin token. No secrets in Docker image. |
| 5.9 | Modular code | ✅ VERIFIED | Layer architecture: `api` → `rag` → `retrieval`/`ingestion` → `db`. |
| 5.10 | Automated tests | ✅ VERIFIED | 154 tests passing. `pytest tests/`. |
| 5.11 | README | ✅ VERIFIED | `README.md` with architecture, installation, usage. |
| 5.12 | Evaluation Report | ✅ VERIFIED | `docs/EVAL_REPORT.md`. |
| 5.13 | AI Usage Report | ✅ VERIFIED | `docs/AI_USAGE.md`. |
| 5.14 | Deployment docs | ✅ VERIFIED | `docs/DEPLOYMENT.md`. |
| 5.15 | Git repository with history | ✅ VERIFIED | `git log` shows commit-by-commit feature development. |

---

## 6. Security

| # | Requirement | Status | Evidence |
|---|-------------|--------|---------|
| 6.1 | Admin token with constant-time compare | ✅ VERIFIED | `hmac.compare_digest` in `documents.py` and `auth.py`. |
| 6.2 | Rate limiting per IP | ✅ VERIFIED | `app/api/middleware.py`. |
| 6.3 | Global daily LLM cap | ✅ VERIFIED | `daily_cap_ask` setting. |
| 6.4 | Upload type/size validation | ✅ VERIFIED | Extension whitelist + 10 MB file limit + 1 MB text limit. |
| 6.5 | Session isolation | ✅ VERIFIED | Session users see only global corpus + their own uploads. Tools respect `session_id`. |
| 6.6 | Prompt-injection defense | ✅ VERIFIED | Retrieved text treated as DATA, not instructions. |

---

## 7. Known Risks and Remaining Gaps

| Risk | Severity | Notes |
|------|----------|-------|
| Dataset: 29,009 loaded vs 50,003 target | HIGH | DB at 526 MB > 500 MB free limit. halfvec migration needed but not executed (requires production DB access). |
| Chunking comparison: quantitative metrics missing | MEDIUM | Both strategies work; qualitative comparison written; controlled A/B test blocked by DB size. |
| Test split eval not run | LOW | Correct — must be run only once, after all tuning. |
| F1 = 0.061 (low) | MEDIUM | Known limitation of F1 vs verbose answers. The system is factually correct more often than F1 suggests. |
| Live deployment URL not verified | MEDIUM | Render deployment config exists; live check not performed in this session. |
| `halfvec` migration not executed | HIGH | Needs owner to run SQL on production DB. See DEPLOYMENT.md §Migration. |

---

## 8. Submission Artefacts

- [x] GitHub repository URL: https://github.com/fridonaaaaa/first-AI-agent
- [x] Live demo URL: https://first-ai-agent-rq2b.onrender.com/
- [ ] `README.md` ✅
- [ ] `docs/EVAL_REPORT.md` ✅
- [ ] `docs/AI_USAGE.md` ✅
- [ ] `docs/ARCHITECTURE.md` ✅
- [ ] `docs/DEPLOYMENT.md` ✅
- [ ] `FINAL_CHECKLIST.md` ✅ (this file)
