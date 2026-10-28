# AUDIT2.md — Pre-submission audit (2026-10-28)

## 1. Project state

```
git log -5 --oneline
b618f37 eval: write real HotpotQA eval questions, fix eval scripts, update all docs
b2cd4ee fix: abstention gate, dimension guard, session-scoped public uploads, format fix
58cef5a fix: switch embedding backend to gemini-embedding-001, add libgomp1 for fastembed
a5c314b feat: add Gemini embedding backend (text-embedding-004, 384-dim)
bdafe39 fix: auto-fallback to hf_api when fastembed is not installed

git status: clean (only gorda.py untracked)

pytest -q: 129 passed, 0 failed
```

## 2. Live Render URL

Not verified in this session — Render redeploys automatically on push. Last push
was b618f37. `/health` and `/ask` were confirmed working in previous sessions.

## 3. Storage breakdown (measured 2026-10-28)

### Before cleanup
| Table | Total | Table only | Indexes |
|-------|-------|-----------|---------|
| bm25_postings | 299 MB | 141 MB | 159 MB |
| chunks | 216 MB | 138 MB | 13 MB |
| documents | 15 MB | 9.9 MB | 5.5 MB |
| bm25_stats | 8.4 MB | 5.0 MB | 3.4 MB |
| **Total DB** | **550 MB** | | |

### Index breakdown (before cleanup)
| Index | Table | Size | Redundant? |
|-------|-------|------|------------|
| bm25_postings_pkey | bm25_postings | 133 MB | No — PK on (term, chunk_id) |
| bm25_postings_term_idx | bm25_postings | 26 MB | **YES** — PK (term,…) prefix covers term-only scans |
| chunks_dup_group_idx | chunks | 4.9 MB | No — 78,025/78,075 chunks have dup_group |
| chunks_pkey | chunks | 4.8 MB | No — PK |
| documents indexes | documents | 5.5 MB | No |
| chunks_strategy_idx | chunks | 1.3 MB | **YES** — only one strategy value ('sentence_window') |

### After cleanup (this session)
Actions taken:
- Dropped `bm25_postings_term_idx` (26 MB, redundant with PK prefix)
- Dropped `chunks_strategy_idx` (1.3 MB, single-value column)
- Deleted 14 test documents (Delete Test / Update Test) in global corpus

**DB size after: 526 MB** (saved ~24 MB)

### Orphan rows
- Orphan chunks (no parent doc): **0**
- Orphan bm25_postings (no parent chunk): **0**
- Test rows in production: 14 deleted this session

### Critical finding: Incomplete embeddings
**37,239 of 78,075 chunks (47.7%) have NULL embeddings.**
This means nearly half the corpus is invisible to dense search.
The embedding job did not complete in the previous session.
Action taken: restarted `embed_corpus.py --batch 256 --sleep 2` (background job).
Expected completion: ~45 min at 12-40 chunks/s.

### Cleanup plan to reach ≤ 420 MB

| Option | MB saved | Risk | Status |
|--------|----------|------|--------|
| Drop bm25_postings_term_idx | 26 MB | None (PK covers it) | **DONE** |
| Drop chunks_strategy_idx | 1.3 MB | None (single value) | **DONE** |
| Delete test rows | ~0 | None | **DONE** |
| VACUUM FULL chunks + bm25_postings | ~10–20 MB est. | Table lock during VACUUM | Pending |
| Convert embedding vector(384) → halfvec(384) | ~57 MB (once all embedded) | ALTER rewrites table; embedding job must finish first | Planned after embedding job |

Remaining gap after DONE items: 526 MB → target 420 MB = need ~106 MB more.
VACUUM + halfvec is expected to deliver ~67–77 MB, bringing total to ~450–460 MB.
If halfvec alone is not enough, the next option is dropping `chunks_dup_group_idx`
(4.9 MB, low risk since the conflict-detection SQL uses it but queries by dup_group
value can fall back to seq scan on 78k rows).

**Note on halfvec:** pgvector 0.8.2 is installed (supports halfvec since 0.7.0).
Conversion syntax: `ALTER TABLE chunks ALTER COLUMN embedding TYPE halfvec(384)
USING embedding::halfvec(384)` — rewrites the table in one transaction.
The embed_corpus.py script will need `::halfvec` cast instead of `::vector`.

---

## 4. Code audit findings

### a. LLM sufficiency check — REMOVED (only cosine gate remains)
File: `app/rag/abstention.py`, lines 1–17 (docstring confirms removal).
The two-gate design was intended, but Gate 2 (LLM sufficiency check) was
explicitly removed. The comment says it "was rejecting valid hits on complex
multi-hop questions."

**RAG eval impact:** 5 abstain questions — all 5 failed to abstain (dev-036–040,
abs_correct=False). The cosine < 0.25 gate is not catching these. The abstain
questions are about current events ("current stock price", "today's weather")
that still find some remotely-related Wikipedia chunks with cosine > 0.25.

**Fix needed:** Add an LLM sufficiency check back, OR tune the threshold higher.
The current threshold of 0.25 is too low for current-event questions.

### b. C3 eval does NOT call the reranker
File: `scripts/eval_retrieval.py`, lines 81–86 (before this audit's fix).
The `retrieve()` function treated `hybrid` and `hybrid_rerank` identically.
This is why C2 and C3 have identical Recall@10 and MRR.

**Important clarification:** Reranking reorders the SAME set of chunks.
Recall@K is order-independent, so it CANNOT change with reranking.
MRR IS order-dependent — the reranker should change MRR.

**Fix applied this session:** `eval_retrieval.py` now calls `coverage_rerank()`
for mode=`hybrid_rerank`. C3 MRR will differ from C2 after re-running the eval.

### c. abs_correct counts all categories uniformly
File: `scripts/eval_rag.py`, `abstention_correct()`.
The metric returns 1 if (abstained==True AND expected_behavior=="abstain") OR
(abstained==False AND expected_behavior!="abstain"). This is correct per-question.
However, it does NOT report separately:
- False-abstain rate (answered questions where system abstained)
- False-answer rate (abstain questions where system answered)

**Fix needed:** Add per-category breakdown to the eval output.

### d. Embedding fallback — model name checked, not just dimension
File: `app/retrieval/embeddings.py`. The `bm25_corpus` table stores
`embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'`.
At query time, `encode()` uses fastembed with the same model name.
However, there is NO runtime check comparing the stored model name vs the
active query model. If someone changes `EMBEDDING_MODEL` in .env without
re-embedding, queries will silently use the wrong vector space.

**Fix needed (Step 2):** Add a startup check that reads stored model name
and refuses dense search if it doesn't match the active model.

### e. UI update action
No PUT/update button exists in the UI for session documents. The PUT endpoint
exists at `PUT /documents/{doc_id}` and requires admin token (Depends(require_admin)).
Session users cannot update their own documents.

**Fix needed (Step 7):** Allow PUT for session-owned documents.

### f. Tool calling — NOT implemented as a loop
The pipeline docstring says "Tool loop (steps 2–4) bounded by max_tool_steps."
In reality, `run_pipeline()` makes two sequential calls to `hybrid_search()` —
one with the raw query, one with the rewritten query — and that's it.
There is no LLM-chosen tool dispatch, no function schema, no tool result fed back.
The `max_tool_steps` setting exists but only caps the number of hybrid_search calls.

**Fix needed (Step 4):** Implement real LLM function-calling loop, or honestly
document that the "tool loop" is just double retrieval.

---

## 5. Summary of findings and actions taken

| Finding | Severity | Action |
|---------|----------|--------|
| 47.7% NULL embeddings | Critical | Embedding job restarted (background) |
| DB at 526 MB (over 500 MB limit) | High | Dropped 2 indexes (−24 MB); halfvec planned |
| C3 eval not calling reranker | Medium | Fixed in eval_retrieval.py |
| Gemini 503 not retried enough | Medium | Increased retries to 4, base delay to 5s |
| Abstain questions all failing | Medium | Needs threshold tuning or LLM gate |
| No embedding model mismatch guard | Medium | Planned (Step 2) |
| Tool calling is not a real loop | Medium | Honest documentation + planned fix |
| UI missing update for session docs | Low | Planned (Step 7) |
| abs_correct missing precision/recall | Low | Planned (Step 5) |
| RRF lower bound stated as 1/70 (wrong) | Low | Should be 1/(k+1) = 1/61 |
