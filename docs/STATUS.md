# STATUS.md

## Current phase
Phase 5 — Evaluation (in progress). 10-step hardening pass in progress.

## Last finished items (2026-10-28)
- Phase 4: Eval questions written — 40 dev + 20 test (real HotpotQA multi-hop + abstain)
- Retrieval eval C1/C2/C3 complete on dev split (see EVAL_REPORT.md)
- RAG eval C4 complete on dev split: F1=0.061, Citation Precision=0.875, Abstention=0.775
- Abstention fix: restored LLM Gate 2 (YES/NO sufficiency check); cosine gate alone can't
  distinguish current-event queries (cosine 0.45–0.53) from answerable ones (0.52–0.56)
- C3 reranker eval fix: hybrid_rerank mode now calls coverage_rerank in eval_retrieval.py
- Gemini retry: 4 retries × 5 s base (was 2 × 1.5 s) to survive 503 bursts
- DB cleanup: dropped 2 redundant indexes (−24 MB); DB at 526 MB
- Step 2 (embedding model mismatch guard): `get_stored_model_name` reads bm25_corpus;
  hybrid_search skips dense and logs a warning if stored model ≠ active model;
  `PipelineResult.retrieval_mode` + `AskResponse.retrieval_mode` expose the flag
- 134 unit tests passing

## In progress
- Background: embed_corpus.py filling 37,239 NULL embeddings (task buv0648jc)

## Next
- Step 3: Add conflict, single-doc, hard-unanswerable, prompt-injection, Georgian questions
- Step 4: Real LLM function-calling tool loop
- Step 5: Abstention precision/recall breakdown in eval; re-run C3 MRR after reranker fix
- Step 6: Chunking strategy comparison (sentence-window vs fixed-token)
- Step 7: Session user PUT endpoint
- Step 8: Load more HotpotQA docs (after halfvec conversion frees space)
- Step 9: Full eval rerun on all configs with expanded question set
- Step 10: Final verification (secret scan, live URL tests, FINAL_CHECKLIST.md)

## Test status
- Unit tests: 134 passed, 0 failed (2026-10-28)
- Integration tests: require live Supabase (`pytest tests/test_registry.py -v`)
- Eval questions: 40 dev + 20 test, schema validated by test suite

## Blockers
- TEST split must not be used until all dev tuning is done
- DB at 526 MB — over Supabase 500 MB free limit; halfvec conversion planned after embed job
- Render free tier: 512 MB RAM — embedding model uses ~420 MB (tight but functional)
