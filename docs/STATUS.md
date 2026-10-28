# STATUS.md

## Current phase
Phase 5 — Evaluation (in progress). Retrieval eval complete; RAG eval running.

## Last finished items (2026-10-28)
- Phase 2: Bug fixes — abstention gate (cosine threshold replaces dead RRF gate),
  dimension guard in vector_store, session-scoped public uploads, format accept fix
- Phase 4: Eval questions written — 40 dev + 20 test, real HotpotQA multi-hop questions
  where all supporting documents exist in the corpus, plus 5/3 abstain questions
- Retrieval eval runs complete for C1, C2, C3 on dev split (see EVAL_REPORT.md)
- RAG eval (C4) running now — ~40 questions, Gemini LLM, ~10-20 min total

## In progress
- RAG eval: `python scripts/eval_rag.py --split dev --config eval/configs/c4_full_pipeline.json`

## Next
1. Fill EVAL_REPORT.md with RAG eval numbers once C4 finishes
2. Run test split evals (after all tuning done on dev)
3. Commit everything
4. Push to GitHub → Render redeploys automatically

## Test status
- Unit tests: 129 passed, 0 failed
- Integration tests: require live Supabase (`pytest tests/test_registry.py -v`)
- Eval questions: 40 dev + 20 test, schema validated by test suite

## Blockers
- TEST split must not be used until all dev tuning is done
- Render free tier: 512 MB RAM — embedding model uses ~420 MB (tight but functional)
- DB at 526 MB — over Supabase 500 MB free limit; reads still work, new uploads may fail
