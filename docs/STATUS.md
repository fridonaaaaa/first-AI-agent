# STATUS.md

## Current phase
Phase 5 — Evaluation complete. Final hardening pass done (2026-10-28/29).

## Last finished items (2026-10-29)

### Tool calling (Step 4 complete)
- `app/rag/tools.py`: three tools with Gemini functionDeclarations schemas —
  `search`, `get_document`, `compare_claims`
- `GeminiClient.complete_with_tools()`: native Gemini function-calling API
- Bounded tool loop in `pipeline.py`: max_tool_steps cap, session/doc isolation
- `PipelineResult.tool_trace` + `AskResponse.tool_trace` (sanitised, no raw text)
- `MockLLMClient.complete_with_tools()` for test scripting
- 20 new tool tests in `tests/test_tools.py` — all pass

### Question set (expanded to assignment minimum)
- Test split: 40 questions (up from 30): bridge×13, comparison×9, single×4,
  abstain×5, conflict×5, unanswerable×3, prompt_injection×1
- Dev split: 63 questions (unchanged)

### Documentation
- `docs/EVAL_REPORT.md` updated: reflects 63 dev + 40 test, all categories,
  chunking comparison section, error analysis, metric limitations
- `FINAL_CHECKLIST.md` created: requirement-to-evidence matrix

### Previous (2026-10-28)
- Retrieval eval C1/C2/C3 complete on dev split (see EVAL_REPORT.md)
- RAG eval C4 complete on dev split: F1=0.061, Citation Precision=0.875, Abstention=0.775
- Abstention fix: restored LLM Gate 2 (YES/NO sufficiency check)
- C3 reranker eval fix: hybrid_rerank mode now calls coverage_rerank in eval_retrieval.py
- Gemini retry: 4 retries × 5 s base
- DB cleanup: dropped 2 redundant indexes (−24 MB); DB at 526 MB
- Step 2: embedding model mismatch guard

## Test status (2026-10-29)
- Unit tests: **154 passed, 0 failed**
- New in this session: 20 tool tests (test_tools.py)
- Integration tests: require live Supabase (`pytest tests/test_registry.py -v`)

## Remaining work / blockers

### BLOCKED — requires owner action on production DB
- halfvec migration: `ALTER TABLE chunks ALTER COLUMN embedding TYPE halfvec(384)` 
  This frees ~150 MB, allowing the remaining 20,994 documents to be loaded.
  See DEPLOYMENT.md §Migration for exact SQL and rollback plan.
- Load remaining HotpotQA documents (after halfvec frees space)
- Full test-split eval (run ONCE after all tuning done)
- Live deployment URL verification

### PENDING — controlled quantitative chunking comparison
- Blocked by DB size (can't add a new chunk set with current capacity).
- Qualitative comparison written in EVAL_REPORT.md.

### DONE
- All mandatory RAG pipeline requirements ✅
- Custom BM25, RRF, coverage reranker ✅  
- Abstention (2-gate), citation verification, conflict detection ✅
- Real LLM tool calling (Gemini functionDeclarations API) ✅
- 103 evaluation questions (63 dev + 40 test) ✅
- 154 automated tests ✅
- Documentation: README, EVAL_REPORT, AI_USAGE, ARCHITECTURE, DEPLOYMENT ✅
- FINAL_CHECKLIST.md ✅
