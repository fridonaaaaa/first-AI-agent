# STATUS.md

## Current phase
Phase 6+7 — Docs + Deploy (COMPLETE). Live pipeline verified working.

## Last finished item
Pipeline fully operational end-to-end. Verified answer returned for "Where is Strasbourg located?"
in ~15s (cold start). DB: 2,182 chunks from 764 HotpotQA docs, all embedded.

Critical fixes applied this session:
  - `gemini.py`              — model corrected to `gemini-3.1-flash-lite` (verified 2026-10-27)
  - `bm25.py`               — batched all per-term DB queries into 2 round-trips (was 2N+1)
  - `vector_store.py`       — fixed column name chunk_id → id
  - `hybrid.py`             — fixed column name chunk_id → id in chunk fetch
  - `embed_corpus.py`       — fixed column name chunk_id → id
  - `db.py`                 — retry loop for DNS failures
  - `.env`                  — ABSTENTION_THRESHOLD lowered to 0.005 (RRF max is ~0.033)
  - `index.html`            — UI redesign

## In progress
None.

## Next (owner actions)
1. Keep loading corpus: `python scripts/load_hotpotqa.py` (764 docs loaded, ~50k total)
2. After load, embed:   `python scripts/embed_corpus.py`
3. Retrieval eval:      `python scripts/eval_retrieval.py --split dev`
4. RAG eval:            `python scripts/eval_rag.py --split dev`
5. Deploy to Render (see docs/DEPLOYMENT.md)

## Blockers
- Full corpus not loaded — only 764 of 50,000 docs; most questions will abstain
- TEST split must not be used until all tuning is done

## Test status
- Unit tests: 129 passed, 0 failed (run `pytest tests/ -v --ignore=tests/test_registry.py`)
- Integration tests: require live Supabase; run `pytest tests/test_registry.py -v`
- Pipeline: verified returning answers at localhost:8000
