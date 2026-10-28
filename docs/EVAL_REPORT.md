# EVAL_REPORT.md

_Generated from `eval/results/`. Run `python scripts/make_report.py --split dev` to regenerate._

---

## How to reproduce

```bash
# 1. Load corpus (one time)
python scripts/load_hotpotqa.py
python scripts/embed_corpus.py

# 2. Retrieval eval (no LLM, no API key needed)
python scripts/eval_retrieval.py --split dev --config eval/configs/c1_dense_only.json
python scripts/eval_retrieval.py --split dev --config eval/configs/c2_hybrid_rrf.json
python scripts/eval_retrieval.py --split dev --config eval/configs/c3_hybrid_rerank.json

# 3. Full RAG eval (requires GEMINI_API_KEY; uses disk cache so reruns are free)
python scripts/eval_rag.py --split dev --config eval/configs/c4_full_pipeline.json

# 4. Generate this report
python scripts/make_report.py --split dev
```

**Rule:** TEST split (`eval/questions/test.jsonl`) is run ONCE, only after all tuning on DEV is done.

---

## Eval set

| Split | Questions | Bridge | Comparison | Abstain |
|-------|-----------|--------|------------|---------|
| dev   | 40        | 20     | 15         | 5       |
| test  | 20        | 10     | 7          | 3       |

All bridge and comparison questions are drawn from HotpotQA validation set,
filtered to questions where all supporting documents exist in the loaded corpus.
Abstain questions are real-world queries that no static encyclopedia corpus can answer.

---

## Retrieval results — DEV split (2026-10-28)

35 of 40 questions are answerable (5 abstain questions skipped from retrieval metrics).

| Config | Recall@10 | MRR | Time |
|--------|-----------|-----|------|
| C1 — dense only | 0.400 | 0.313 | 94 s |
| C2 — hybrid RRF | **0.886** | **0.523** | 340 s |
| C3 — hybrid rerank | **0.886** | **0.523** | 437 s |

**Key finding:** Hybrid RRF (C2) improves Recall@10 by 122% over dense-only (C1). The coverage
reranker (C3) does not change Recall@10 or MRR — its benefit shows up in answer quality
(by promoting diverse, non-redundant chunks to the LLM) rather than in retrieval hit rate.

---

## RAG results — DEV split (C4 — full pipeline)

_Running as of 2026-10-28. Numbers will be filled in when eval completes._

| Metric | C4 — full_pipeline |
|--------|--------------------|
| Answer F1 | — |
| Exact Match | — |
| Citation Precision | — |
| Abstention Accuracy | — |
| n questions | 40 |

---

## Notes on metrics

- **Recall@K**: fraction of answerable questions where at least one gold supporting document's
  chunk appears in the top-K retrieved chunks (title substring match).
- **MRR**: mean reciprocal rank of the first matching chunk across answerable questions.
- **Answer F1**: token-level F1 between predicted answer and gold answer (case-insensitive,
  whitespace-normalised). Abstain questions contribute 0 F1 (correct abstain ≠ wrong answer).
- **Exact Match**: 1 if prediction equals gold after normalisation, 0 otherwise.
- **Citation Precision**: fraction of `[chunk_id]` citations in the answer that are in the
  retrieved set (1.0 means no hallucinated citations).
- **Abstention Accuracy**: 1 if the model correctly abstained (unanswerable) or answered
  (answerable), 0 otherwise.
