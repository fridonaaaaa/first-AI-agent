"""
Retrieval evaluation — Recall@K and MRR.  No LLM calls needed.

For each question in the eval set, retrieves top-k chunks and checks whether
any chunk's title matches a gold supporting title from the question.

Usage:
    python scripts/eval_retrieval.py --split dev --config eval/configs/c2_hybrid_rrf.json
    python scripts/eval_retrieval.py --split test --config eval/configs/c3_hybrid_rerank.json

Writes results to eval/results/<config_id>_<split>_retrieval.json.

Metrics:
  Recall@K  — fraction of questions where at least one gold title appears in top-K chunks.
  MRR       — mean reciprocal rank of the first matching chunk (0 if not found in top-K).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.core.db import open_connection
from app.core.logging import configure_logging, get_logger
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.embeddings import encode_one
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vector_store import search_dense

log = get_logger("eval_retrieval")
EVAL_DIR  = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions"
RESULTS   = EVAL_DIR / "results"
RESULTS.mkdir(exist_ok=True)


def _load_questions(split: str) -> list[dict]:
    path = QUESTIONS / f"{split}.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fetch_chunk_titles(conn, chunk_ids: list[str]) -> dict[str, str]:
    """Return {chunk_id: document_title} for the given ids."""
    if not chunk_ids:
        return {}
    ph = ",".join(["%s"] * len(chunk_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, d.title
            FROM   chunks c
            JOIN   documents d ON c.doc_id = d.id
            WHERE  c.id IN ({ph})
            """,
            chunk_ids,
        )
        return {r[0]: r[1] for r in cur.fetchall()}


def retrieve(query: str, cfg: dict, conn, top_k: int = 10) -> list[str]:
    """Return ordered list of chunk_ids for the query under the given config."""
    mode = cfg.get("retrieval_mode", "hybrid")
    k1   = cfg.get("bm25_k1", 1.5)
    b    = cfg.get("bm25_b", 0.75)
    rrf_k = cfg.get("rrf_k", 60)

    if mode == "dense_only":
        vec = encode_one(query)
        results = search_dense(vec, top_k=top_k, conn=conn)
        return [cid for cid, _ in results]

    if mode in ("hybrid", "hybrid_rerank"):
        vec = encode_one(query)
        dense = search_dense(vec, top_k=top_k * 2, conn=conn)
        bm25  = BM25Searcher(conn=conn, k1=k1, b=b).search(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion([dense, bm25], k=rrf_k)
        return [cid for cid, _ in fused[:top_k]]

    raise ValueError(f"Unknown retrieval_mode: {mode!r}")


def evaluate(questions: list[dict], cfg: dict, conn, top_k: int = 10) -> dict:
    hits_at_k = 0
    rr_sum    = 0.0
    per_q     = []

    for q in questions:
        gold_titles = {t.lower() for t in q.get("supporting_titles", [])}
        if not gold_titles or q.get("expected_behavior") == "abstain":
            # Abstain questions or questions with no supporting titles — skip from retrieval metrics
            per_q.append({"id": q["id"], "skipped": True})
            continue

        query = q["question"]
        chunk_ids = retrieve(query, cfg, conn, top_k=top_k)
        titles    = _fetch_chunk_titles(conn, chunk_ids)

        rank = None
        for i, cid in enumerate(chunk_ids, start=1):
            title = titles.get(cid, "").lower()
            if any(gt in title or title in gt for gt in gold_titles):
                rank = i
                break

        hit = rank is not None
        rr  = 1.0 / rank if rank else 0.0
        hits_at_k += int(hit)
        rr_sum    += rr
        per_q.append({
            "id": q["id"], "type": q.get("type"),
            "hit": hit, "rank": rank, "rr": rr,
        })

    answerable = [r for r in per_q if not r.get("skipped")]
    n = len(answerable)
    return {
        "recall_at_k": hits_at_k / n if n else 0.0,
        "mrr":         rr_sum    / n if n else 0.0,
        "n_answerable": n,
        "n_total": len(questions),
        "top_k": top_k,
        "per_question": per_q,
    }


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="dev",  choices=["dev", "test"])
    parser.add_argument("--config", default="eval/configs/c2_hybrid_rrf.json")
    parser.add_argument("--top-k",  type=int, default=10)
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    questions = _load_questions(args.split)

    log.info("eval_start", config=cfg["config_id"], split=args.split, n=len(questions))
    t0 = time.time()

    conn = open_connection()
    metrics = evaluate(questions, cfg, conn, top_k=args.top_k)
    conn.close()

    elapsed = time.time() - t0
    result = {
        "config":  cfg,
        "split":   args.split,
        "elapsed": round(elapsed, 1),
        **{k: v for k, v in metrics.items() if k != "per_question"},
        "per_question": metrics["per_question"],
    }

    out_path = RESULTS / f"{cfg['config_id']}_{args.split}_retrieval.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*50}")
    print(f"Config:     {cfg['config_id']} — {cfg['name']}")
    print(f"Split:      {args.split}  ({metrics['n_answerable']} answerable / {metrics['n_total']} total)")
    print(f"Recall@{args.top_k}:  {metrics['recall_at_k']:.3f}")
    print(f"MRR:        {metrics['mrr']:.3f}")
    print(f"Time:       {elapsed:.1f}s")
    print(f"Results:    {out_path}")


if __name__ == "__main__":
    main()
