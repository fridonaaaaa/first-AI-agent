"""
RAG pipeline evaluation — answer correctness, citation accuracy, abstention.

Uses the full run_pipeline() for each question and caches LLM responses
(via GeminiClient's disk cache) so reruns are free.

IMPORTANT: never run this on the TEST split until all tuning is done.
Use the DEV split for all tuning and threshold adjustments.

Usage:
    python scripts/eval_rag.py --split dev --config eval/configs/c4_full_pipeline.json

Metrics produced:
  answer_f1          — token-level F1 between generated and gold answer
  exact_match        — strict exact match (lower-cased, stripped)
  citation_precision — fraction of cited chunks that appear in retrieved set
  abstention_accuracy — correct abstain/answer decision
  conflict_recall    — (placeholder; requires labelled conflict eval set)

Results written to eval/results/<config_id>_<split>_rag.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging, get_logger
from app.rag.llm.factory import get_llm_client
from app.rag.pipeline import run_pipeline

log = get_logger("eval_rag")
EVAL_DIR  = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions"
RESULTS   = EVAL_DIR / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _normalise(pred).split()
    gold_tokens = _normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def citation_precision(answer: str, retrieved_ids: set[str]) -> float:
    """Fraction of [chunk_id] citations in answer that exist in retrieved_ids."""
    from app.rag.citation import extract_citations
    cited = extract_citations(answer)
    if not cited:
        return 1.0   # no citations → no wrong citations
    valid = sum(1 for c in cited if c in retrieved_ids)
    return valid / len(cited)


def abstention_correct(abstained: bool, gold_answer: str, expected_behavior: str = "") -> bool:
    """True if the system correctly abstained or answered per expected_behavior."""
    if expected_behavior:
        should_abstain = expected_behavior == "abstain"
    else:
        should_abstain = gold_answer.strip() in ("", "abstain")
    return abstained == should_abstain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_questions(split: str) -> list[dict]:
    path = QUESTIONS / f"{split}.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="dev", choices=["dev", "test"])
    parser.add_argument("--config", default="eval/configs/c4_full_pipeline.json")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Evaluate only first N questions (0 = all). Use for quick smoke tests.")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    questions = _load_questions(args.split)
    if args.limit:
        questions = questions[:args.limit]

    llm = get_llm_client()
    log.info("eval_start", config=cfg["config_id"], split=args.split, n=len(questions))
    t0 = time.time()

    per_q: list[dict] = []
    total_f1 = total_em = total_cite_p = total_abs = 0.0

    for q in questions:
        q_start = time.time()
        try:
            result = run_pipeline(q["question"], llm)
        except Exception as exc:
            log.warning("pipeline_error", qid=q["id"], error=str(exc))
            per_q.append({"id": q["id"], "error": str(exc)})
            continue

        gold    = q.get("gold_answer", q.get("answer", ""))
        expected = q.get("expected_behavior", "")
        abstained = bool(result.abstention and result.abstention.should_abstain)
        retrieved_ids = {c.chunk_id for c in result.chunks}

        is_answerable = expected != "abstain" and gold.strip() not in ("", "abstain")
        f1  = token_f1(result.answer, gold) if is_answerable else 0.0
        em  = exact_match(result.answer, gold) if is_answerable else False
        cp  = citation_precision(result.answer, retrieved_ids)
        abs_ok = abstention_correct(abstained, gold, expected)

        total_f1    += f1
        total_em    += int(em)
        total_cite_p += cp
        total_abs   += int(abs_ok)

        per_q.append({
            "id":          q["id"],
            "category":    q.get("category", q.get("type")),
            "language":    q.get("language"),
            "question":    q["question"],
            "gold":        gold,
            "answer":      result.answer,
            "abstained":   abstained,
            "f1":          round(f1, 4),
            "em":          em,
            "cite_p":      round(cp, 4),
            "abs_correct": abs_ok,
            "steps":       result.steps_used,
            "sub_queries": result.sub_queries,
            "elapsed_s":   round(time.time() - q_start, 2),
        })
        log.info("question_done", qid=q["id"], f1=round(f1, 3), abs_correct=abs_ok)

    n = len(per_q)
    elapsed = time.time() - t0
    metrics = {
        "answer_f1":            round(total_f1    / n, 4) if n else 0,
        "exact_match":          round(total_em    / n, 4) if n else 0,
        "citation_precision":   round(total_cite_p / n, 4) if n else 0,
        "abstention_accuracy":  round(total_abs   / n, 4) if n else 0,
        "n": n,
    }

    result_doc = {
        "config":  cfg,
        "split":   args.split,
        "elapsed": round(elapsed, 1),
        **metrics,
        "per_question": per_q,
    }

    out_path = RESULTS / f"{cfg['config_id']}_{args.split}_rag.json"
    out_path.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False))

    print(f"\n{'='*50}")
    print(f"Config:               {cfg['config_id']} — {cfg['name']}")
    print(f"Split:                {args.split}  (n={n})")
    print(f"Answer F1:            {metrics['answer_f1']:.3f}")
    print(f"Exact Match:          {metrics['exact_match']:.3f}")
    print(f"Citation Precision:   {metrics['citation_precision']:.3f}")
    print(f"Abstention Accuracy:  {metrics['abstention_accuracy']:.3f}")
    print(f"Time:                 {elapsed:.1f}s")
    print(f"Results:              {out_path}")


if __name__ == "__main__":
    main()
