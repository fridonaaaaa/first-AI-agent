"""
Conflict detection — finds and adjudicates contradictions between sources.

Two chunks are "conflict candidates" if they share a dup_group (near-duplicate
of the same source paragraph) OR if they both contain numeric values or dates
for the same entity. The LLM is asked to give a structured verdict.

The verdict is one of:
  "agree"     — the chunks say the same thing (or compatible things)
  "conflict"  — they make contradictory factual claims
  "unrelated" — they look related but don't actually disagree

If a conflict is found, the pipeline surfaces both sources and tells the user
rather than silently picking one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.llm.base import LLMClient
from app.retrieval.rerank import CandidateChunk

_CONFLICT_SYSTEM = """You compare two text passages and decide if they contradict each other.
Return a JSON object with two keys:
  "verdict": one of "agree", "conflict", or "unrelated"
  "reason":  one sentence explaining why

Output ONLY valid JSON."""

_NUMERIC_RE = re.compile(r"\b\d{4}\b|\b\d+[,.]?\d*\s*(?:km|m|kg|lb|mi|%)\b")


@dataclass
class ConflictVerdict:
    chunk_a: str
    chunk_b: str
    verdict: str          # "agree" | "conflict" | "unrelated"
    reason: str


def _is_conflict_candidate(a: CandidateChunk, b: CandidateChunk) -> bool:
    """True if the pair is worth checking for contradictions."""
    # Same near-duplicate group — near-copies of the same source may differ
    if a.dup_group and b.dup_group and a.dup_group == b.dup_group:
        return True
    # Both contain numeric/date values — potential factual disagreement
    if _NUMERIC_RE.search(a.text) and _NUMERIC_RE.search(b.text):
        return True
    return False


def _check_pair(a: CandidateChunk, b: CandidateChunk, llm: LLMClient) -> ConflictVerdict:
    prompt = (
        f"Passage A:\n{a.text}\n\n"
        f"Passage B:\n{b.text}\n\n"
        "Do these passages contradict each other?"
    )
    try:
        data = llm.complete_json(prompt, system=_CONFLICT_SYSTEM, temperature=0.0, max_tokens=256)
        verdict = str(data.get("verdict", "unrelated")).lower()
        reason = str(data.get("reason", ""))
        if verdict not in ("agree", "conflict", "unrelated"):
            verdict = "unrelated"
    except Exception:
        verdict = "unrelated"
        reason = ""

    return ConflictVerdict(
        chunk_a=a.chunk_id,
        chunk_b=b.chunk_id,
        verdict=verdict,
        reason=reason,
    )


_MAX_PAIRS = 6  # hard cap on LLM calls; prevents O(n²) blowup on large result sets
_TOP_N = 6      # only consider the top-N ranked chunks when building candidate pairs


def detect_conflicts(
    chunks: list[CandidateChunk],
    llm: LLMClient,
) -> list[ConflictVerdict]:
    """
    Check candidate pairs for contradictions, bounded by _MAX_PAIRS LLM calls.

    Only the top _TOP_N chunks are considered (highest RRF score first) and
    only pairs that pass _is_conflict_candidate() are sent to the LLM.
    This keeps worst-case cost at _MAX_PAIRS calls regardless of top_k.

    Returns a list of ConflictVerdict — only "conflict" verdicts matter
    to callers, but all verdicts are returned for logging/eval.
    """
    candidates = chunks[:_TOP_N]
    verdicts: list[ConflictVerdict] = []
    for i in range(len(candidates)):
        if len(verdicts) >= _MAX_PAIRS:
            break
        for j in range(i + 1, len(candidates)):
            if len(verdicts) >= _MAX_PAIRS:
                break
            if _is_conflict_candidate(candidates[i], candidates[j]):
                verdicts.append(_check_pair(candidates[i], candidates[j], llm))
    return verdicts
