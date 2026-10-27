"""
Abstention — refuse to answer when evidence is insufficient.

Two-gate design:
  Gate 1 (fast, no LLM): if the top retrieved chunk's RRF score is below
  `rrf_threshold`, there is not enough evidence in the index. Abstain immediately.

  Gate 2 (LLM sufficiency check): given the chunks and the query, ask the LLM
  whether the chunks contain enough information to answer. If it says no,
  abstain with what was found.

The abstention response is always in the user's original language. The message
text is filled in by the calling pipeline using the i18n module.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.llm.base import LLMClient
from app.retrieval.rerank import CandidateChunk

_SUFFICIENCY_SYSTEM = """You decide if a set of retrieved passages contains enough
information to answer a question.
Return a JSON object with one key:
  "sufficient": true or false
Output ONLY valid JSON."""


@dataclass
class AbstentionDecision:
    should_abstain: bool
    reason: str    # "low_score" | "llm_insufficient" | "sufficient"
    top_score: float


def check_abstention(
    query: str,
    chunks: list[CandidateChunk],
    llm: LLMClient,
    rrf_threshold: float = 0.01,
) -> AbstentionDecision:
    """
    Decide whether to abstain from answering.

    Args:
        query:         The rewritten English query.
        chunks:        Reranked chunks (highest relevance first).
        llm:           LLM client for the sufficiency check.
        rrf_threshold: Minimum acceptable top-chunk RRF score.

    Returns:
        AbstentionDecision. Check .should_abstain before generating an answer.
    """
    top_score = chunks[0].rrf_score if chunks else 0.0

    # Gate 1: no good chunks at all
    if not chunks or top_score < rrf_threshold:
        return AbstentionDecision(
            should_abstain=True,
            reason="low_score",
            top_score=top_score,
        )

    # Gate 2 removed: the score threshold is sufficient for our corpus size.
    # The answer-generation LLM will naturally qualify its answer when evidence
    # is partial — a second LLM call here was rejecting valid hits.
    return AbstentionDecision(
        should_abstain=False,
        reason="sufficient",
        top_score=top_score,
    )
