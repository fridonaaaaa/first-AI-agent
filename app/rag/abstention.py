"""
Abstention — refuse to answer when evidence is insufficient.

Two-gate design:
  Gate 1 (fast, no LLM): if the top retrieved chunk's cosine similarity is below
  `cosine_threshold`, the search found nothing genuinely relevant. Abstain
  immediately without calling the LLM.

  Why cosine similarity instead of RRF score: RRF scores are bounded below by
  1/(k+top_k) ≈ 0.014 regardless of relevance, so a threshold below that value
  can never fire.  Cosine similarity is a real relevance signal: values below
  ~0.25 indicate the query vector and chunk vector have little in common.

  Gate 2 (LLM sufficiency check): REMOVED. The answer-generation LLM naturally
  qualifies its answer when evidence is partial, and a second LLM call was
  rejecting valid hits on complex multi-hop questions.

The abstention response text is filled in by the calling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.llm.base import LLMClient
from app.retrieval.rerank import CandidateChunk


@dataclass
class AbstentionDecision:
    should_abstain: bool
    reason: str    # "low_score" | "sufficient"
    top_score: float


def check_abstention(
    query: str,
    chunks: list[CandidateChunk],
    llm: LLMClient,
    cosine_threshold: float = 0.25,
) -> AbstentionDecision:
    """
    Decide whether to abstain from answering.

    Args:
        query:            The rewritten English query (unused; kept for API compat).
        chunks:           Reranked chunks (highest relevance first).
        llm:              LLM client (unused; kept for API compat).
        cosine_threshold: Minimum acceptable cosine similarity for the top chunk.

    Returns:
        AbstentionDecision. Check .should_abstain before generating an answer.
    """
    if not chunks:
        return AbstentionDecision(
            should_abstain=True,
            reason="low_score",
            top_score=0.0,
        )

    top_cosine = max((c.cosine_sim for c in chunks), default=0.0)

    if top_cosine < cosine_threshold:
        return AbstentionDecision(
            should_abstain=True,
            reason="low_score",
            top_score=top_cosine,
        )

    return AbstentionDecision(
        should_abstain=False,
        reason="sufficient",
        top_score=top_cosine,
    )
