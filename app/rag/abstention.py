"""
Abstention — refuse to answer when evidence is insufficient.

Two-gate design:

  Gate 1 (fast, no LLM): if the top retrieved chunk's cosine similarity is below
  `cosine_threshold`, the search found nothing genuinely relevant. Abstain
  immediately without calling the LLM.

  Why cosine similarity instead of RRF score: RRF scores are always positive and
  bounded below by 1/(k + search_depth) regardless of relevance, so an RRF threshold
  cannot distinguish "irrelevant" from "relevant". Cosine similarity is a real
  relevance signal, but it too has limits: a query about "weather in Tbilisi today"
  finds chunks about Tbilisi (cosine ~0.47) even though the corpus has no weather data.

  Gate 2 (LLM sufficiency check): ask the LLM whether the retrieved passages
  actually contain a direct answer to the question. This catches "related entity,
  wrong fact" cases that the cosine gate misses — e.g. time-sensitive queries
  ("stock price right now"), personal queries, or facts simply not in the corpus.

  The prompt is binary (YES/NO) to minimise false-abstains on valid multi-hop
  questions: as long as the passages contain a specific answerable fact, the LLM
  returns YES even if retrieval is imperfect.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.llm.base import LLMClient
from app.retrieval.rerank import CandidateChunk

_SUFFICIENCY_SYSTEM = (
    "You are a fact-checker. You will be given a question and up to 5 passages "
    "from a knowledge base.\n"
    "Reply with exactly one word:\n"
    "YES — if ANY of the passages contain information relevant to the question, "
    "even partially.\n"
    "NO  — ONLY if ALL passages are completely unrelated to the question and share "
    "no topic, entity, or fact with it whatsoever.\n"
    "Reply YES or NO only. No explanation."
)


@dataclass
class AbstentionDecision:
    should_abstain: bool
    reason: str    # "no_results" | "low_score" | "insufficient" | "sufficient"
    top_score: float


def check_abstention(
    query: str,
    chunks: list[CandidateChunk],
    llm: LLMClient,
    cosine_threshold: float = 0.25,
) -> AbstentionDecision:
    """
    Decide whether to abstain from answering.

    Two gates are checked in order:
    1. Fast cosine check — no LLM call.
    2. LLM sufficiency check — one LLM call on the top-5 chunks.

    Args:
        query:            The rewritten English query.
        chunks:           Reranked chunks (highest relevance first).
        llm:              LLM client for Gate 2.
        cosine_threshold: Minimum acceptable cosine similarity (Gate 1).

    Returns:
        AbstentionDecision. Check .should_abstain before generating an answer.
    """
    if not chunks:
        return AbstentionDecision(
            should_abstain=True,
            reason="no_results",
            top_score=0.0,
        )

    top_cosine = max((c.cosine_sim for c in chunks), default=0.0)

    # Gate 1: fast cosine check — catches clearly off-topic queries with no LLM cost.
    if top_cosine < cosine_threshold:
        return AbstentionDecision(
            should_abstain=True,
            reason="low_score",
            top_score=top_cosine,
        )

    # Gate 2: LLM sufficiency check — catches "related entity, wrong fact" cases.
    # Uses only the top-5 chunks to keep the prompt short.
    top5_text = "\n\n".join(
        f"[{c.chunk_id}] {c.text[:400]}" for c in chunks[:5]
    )
    prompt = f"Question: {query}\n\nPassages:\n{top5_text}"
    try:
        resp = llm.complete(
            prompt, system=_SUFFICIENCY_SYSTEM, temperature=0.0, max_tokens=5
        )
        verdict = resp.text.strip().upper()
        if verdict.startswith("NO"):
            return AbstentionDecision(
                should_abstain=True,
                reason="insufficient",
                top_score=top_cosine,
            )
    except Exception:
        # If the LLM check itself fails, allow the pipeline to continue rather
        # than silently turning an LLM error into a false abstain.
        pass

    return AbstentionDecision(
        should_abstain=False,
        reason="sufficient",
        top_score=top_cosine,
    )
