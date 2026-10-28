"""
Coverage-aware greedy reranker for multi-hop retrieval.

After RRF produces a ranked list, this reranker re-orders it so that
the final top-k set covers as many sub-queries as possible, while still
preferring high-relevance chunks and penalising near-duplicates.

Algorithm (MMR-style greedy selection):
  At each step, pick the candidate that maximises:

      score(c) = lambda_ * relevance(c)
               + (1 - lambda_) * coverage_gain(c)
               - dup_penalty  * is_near_dup(c)

  Where:
    relevance(c)    = c.rrf_score  (already normalised by RRF)
    coverage_gain(c) = fraction of sub-queries c covers that are not yet
                       covered by any already-selected chunk
    is_near_dup(c)  = 1 if c.dup_group matches a group already selected

  "Covers" is defined as: the overlap between the chunk's token set and
  the sub-query's token set reaches `coverage_threshold` of the query tokens.

  lambda_ = 1.0 → pure relevance (same as sorting by rrf_score).
  lambda_ = 0.0 → maximise coverage, ignore relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateChunk:
    chunk_id: str
    rrf_score: float
    text: str
    dup_group: str | None = field(default=None)
    doc_id: str | None = field(default=None)
    title: str | None = field(default=None)
    cosine_sim: float = field(default=0.0)


def _tokenise(text: str) -> set[str]:
    """Lowercase alphabetic tokens of length >= 2."""
    return {w for w in text.lower().split() if w.isalpha() and len(w) >= 2}


def _covers_query(chunk_tokens: set[str], query_tokens: set[str], threshold: float) -> bool:
    """True if chunk overlaps ≥ threshold fraction of the query's tokens."""
    if not query_tokens:
        return False
    return len(chunk_tokens & query_tokens) / len(query_tokens) >= threshold


def coverage_rerank(
    candidates: list[CandidateChunk],
    sub_queries: list[str],
    top_k: int = 10,
    lambda_: float = 0.5,
    dup_penalty: float = 0.5,
    coverage_threshold: float = 0.4,
) -> list[CandidateChunk]:
    """
    Greedy coverage-aware reranking.

    Args:
        candidates:         Chunks from RRF, any order.
        sub_queries:        Decomposed sub-queries the answer must cover.
        top_k:              Maximum number of chunks to return.
        lambda_:            Weight for relevance vs coverage (0=coverage only, 1=relevance only).
        dup_penalty:        Score deduction when a near-duplicate group is already selected.
        coverage_threshold: Fraction of query tokens a chunk must share to count as "covering" it.

    Returns:
        Up to top_k chunks, ordered by the greedy selection score.
    """
    if not candidates:
        return []

    query_token_sets: list[set[str]] = [_tokenise(q) for q in sub_queries]
    n_queries = len(query_token_sets)

    covered: list[bool] = [False] * n_queries
    selected_dup_groups: set[str] = set()

    # dup_penalty scales with the coverage weight so that lambda_=1.0 (pure
    # relevance) never penalises near-duplicates, and n_queries=0 (no diversity
    # goal) also skips the penalty entirely.
    effective_dup_penalty = (1 - lambda_) * dup_penalty if n_queries > 0 else 0.0

    remaining = list(candidates)
    result: list[CandidateChunk] = []

    while remaining and len(result) < top_k:
        best_score = -1.0
        best_gain = -1.0
        best_idx = 0

        for i, chunk in enumerate(remaining):
            chunk_tokens = _tokenise(chunk.text)

            # How many new sub-queries does this chunk cover?
            new_covered = sum(
                1
                for j, qtoks in enumerate(query_token_sets)
                if not covered[j] and _covers_query(chunk_tokens, qtoks, coverage_threshold)
            )
            gain = new_covered / n_queries if n_queries > 0 else 0.0

            # Near-duplicate penalty
            penalty = (
                effective_dup_penalty
                if (chunk.dup_group is not None and chunk.dup_group in selected_dup_groups)
                else 0.0
            )

            score = lambda_ * chunk.rrf_score + (1 - lambda_) * gain - penalty

            # Tie-break by coverage gain so equal-score chunks that cover new
            # sub-queries are preferred over those that do not.
            if (score, gain) > (best_score, best_gain):
                best_score = score
                best_gain = gain
                best_idx = i

        chosen = remaining.pop(best_idx)
        result.append(chosen)

        # Mark sub-queries as covered
        chosen_tokens = _tokenise(chosen.text)
        for j, qtoks in enumerate(query_token_sets):
            if not covered[j] and _covers_query(chosen_tokens, qtoks, coverage_threshold):
                covered[j] = True

        if chosen.dup_group is not None:
            selected_dup_groups.add(chosen.dup_group)

    return result
