"""
Reciprocal Rank Fusion (RRF) — hand-written, no library.

Merges multiple ranked lists into one by assigning each document a score
based on its position in every list it appears in:

    rrf_score(d) = Σ  1 / (k + rank_i(d))
                    i

Where:
  rank_i(d) = position of document d in list i (1-indexed; absent = skipped)
  k          = smoothing constant (default 60) that controls how much
               weight rank-1 has over rank-2

Key properties:
  - The original relevance scores are ignored — only rank matters.
  - A document appearing in multiple lists is boosted: its contributions add up.
  - k=60 means rank 1 → 1/61 ≈ 0.0164; rank 61 → 1/121 ≈ 0.0083.
    The gap between consecutive ranks shrinks as k increases.

Typical use: fuse the dense (pgvector) ranked list with the BM25 ranked list.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a list of (chunk_id, score) pairs
                      already sorted by score descending.  The scores are
                      not used — only the order within each list matters.
        k:            Smoothing constant. Higher k → smaller rank differences.

    Returns:
        A merged list of (chunk_id, rrf_score) sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
