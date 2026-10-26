"""
Unit tests for Reciprocal Rank Fusion.

Hand-computed example (k=60):

  Dense list:  [A(rank 1), B(rank 2), C(rank 3)]
  BM25  list:  [B(rank 1), C(rank 2), D(rank 3)]

  RRF scores:
    A = 1/(60+1)            = 0.016393  (dense only)
    B = 1/(60+2)+1/(60+1)   = 0.032520  (both lists)
    C = 1/(60+3)+1/(60+2)   = 0.032003  (both lists)
    D = 1/(60+3)            = 0.015873  (BM25 only)

  Expected ranking: B > C > A > D
"""

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


DENSE = [("A", 0.9), ("B", 0.7), ("C", 0.5)]
BM25  = [("B", 8.0), ("C", 6.0), ("D", 4.0)]
K = 60


def _scores(results):
    return {cid: s for cid, s in results}


def test_rrf_correct_ranking():
    results = reciprocal_rank_fusion([DENSE, BM25], k=K)
    ids = [cid for cid, _ in results]
    assert ids[0] == "B"
    assert ids[1] == "C"
    assert ids[2] == "A"
    assert ids[3] == "D"


def test_rrf_hand_computed_scores():
    results = _scores(reciprocal_rank_fusion([DENSE, BM25], k=K))
    assert abs(results["A"] - 1 / (K + 1))                       < 1e-9
    assert abs(results["B"] - (1 / (K + 2) + 1 / (K + 1)))      < 1e-9
    assert abs(results["C"] - (1 / (K + 3) + 1 / (K + 2)))      < 1e-9
    assert abs(results["D"] - 1 / (K + 3))                       < 1e-9


def test_rrf_document_in_one_list_only():
    results = _scores(reciprocal_rank_fusion([DENSE, BM25], k=K))
    # A is only in dense (rank 1); D is only in BM25 (rank 3)
    assert results["A"] > results["D"]  # rank 1 > rank 3


def test_rrf_document_in_both_lists_beats_single_list():
    results = _scores(reciprocal_rank_fusion([DENSE, BM25], k=K))
    # B appears in both lists; A appears in one list at rank 1
    # B should beat A because appearing in both lists compounds its score
    assert results["B"] > results["A"]


def test_rrf_original_scores_ignored():
    # Swap the scores but keep the same ranks — result must be identical
    dense_diff_scores = [("A", 99.9), ("B", 0.001), ("C", 0.0001)]
    bm25_diff_scores  = [("B", 0.001), ("C", 0.0001), ("D", 0.00001)]
    results1 = reciprocal_rank_fusion([DENSE, BM25], k=K)
    results2 = reciprocal_rank_fusion([dense_diff_scores, bm25_diff_scores], k=K)
    assert results1 == results2


def test_rrf_empty_list_ignored():
    results = reciprocal_rank_fusion([DENSE, [], BM25], k=K)
    ids = [cid for cid, _ in results]
    assert ids[0] == "B"


def test_rrf_all_empty_returns_empty():
    assert reciprocal_rank_fusion([[], []], k=K) == []


def test_rrf_single_list_preserves_order():
    results = reciprocal_rank_fusion([DENSE], k=K)
    ids = [cid for cid, _ in results]
    assert ids == ["A", "B", "C"]


def test_rrf_scores_sorted_descending():
    results = reciprocal_rank_fusion([DENSE, BM25], k=K)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_larger_k_reduces_rank_difference():
    # With larger k, the gap between rank 1 and rank 2 shrinks
    r_k1   = _scores(reciprocal_rank_fusion([DENSE], k=1))
    r_k100 = _scores(reciprocal_rank_fusion([DENSE], k=100))
    gap_small_k = r_k1["A"]   - r_k1["B"]
    gap_large_k = r_k100["A"] - r_k100["B"]
    assert gap_small_k > gap_large_k


def test_rrf_three_lists():
    list3 = [("C", 1.0), ("A", 0.8), ("E", 0.6)]
    results = reciprocal_rank_fusion([DENSE, BM25, list3], k=K)
    scores = _scores(results)
    # C appears in all 3 lists (rank 3, rank 2, rank 1) — should rank very high
    assert scores["C"] > scores["D"]  # D only in one list
