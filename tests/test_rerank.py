"""
Unit tests for the coverage-aware reranker.

Key test: plain RRF order is [A, B, D, C].
After reranking with sub-queries, C (which covers the second hop) jumps above B and D.

Toy corpus:
  A (rrf=0.9, dup_group=X): "bell invented telephone"        → covers sub-query 1
  B (rrf=0.8, dup_group=None): "telephone changed world"     → covers sub-query 1 weakly
  C (rrf=0.3, dup_group=None): "bell born 1847 edinburgh"    → covers sub-query 2
  D (rrf=0.7, dup_group=X): "bell telephone invention"       → covers sub-query 1, near-dup of A

Sub-queries:
  sq1: "bell telephone inventor"
  sq2: "bell born year"
"""

import pytest

from app.retrieval.rerank import CandidateChunk, coverage_rerank


def _make(chunk_id, score, text, dup_group=None):
    return CandidateChunk(chunk_id=chunk_id, rrf_score=score, text=text, dup_group=dup_group)


A = _make("A", 0.9, "bell invented telephone", dup_group="X")
B = _make("B", 0.8, "telephone changed world")
C = _make("C", 0.3, "bell born 1847 edinburgh")
D = _make("D", 0.7, "bell telephone invention", dup_group="X")

SQ1 = "bell telephone inventor"
SQ2 = "bell born year"


def _ids(results):
    return [c.chunk_id for c in results]


def test_reranker_second_hop_promoted():
    """C covers sq2 (not yet covered after A) — must jump above B and D."""
    results = coverage_rerank([A, B, C, D], sub_queries=[SQ1, SQ2], top_k=4)
    ids = _ids(results)
    assert ids[0] == "A"            # highest rrf + covers sq1
    assert ids[1] == "C"            # covers uncovered sq2 → boosted
    assert ids.index("C") < ids.index("B")
    assert ids.index("C") < ids.index("D")


def test_reranker_near_duplicate_penalised():
    """D is a near-dup of A (same dup_group X) — must rank below C."""
    results = coverage_rerank([A, B, C, D], sub_queries=[SQ1, SQ2], top_k=4)
    ids = _ids(results)
    assert ids.index("C") < ids.index("D")


def test_reranker_no_sub_queries_preserves_rrf_order():
    """With no sub-queries, coverage gain = 0 for all → pure relevance order."""
    results = coverage_rerank([A, B, C, D], sub_queries=[], top_k=4)
    scores = [c.rrf_score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_reranker_empty_candidates_returns_empty():
    assert coverage_rerank([], sub_queries=[SQ1], top_k=5) == []


def test_reranker_top_k_limits_output():
    results = coverage_rerank([A, B, C, D], sub_queries=[SQ1, SQ2], top_k=2)
    assert len(results) == 2


def test_reranker_all_queries_covered_uses_relevance():
    """Once all sub-queries are covered, remaining picks go by relevance score."""
    # Use only sq1; A covers it first. Then B and D compete on relevance only (D penalised).
    results = coverage_rerank([A, B, C, D], sub_queries=[SQ1], top_k=4)
    ids = _ids(results)
    assert ids[0] == "A"
    # After sq1 covered: B (0.8, no dup) vs D (0.7, dup penalty) vs C (0.3)
    assert ids[1] == "B"


def test_reranker_lambda_1_is_pure_relevance():
    """lambda_=1.0 means coverage weight=0 → same as sorting by rrf_score."""
    results = coverage_rerank([A, B, C, D], sub_queries=[SQ1, SQ2], top_k=4, lambda_=1.0)
    scores = [c.rrf_score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_reranker_lambda_0_maximises_coverage():
    """lambda_=0.0 means only coverage matters — C (covers new sq2) should be picked early."""
    results = coverage_rerank([A, B, C, D], sub_queries=[SQ1, SQ2], top_k=4, lambda_=0.0)
    # Both A and C cover a unique query. With lambda=0, coverage_gain drives selection.
    # One of them must come first, and C must appear before B (B covers no new query).
    ids = _ids(results)
    assert ids.index("C") < ids.index("B")


def test_reranker_single_candidate():
    results = coverage_rerank([A], sub_queries=[SQ1], top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "A"


def test_reranker_no_dup_group_no_penalty():
    """Chunks with dup_group=None are never penalised."""
    only_b = _make("B", 0.5, "telephone changed world", dup_group=None)
    results = coverage_rerank([only_b], sub_queries=[SQ1], top_k=1)
    assert results[0].chunk_id == "B"
