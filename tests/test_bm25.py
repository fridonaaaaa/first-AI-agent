"""
Unit tests for BM25 scorer.  No database — tests the pure scoring function
directly with hand-computed expected values.

Corpus used across tests:
  Chunk A: "cat jumped fence twice quickly"   tokens: [cat,jumped,fence,twice,quickly]  dl=5
  Chunk B: "cat cat fence"                    tokens: [cat,cat,fence]                   dl=3
  Chunk C: "dog ran park"                     tokens: [dog,ran,park]                    dl=3

Term frequencies (normalised = count/dl):
  tf("cat",  A) = 1/5 = 0.2
  tf("cat",  B) = 2/3 ≈ 0.667
  tf("fence",A) = 1/5 = 0.2
  tf("fence",B) = 1/3 ≈ 0.333

Corpus stats:
  N = 3,  avgdl = (5+3+3)/3 ≈ 3.667
  df("cat")   = 2,  df("fence") = 2,  df("dog") = 1

Default k1=1.5, b=0.75.
"""

import math
import pytest

from app.retrieval.bm25 import bm25_score, BM25Searcher


# ── hand-computed expected scores ────────────────────────────────────────────

K1, B = 1.5, 0.75
N, AVGDL = 3, (5 + 3 + 3) / 3      # 3.667

TF_CAT_A,   DL_A = 1 / 5, 5
TF_CAT_B,   DL_B = 2 / 3, 3
TF_FENCE_A       = 1 / 5
TF_FENCE_B       = 1 / 3
DF_CAT,   DF_FENCE,  DF_DOG = 2, 2, 1


def _expected(tf, dl, df):
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    tf_component = tf * (K1 + 1) / (tf + K1 * ((1 - B) / dl + B / AVGDL))
    return idf * tf_component


# ── pure scoring function tests ───────────────────────────────────────────────

def test_bm25_score_known_values():
    score = bm25_score(TF_CAT_A, DL_A, AVGDL, DF_CAT, N, K1, B)
    assert abs(score - _expected(TF_CAT_A, DL_A, DF_CAT)) < 1e-9


def test_bm25_score_higher_tf_scores_higher():
    score_a = bm25_score(TF_CAT_A, DL_A, AVGDL, DF_CAT, N, K1, B)
    score_b = bm25_score(TF_CAT_B, DL_B, AVGDL, DF_CAT, N, K1, B)
    # B has higher tf and shorter length → should score higher
    assert score_b > score_a


def test_bm25_score_rare_term_scores_higher():
    # dog appears in only 1 doc (df=1); cat in 2 (df=2)
    # same tf and dl → rare term should score higher
    score_common = bm25_score(1 / 3, 3, AVGDL, DF_CAT, N, K1, B)
    score_rare   = bm25_score(1 / 3, 3, AVGDL, DF_DOG, N, K1, B)
    assert score_rare > score_common


def test_bm25_score_zero_tf_returns_zero():
    assert bm25_score(0.0, 3, AVGDL, DF_CAT, N, K1, B) == 0.0


def test_bm25_score_all_docs_contain_term_near_zero():
    # When df = N, IDF = log(0.5/2.5 + 1) = log(1.2) ≈ 0.182 — positive but small
    score = bm25_score(1 / 3, 3, AVGDL, N, N, K1, B)
    assert 0 < score < 0.3


# ── BM25Searcher (in-memory) tests ────────────────────────────────────────────

def _make_searcher(postings, stats, corpus):
    """
    Build a BM25Searcher from plain dicts (no DB).
    postings: {term: [(chunk_id, tf, dl), ...]}
    stats:    {term: df}
    corpus:   (N, avgdl)
    """
    return BM25Searcher.from_index(postings, stats, corpus, k1=K1, b=B)


def test_searcher_single_term_correct_ranking():
    postings = {
        "cat":   [("A", TF_CAT_A, DL_A), ("B", TF_CAT_B, DL_B)],
        "fence": [("A", TF_FENCE_A, DL_A), ("B", TF_FENCE_B, DL_B)],
        "dog":   [("C", 1 / 3, 3)],
    }
    stats   = {"cat": DF_CAT, "fence": DF_FENCE, "dog": DF_DOG}
    corpus  = (N, AVGDL)
    searcher = _make_searcher(postings, stats, corpus)

    results = searcher.search_index("cat", top_k=3)
    chunk_ids = [cid for cid, _ in results]
    assert chunk_ids[0] == "B"   # B has higher tf and shorter length
    assert "C" not in chunk_ids  # C has no "cat"


def test_searcher_unknown_term_returns_empty():
    postings = {"cat": [("A", 0.2, 5)]}
    stats    = {"cat": 1}
    corpus   = (1, 5)
    searcher = _make_searcher(postings, stats, corpus)
    assert searcher.search_index("zebra", top_k=5) == []


def test_searcher_multi_term_sums_scores():
    postings = {
        "cat":   [("A", TF_CAT_A, DL_A), ("B", TF_CAT_B, DL_B)],
        "fence": [("A", TF_FENCE_A, DL_A), ("B", TF_FENCE_B, DL_B)],
    }
    stats  = {"cat": DF_CAT, "fence": DF_FENCE}
    corpus = (N, AVGDL)
    searcher = _make_searcher(postings, stats, corpus)

    results = searcher.search_index("cat fence", top_k=3)
    assert len(results) == 2
    # B should still rank first (higher scores on both terms)
    assert results[0][0] == "B"


def test_searcher_top_k_limits_results():
    postings = {"cat": [("A", 0.2, 5), ("B", 0.5, 3), ("C", 0.1, 4)]}
    stats    = {"cat": 3}
    corpus   = (3, 4)
    searcher = _make_searcher(postings, stats, corpus)
    results  = searcher.search_index("cat", top_k=2)
    assert len(results) == 2


def test_searcher_scores_are_sorted_descending():
    postings = {"cat": [("A", 0.1, 5), ("B", 0.5, 3), ("C", 0.3, 4)]}
    stats    = {"cat": 3}
    corpus   = (3, 4)
    searcher = _make_searcher(postings, stats, corpus)
    results  = searcher.search_index("cat", top_k=3)
    scores   = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
