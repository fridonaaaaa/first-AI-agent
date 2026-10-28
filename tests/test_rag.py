"""
Unit tests for the RAG pipeline modules.

All tests use MockLLMClient — no real LLM calls, no network, no DB.
The pipeline's hybrid_search is also mocked so tests are fully offline.
"""

import pytest
from unittest.mock import patch

from app.rag.llm.mock import MockLLMClient
from app.rag.llm.base import LLMResponse
from app.rag.query_understanding import (
    is_georgian,
    translate_to_english,
    decompose_query,
    understand_query,
)
from app.rag.citation import extract_citations, verify_citations
from app.rag.conflict import detect_conflicts, ConflictVerdict
from app.rag.abstention import check_abstention
from app.rag.pipeline import run_pipeline
from app.retrieval.rerank import CandidateChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(cid, score=0.5, text="some text", dup=None, cosine_sim=0.5):
    return CandidateChunk(chunk_id=cid, rrf_score=score, text=text, dup_group=dup, cosine_sim=cosine_sim)


# ---------------------------------------------------------------------------
# is_georgian
# ---------------------------------------------------------------------------

def test_georgian_detection_true():
    assert is_georgian("ვინ გამოიგონა ტელეფონი?") is True


def test_georgian_detection_false():
    assert is_georgian("Who invented the telephone?") is False


def test_georgian_mixed_is_true():
    assert is_georgian("Hello ვინ?") is True


# ---------------------------------------------------------------------------
# translate_to_english
# ---------------------------------------------------------------------------

def test_translate_georgian_calls_llm():
    llm = MockLLMClient(fallback="Who invented the telephone?")
    result = translate_to_english("ვინ გამოიგონა ტელეფონი?", llm)
    assert "telephone" in result.lower()
    assert len(llm.calls) == 1


def test_translate_english_unchanged():
    llm = MockLLMClient()
    result = translate_to_english("Who invented the telephone?", llm)
    assert result == "Who invented the telephone?"
    assert len(llm.calls) == 0


# ---------------------------------------------------------------------------
# decompose_query
# ---------------------------------------------------------------------------

def test_decompose_returns_sub_queries():
    llm = MockLLMClient(
        responses={"telephone": '{"sub_queries": ["who invented telephone", "when was bell born"]}'}
    )
    subs = decompose_query("Who invented the telephone and when was he born?", llm)
    assert len(subs) == 2
    assert "telephone" in subs[0]


def test_decompose_malformed_json_falls_back():
    llm = MockLLMClient(fallback="not json at all")
    subs = decompose_query("some question", llm)
    assert subs == ["some question"]


def test_decompose_missing_key_falls_back():
    llm = MockLLMClient(fallback='{"wrong_key": []}')
    subs = decompose_query("some question", llm)
    assert subs == ["some question"]


def test_decompose_respects_max():
    llm = MockLLMClient(
        fallback='{"sub_queries": ["a", "b", "c", "d", "e"]}'
    )
    subs = decompose_query("complex question", llm, max_sub_queries=3)
    assert len(subs) <= 3


# ---------------------------------------------------------------------------
# extract_citations / verify_citations
# ---------------------------------------------------------------------------

def test_extract_citations_basic():
    ids = extract_citations("Bell invented it [chunk-1] in 1876 [chunk-2].")
    assert ids == ["chunk-1", "chunk-2"]


def test_extract_citations_deduplicates():
    ids = extract_citations("[a] [b] [a]")
    assert ids == ["a", "b"]


def test_verify_removes_invalid():
    result = verify_citations("Bell [real-1] invented [fake-99] the telephone.", {"real-1"})
    assert "fake-99" not in result.verified_answer
    assert "real-1" in result.verified_answer
    assert "fake-99" in result.invalid_ids


def test_verify_keeps_all_valid():
    result = verify_citations("Bell [a] born [b].", {"a", "b"})
    assert result.invalid_ids == []
    assert "[a]" in result.verified_answer
    assert "[b]" in result.verified_answer


def test_verify_empty_retrieved_strips_all():
    result = verify_citations("Answer [chunk-1].", set())
    assert "[chunk-1]" not in result.verified_answer
    assert "chunk-1" in result.invalid_ids


# ---------------------------------------------------------------------------
# detect_conflicts
# ---------------------------------------------------------------------------

def test_detect_conflicts_finds_conflict():
    llm = MockLLMClient(fallback='{"verdict": "conflict", "reason": "different dates"}')
    a = _chunk("a", text="Bell was born in 1847.", dup="X")
    b = _chunk("b", text="Bell was born in 1850.", dup="X")
    verdicts = detect_conflicts([a, b], llm)
    assert any(v.verdict == "conflict" for v in verdicts)


def test_detect_conflicts_agree():
    llm = MockLLMClient(fallback='{"verdict": "agree", "reason": "same info"}')
    a = _chunk("a", text="The telephone was invented in 1876.", dup="X")
    b = _chunk("b", text="Bell invented the telephone in 1876.", dup="X")
    verdicts = detect_conflicts([a, b], llm)
    assert all(v.verdict == "agree" for v in verdicts)


def test_detect_no_candidates_no_llm_calls():
    llm = MockLLMClient()
    # Chunks with no dup_group and no numeric values — not candidates
    a = _chunk("a", text="The sky is blue.")
    b = _chunk("b", text="Cats eat fish.")
    detect_conflicts([a, b], llm)
    assert len(llm.calls) == 0


# ---------------------------------------------------------------------------
# check_abstention
# ---------------------------------------------------------------------------

def test_abstain_on_empty_chunks():
    llm = MockLLMClient()
    decision = check_abstention("some query", [], llm)
    assert decision.should_abstain is True
    assert decision.reason == "no_results"


def test_abstain_when_llm_says_no():
    # Gate 1 passes (cosine=0.5 >= 0.25), but LLM Gate 2 says NO → abstain.
    llm = MockLLMClient(fallback="NO")
    chunks = [_chunk("a", score=0.5, text="Tbilisi is the capital of Georgia.", cosine_sim=0.5)]
    decision = check_abstention("What is the weather in Tbilisi right now?", chunks, llm, cosine_threshold=0.25)
    assert decision.should_abstain is True
    assert decision.reason == "insufficient"


def test_abstain_on_low_cosine():
    # cosine_sim=0.1 is below the default threshold of 0.25 → should abstain.
    llm = MockLLMClient()
    chunks = [_chunk("a", score=0.5, cosine_sim=0.1)]
    decision = check_abstention("query", chunks, llm, cosine_threshold=0.25)
    assert decision.should_abstain is True
    assert decision.reason == "low_score"


def test_no_abstain_when_sufficient():
    llm = MockLLMClient()
    chunks = [_chunk("a", score=0.5, text="Bell invented the telephone in 1876.", cosine_sim=0.8)]
    decision = check_abstention("who invented telephone", chunks, llm, cosine_threshold=0.25)
    assert decision.should_abstain is False


def test_abstain_when_cosine_below_threshold():
    llm = MockLLMClient()
    chunks = [_chunk("a", score=0.9, text="Some unrelated text.", cosine_sim=0.05)]
    decision = check_abstention("specific question", chunks, llm, cosine_threshold=0.1)
    assert decision.should_abstain is True
    assert decision.reason == "low_score"


# ---------------------------------------------------------------------------
# run_pipeline (fully mocked, no DB)
# ---------------------------------------------------------------------------

_FAKE_CHUNKS = [
    _chunk("c1", score=0.9, text="Bell invented the telephone in 1876.", cosine_sim=0.85),
    _chunk("c2", score=0.5, text="Bell was born in Edinburgh in 1847.", cosine_sim=0.60),
]


def _mock_hybrid(query, sub_queries, top_k=10, lambda_=0.5, conn=None, doc_id=None, session_id=None):
    return _FAKE_CHUNKS


def test_pipeline_returns_answer():
    llm = MockLLMClient(
        responses={
            "translate": "Who invented the telephone?",
            "search query": "Bell telephone inventor",
            "sub_queries": '{"sub_queries": ["who invented telephone"]}',
            "sufficient": '{"sufficient": true}',
            "verdict": '{"verdict": "unrelated", "reason": "ok"}',
            "Passages": "Bell invented the telephone. [c1]",
        },
        fallback='{"sufficient": true}',
    )
    with patch("app.rag.pipeline.hybrid_search", side_effect=_mock_hybrid):
        result = run_pipeline("Who invented the telephone?", llm)
    assert result.answer
    assert result.language == "english"
    assert result.abstention is not None
    assert result.abstention.should_abstain is False


def test_pipeline_abstains_on_no_chunks():
    llm = MockLLMClient(fallback='{"sub_queries": ["q"]}')

    def _no_chunks(*a, **kw):
        return []

    with patch("app.rag.pipeline.hybrid_search", side_effect=_no_chunks):
        result = run_pipeline("impossible question", llm)
    assert result.abstention.should_abstain is True


def test_pipeline_georgian_query():
    llm = MockLLMClient(
        responses={
            "ვინ": "Who invented the telephone?",
            "sufficient": '{"sufficient": true}',
            "Passages": "Bell invented it. [c1]",
        },
        fallback='{"sub_queries": ["who invented telephone"]}',
    )
    with patch("app.rag.pipeline.hybrid_search", side_effect=_mock_hybrid):
        result = run_pipeline("ვინ გამოიგონა ტელეფონი?", llm)
    assert result.language == "georgian"


def test_pipeline_retrieval_mode_bm25_only():
    """When stored model ≠ active model, PipelineResult.retrieval_mode is 'bm25_only'."""
    from unittest.mock import patch as _patch
    from app.retrieval.embeddings import get_stored_model_name as _gsm

    llm = MockLLMClient(
        responses={
            "sufficient": '{"sufficient": true}',
            "Passages": "Bell invented it. [c1]",
        },
        fallback='{"sub_queries": ["q"]}',
    )

    def _stored_mismatch(conn):
        return "some-other-model/v1"

    with _patch("app.rag.pipeline.hybrid_search", side_effect=_mock_hybrid), \
         _patch("app.rag.pipeline.get_stored_model_name", side_effect=_stored_mismatch), \
         _patch("app.rag.pipeline.get_db"):
        result = run_pipeline("who invented telephone", llm)

    assert result.retrieval_mode == "bm25_only"


def test_pipeline_strips_hallucinated_citations():
    llm = MockLLMClient(
        responses={
            "sufficient": '{"sufficient": true}',
            "Passages": "Bell invented it [c1] and also [fake-999].",
        },
        fallback='{"sub_queries": ["q"]}',
    )
    with patch("app.rag.pipeline.hybrid_search", side_effect=_mock_hybrid):
        result = run_pipeline("who invented telephone", llm)
    assert "fake-999" not in result.answer
    assert result.citation_check is not None
    assert "fake-999" in result.citation_check.invalid_ids
