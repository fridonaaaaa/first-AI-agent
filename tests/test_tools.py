"""
Tests for the RAG tool-calling infrastructure.

Covers:
- Tool schema structure
- execute_tool dispatching for known and unknown names
- compare_claims with MockLLMClient
- complete_with_tools on MockLLMClient (text, tool call, scripted sequence)
- Tool loop integration via run_pipeline with MockLLMClient scripting
- Iteration cap: tool loop does not exceed max_tool_steps
- Session isolation: search tool respects session_id filter
- Invalid tool args produce error dicts rather than exceptions
"""

from __future__ import annotations

import pytest

from app.rag.llm.mock import MockLLMClient
from app.rag.pipeline import PipelineResult, run_pipeline
from app.rag.tools import TOOL_SCHEMAS, execute_tool


# ---------------------------------------------------------------------------
# Tool schema validation
# ---------------------------------------------------------------------------

def test_tool_schemas_defined():
    assert len(TOOL_SCHEMAS) == 3
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert names == {"search", "get_document", "compare_claims"}


def test_tool_schemas_have_required_parameters():
    for tool in TOOL_SCHEMAS:
        assert "name" in tool
        assert "description" in tool
        params = tool.get("parameters", {})
        assert params.get("type") == "object"
        assert "properties" in params
        assert "required" in params


def test_tool_schemas_required_fields():
    by_name = {t["name"]: t for t in TOOL_SCHEMAS}
    assert "query" in by_name["search"]["parameters"]["required"]
    assert "chunk_id" in by_name["get_document"]["parameters"]["required"]
    assert "passage_a" in by_name["compare_claims"]["parameters"]["required"]
    assert "passage_b" in by_name["compare_claims"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# execute_tool — unknown tool
# ---------------------------------------------------------------------------

def test_execute_tool_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# execute_tool — search without DB connection
# ---------------------------------------------------------------------------

def test_execute_tool_search_no_conn():
    result = execute_tool("search", {"query": "test"}, conn=None)
    assert "error" in result


def test_execute_tool_search_empty_query():
    result = execute_tool("search", {"query": ""}, conn=None)
    assert "error" in result


# ---------------------------------------------------------------------------
# execute_tool — get_document without DB connection
# ---------------------------------------------------------------------------

def test_execute_tool_get_document_no_conn():
    result = execute_tool("get_document", {"chunk_id": "abc"}, conn=None)
    assert "error" in result


def test_execute_tool_get_document_empty_id():
    result = execute_tool("get_document", {"chunk_id": ""}, conn=None)
    assert "error" in result


# ---------------------------------------------------------------------------
# execute_tool — compare_claims
# ---------------------------------------------------------------------------

def test_compare_claims_no_llm():
    result = execute_tool("compare_claims", {"passage_a": "A", "passage_b": "B"}, llm=None)
    assert result["verdict"] == "unrelated"
    assert "reason" in result


def test_compare_claims_missing_passage():
    result = execute_tool("compare_claims", {"passage_a": "A"}, llm=None)
    assert "error" in result


def test_compare_claims_with_mock_llm_agree():
    llm = MockLLMClient(
        responses={"passage": '{"verdict": "agree", "reason": "same claim"}'}
    )
    result = execute_tool(
        "compare_claims",
        {"passage_a": "Paris is in France.", "passage_b": "France contains Paris."},
        llm=llm,
    )
    assert result["verdict"] == "agree"


def test_compare_claims_with_mock_llm_conflict():
    llm = MockLLMClient(
        responses={"passage": '{"verdict": "conflict", "reason": "contradictory dates"}'}
    )
    result = execute_tool(
        "compare_claims",
        {"passage_a": "The war ended in 1945.", "passage_b": "The war ended in 1946."},
        llm=llm,
    )
    assert result["verdict"] == "conflict"


# ---------------------------------------------------------------------------
# MockLLMClient.complete_with_tools
# ---------------------------------------------------------------------------

def test_mock_llm_complete_with_tools_no_script_returns_text():
    llm = MockLLMClient(fallback="final answer")
    text, calls = llm.complete_with_tools(
        [{"role": "user", "content": "question?"}],
        TOOL_SCHEMAS,
    )
    assert text == "final answer"
    assert calls == []


def test_mock_llm_complete_with_tools_scripted_tool_call():
    llm = MockLLMClient(
        tool_calls=[[{"name": "search", "args": {"query": "capital of France"}}]],
        fallback="Paris is the capital.",
    )
    text, calls = llm.complete_with_tools(
        [{"role": "user", "content": "What is the capital of France?"}],
        TOOL_SCHEMAS,
    )
    assert text is None
    assert len(calls) == 1
    assert calls[0]["name"] == "search"


def test_mock_llm_complete_with_tools_scripted_then_text():
    llm = MockLLMClient(
        tool_calls=[
            [{"name": "search", "args": {"query": "q"}}],
            [],  # second call → text
        ],
        fallback="final answer",
    )
    # First call → tool
    text1, calls1 = llm.complete_with_tools(
        [{"role": "user", "content": "q"}], TOOL_SCHEMAS
    )
    assert text1 is None
    assert calls1

    # Second call → text
    text2, calls2 = llm.complete_with_tools(
        [{"role": "user", "content": "q"}], TOOL_SCHEMAS
    )
    assert text2 is not None
    assert calls2 == []


def test_mock_llm_complete_with_tools_records_calls():
    llm = MockLLMClient(fallback="answer")
    llm.complete_with_tools([{"role": "user", "content": "hello"}], TOOL_SCHEMAS)
    assert "hello" in llm.calls


# ---------------------------------------------------------------------------
# Pipeline integration — tool_trace in PipelineResult
# ---------------------------------------------------------------------------

def test_pipeline_tool_trace_populated(monkeypatch):
    """
    When MockLLMClient returns one tool call then a final answer,
    PipelineResult.tool_trace should contain the tool call entry.
    """
    import app.rag.pipeline as _pipe

    # Stub hybrid_search to return one chunk with cosine_sim=0.9 (passes abstention gate).
    from app.retrieval.rerank import CandidateChunk
    fake_chunks = [
        CandidateChunk(
            chunk_id="chunk-001",
            rrf_score=0.02,
            text="Paris is the capital of France.",
            cosine_sim=0.9,
            title="France",
        )
    ]
    monkeypatch.setattr(_pipe, "hybrid_search", lambda *a, **kw: fake_chunks)

    # Stub understand_query
    monkeypatch.setattr(
        _pipe, "understand_query", lambda q, llm, **kw: ("english", q, [q])
    )

    # Stub abstention to not abstain.
    from app.rag.abstention import AbstentionDecision
    monkeypatch.setattr(
        _pipe, "check_abstention",
        lambda *a, **kw: AbstentionDecision(should_abstain=False, reason="sufficient", top_score=0.9),
    )
    # Stub detect_conflicts.
    monkeypatch.setattr(_pipe, "detect_conflicts", lambda *a, **kw: [])

    # Stub execute_tool to return a fake search result.
    monkeypatch.setattr(
        _pipe, "execute_tool",
        lambda name, args, **kw: {"count": 1, "results": [{"chunk_id": "chunk-002", "text": "hi", "title": "T", "rrf_score": 0.01}]},
    )

    # MockLLMClient: first complete_with_tools call → search tool, second → final text.
    llm = MockLLMClient(
        tool_calls=[
            [{"name": "search", "args": {"query": "capital France"}}],
            [],  # second turn → text
        ],
        fallback="Paris is the capital of France. [chunk-001]",
    )

    result = run_pipeline("What is the capital of France?", llm, conn=None, max_tool_steps=4)
    assert isinstance(result, PipelineResult)
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["tool"] == "search"
    assert "result_count" in result.tool_trace[0]


def test_pipeline_tool_trace_empty_when_no_tools_called(monkeypatch):
    """When the LLM skips tools, tool_trace should be empty."""
    import app.rag.pipeline as _pipe

    from app.retrieval.rerank import CandidateChunk
    fake_chunks = [
        CandidateChunk(
            chunk_id="chunk-001",
            rrf_score=0.02,
            text="The sky is blue.",
            cosine_sim=0.9,
            title="Sky",
        )
    ]
    monkeypatch.setattr(_pipe, "hybrid_search", lambda *a, **kw: fake_chunks)
    monkeypatch.setattr(
        _pipe, "understand_query", lambda q, llm, **kw: ("english", q, [q])
    )
    from app.rag.abstention import AbstentionDecision
    monkeypatch.setattr(
        _pipe, "check_abstention",
        lambda *a, **kw: AbstentionDecision(should_abstain=False, reason="sufficient", top_score=0.9),
    )
    monkeypatch.setattr(_pipe, "detect_conflicts", lambda *a, **kw: [])

    # No tool calls scripted → complete_with_tools returns text immediately.
    llm = MockLLMClient(fallback="The sky is blue. [chunk-001]")

    result = run_pipeline("What color is the sky?", llm, conn=None, max_tool_steps=4)
    assert result.tool_trace == []


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------

def test_pipeline_tool_loop_iteration_cap(monkeypatch):
    """
    When the LLM keeps returning tool calls, the loop must stop at max_tool_steps
    and still produce a final answer without infinite looping.
    """
    import app.rag.pipeline as _pipe

    from app.retrieval.rerank import CandidateChunk
    fake_chunks = [
        CandidateChunk(
            chunk_id="c1", rrf_score=0.02, text="some text", cosine_sim=0.9
        )
    ]
    monkeypatch.setattr(_pipe, "hybrid_search", lambda *a, **kw: fake_chunks)
    monkeypatch.setattr(
        _pipe, "understand_query", lambda q, llm, **kw: ("english", q, [q])
    )
    from app.rag.abstention import AbstentionDecision
    monkeypatch.setattr(
        _pipe, "check_abstention",
        lambda *a, **kw: AbstentionDecision(should_abstain=False, reason="sufficient", top_score=0.9),
    )
    monkeypatch.setattr(_pipe, "detect_conflicts", lambda *a, **kw: [])
    monkeypatch.setattr(
        _pipe, "execute_tool",
        lambda *a, **kw: {"count": 0, "results": []},
    )

    # Always returns a tool call — loop must cap at max_tool_steps.
    always_search = [{"name": "search", "args": {"query": "q"}}]
    llm = MockLLMClient(
        tool_calls=[always_search] * 20,  # 20 scripted, but cap should fire
        fallback="done",
    )

    result = run_pipeline("test", llm, conn=None, max_tool_steps=4)
    assert isinstance(result, PipelineResult)
    # The pipeline must return an answer string even when the cap fires.
    assert isinstance(result.answer, str)
    assert result.steps_used <= 6  # initial(1) + rewritten(1) + up to 4 tool steps


# ---------------------------------------------------------------------------
# Session isolation in tool execution
# ---------------------------------------------------------------------------

def test_execute_tool_search_passes_session_id(monkeypatch):
    """search tool must forward session_id to hybrid_search."""
    from app.rag import tools as _tools
    captured: dict = {}

    def fake_hybrid_search(query, sub_queries, top_k, conn, session_id, doc_id):
        captured["session_id"] = session_id
        captured["doc_id"] = doc_id
        return []

    monkeypatch.setattr(_tools, "hybrid_search", fake_hybrid_search)

    fake_conn = object()  # non-None sentinel
    execute_tool(
        "search",
        {"query": "test"},
        conn=fake_conn,  # type: ignore[arg-type]
        session_id="sess-abc",
        doc_id=None,
    )
    assert captured["session_id"] == "sess-abc"
    assert captured["doc_id"] is None
