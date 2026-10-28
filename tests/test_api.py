"""
Unit tests for the FastAPI endpoints.

All DB and pipeline calls are mocked — no real DB, no LLM, no network.
Uses the FastAPI TestClient (httpx-backed, synchronous).
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.settings import get_settings

client = TestClient(app, raise_server_exceptions=False)

ADMIN_TOKEN = get_settings().admin_token


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

def _mock_pipeline_result(answer="Test answer.", abstained=False):
    from app.rag.pipeline import PipelineResult
    from app.rag.abstention import AbstentionDecision
    from app.rag.citation import CitationResult

    return PipelineResult(
        answer=answer,
        language="english",
        rewritten_query="test query",
        sub_queries=["test sub-query"],
        chunks=[],
        conflicts=[],
        abstention=AbstentionDecision(
            should_abstain=abstained,
            reason="sufficient" if not abstained else "low_score",
            top_score=0.5,
        ),
        citation_check=CitationResult(
            verified_answer=answer,
            valid_ids=[],
            invalid_ids=[],
        ),
        steps_used=1,
    )


def test_ask_returns_answer():
    with patch("app.api.routes.ask.run_pipeline", return_value=_mock_pipeline_result()):
        res = client.post("/ask", json={"question": "Who invented the telephone?"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "Test answer."
    assert data["language"] == "english"
    assert data["abstained"] is False


def test_ask_empty_question_rejected():
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422


def test_ask_abstained_response():
    with patch("app.api.routes.ask.run_pipeline",
               return_value=_mock_pipeline_result(abstained=True)):
        res = client.post("/ask", json={"question": "impossible question"})
    assert res.status_code == 200
    assert res.json()["abstained"] is True


def test_ask_pipeline_error_returns_500_message():
    from app.core.errors import InvestigationEngineError
    with patch("app.api.routes.ask.run_pipeline",
               side_effect=InvestigationEngineError("db down")):
        res = client.post("/ask", json={"question": "test"})
    assert res.status_code == 200   # graceful error response, not 500
    assert res.json()["answer"]     # non-empty error message


# ---------------------------------------------------------------------------
# /documents (admin-protected writes)
# ---------------------------------------------------------------------------

def _mock_add_document(conn, parsed, chunks, dedupe, session_id=None):
    return "doc-abc-123"


def test_upload_document_success():
    mock_chunks = [MagicMock()]
    with patch("app.api.routes.documents.add_document", side_effect=_mock_add_document), \
         patch("app.api.routes.documents.chunk_document", return_value=mock_chunks), \
         patch("app.api.routes.documents.get_db") as mock_db:
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.__exit__  = MagicMock(return_value=False)
        res = client.post(
            "/documents",
            json={"title": "Test", "text": "Some content here.", "format": "text",
                  "source": "", "language": "en", "chunking_strategy": "sentence_window"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
    assert res.status_code == 200
    assert res.json()["doc_id"] == "doc-abc-123"


def test_upload_document_missing_token():
    res = client.post(
        "/documents",
        json={"title": "T", "text": "content", "format": "text",
              "source": "", "language": "en", "chunking_strategy": "sentence_window"},
    )
    assert res.status_code == 422   # missing required header


def test_upload_document_wrong_token():
    res = client.post(
        "/documents",
        json={"title": "T", "text": "content", "format": "text",
              "source": "", "language": "en", "chunking_strategy": "sentence_window"},
        headers={"X-Admin-Token": "wrong-token"},
    )
    assert res.status_code == 401


def test_upload_duplicate_returns_409():
    with patch("app.api.routes.documents.add_document", return_value=None), \
         patch("app.api.routes.documents.chunk_document", return_value=[MagicMock()]), \
         patch("app.api.routes.documents.get_db") as mock_db:
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.__exit__  = MagicMock(return_value=False)
        res = client.post(
            "/documents",
            json={"title": "T", "text": "content", "format": "text",
                  "source": "", "language": "en", "chunking_strategy": "sentence_window"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
    assert res.status_code == 409


def test_delete_document_wrong_token():
    res = client.delete("/documents/some-id", headers={"X-Admin-Token": "bad"})
    assert res.status_code == 401


def test_list_documents_open():
    with patch("app.api.routes.documents.list_documents", return_value=[]), \
         patch("app.api.routes.documents.get_db") as mock_db:
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.__exit__  = MagicMock(return_value=False)
        res = client.get("/documents")
    assert res.status_code == 200
    assert res.json() == []


# ---------------------------------------------------------------------------
# Rate limiter (basic check)
# ---------------------------------------------------------------------------

def test_rate_limit_middleware_imported():
    from app.api.middleware import check_rate_limit
    assert callable(check_rate_limit)
