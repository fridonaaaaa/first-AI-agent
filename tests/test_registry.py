"""
Integration tests for the document registry.
These tests connect to the real database — they are skipped when DATABASE_URL
is not configured or the DB is unreachable.

Run with: pytest tests/test_registry.py -v
"""

import os
import uuid
import pytest

from app.ingestion.chunkers import chunk_document
from app.ingestion.dedupe import DedupeIndex
from app.ingestion.parsers import ParsedDocument
from app.ingestion.registry import (
    add_document,
    delete_document,
    get_document,
    list_documents,
    update_document,
)


def _db_available() -> bool:
    from dotenv import load_dotenv
    load_dotenv()
    url = os.getenv("DATABASE_URL", "")
    return bool(url) and "[YOUR-PASSWORD]" not in url and "[PASSWORD]" not in url


requires_db = pytest.mark.skipif(not _db_available(), reason="DB not available")


@pytest.fixture(scope="module")
def conn():
    from dotenv import load_dotenv
    load_dotenv()
    import psycopg2
    from app.core.db import open_connection
    try:
        c = open_connection()
    except psycopg2.OperationalError as exc:
        pytest.skip(f"DB unreachable: {exc}")
    yield c
    c.rollback()
    c.close()


def _make_doc(title: str = "Test Doc", text: str = "") -> ParsedDocument:
    # UUID suffix ensures each test run produces a fresh content_hash
    unique_text = text + f" run={str(uuid.uuid4())[:8]}"
    return ParsedDocument(
        title=title,
        text=unique_text,
        source=f"test://registry/{title}",
        format="txt",
    )


@requires_db
def test_add_document_creates_chunks(conn):
    doc = _make_doc("Registry Add Test", "The quick brown fox jumps over the lazy dog repeatedly.")
    chunks = chunk_document(doc, strategy="sentence_window")
    doc_id = add_document(conn, doc, chunks)
    assert doc_id is not None

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (doc_id,))
    assert cur.fetchone()[0] > 0


@requires_db
def test_add_document_exact_dup_returns_none(conn):
    doc = _make_doc("Exact Dup Test", "Exact duplicate content for deduplication testing.")
    chunks = chunk_document(doc, strategy="sentence_window")
    doc_id1 = add_document(conn, doc, chunks)
    assert doc_id1 is not None

    doc_id2 = add_document(conn, doc, chunks)
    assert doc_id2 is None


@requires_db
def test_add_document_creates_bm25_postings(conn):
    doc = _make_doc("BM25 Postings Test", "PostgreSQL stores vectors efficiently with pgvector extension.")
    chunks = chunk_document(doc, strategy="sentence_window")
    doc_id = add_document(conn, doc, chunks)
    assert doc_id is not None

    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM bm25_postings bp JOIN chunks c ON bp.chunk_id = c.id WHERE c.doc_id = %s",
        (doc_id,),
    )
    assert cur.fetchone()[0] > 0


@requires_db
def test_delete_document_removes_chunks_and_updates_stats(conn):
    doc = _make_doc("Delete Test", "This document will be deleted after insertion for testing.")
    chunks = chunk_document(doc, strategy="sentence_window")
    doc_id = add_document(conn, doc, chunks)
    assert doc_id is not None
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT total_docs FROM bm25_corpus WHERE id = 1")
    total_before = cur.fetchone()[0]

    delete_document(conn, doc_id)

    cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (doc_id,))
    assert cur.fetchone()[0] == 0

    cur.execute("SELECT total_docs FROM bm25_corpus WHERE id = 1")
    total_after = cur.fetchone()[0]
    assert total_after < total_before


@requires_db
def test_update_document_replaces_chunks(conn):
    doc = _make_doc("Update Test", "Original content before update operation.")
    chunks = chunk_document(doc, strategy="sentence_window")
    doc_id = add_document(conn, doc, chunks)
    assert doc_id is not None
    conn.commit()

    new_doc = _make_doc("Update Test v2", "Completely new content after the update was applied.")
    new_chunks = chunk_document(new_doc, strategy="sentence_window")
    update_document(conn, doc_id, new_doc, new_chunks)

    result = get_document(conn, doc_id)
    assert result is not None
    assert result["version"] == 2


@requires_db
def test_get_document_returns_correct_fields(conn):
    doc = _make_doc("Get Test", "Document for get_document function testing purposes.")
    chunks = chunk_document(doc, strategy="sentence_window")
    doc_id = add_document(conn, doc, chunks)
    assert doc_id is not None

    result = get_document(conn, doc_id)
    assert result is not None
    assert result["title"] == "Get Test"
    assert result["chunk_count"] > 0


@requires_db
def test_list_documents_returns_results(conn):
    doc = _make_doc("List Test", "Document for list_documents function testing.")
    chunks = chunk_document(doc, strategy="sentence_window")
    add_document(conn, doc, chunks)

    results = list_documents(conn, limit=10)
    assert isinstance(results, list)
    assert len(results) > 0
