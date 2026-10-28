"""
Document registry: add, update, delete, list documents in Postgres.

All mutations that touch chunks also keep bm25_postings, bm25_stats,
and bm25_corpus consistent — always inside a single transaction.

Callers pass an open psycopg2 connection.  Commit/rollback is the
caller's responsibility (use app.core.db.get_db for single requests,
or manage manually for bulk jobs).
"""

import hashlib
import json
import uuid
from typing import Any

import psycopg2.extras

from app.ingestion.chunkers import Chunk
from app.ingestion.dedupe import DedupeIndex, content_hash as compute_content_hash
from app.ingestion.parsers import ParsedDocument
from app.ingestion.tokeniser import tokenise


# ── helpers ──────────────────────────────────────────────────────────────────

def _tf(tokens: list[str]) -> dict[str, float]:
    """Compute normalised term frequency: count / total_tokens."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return {term: cnt / n for term, cnt in counts.items()}


def _insert_chunk(cur, doc_id: str, chunk: Chunk, dup_group: str | None) -> tuple[str, int]:
    """Insert one chunk row. Returns (chunk_id, token_count)."""
    chunk_id = str(uuid.uuid4())
    chash = hashlib.sha256(chunk.text.encode()).hexdigest()
    tokens = tokenise(chunk.text)
    token_count = len(tokens)
    # token_count stored in metadata so BM25 can compute length normalisation
    meta = json.dumps({**chunk.metadata, "token_count": token_count})
    cur.execute(
        """
        INSERT INTO chunks (id, doc_id, text, strategy, dup_group, position, content_hash, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (chunk_id, doc_id, chunk.text, chunk.strategy,
         dup_group, chunk.position, chash, meta),
    )
    return chunk_id, token_count


def _index_chunk(cur, chunk_id: str, chunk_text: str) -> int:
    """Insert BM25 postings for one chunk. Returns token count."""
    tokens = tokenise(chunk_text)
    tf_dict = _tf(tokens)
    if tf_dict:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO bm25_postings (term, chunk_id, tf) VALUES %s ON CONFLICT DO NOTHING",
            [(term, chunk_id, tf) for term, tf in tf_dict.items()],
        )
    return len(tokens)


def _stats_increment(cur, terms: set[str]) -> None:
    """Increment df for each term; insert with df=1 if new. Single round-trip via execute_values."""
    if not terms:
        return
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO bm25_stats (term, df, idf) VALUES %s "
        "ON CONFLICT (term) DO UPDATE SET df = bm25_stats.df + 1",
        [(t, 1, 0) for t in terms],
    )


def _stats_decrement(cur, terms: set[str]) -> None:
    """Decrement df for each term; delete the row when df hits 0."""
    if not terms:
        return
    psycopg2.extras.execute_values(
        cur,
        "UPDATE bm25_stats SET df = GREATEST(df - 1, 0) FROM (VALUES %s) AS v(t) WHERE term = v.t",
        [(t,) for t in terms],
    )
    cur.execute("DELETE FROM bm25_stats WHERE df = 0")


def _corpus_add(cur, token_count: int) -> None:
    """Increment total_docs and update rolling avg_doc_len."""
    cur.execute(
        """
        UPDATE bm25_corpus SET
            avg_doc_len = (avg_doc_len * total_docs + %s) / (total_docs + 1),
            total_docs  = total_docs + 1,
            updated_at  = NOW()
        WHERE id = 1
        """,
        (token_count,),
    )


def _corpus_remove(cur, token_count: int) -> None:
    """Decrement total_docs and update rolling avg_doc_len."""
    cur.execute(
        """
        UPDATE bm25_corpus SET
            avg_doc_len = CASE
                WHEN total_docs > 1 THEN (avg_doc_len * total_docs - %s) / (total_docs - 1)
                ELSE 0
            END,
            total_docs = GREATEST(total_docs - 1, 0),
            updated_at = NOW()
        WHERE id = 1
        """,
        (token_count,),
    )


# ── public API ────────────────────────────────────────────────────────────────

def add_document(
    conn,
    parsed: ParsedDocument,
    chunks: list[Chunk],
    dedupe_index: DedupeIndex | None = None,
    session_id: str | None = None,
) -> str | None:
    """
    Insert a document and its chunks in one transaction.

    Returns the new doc_id (UUID string), or None if the document
    is an exact duplicate (same content_hash already in the DB).

    session_id: when set, document is scoped to that chat session.
    Dedup is scoped to the same session (different sessions may have identical docs).
    """
    chash = parsed.content_hash
    cur = conn.cursor()

    # Duplicate = same content AND same title, within the same session scope.
    if session_id is not None:
        cur.execute(
            "SELECT id FROM documents WHERE content_hash = %s AND title = %s AND session_id = %s",
            (chash, parsed.title, session_id),
        )
    else:
        cur.execute(
            "SELECT id FROM documents WHERE content_hash = %s AND title = %s AND session_id IS NULL",
            (chash, parsed.title),
        )
    if cur.fetchone():
        return None  # exact duplicate — skip

    doc_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO documents (id, "source", title, format, version,
                               content_hash, language, synthetic, metadata, session_id)
        VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s)
        """,
        (doc_id, parsed.source, parsed.title, parsed.format,
         chash, parsed.language, parsed.metadata.get("synthetic", False),
         json.dumps(parsed.metadata), session_id),
    )

    for chunk in chunks:
        result = dedupe_index.check(chunk.text) if dedupe_index else None
        dup_group = None
        if result:
            if result.is_exact_dup:
                continue  # skip exact-dup chunk
            dup_group = dedupe_index.register(result)

        chunk_id, tok_count = _insert_chunk(cur, doc_id, chunk, dup_group)
        _index_chunk(cur, chunk_id, chunk.text)
        terms = set(tokenise(chunk.text))
        _stats_increment(cur, terms)
        _corpus_add(cur, tok_count)

    return doc_id


def update_document(
    conn,
    doc_id: str,
    parsed: ParsedDocument,
    chunks: list[Chunk],
) -> None:
    """
    Replace an existing document's chunks in one transaction.
    BM25 stats are kept consistent: old terms decremented, new terms incremented.
    """
    cur = conn.cursor()

    # collect terms and token counts from old chunks before deleting
    cur.execute(
        "SELECT c.id, c.text FROM chunks c WHERE c.doc_id = %s",
        (doc_id,),
    )
    old_chunks = cur.fetchall()
    for old_chunk_id, old_text in old_chunks:
        terms = set(tokenise(old_text))
        tok_count = len(tokenise(old_text))
        _stats_decrement(cur, terms)
        _corpus_remove(cur, tok_count)

    cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    # update document row
    chash = parsed.content_hash
    cur.execute(
        """
        UPDATE documents SET title=%s, "source"=%s, content_hash=%s,
            version=version+1, metadata=%s, updated_at=NOW()
        WHERE id = %s
        """,
        (parsed.title, parsed.source, chash, json.dumps(parsed.metadata), doc_id),
    )

    for chunk in chunks:
        chunk_id, tok_count = _insert_chunk(cur, doc_id, chunk, None)
        _index_chunk(cur, chunk_id, chunk.text)
        terms = set(tokenise(chunk.text))
        _stats_increment(cur, terms)
        _corpus_add(cur, tok_count)


def delete_document(conn, doc_id: str) -> None:
    """
    Delete a document. Chunks and postings are removed via CASCADE.
    BM25 stats are updated for all affected terms.
    """
    cur = conn.cursor()

    cur.execute("SELECT text FROM chunks WHERE doc_id = %s", (doc_id,))
    old_texts = [row[0] for row in cur.fetchall()]

    for text in old_texts:
        terms = set(tokenise(text))
        tok_count = len(tokenise(text))
        _stats_decrement(cur, terms)
        _corpus_remove(cur, tok_count)

    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def list_documents(conn, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """Return a page of document summaries."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, "source", format, version, language, created_at
        FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s
        """,
        (limit, offset),
    )
    cols = ["id", "source", "title", "format", "version", "language", "created_at"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_document(conn, doc_id: str) -> dict[str, Any] | None:
    """Return one document with its chunk count, or None if not found."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT d.id, d.title, d."source", d.format, d.version, d.language,
               d.content_hash, d.metadata, d.created_at,
               COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.doc_id = d.id
        WHERE d.id = %s
        GROUP BY d.id
        """,
        (doc_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = ["id", "title", "source", "format", "version", "language",
            "content_hash", "metadata", "created_at", "chunk_count"]
    return dict(zip(cols, row))
