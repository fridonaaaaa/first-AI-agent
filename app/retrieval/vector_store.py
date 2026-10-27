"""
Vector store — dense nearest-neighbour search via pgvector.

Uses the `<=>` cosine distance operator from the pgvector extension.
Returns (chunk_id, similarity_score) pairs sorted by similarity descending.

Similarity is computed as 1 − cosine_distance so callers always get a
higher = better score, consistent with BM25 and RRF conventions.

The chunks table stores embeddings as `VECTOR(384)`. The column was created
with `CREATE INDEX … USING ivfflat … WITH (lists = 100)` in schema.sql.
"""

from __future__ import annotations

import psycopg2

from app.core.db import get_db
from app.core.errors import DatabaseError, EmbeddingError
from app.retrieval.embeddings import encode


def search_dense(
    query_vector: list[float],
    top_k: int = 20,
    conn: "psycopg2.extensions.connection | None" = None,
    doc_id: str | None = None,
) -> list[tuple[str, float]]:
    """
    Find the top_k most similar chunks to query_vector.

    Args:
        query_vector: The query embedding (must match the stored dimension).
        top_k:        Maximum number of results.
        conn:         Optional existing connection (for testing / bulk jobs).
                      If None, a fresh connection is opened and closed.
        doc_id:       When set, restrict search to chunks from this document.

    Returns:
        List of (chunk_id, similarity) sorted by similarity descending.
        similarity = 1 − cosine_distance ∈ [−1, 1]; 1 = identical.
    """
    vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"

    if doc_id is not None:
        sql = """
            SELECT id,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM   chunks
            WHERE  doc_id = %s::uuid
              AND  embedding IS NOT NULL
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """
        params = (vec_str, doc_id, vec_str, top_k)
    else:
        sql = """
            SELECT id,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM   chunks
            WHERE  embedding IS NOT NULL
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """
        params = (vec_str, vec_str, top_k)

    def _run(c: "psycopg2.extensions.connection") -> list[tuple[str, float]]:
        with c.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [(row[0], float(row[1])) for row in rows]

    if conn is not None:
        return _run(conn)

    try:
        with get_db() as c:
            return _run(c)
    except Exception as exc:
        raise DatabaseError(f"Dense search failed: {exc}") from exc


def embed_new_chunks(conn: "psycopg2.extensions.connection", doc_id: str) -> int:
    """
    Compute and store embeddings for all unembedded chunks of a document.

    Called immediately after add_document so user-uploaded docs are
    searchable right away. Returns the number of chunks embedded.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, text FROM chunks WHERE doc_id = %s::uuid AND embedding IS NULL",
            (doc_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return 0

    chunk_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    vectors = encode(texts)

    with conn.cursor() as cur:
        for chunk_id, vec in zip(chunk_ids, vectors):
            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
            cur.execute(
                "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                (vec_str, chunk_id),
            )

    return len(chunk_ids)
