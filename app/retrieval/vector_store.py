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
from app.core.errors import DatabaseError


def search_dense(
    query_vector: list[float],
    top_k: int = 20,
    conn: "psycopg2.extensions.connection | None" = None,
) -> list[tuple[str, float]]:
    """
    Find the top_k most similar chunks to query_vector.

    Args:
        query_vector: The query embedding (must match the stored dimension).
        top_k:        Maximum number of results.
        conn:         Optional existing connection (for testing / bulk jobs).
                      If None, a fresh connection is opened and closed.

    Returns:
        List of (chunk_id, similarity) sorted by similarity descending.
        similarity = 1 − cosine_distance ∈ [−1, 1]; 1 = identical.
    """
    sql = """
        SELECT chunk_id,
               1 - (embedding <=> %s::vector) AS similarity
        FROM   chunks
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
    """
    vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"

    def _run(c: "psycopg2.extensions.connection") -> list[tuple[str, float]]:
        with c.cursor() as cur:
            cur.execute(sql, (vec_str, vec_str, top_k))
            rows = cur.fetchall()
        return [(row[0], float(row[1])) for row in rows]

    if conn is not None:
        return _run(conn)

    try:
        with get_db() as c:
            return _run(c)
    except Exception as exc:
        raise DatabaseError(f"Dense search failed: {exc}") from exc
