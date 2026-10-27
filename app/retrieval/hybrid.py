"""
Hybrid retriever — wires dense search, BM25, RRF, and coverage reranking.

Pipeline:
  1. Embed the query (dense search).
  2. BM25 keyword search.
  3. Fuse both ranked lists with Reciprocal Rank Fusion.
  4. Fetch chunk text for the top fused results.
  5. Coverage-aware rerank using the sub-queries.
  6. Return top_k CandidateChunk objects.

This module owns no algorithm logic; it only calls the four specialised
modules in order and translates between their data shapes.
"""

from __future__ import annotations

import psycopg2

from app.core.db import get_db
from app.core.errors import DatabaseError
from app.core.settings import get_settings
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.embeddings import encode_one
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import CandidateChunk, coverage_rerank
from app.retrieval.vector_store import search_dense

# How many candidates to fetch from each retriever before fusing.
# Larger → better recall at the cost of more BM25 DB rows fetched.
_CANDIDATE_POOL = 40


def _fetch_chunk_rows(
    chunk_ids: list[str],
    conn: "psycopg2.extensions.connection",
) -> dict[str, dict]:
    """Return {chunk_id: {text, dup_group}} for the given ids."""
    if not chunk_ids:
        return {}
    placeholders = ",".join(["%s"] * len(chunk_ids))
    sql = f"""
        SELECT id, text, dup_group
        FROM   chunks
        WHERE  id IN ({placeholders})
    """
    with conn.cursor() as cur:
        cur.execute(sql, chunk_ids)
        rows = cur.fetchall()
    return {r[0]: {"text": r[1], "dup_group": r[2]} for r in rows}


def hybrid_search(
    query: str,
    sub_queries: list[str],
    top_k: int = 10,
    lambda_: float = 0.5,
    conn: "psycopg2.extensions.connection | None" = None,
) -> list[CandidateChunk]:
    """
    Full hybrid retrieval for one query.

    Args:
        query:       The main question (used for dense + BM25 search).
        sub_queries: Decomposed hops for coverage reranking.
        top_k:       Number of chunks to return.
        lambda_:     Relevance vs coverage trade-off (passed to coverage_rerank).
        conn:        Optional existing DB connection.

    Returns:
        Up to top_k CandidateChunk objects, ordered by the reranker.
    """
    settings = get_settings()

    def _run(c: "psycopg2.extensions.connection") -> list[CandidateChunk]:
        # 1. Dense search
        query_vec = encode_one(query)
        dense_results = search_dense(query_vec, top_k=_CANDIDATE_POOL, conn=c)

        # 2. BM25 search
        bm25 = BM25Searcher(conn=c, k1=settings.bm25_k1, b=settings.bm25_b)
        bm25_results = bm25.search(query, top_k=_CANDIDATE_POOL)

        # 3. Fuse
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=settings.rrf_k,
        )
        fused_ids = [cid for cid, _ in fused[:_CANDIDATE_POOL]]

        # 4. Fetch text + dup_group for fused candidates
        rows = _fetch_chunk_rows(fused_ids, c)

        # Build CandidateChunk list in fused order, preserving RRF scores
        rrf_score_map = {cid: score for cid, score in fused}
        candidates: list[CandidateChunk] = []
        for cid in fused_ids:
            if cid not in rows:
                continue
            candidates.append(
                CandidateChunk(
                    chunk_id=cid,
                    rrf_score=rrf_score_map[cid],
                    text=rows[cid]["text"],
                    dup_group=rows[cid]["dup_group"],
                )
            )

        # 5. Coverage rerank
        return coverage_rerank(
            candidates,
            sub_queries=sub_queries,
            top_k=top_k,
            lambda_=lambda_,
        )

    if conn is not None:
        return _run(conn)

    try:
        with get_db() as c:
            return _run(c)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Hybrid search failed: {exc}") from exc
