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
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.embeddings import encode_one, get_stored_model_name
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import CandidateChunk, coverage_rerank
from app.retrieval.vector_store import search_dense

log = get_logger("hybrid")

# How many candidates to fetch from each retriever before fusing.
# Larger → better recall at the cost of more BM25 DB rows fetched.
_CANDIDATE_POOL = 40


def _fetch_chunk_rows(
    chunk_ids: list[str],
    conn: "psycopg2.extensions.connection",
) -> dict[str, dict]:
    """Return {chunk_id: {text, dup_group, doc_id, title}} for the given ids."""
    if not chunk_ids:
        return {}
    placeholders = ",".join(["%s"] * len(chunk_ids))
    sql = f"""
        SELECT c.id, c.text, c.dup_group, c.doc_id::text, d.title
        FROM   chunks c
        JOIN   documents d ON d.id = c.doc_id
        WHERE  c.id IN ({placeholders})
    """
    with conn.cursor() as cur:
        cur.execute(sql, chunk_ids)
        rows = cur.fetchall()
    return {r[0]: {"text": r[1], "dup_group": r[2], "doc_id": r[3], "title": r[4]} for r in rows}


def hybrid_search(
    query: str,
    sub_queries: list[str],
    top_k: int = 10,
    lambda_: float = 0.5,
    conn: "psycopg2.extensions.connection | None" = None,
    doc_id: str | None = None,
    session_id: str | None = None,
) -> list[CandidateChunk]:
    """
    Full hybrid retrieval for one query.

    Args:
        query:       The main question (used for dense + BM25 search).
        sub_queries: Decomposed hops for coverage reranking.
        top_k:       Number of chunks to return.
        lambda_:     Relevance vs coverage trade-off (passed to coverage_rerank).
        conn:        Optional existing DB connection.
        doc_id:      When set, restrict search to chunks from this document only.
        session_id:  When set (and doc_id is not set), include only global docs
                     (session_id IS NULL) plus this session's uploaded docs.

    Returns:
        Up to top_k CandidateChunk objects, ordered by the reranker.
    """
    settings = get_settings()

    def _run(c: "psycopg2.extensions.connection") -> list[CandidateChunk]:
        # Guard: skip dense search if the stored embedding model doesn't match the
        # active model — searching across different vector spaces returns garbage.
        stored_model = get_stored_model_name(c)
        active_model = settings.embedding_model
        use_dense = not (stored_model and stored_model != active_model)
        if not use_dense:
            log.warning(
                "embedding_model_mismatch",
                stored=stored_model,
                active=active_model,
                fallback="bm25_only",
            )

        # 1. Dense search (skipped when embedding models don't match)
        dense_results: list[tuple[str, float]] = []
        if use_dense:
            query_vec = encode_one(query)
            dense_results = search_dense(
                query_vec, top_k=_CANDIDATE_POOL, conn=c, doc_id=doc_id, session_id=session_id
            )

        # 2. BM25 search
        bm25 = BM25Searcher(conn=c, k1=settings.bm25_k1, b=settings.bm25_b)
        bm25_results = bm25.search(
            query, top_k=_CANDIDATE_POOL, doc_id=doc_id, session_id=session_id
        )

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
        cosine_map = {cid: sim for cid, sim in dense_results}
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
                    doc_id=rows[cid]["doc_id"],
                    title=rows[cid]["title"],
                    cosine_sim=cosine_map.get(cid, 0.0),
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
