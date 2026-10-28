"""
BM25 scorer — hand-written, no search library.

Scoring formula (Okapi BM25):

  IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )

  TF(t, d) = tf_norm * (k1 + 1)
             ─────────────────────────────────────────────────────
             tf_norm + k1 * ( (1-b)/dl  +  b/avgdl )

  score(q, d) = Σ  IDF(t) * TF(t, d)   for each query term t

Where:
  tf_norm  = (term count in chunk) / (total tokens in chunk)  — stored in bm25_postings
  dl       = token count of this chunk                        — stored in chunk metadata
  avgdl    = average token count across all chunks            — from bm25_corpus
  N        = total number of indexed chunks                   — from bm25_corpus
  df       = number of chunks containing term t               — from bm25_stats
  k1       = term frequency saturation (default 1.5)
  b        = length normalisation strength (default 0.75)

This formulation uses stored normalised TF directly; multiplying back by dl
gives the same result as the classical raw-tf formula.
"""

import math
from typing import Any

from app.ingestion.tokeniser import tokenise


def bm25_score(
    tf_norm: float,
    dl: int,
    avgdl: float,
    df: int,
    N: int,
    k1: float,
    b: float,
) -> float:
    """
    Compute BM25 score for one (term, document) pair.

    Returns 0.0 when tf_norm is 0 (term absent from document).
    """
    if tf_norm == 0.0 or dl == 0:
        return 0.0
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    tf_component = tf_norm * (k1 + 1) / (tf_norm + k1 * ((1 - b) / dl + b / avgdl))
    return idf * tf_component


class BM25Searcher:
    """
    Scores chunks for a query using stored BM25 postings.

    Two construction paths:
      - BM25Searcher(conn, k1, b)  — reads from Postgres at search time
      - BM25Searcher.from_index(...)  — in-memory index for unit tests
    """

    def __init__(self, conn=None, k1: float = 1.5, b: float = 0.75) -> None:
        self._conn = conn
        self.k1 = k1
        self.b = b
        self._index: dict[str, list[tuple[str, float, int]]] | None = None
        self._stats: dict[str, int] | None = None
        self._corpus: tuple[int, float] | None = None

    @classmethod
    def from_index(
        cls,
        postings: dict[str, list[tuple[str, float, int]]],
        stats: dict[str, int],
        corpus: tuple[int, float],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Searcher":
        """Build an in-memory searcher for testing (no DB needed)."""
        obj = cls(conn=None, k1=k1, b=b)
        obj._index = postings
        obj._stats = stats
        obj._corpus = corpus
        return obj

    def _get_corpus_stats(self) -> tuple[int, float]:
        if self._corpus is not None:
            return self._corpus
        cur = self._conn.cursor()
        cur.execute("SELECT total_docs, avg_doc_len FROM bm25_corpus WHERE id = 1")
        row = cur.fetchone()
        return (row[0], row[1]) if row else (0, 0.0)

    def _get_term_df(self, term: str) -> int:
        if self._stats is not None:
            return self._stats.get(term, 0)
        cur = self._conn.cursor()
        cur.execute("SELECT df FROM bm25_stats WHERE term = %s", (term,))
        row = cur.fetchone()
        return row[0] if row else 0

    def _get_postings(
        self, term: str, doc_id: str | None = None, session_id: str | None = None
    ) -> list[tuple[str, float, int]]:
        """Return [(chunk_id, tf_norm, dl), ...] for a term."""
        if self._index is not None:
            return self._index.get(term, [])
        cur = self._conn.cursor()
        if doc_id is not None:
            cur.execute(
                """
                SELECT bp.chunk_id, bp.tf,
                       COALESCE((c.metadata->>'token_count')::int, 1) AS dl
                FROM bm25_postings bp
                JOIN chunks c ON c.id = bp.chunk_id
                WHERE bp.term = %s AND c.doc_id = %s::uuid
                """,
                (term, doc_id),
            )
        elif session_id is not None:
            cur.execute(
                """
                SELECT bp.chunk_id, bp.tf,
                       COALESCE((c.metadata->>'token_count')::int, 1) AS dl
                FROM bm25_postings bp
                JOIN chunks c ON c.id = bp.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE bp.term = %s AND (d.session_id IS NULL OR d.session_id = %s)
                """,
                (term, session_id),
            )
        else:
            cur.execute(
                """
                SELECT bp.chunk_id, bp.tf,
                       COALESCE((c.metadata->>'token_count')::int, 1) AS dl
                FROM bm25_postings bp
                JOIN chunks c ON c.id = bp.chunk_id
                WHERE bp.term = %s
                """,
                (term,),
            )
        return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    def _fetch_all_terms(
        self,
        tokens: list[str],
        doc_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[dict[str, int], dict[str, list[tuple[str, float, int]]]]:
        """
        Batch-fetch df and postings for all tokens in two queries instead of 2N.
        Returns (df_map, postings_map).
        Only used when a DB connection is available (not the in-memory path).
        """
        placeholders = ",".join(["%s"] * len(tokens))
        cur = self._conn.cursor()

        cur.execute(
            f"SELECT term, df FROM bm25_stats WHERE term IN ({placeholders})",
            tokens,
        )
        df_map: dict[str, int] = {row[0]: row[1] for row in cur.fetchall()}

        if doc_id is not None:
            cur.execute(
                f"""
                SELECT bp.term, bp.chunk_id, bp.tf,
                       COALESCE((c.metadata->>'token_count')::int, 1) AS dl
                FROM bm25_postings bp
                JOIN chunks c ON c.id = bp.chunk_id
                WHERE bp.term IN ({placeholders}) AND c.doc_id = %s::uuid
                """,
                tokens + [doc_id],
            )
        elif session_id is not None:
            cur.execute(
                f"""
                SELECT bp.term, bp.chunk_id, bp.tf,
                       COALESCE((c.metadata->>'token_count')::int, 1) AS dl
                FROM bm25_postings bp
                JOIN chunks c ON c.id = bp.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE bp.term IN ({placeholders})
                  AND (d.session_id IS NULL OR d.session_id = %s)
                """,
                tokens + [session_id],
            )
        else:
            cur.execute(
                f"""
                SELECT bp.term, bp.chunk_id, bp.tf,
                       COALESCE((c.metadata->>'token_count')::int, 1) AS dl
                FROM bm25_postings bp
                JOIN chunks c ON c.id = bp.chunk_id
                WHERE bp.term IN ({placeholders})
                """,
                tokens,
            )
        postings_map: dict[str, list[tuple[str, float, int]]] = {}
        for term, chunk_id, tf, dl in cur.fetchall():
            postings_map.setdefault(term, []).append((chunk_id, tf, dl))

        return df_map, postings_map

    def search_index(
        self, query: str, top_k: int = 10, doc_id: str | None = None, session_id: str | None = None
    ) -> list[tuple[str, float]]:
        """
        Score all chunks matching any query term.
        Returns [(chunk_id, score), ...] sorted by score descending.
        """
        tokens = list(dict.fromkeys(tokenise(query)))  # unique, order preserved
        if not tokens:
            return []

        N, avgdl = self._get_corpus_stats()
        if N == 0:
            return []

        scores: dict[str, float] = {}

        if self._conn is not None:
            # DB path: fetch all terms in two queries instead of 2N round-trips.
            df_map, postings_map = self._fetch_all_terms(
                tokens, doc_id=doc_id, session_id=session_id
            )
            for term in tokens:
                df = df_map.get(term, 0)
                if df == 0:
                    continue
                for chunk_id, tf_norm, dl in postings_map.get(term, []):
                    s = bm25_score(tf_norm, dl, avgdl, df, N, self.k1, self.b)
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + s
        else:
            # In-memory path (used by unit tests via from_index()).
            for term in tokens:
                df = self._get_term_df(term)
                if df == 0:
                    continue
                for chunk_id, tf_norm, dl in self._get_postings(term):
                    s = bm25_score(tf_norm, dl, avgdl, df, N, self.k1, self.b)
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search(
        self, query: str, top_k: int = 10, doc_id: str | None = None, session_id: str | None = None
    ) -> list[tuple[str, float]]:
        """Alias used by hybrid.py."""
        return self.search_index(query, top_k, doc_id=doc_id, session_id=session_id)
