"""
Bulk embedding job — compute and store vectors for every chunk in the DB.

Usage:
    python scripts/embed_corpus.py [--batch 256] [--dry-run]

Reads all chunks that have embedding = NULL (or that were added since the last
run), encodes them in batches with the configured embedding model, and writes
the vector back to the chunks table via pgvector.

The model name is read from settings (EMBEDDING_MODEL env var). It must be
the same model used for all future queries — changing it requires re-running
this script on the full corpus.

Resume-safe: chunks with an existing embedding are skipped. Run again after
a failure to pick up where it left off.
"""

from __future__ import annotations

import argparse
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import psycopg2.extras

from app.core.db import open_connection
from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings
from app.retrieval.embeddings import encode

log = get_logger("embed_corpus")


def _fetch_unembedded(conn, limit: int) -> list[tuple[str, str]]:
    """Return up to `limit` (chunk_id, text) pairs with no embedding yet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, text
            FROM   chunks
            WHERE  embedding IS NULL
            LIMIT  %s
            """,
            (limit,),
        )
        return cur.fetchall()


def _write_embeddings(
    conn,
    pairs: list[tuple[str, list[float]]],
) -> None:
    """Write (chunk_id, vector) pairs back to the DB in one round-trip using a temp table."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE _emb_batch (id uuid, vec text) ON COMMIT DROP"
        )
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO _emb_batch (id, vec) VALUES %s",
            [
                (chunk_id, "[" + ",".join(str(x) for x in vec) + "]")
                for chunk_id, vec in pairs
            ],
        )
        cur.execute(
            "UPDATE chunks SET embedding = b.vec::vector FROM _emb_batch b WHERE chunks.id = b.id"
        )


def _record_model(conn, model_name: str, dim: int) -> None:
    """Upsert the embedding model record into bm25_corpus for future consistency checks."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE bm25_corpus
            SET    embedding_model = %s
            WHERE  embedding_model IS NULL OR embedding_model = ''
            """,
            (model_name,),
        )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Embed all un-vectorised chunks.")
    parser.add_argument("--batch", type=int, default=256,
                        help="Chunks per encode call (default 256)")
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="Seconds to wait between batches to avoid rate limits (default 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count un-embedded chunks but do not write anything")
    args = parser.parse_args()

    settings = get_settings()
    log.info("starting", model=settings.embedding_model, dim=settings.embedding_dim,
             batch=args.batch, dry_run=args.dry_run)

    conn = open_connection()

    if args.dry_run:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
            count = cur.fetchone()[0]
        print(f"Chunks without embeddings: {count}")
        conn.close()
        return

    _record_model(conn, settings.embedding_model, settings.embedding_dim)
    conn.commit()

    total = 0
    t0 = time.time()

    while True:
        rows = _fetch_unembedded(conn, limit=args.batch)
        if not rows:
            break

        chunk_ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        try:
            vectors = encode(texts)
        except Exception as exc:
            log.warning("encode_failed_retrying", batch_start=chunk_ids[0], error=str(exc))
            time.sleep(60.0)
            try:
                vectors = encode(texts)
            except Exception as exc2:
                log.error("encode_failed", batch_start=chunk_ids[0], error=str(exc2))
                conn.close()
                sys.exit(1)

        _write_embeddings(conn, list(zip(chunk_ids, vectors)))
        conn.commit()
        total += len(rows)

        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        log.info("progress", embedded=total, rate=f"{rate:.1f} chunks/s")

        if args.sleep > 0:
            time.sleep(args.sleep)

    conn.close()
    elapsed = time.time() - t0
    log.info("done", total=total, elapsed_s=f"{elapsed:.1f}")
    print(f"\nEmbedded {total} chunks in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
