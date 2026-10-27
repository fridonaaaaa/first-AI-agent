"""
Bulk-load HotpotQA paragraphs into the database.

Uses batched INSERT via execute_values so the entire batch of BATCH_SIZE
documents costs only ~5 round-trips regardless of corpus size.

Usage:
    python scripts/load_hotpotqa.py [--batch 50] [--max-docs N]
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import uuid

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import psycopg2.extras

from app.core.db import open_connection
from app.core.logging import configure_logging, get_logger
from app.ingestion.chunkers import chunk_document
from app.ingestion.dedupe import DedupeIndex
from app.ingestion.parsers import parse_hotpotqa_record
from app.ingestion.tokeniser import tokenise

MANIFEST_PATH = ROOT / "data" / "manifest.json"
log = get_logger("load_hotpotqa")


def _tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return {term: cnt / n for term, cnt in counts.items()}


def flush_batch(conn, docs, chunks, postings, total_tokens: int) -> None:
    """Write one batch to DB in 5 round-trips then commit."""
    if not docs:
        return
    cur = conn.cursor()

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO documents
             (id, "source", title, format, version, content_hash, language, synthetic, metadata)
           VALUES %s ON CONFLICT DO NOTHING""",
        docs,
    )

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO chunks
             (id, doc_id, text, strategy, dup_group, position, content_hash, metadata)
           VALUES %s ON CONFLICT DO NOTHING""",
        chunks,
    )

    if postings:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO bm25_postings (term, chunk_id, tf) VALUES %s ON CONFLICT DO NOTHING",
            postings,
        )

    # Aggregate term counts across all chunks in this batch
    term_counts: dict[str, int] = {}
    for term, _chunk_id, _tf in postings:
        term_counts[term] = term_counts.get(term, 0) + 1

    if term_counts:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO bm25_stats (term, df, idf) VALUES %s "
            "ON CONFLICT (term) DO UPDATE SET df = bm25_stats.df + EXCLUDED.df",
            [(t, cnt, 0) for t, cnt in term_counts.items()],
        )

    n_chunks = len(chunks)
    cur.execute(
        """UPDATE bm25_corpus SET
               avg_doc_len = (avg_doc_len * total_docs + %s) / (total_docs + %s),
               total_docs  = total_docs + %s,
               updated_at  = NOW()
           WHERE id = 1""",
        (total_tokens, n_chunks, n_chunks),
    )

    conn.commit()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=50,
                        help="Documents per DB flush (default 50)")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Stop after N documents (0 = load all)")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print("ERROR: data/manifest.json not found. Run scripts/build_subset.py first.")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text())
    target_ids = set(manifest["question_ids"])
    log.info("manifest_loaded", question_count=len(target_ids))

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    conn = open_connection()

    # Load all existing hashes once — avoids per-doc SELECT round-trips
    cur = conn.cursor()
    cur.execute("SELECT content_hash FROM documents")
    seen_hashes: set[str] = {row[0] for row in cur.fetchall()}
    log.info("existing_docs", count=len(seen_hashes))

    dedupe = DedupeIndex()

    # Pending batch accumulators
    b_docs: list = []
    b_chunks: list = []
    b_postings: list = []
    b_total_tokens = 0

    added = skipped = errors = 0
    batch_start = time.time()

    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    for split in ("validation", "train"):
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        for row in dataset:
            if row["id"] not in target_ids:
                continue
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"]
            ):
                record = {
                    "question_id": row["id"],
                    "title": title,
                    "text": " ".join(sentences),
                    "answer": row["answer"],
                    "type": row["type"],
                    "level": row["level"],
                }
                try:
                    parsed = parse_hotpotqa_record(record)

                    if parsed.content_hash in seen_hashes:
                        skipped += 1
                        continue

                    chunks = chunk_document(parsed, strategy="sentence_window")
                    doc_id = str(uuid.uuid4())

                    b_docs.append((
                        doc_id, parsed.source, parsed.title, parsed.format,
                        1, parsed.content_hash, parsed.language,
                        parsed.metadata.get("synthetic", False),
                        json.dumps(parsed.metadata),
                    ))

                    for chunk in chunks:
                        result = dedupe.check(chunk.text)
                        dup_group = None
                        if result and not result.is_exact_dup:
                            dup_group = dedupe.register(result)
                        elif result and result.is_exact_dup:
                            continue

                        chunk_id = str(uuid.uuid4())
                        chash = hashlib.sha256(chunk.text.encode()).hexdigest()
                        tokens = tokenise(chunk.text)
                        tok_count = len(tokens)
                        meta = json.dumps({**chunk.metadata, "token_count": tok_count})

                        b_chunks.append((
                            chunk_id, doc_id, chunk.text, chunk.strategy,
                            dup_group, chunk.position, chash, meta,
                        ))

                        for term, tf in _tf(tokens).items():
                            b_postings.append((term, chunk_id, tf))

                        b_total_tokens += tok_count

                    seen_hashes.add(parsed.content_hash)
                    added += 1

                    if added % args.batch == 0:
                        flush_batch(conn, b_docs, b_chunks, b_postings, b_total_tokens)
                        b_docs, b_chunks, b_postings, b_total_tokens = [], [], [], 0
                        elapsed = time.time() - batch_start
                        log.info("progress",
                                 added=added, skipped=skipped, errors=errors,
                                 rate=f"{args.batch / elapsed:.1f} docs/s")
                        batch_start = time.time()

                    if args.max_docs and added >= args.max_docs:
                        break

                except Exception as exc:
                    errors += 1
                    log.warning("row_error", error=str(exc))

            if args.max_docs and added >= args.max_docs:
                break

        if added >= manifest["unique_paragraphs"] - len(seen_hashes) + added:
            break
        if args.max_docs and added >= args.max_docs:
            break

    # Flush remaining
    flush_batch(conn, b_docs, b_chunks, b_postings, b_total_tokens)
    conn.close()

    log.info("load_complete", added=added, skipped=skipped, errors=errors)
    print(f"\nDone. Added: {added}  Skipped: {skipped}  Errors: {errors}")


if __name__ == "__main__":
    main()
