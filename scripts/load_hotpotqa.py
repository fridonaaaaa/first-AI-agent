"""
Bulk-load HotpotQA paragraphs into the database.

Usage:
    python scripts/load_hotpotqa.py [--strategy sentence_window] [--batch 500]

Reads data/manifest.json (produced by build_subset.py) to know which
question IDs to load.  Skips documents that are already in the DB
(resumable via content_hash check in add_document).

Progress is logged every --batch rows.  Run again after a failure
to resume from where it stopped.
"""

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.core.db import open_connection
from app.core.logging import configure_logging, get_logger
from app.ingestion.chunkers import chunk_document
from app.ingestion.dedupe import DedupeIndex
from app.ingestion.parsers import parse_hotpotqa_record
from app.ingestion.registry import add_document

MANIFEST_PATH = ROOT / "data" / "manifest.json"
log = get_logger("load_hotpotqa")


def load_existing_hashes(conn) -> set[str]:
    """Fetch all content hashes already in the DB for resume support."""
    cur = conn.cursor()
    cur.execute("SELECT content_hash FROM documents")
    return {row[0] for row in cur.fetchall()}


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="sentence_window",
                        choices=["sentence_window", "fixed_token"])
    parser.add_argument("--batch", type=int, default=500,
                        help="Commit every N documents")
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
    existing_hashes = load_existing_hashes(conn)
    log.info("existing_docs", count=len(existing_hashes))

    dedupe = DedupeIndex()
    added = 0
    skipped = 0
    errors = 0
    batch_start = time.time()

    seen_hashes: set[str] = set(existing_hashes)

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
                    chunks = chunk_document(parsed, strategy=args.strategy)
                    doc_id = add_document(conn, parsed, chunks, dedupe)
                    if doc_id:
                        seen_hashes.add(parsed.content_hash)
                        added += 1
                    else:
                        skipped += 1

                    if (added + skipped) % args.batch == 0:
                        conn.commit()
                        elapsed = time.time() - batch_start
                        log.info("progress",
                                 added=added, skipped=skipped, errors=errors,
                                 rate=f"{args.batch / elapsed:.0f} docs/s")
                        batch_start = time.time()

                except Exception as exc:
                    errors += 1
                    conn.rollback()
                    log.warning("row_error", error=str(exc))

        if added >= manifest["unique_paragraphs"] - len(existing_hashes):
            break

    conn.commit()
    conn.close()

    log.info("load_complete", added=added, skipped=skipped, errors=errors)
    print(f"\nDone. Added: {added}  Skipped: {skipped}  Errors: {errors}")


if __name__ == "__main__":
    main()
