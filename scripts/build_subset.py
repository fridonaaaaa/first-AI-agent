"""
Build the HotpotQA paragraph subset and write data/manifest.json.

Usage:
    python scripts/build_subset.py [--seed 42] [--target 50000]

The script loads HotpotQA distractor config from Hugging Face, extracts
unique paragraphs from the context field, and keeps adding questions
(validation first, then train) until the target unique-paragraph count
is reached.  It writes data/manifest.json (committed) but NOT the text
itself (gitignored via data/).
"""

import argparse
import hashlib
import json
import pathlib
import random
import sys


TARGET_DEFAULT = 50_000
SEED_DEFAULT = 42

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def paragraph_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def extract_paragraphs(row: dict) -> list[dict]:
    """Return one dict per context paragraph with its source question id."""
    paragraphs = []
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        text = " ".join(sentences)
        paragraphs.append(
            {
                "question_id": row["id"],
                "title": title,
                "text": text,
                "hash": paragraph_hash(text),
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
            }
        )
    return paragraphs


def build_subset(seed: int, target: int) -> dict:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)

    seen_hashes: set[str] = set()
    kept_question_ids: list[str] = []
    unique_count = 0

    for split in ("validation", "train"):
        print(f"Loading HotpotQA distractor/{split} …")
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=True)

        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        for idx in indices:
            row = dataset[idx]
            paras = extract_paragraphs(row)
            new_hashes = {p["hash"] for p in paras} - seen_hashes
            if new_hashes:
                seen_hashes.update(new_hashes)
                unique_count += len(new_hashes)
                kept_question_ids.append(row["id"])

            if unique_count >= target:
                break

        if unique_count >= target:
            print(f"Reached {unique_count} unique paragraphs after {split} split.")
            break

    return {
        "seed": seed,
        "target": target,
        "unique_paragraphs": unique_count,
        "question_ids": kept_question_ids,
        "question_count": len(kept_question_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HotpotQA paragraph subset manifest.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"

    manifest = build_subset(args.seed, args.target)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Unique paragraphs : {manifest['unique_paragraphs']}")
    print(f"Questions used    : {manifest['question_count']}")
    print(f"Seed              : {manifest['seed']}")


if __name__ == "__main__":
    main()
