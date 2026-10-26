# dedupe.py — explained

## What it does
Detects duplicate chunks during ingestion so we don't store the same text twice
(exact duplicates) and so we can find near-identical texts that might be conflicting
sources (near-duplicates).

## Two levels of detection

### Level 1 — Exact duplicate
Normalise the text (lowercase, collapse whitespace), compute SHA-256.
If the hash already exists in the index, this is an exact duplicate.
**Action:** skip storing it again; record the original's sources as aliases.

### Level 2 — Near-duplicate (SimHash)
Used for texts like "The outage happened on 2021-01-01" vs "…on 2021-01-02".
Same structure, one fact changed. SHA-256 will differ, but they are near-duplicates.

## SimHash algorithm

**Step 1 — Shingles.** Break the normalised text into overlapping character n-grams
of size k (default k=4). Example: "hello" → ["hell", "ello"].

**Step 2 — Hash each shingle.** Use MD5, take the first 8 bytes as a 64-bit integer.

**Step 3 — Vote.** Maintain a vector v of 64 integers (one per bit).
For each shingle hash h, and each bit position b:
  - if bit b of h is 1 → v[b] += 1
  - if bit b of h is 0 → v[b] -= 1

**Step 4 — Fingerprint.** Final bit b = 1 if v[b] > 0, else 0.
Pack the 64 bits into one integer.

**Why it works:** similar texts share most shingles, so their vote vectors point in
the same direction, producing fingerprints with few differing bits.

## Hamming distance
Count the number of bit positions where two fingerprints differ.
`hamming(a, b) = popcount(a XOR b)`

Implemented with a simple loop (no numpy needed, corpus is ingested once).

**Threshold:** distance ≤ 5 → near-duplicate (configurable).

## Complexity
- Shingle generation: O(n) where n = text length
- Hamming distance: O(64) = O(1)
- Index lookup: O(E) where E = number of existing fingerprints
  (acceptable for a one-time bulk ingestion; could use LSH for millions of docs)

## Why not delete near-duplicates?
Near-duplicates may be genuine conflicting sources — a synthetic doc that changes
one fact to test contradiction detection. Deleting them would destroy the conflict signal.
We link them via `dup_group` UUID so the RAG pipeline can find and compare them.

## Edge cases
- Text shorter than k chars: treated as a single shingle.
- Empty text: returns fingerprint 0 (all votes tie at 0 → bit = 0).
- Identical texts always produce the same fingerprint (deterministic).

## 5 questions an interviewer would ask

**Q1: Why SimHash and not Jaccard similarity?**
Jaccard requires comparing all pairs (O(n²)). SimHash produces a fixed-size
fingerprint once per document; comparing two fingerprints is O(1).

**Q2: Why char n-grams instead of word n-grams?**
Char n-grams are robust to minor spelling variations and work for any language.
Word n-grams would miss "2021-01-01" vs "2021-01-02" changes.

**Q3: What happens if the threshold is too high?**
Unrelated documents get linked as near-duplicates. The retriever would then
flag false conflicts. Threshold must be tuned on a sample — default 5 is conservative.

**Q4: What if two exact duplicates have different sources?**
`content_hash` collapses them to one DB record. The sources are stored as aliases
in the document's metadata so citations still point to the real origin.

**Q5: How does dup_group work at query time?**
The reranker and conflict detector fetch all chunks sharing a `dup_group` UUID.
If their texts disagree on a key fact, the conflict detector flags it for the user.
