"""
Duplicate detection for ingested chunks.

Two levels:
1. Exact duplicate — identical normalised text hash (SHA-256).
   Action: collapse to one record; keep all source URLs as aliases.

2. Near-duplicate — similar but not identical text (e.g. same paragraph
   with one changed number).  Detected with a hand-written SimHash.
   Action: assign the same dup_group UUID so retrievers can find them.
   Never delete near-duplicates — conflicting sources are a feature.

SimHash algorithm (64-bit):
  - Tokenise text into shingles (overlapping char n-grams of size k).
  - For each shingle, compute a 64-bit hash.
  - For each bit position b (0..63):
      v[b] += +1 if bit b of hash is 1, else -1
  - Final fingerprint: bit b = 1 if v[b] > 0 else 0.
  - Two texts are near-duplicates when hamming_distance(fp_a, fp_b) <= threshold.
"""

import hashlib
import struct
import uuid
from dataclasses import dataclass


SIMHASH_BITS = 64
SHINGLE_SIZE = 4       # char n-gram size
NEAR_DUP_THRESHOLD = 5 # hamming distance <= this → near-duplicate


@dataclass
class DedupeResult:
    content_hash: str        # SHA-256 of normalised text
    simhash: int             # 64-bit fingerprint
    is_exact_dup: bool       # True if an identical hash already exists
    dup_group: str | None    # UUID string linking near-duplicates; None if unique


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()


def _shingles(text: str, k: int = SHINGLE_SIZE) -> list[str]:
    normalised = _normalise(text)
    if len(normalised) < k:
        return [normalised]
    return [normalised[i : i + k] for i in range(len(normalised) - k + 1)]


def _hash64(token: str) -> int:
    digest = hashlib.md5(token.encode()).digest()
    return struct.unpack("<Q", digest[:8])[0]


def simhash(text: str, k: int = SHINGLE_SIZE) -> int:
    """
    Compute a 64-bit SimHash fingerprint for the given text.

    Steps:
      1. Build overlapping char-k-grams (shingles).
      2. Hash each shingle to a 64-bit integer.
      3. Accumulate a weighted vote vector v[0..63].
      4. Set bit b = 1 if v[b] > 0, else 0.

    Similar texts produce fingerprints with small Hamming distance.
    """
    v = [0] * SIMHASH_BITS
    for shingle in _shingles(text, k):
        h = _hash64(shingle)
        for b in range(SIMHASH_BITS):
            v[b] += 1 if (h >> b) & 1 else -1
    fingerprint = 0
    for b in range(SIMHASH_BITS):
        if v[b] > 0:
            fingerprint |= 1 << b
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers."""
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


class DedupeIndex:
    """
    In-memory index used during a single ingestion run.

    Tracks:
      - seen_hashes: exact-dup detection (hash → dup_group)
      - seen_fingerprints: near-dup detection (list of (simhash, dup_group))

    For production use, pre-load existing hashes and fingerprints from the DB
    before starting a bulk ingestion run.
    """

    def __init__(self, threshold: int = NEAR_DUP_THRESHOLD) -> None:
        self.threshold = threshold
        self._hashes: dict[str, str] = {}           # content_hash → dup_group
        self._fingerprints: list[tuple[int, str]] = []  # (simhash, dup_group)

    def add_existing(self, content_hash: str, sim: int, dup_group: str) -> None:
        """Pre-load a record that already exists in the DB."""
        self._hashes[content_hash] = dup_group
        self._fingerprints.append((sim, dup_group))

    def check(self, text: str) -> DedupeResult:
        """
        Check text against the index.  Returns a DedupeResult describing
        whether this text is an exact dup, a near-dup, or unique.
        After calling check(), call register() if you decide to keep the record.
        """
        chash = content_hash(text)
        sim = simhash(text)

        if chash in self._hashes:
            return DedupeResult(
                content_hash=chash,
                simhash=sim,
                is_exact_dup=True,
                dup_group=self._hashes[chash],
            )

        for existing_sim, existing_group in self._fingerprints:
            if hamming_distance(sim, existing_sim) <= self.threshold:
                return DedupeResult(
                    content_hash=chash,
                    simhash=sim,
                    is_exact_dup=False,
                    dup_group=existing_group,
                )

        return DedupeResult(
            content_hash=chash,
            simhash=sim,
            is_exact_dup=False,
            dup_group=None,
        )

    def register(self, result: DedupeResult) -> str:
        """
        Record a newly kept chunk in the index.
        Assigns a new dup_group UUID if this chunk is the first in its group.
        Returns the dup_group UUID string.
        """
        group = result.dup_group or str(uuid.uuid4())
        self._hashes[result.content_hash] = group
        self._fingerprints.append((result.simhash, group))
        return group
