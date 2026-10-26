from app.ingestion.dedupe import (
    content_hash,
    simhash,
    hamming_distance,
    DedupeIndex,
)


def test_content_hash_identical_texts():
    assert content_hash("Hello World") == content_hash("hello   world")


def test_content_hash_different_texts():
    assert content_hash("Hello World") != content_hash("Hello Earth")


def test_simhash_identical_texts_same_fingerprint():
    assert simhash("The quick brown fox") == simhash("The quick brown fox")


def test_simhash_similar_texts_small_distance():
    # Change one word — distance should be small (near-duplicate)
    fp1 = simhash("The outage was caused by a config change on 2021-01-01.")
    fp2 = simhash("The outage was caused by a config change on 2021-01-02.")
    assert hamming_distance(fp1, fp2) <= 10


def test_simhash_different_texts_large_distance():
    fp1 = simhash("PostgreSQL is a relational database.")
    fp2 = simhash("The moon orbits the Earth once every 27 days.")
    assert hamming_distance(fp1, fp2) > 5


def test_hamming_distance_identical():
    assert hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_all_different():
    assert hamming_distance(0, 0xFFFFFFFFFFFFFFFF) == 64


def test_dedupe_index_exact_duplicate():
    index = DedupeIndex()
    text = "Exact duplicate text for testing purposes."
    result1 = index.check(text)
    assert not result1.is_exact_dup
    index.register(result1)

    result2 = index.check(text)
    assert result2.is_exact_dup
    assert result2.dup_group is not None


def test_dedupe_index_near_duplicate_linked():
    index = DedupeIndex(threshold=10)
    text1 = "The failure occurred on 2021-03-15 due to a misconfigured load balancer."
    text2 = "The failure occurred on 2021-03-16 due to a misconfigured load balancer."

    result1 = index.check(text1)
    group1 = index.register(result1)

    result2 = index.check(text2)
    assert not result2.is_exact_dup
    assert result2.dup_group == group1


def test_dedupe_index_exact_dup_same_group_as_near_dup():
    """Exact duplicate gets the same dup_group as the original."""
    index = DedupeIndex()
    text = "Some text for deduplication."
    result = index.check(text)
    group = index.register(result)

    result_dup = index.check(text)
    assert result_dup.is_exact_dup
    assert result_dup.dup_group == group


def test_dedupe_index_unique_texts_get_different_groups():
    index = DedupeIndex()
    texts = [
        "PostgreSQL stores data reliably.",
        "The moon is 384400 km from Earth.",
        "Python is a programming language.",
    ]
    groups = []
    for t in texts:
        r = index.check(t)
        groups.append(index.register(r))
    assert len(set(groups)) == 3


def test_dedupe_index_near_dup_not_deleted():
    """Near-duplicates must be kept (linked, not removed)."""
    index = DedupeIndex(threshold=10)
    text1 = "Service went down at 14:00 due to disk failure."
    text2 = "Service went down at 15:00 due to disk failure."

    r1 = index.check(text1)
    index.register(r1)
    r2 = index.check(text2)

    # Near-dup should NOT be flagged as exact dup — it must be kept
    assert not r2.is_exact_dup
    # But it should be linked to the same group
    assert r2.dup_group is not None
