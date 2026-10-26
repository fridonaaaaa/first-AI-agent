"""
Shared tokeniser used by both the ingestion registry (for BM25 indexing)
and the BM25 retriever (for query tokenisation).
They must always use the same function so index and query match.

Strategy: lowercase, keep alphanumeric tokens of length >= 2,
remove a small English stopword list.
Documents are English; Georgian queries are translated before BM25.
"""

import re

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "not", "this", "that", "it", "its", "as",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "also", "which", "who", "when", "where", "what", "how", "its", "been",
})


def tokenise(text: str) -> list[str]:
    """
    Split text into lowercase alphanumeric tokens, drop stopwords and
    single-character tokens.  Returns a list (not a set) to preserve count
    for TF computation.
    """
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in raw if len(t) >= 2 and t not in STOPWORDS]
