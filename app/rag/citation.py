"""
Citation verifier — post-generation check.

After the LLM produces an answer with [chunk_id] markers, this module:
  1. Extracts all cited chunk IDs from the answer text.
  2. Checks each one exists in the retrieved set.
  3. Removes or flags any citation not in the retrieved set (hallucinated IDs).

This is the "post-check" step described in the RAG design: it prevents the
model from fabricating chunk IDs that were never retrieved.

The answer text uses the format: "... some claim [abc-123] ..."
Multiple citations on one claim: "... claim [abc-123][def-456] ..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE_RE = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")


@dataclass
class CitationResult:
    verified_answer: str
    valid_ids: list[str]
    invalid_ids: list[str] = field(default_factory=list)


def extract_citations(text: str) -> list[str]:
    """Return all [chunk_id] references found in text, in order, deduplicated."""
    seen: dict[str, None] = {}
    for m in _CITE_RE.finditer(text):
        seen[m.group(1)] = None
    return list(seen)


def verify_citations(answer: str, retrieved_ids: set[str]) -> CitationResult:
    """
    Remove citations not in retrieved_ids from the answer.

    Any [chunk_id] that does not appear in retrieved_ids is stripped from the
    answer text. The caller can log the invalid_ids for debugging.

    Args:
        answer:        The LLM-generated answer string.
        retrieved_ids: The set of chunk_ids that were actually retrieved and
                       passed to the LLM as context.

    Returns:
        CitationResult with the cleaned answer and lists of valid/invalid IDs.
    """
    cited = extract_citations(answer)
    valid = [cid for cid in cited if cid in retrieved_ids]
    invalid = [cid for cid in cited if cid not in retrieved_ids]

    cleaned = answer
    for cid in invalid:
        cleaned = cleaned.replace(f"[{cid}]", "")

    # Collapse double spaces left by removed citations
    cleaned = re.sub(r"  +", " ", cleaned).strip()

    return CitationResult(
        verified_answer=cleaned,
        valid_ids=valid,
        invalid_ids=invalid,
    )
