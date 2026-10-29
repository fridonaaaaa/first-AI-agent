"""
Query understanding — language detection, rewriting, and decomposition.

Step 1: Detect whether the query is Georgian (simple heuristic: Georgian
Unicode block U+10D0–U+10FF).  If Georgian, translate to English first.

Step 2: Rewrite the English query into a clean, unambiguous search query
(removes filler, expands abbreviations, normalises dates).

Step 3: Decompose the rewritten query into ≤ max_sub_queries sub-queries,
each targeting one "hop" of the answer (e.g. "who invented X" + "when was
that person born").

All LLM calls use temperature=0 for deterministic, parseable output.
"""

from __future__ import annotations

import re

from app.core.errors import LLMMalformedResponseError, LLMTimeoutError, LLMError
from app.rag.llm.base import LLMClient

_GEORGIAN_RE = re.compile(r"[ა-ჿ]")

_TRANSLATE_SYSTEM = (
    "You are a precise translator. Translate the user's Georgian text to English. "
    "Output ONLY the English translation, nothing else."
)

_REWRITE_DECOMPOSE_SYSTEM = """You are a search query optimizer.
Given a question, return a JSON object with two keys:
  "rewritten": a clean, keyword-rich English search query (one string)
  "sub_queries": a list of 1–{max_sub} self-contained search queries, each
                 targeting one step of the answer. Do NOT repeat the rewritten query.
Output ONLY valid JSON."""

_DECOMPOSE_SYSTEM = """You decompose multi-hop questions into simple sub-queries.
Return a JSON object with one key "sub_queries": a list of 1–{max_sub} strings.
Each string is one self-contained search query that retrieves evidence for one
step of the answer. Do NOT include the original question. Output ONLY valid JSON."""


def is_georgian(text: str) -> bool:
    return bool(_GEORGIAN_RE.search(text))


def translate_to_english(text: str, llm: LLMClient) -> str:
    """Translate Georgian text to English. Falls back to original on any LLM failure."""
    if not is_georgian(text):
        return text
    try:
        resp = llm.complete(text, system=_TRANSLATE_SYSTEM, temperature=0.0, max_tokens=512)
        return resp.text.strip()
    except (LLMTimeoutError, LLMError):
        return text


def decompose_query(
    query: str,
    llm: LLMClient,
    max_sub_queries: int = 2,
) -> list[str]:
    """Decompose query into sub-queries. Used directly by tests."""
    system = _DECOMPOSE_SYSTEM.format(max_sub=max_sub_queries)
    try:
        data = llm.complete_json(query, system=system, temperature=0.0, max_tokens=512)
        subs = data.get("sub_queries", [])
        if not isinstance(subs, list) or not subs:
            return [query]
        return [str(s).strip() for s in subs[:max_sub_queries] if str(s).strip()]
    except (LLMMalformedResponseError, LLMTimeoutError, LLMError):
        return [query]


def rewrite_and_decompose(
    query: str,
    llm: LLMClient,
    max_sub_queries: int = 2,
) -> tuple[str, list[str]]:
    """
    Rewrite and decompose in one LLM call instead of two.

    Returns (rewritten_query, sub_queries). Falls back gracefully on failure.
    """
    system = _REWRITE_DECOMPOSE_SYSTEM.format(max_sub=max_sub_queries)
    try:
        data = llm.complete_json(query, system=system, temperature=0.0, max_tokens=512)
        rewritten = str(data.get("rewritten", query)).strip() or query
        subs = data.get("sub_queries", [])
        if not isinstance(subs, list) or not subs:
            subs = [rewritten]
        subs = [str(s).strip() for s in subs[:max_sub_queries] if str(s).strip()]
        return rewritten, subs or [rewritten]
    except (LLMMalformedResponseError, LLMTimeoutError, LLMError):
        return query, [query]


def understand_query(
    raw_query: str,
    llm: LLMClient,
    max_sub_queries: int = 2,
) -> tuple[str, str, list[str]]:
    """
    Full query understanding pipeline — one translation call (Georgian only)
    plus one combined rewrite+decompose call.

    Returns:
        (original_language, rewritten_english_query, sub_queries)
    """
    lang = "georgian" if is_georgian(raw_query) else "english"
    english = translate_to_english(raw_query, llm)
    rewritten, sub_queries = rewrite_and_decompose(english, llm, max_sub_queries)
    return lang, rewritten, sub_queries
