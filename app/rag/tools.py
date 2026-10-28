"""
Tool schemas and executors for the RAG tool loop.

Three tools the LLM can call during investigation:
  search        — retrieve relevant passages from the corpus.
  get_document  — fetch the full text + metadata of one chunk by ID.
  compare_claims — determine whether two text passages agree or conflict.

All tools respect session_id / doc_id boundaries: they only access chunks
the current session is permitted to see.

Tool schemas are expressed in Gemini's functionDeclarations format.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2
    from app.rag.llm.base import LLMClient

from app.core.logging import get_logger
from app.retrieval.hybrid import hybrid_search

log = get_logger("tools")

# ---------------------------------------------------------------------------
# Schemas (Gemini functionDeclarations format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search",
        "description": (
            "Search the knowledge base for text passages relevant to a query. "
            "Returns up to `top_k` passages with their chunk IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": (
            "Fetch the full text and source metadata for a specific passage by its chunk ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "The chunk ID to retrieve (e.g. 'abc123-...').",
                },
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "compare_claims",
        "description": (
            "Compare two text passages to determine whether they agree, conflict, "
            "or are unrelated. Use this to detect contradictions between sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "passage_a": {
                    "type": "string",
                    "description": "First passage.",
                },
                "passage_b": {
                    "type": "string",
                    "description": "Second passage.",
                },
            },
            "required": ["passage_a", "passage_b"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

_COMPARE_SYSTEM = (
    "You compare two text passages. Return a JSON object with:\n"
    '  "verdict": "agree" | "conflict" | "unrelated"\n'
    '  "reason": one sentence explaining why\n'
    "Output ONLY valid JSON."
)

_MAX_CHUNK_TEXT = 1000  # chars; keeps tool results compact in the conversation


def _search(
    args: dict,
    conn: "psycopg2.extensions.connection",
    session_id: str | None,
    doc_id: str | None,
) -> dict:
    """Execute the `search` tool."""
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}
    top_k = min(int(args.get("top_k", 5)), 10)

    chunks = hybrid_search(
        query,
        sub_queries=[query],
        top_k=top_k,
        conn=conn,
        session_id=session_id,
        doc_id=doc_id,
    )
    return {
        "results": [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "text": c.text[:_MAX_CHUNK_TEXT],
                "rrf_score": round(c.rrf_score, 5),
            }
            for c in chunks
        ],
        "count": len(chunks),
    }


def _get_document(
    args: dict,
    conn: "psycopg2.extensions.connection",
    session_id: str | None,
    doc_id: str | None,
) -> dict:
    """Execute the `get_document` tool — fetch one chunk by ID."""
    chunk_id = str(args.get("chunk_id", "")).strip()
    if not chunk_id:
        return {"error": "chunk_id is required"}

    with conn.cursor() as cur:
        if doc_id is not None:
            cur.execute(
                """
                SELECT c.id, c.text, c.doc_id::text, d.title, d.source
                FROM   chunks c
                JOIN   documents d ON d.id = c.doc_id
                WHERE  c.id = %s AND c.doc_id = %s::uuid
                """,
                (chunk_id, doc_id),
            )
        elif session_id is not None:
            cur.execute(
                """
                SELECT c.id, c.text, c.doc_id::text, d.title, d.source
                FROM   chunks c
                JOIN   documents d ON d.id = c.doc_id
                WHERE  c.id = %s
                  AND  (d.session_id IS NULL OR d.session_id = %s)
                """,
                (chunk_id, session_id),
            )
        else:
            cur.execute(
                """
                SELECT c.id, c.text, c.doc_id::text, d.title, d.source
                FROM   chunks c
                JOIN   documents d ON d.id = c.doc_id
                WHERE  c.id = %s
                """,
                (chunk_id,),
            )
        row = cur.fetchone()

    if not row:
        return {"error": f"chunk_id '{chunk_id}' not found or not accessible"}

    return {
        "chunk_id": row[0],
        "text": row[1][:_MAX_CHUNK_TEXT],
        "doc_id": row[2],
        "title": row[3],
        "source": row[4],
    }


def _compare_claims(
    args: dict,
    llm: "LLMClient | None",
) -> dict:
    """Execute the `compare_claims` tool — LLM compares two passages."""
    passage_a = str(args.get("passage_a", "")).strip()
    passage_b = str(args.get("passage_b", "")).strip()
    if not passage_a or not passage_b:
        return {"error": "passage_a and passage_b are both required"}

    if llm is None:
        return {"verdict": "unrelated", "reason": "comparison unavailable (no LLM)"}

    prompt = (
        f"Passage A:\n{passage_a[:600]}\n\n"
        f"Passage B:\n{passage_b[:600]}\n\n"
        "Do these passages contradict each other?"
    )
    try:
        result = llm.complete_json(
            prompt,
            system=_COMPARE_SYSTEM,
            temperature=0.0,
            max_tokens=256,
        )
        verdict = str(result.get("verdict", "unrelated")).lower()
        if verdict not in ("agree", "conflict", "unrelated"):
            verdict = "unrelated"
        return {"verdict": verdict, "reason": str(result.get("reason", ""))}
    except Exception as exc:
        log.warning("compare_claims_failed", error=str(exc))
        return {"verdict": "unrelated", "reason": "comparison error"}


def execute_tool(
    name: str,
    args: dict,
    *,
    conn: "psycopg2.extensions.connection | None" = None,
    session_id: str | None = None,
    doc_id: str | None = None,
    llm: "LLMClient | None" = None,
) -> dict:
    """
    Dispatch a tool call by name.

    Returns a dict that becomes the functionResponse payload sent back to the LLM.
    Raises ValueError for unknown tool names so callers can handle bad calls safely.
    """
    if name == "search":
        if conn is None:
            return {"error": "no database connection available"}
        return _search(args, conn, session_id, doc_id)
    if name == "get_document":
        if conn is None:
            return {"error": "no database connection available"}
        return _get_document(args, conn, session_id, doc_id)
    if name == "compare_claims":
        return _compare_claims(args, llm)
    raise ValueError(f"Unknown tool: {name!r}")
