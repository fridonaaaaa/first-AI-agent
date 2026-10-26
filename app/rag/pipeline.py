"""
RAG pipeline — multi-step tool loop that answers a user question.

Full flow (from PLAN.md §7):
  1. understand_query  → language, rewritten query, sub-queries
  2. hybrid_search     → top-k CandidateChunk per sub-query, merged & reranked
  3. check_abstention  → refuse if evidence is insufficient
  4. detect_conflicts  → find contradictions in retrieved chunks
  5. generate answer   → LLM reads chunks + query, writes answer with [id] citations
  6. verify_citations  → strip any [id] not in the retrieved set
  7. Return PipelineResult

Tool loop (steps 2–4) is bounded by max_tool_steps so the LLM cannot loop
infinitely. Each iteration may call search_documents (hybrid_search) or
get_document (fetch one chunk by id). compare_claims (conflict check) runs
once after retrieval.

All retrieved text is passed to the LLM as inert DATA in the prompt, never
executed or interpolated into code — prompt-injection defense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg2

from app.core.settings import get_settings
from app.rag.abstention import AbstentionDecision, check_abstention
from app.rag.citation import CitationResult, verify_citations
from app.rag.conflict import ConflictVerdict, detect_conflicts
from app.rag.llm.base import LLMClient
from app.rag.query_understanding import understand_query
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import CandidateChunk

_ANSWER_SYSTEM = """You are a research assistant. Answer the question using ONLY the
provided passages. For every factual claim cite the passage ID like [chunk-id].
If the passages contradict each other, say so and show both sides.
Answer in {language}. Do NOT add information not found in the passages."""


@dataclass
class PipelineResult:
    answer: str
    language: str
    rewritten_query: str
    sub_queries: list[str]
    chunks: list[CandidateChunk]
    conflicts: list[ConflictVerdict]
    abstention: AbstentionDecision | None
    citation_check: CitationResult | None
    steps_used: int


def _format_context(chunks: list[CandidateChunk]) -> str:
    """Format chunks as labelled passages for the LLM prompt."""
    return "\n\n".join(
        f"[{c.chunk_id}]\n{c.text}" for c in chunks
    )


def _conflict_notice(verdicts: list[ConflictVerdict]) -> str:
    """Return a warning paragraph if any conflicts were found."""
    real = [v for v in verdicts if v.verdict == "conflict"]
    if not real:
        return ""
    lines = ["Note: the following passages appear to contradict each other:"]
    for v in real:
        lines.append(f"  [{v.chunk_a}] vs [{v.chunk_b}]: {v.reason}")
    return "\n".join(lines) + "\n\n"


def run_pipeline(
    raw_query: str,
    llm: LLMClient,
    conn: "psycopg2.extensions.connection | None" = None,
    top_k: int | None = None,
    max_tool_steps: int | None = None,
) -> PipelineResult:
    """
    Run the full RAG pipeline for one user question.

    Args:
        raw_query:      The user's question (Georgian or English).
        llm:            LLM client (injected so tests can use MockLLMClient).
        conn:           Optional DB connection (injected for tests).
        top_k:          Override retrieval_top_k from settings.
        max_tool_steps: Override max_tool_steps from settings.

    Returns:
        PipelineResult with the final answer and all intermediate artefacts.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    max_steps = max_tool_steps or settings.max_tool_steps

    # 1. Query understanding
    lang, rewritten, sub_queries = understand_query(
        raw_query, llm, max_sub_queries=settings.max_sub_queries
    )

    # 2. Tool loop: retrieve chunks (each sub-query counted as one step)
    steps = 0
    all_chunks: list[CandidateChunk] = []
    seen_ids: set[str] = set()

    def _do_search(query: str, subs: list[str]) -> None:
        nonlocal steps
        if steps >= max_steps:
            return
        steps += 1
        # Pass conn only when the caller provided one (e.g. integration tests /
        # bulk jobs). Otherwise let hybrid_search open its own connection so
        # the pipeline itself never holds an open connection between steps.
        new_chunks = hybrid_search(
            query, sub_queries=subs, top_k=k, conn=conn
        )

        for chunk in new_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)

    _do_search(rewritten, sub_queries)

    # If sub-queries produced nothing useful, retry with the raw query
    if not all_chunks and raw_query != rewritten:
        _do_search(raw_query, [raw_query])

    # 3. Abstention check
    abstention = check_abstention(rewritten, all_chunks, llm, settings.abstention_threshold)
    if abstention.should_abstain:
        abstain_msg = (
            "ვერ ვიპოვე საკმარისი მტკიცებულება ამ კითხვაზე პასუხის გასაცემად."
            if lang == "georgian"
            else "I could not find sufficient evidence to answer this question."
        )
        return PipelineResult(
            answer=abstain_msg,
            language=lang,
            rewritten_query=rewritten,
            sub_queries=sub_queries,
            chunks=all_chunks,
            conflicts=[],
            abstention=abstention,
            citation_check=None,
            steps_used=steps,
        )

    # 4. Conflict detection
    conflicts = detect_conflicts(all_chunks, llm)

    # 5. Generate answer
    context = _format_context(all_chunks)
    conflict_notice = _conflict_notice(conflicts)
    answer_lang = "Georgian" if lang == "georgian" else "English"
    system = _ANSWER_SYSTEM.format(language=answer_lang)

    prompt = (
        f"{conflict_notice}"
        f"Question: {raw_query}\n\n"
        f"Passages:\n{context}"
    )
    resp = llm.complete(prompt, system=system, temperature=0.0, max_tokens=1024)
    raw_answer = resp.text.strip()

    # 6. Citation verification
    retrieved_ids = {c.chunk_id for c in all_chunks}
    citation_result = verify_citations(raw_answer, retrieved_ids)

    return PipelineResult(
        answer=citation_result.verified_answer,
        language=lang,
        rewritten_query=rewritten,
        sub_queries=sub_queries,
        chunks=all_chunks,
        conflicts=conflicts,
        abstention=abstention,
        citation_check=citation_result,
        steps_used=steps,
    )
