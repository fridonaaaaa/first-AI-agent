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
provided passages. Cite each claim with the passage ID like [chunk-id].

Rules for conflicting passages:
- Count how many passages support each position.
- Majority: "Most sources say [A] (N sources). However, M source(s) say [B]. There is conflicting information."
- Equal split — use EXACTLY this format, no "according to":
  "Sources disagree:
  • [state position A as a plain sentence]
  • [state position B as a plain sentence]
  I cannot determine which is correct — please decide which source you trust."
- NEVER pick a winner silently. NEVER write "according to" inside the bullet points.

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
    doc_id: str | None = None,
    session_id: str | None = None,
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
    is_georgian = any(ord(c) > 0x10CF for c in raw_query)

    def _abstain_result(reason: str) -> PipelineResult:
        msg = (
            "ვერ ვიპოვე საკმარისი მტკიცებულება ამ კითხვაზე პასუხის გასაცემად."
            if is_georgian
            else "I could not find any information about this in my knowledge base."
        )
        return PipelineResult(
            answer=msg,
            language="georgian" if is_georgian else "english",
            rewritten_query=raw_query,
            sub_queries=[],
            chunks=[],
            conflicts=[],
            abstention=AbstentionDecision(should_abstain=True, reason=reason, top_score=0.0),
            citation_check=None,
            steps_used=0,
        )

    # 1. Initial search with the raw query — used for both early-exit and as
    #    the seed result set.  Only one DB round-trip before the LLM is called.
    initial_chunks = hybrid_search(
        raw_query, sub_queries=[raw_query], top_k=k, conn=conn, doc_id=doc_id, session_id=session_id
    )

    # Early exit: nothing found at all, or the best cosine similarity is below
    # the threshold.  Cosine similarity is a real relevance signal; RRF scores
    # are bounded below by ~0.014 and cannot distinguish "irrelevant" results.
    if not initial_chunks:
        return _abstain_result("no_results")

    top_cosine = max((c.cosine_sim for c in initial_chunks), default=0.0)
    if top_cosine < settings.abstention_threshold:
        return _abstain_result("low_score")

    # 2. Query understanding — only reached when the search looks promising.
    lang, rewritten, sub_queries = understand_query(
        raw_query, llm, max_sub_queries=settings.max_sub_queries
    )

    # 3. Refined search with the rewritten query + sub-queries (if different).
    steps = 1  # initial_chunks counts as step 1
    all_chunks: list[CandidateChunk] = list(initial_chunks)
    seen_ids: set[str] = {c.chunk_id for c in initial_chunks}

    def _do_search(query: str, subs: list[str]) -> None:
        nonlocal steps
        if steps >= max_steps:
            return
        steps += 1
        new_chunks = hybrid_search(
            query, sub_queries=subs, top_k=k, conn=conn, doc_id=doc_id, session_id=session_id
        )
        for chunk in new_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)

    if rewritten != raw_query or sub_queries != [raw_query]:
        _do_search(rewritten, sub_queries)

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
