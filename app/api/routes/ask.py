"""
POST /ask — the main question-answering endpoint.

Rate-limited per IP and globally. Calls run_pipeline() with the configured
LLM client. Returns the answer, source citations, and the pipeline trace
(sub-queries used, steps taken, conflicts found).

All retrieved chunk text is treated as untrusted data by the pipeline;
it never reaches this layer as code.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.api.middleware import check_rate_limit
from app.core.errors import InvestigationEngineError
from app.core.logging import get_logger
from app.rag.llm.factory import get_llm_client
from app.rag.pipeline import run_pipeline

router = APIRouter()
log = get_logger("ask")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: str | None = Field(None, description="Restrict search to this document's chunks.")
    session_id: str | None = Field(None, max_length=100, description="Chat session ID for per-session document isolation.")


class SourceInfo(BaseModel):
    chunk_id: str
    text: str
    rrf_score: float
    title: str | None = None


class ConflictInfo(BaseModel):
    chunk_a: str
    chunk_b: str
    verdict: str
    reason: str


class AskResponse(BaseModel):
    answer: str
    language: str
    sources: list[SourceInfo]
    conflicts: list[ConflictInfo]
    sub_queries: list[str]
    steps_used: int
    abstained: bool
    retrieval_mode: str = "hybrid_rerank"  # "bm25_only" when embedding model mismatch
    tool_trace: list[dict] = []  # sanitised tool-call log (no raw chunk text)


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    limit_response = check_rate_limit(request)
    if limit_response:
        return limit_response  # type: ignore[return-value]

    llm = get_llm_client()

    try:
        result = run_pipeline(body.question, llm, doc_id=body.doc_id, session_id=body.session_id)
    except InvestigationEngineError as exc:
        log.error("pipeline_error", question=body.question[:80], error=str(exc))
        return AskResponse(
            answer="სამწუხაროდ, პასუხის გაცემა ვერ მოხდა. გთხოვთ სცადოთ მოგვიანებით."
                   if any(ord(c) > 0x10CF for c in body.question)
                   else "An error occurred. Please try again.",
            language="unknown",
            sources=[],
            conflicts=[],
            sub_queries=[],
            steps_used=0,
            abstained=False,
        )

    log.info(
        "ask_complete",
        language=result.language,
        steps=result.steps_used,
        chunks=len(result.chunks),
        abstained=result.abstention.should_abstain if result.abstention else False,
    )

    cited_ids: set[str] = set(result.citation_check.valid_ids) if result.citation_check else set()
    return AskResponse(
        answer=result.answer,
        language=result.language,
        sources=[
            SourceInfo(chunk_id=c.chunk_id, text=c.text, rrf_score=c.rrf_score, title=c.title)
            for c in result.chunks
            if c.chunk_id in cited_ids
        ],
        conflicts=[
            ConflictInfo(
                chunk_a=v.chunk_a,
                chunk_b=v.chunk_b,
                verdict=v.verdict,
                reason=v.reason,
            )
            for v in result.conflicts
        ],
        sub_queries=result.sub_queries,
        steps_used=result.steps_used,
        abstained=bool(result.abstention and result.abstention.should_abstain),
        retrieval_mode=result.retrieval_mode,
        tool_trace=result.tool_trace,
    )
