"""
/documents CRUD — upload, update, delete, list documents.

All write endpoints require the X-Admin-Token header (compared with
hmac.compare_digest). Listing is open (read-only).

Upload accepts raw text body with metadata in query params; the caller is
responsible for pre-parsing their file. Large files should be sent as
multipart/form-data (future enhancement).

Input validation: text is stripped and truncated at 1 MB. Type is validated
against known formats. Content is never executed or rendered as HTML.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.auth import require_admin
from app.core.db import get_db
from app.core.errors import DocumentNotFoundError, DuplicateDocumentError, IngestionError
from app.core.logging import get_logger
from app.ingestion.chunkers import chunk_document
from app.ingestion.dedupe import DedupeIndex
from app.ingestion.parsers import ParsedDocument, parse_text
from app.ingestion.registry import (
    add_document,
    delete_document,
    get_document,
    list_documents,
    update_document,
)

router = APIRouter(prefix="/documents")
log = get_logger("documents")

_MAX_TEXT_BYTES = 1_000_000  # 1 MB


class UploadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1)
    source: str = Field("", max_length=1000)
    format: str = Field("text", pattern=r"^(text|markdown|html|pdf|jsonl)$")
    language: str = Field("en", max_length=10)
    chunking_strategy: str = Field("sentence_window", pattern=r"^(sentence_window|fixed_token)$")


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source: str
    format: str
    content_hash: str
    chunk_count: int | None = None


class UploadResponse(BaseModel):
    doc_id: str
    chunk_count: int
    message: str


@router.post("", response_model=UploadResponse, dependencies=[Depends(require_admin)])
def upload_document(body: UploadRequest) -> UploadResponse:
    text = body.text.strip()
    if len(text.encode()) > _MAX_TEXT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Text exceeds 1 MB limit.",
        )

    parsed = ParsedDocument(
        title=body.title,
        text=text,
        source=body.source,
        format=body.format,
        language=body.language,
        metadata={},
    )
    chunks = chunk_document(parsed, strategy=body.chunking_strategy)
    dedupe = DedupeIndex()

    try:
        with get_db() as conn:
            doc_id = add_document(conn, parsed, chunks, dedupe)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if doc_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Exact duplicate — document already exists.",
        )

    log.info("document_uploaded", doc_id=doc_id, chunks=len(chunks))
    return UploadResponse(doc_id=doc_id, chunk_count=len(chunks), message="Uploaded.")


@router.get("", response_model=list[DocumentSummary])
def list_docs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[DocumentSummary]:
    with get_db() as conn:
        docs = list_documents(conn, limit=limit, offset=offset)
    return [
        DocumentSummary(
            doc_id=d["doc_id"],
            title=d["title"],
            source=d.get("source", ""),
            format=d.get("format", ""),
            content_hash=d.get("content_hash", ""),
        )
        for d in docs
    ]


@router.get("/{doc_id}", response_model=DocumentSummary)
def get_doc(doc_id: str) -> DocumentSummary:
    try:
        with get_db() as conn:
            d = get_document(conn, doc_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentSummary(
        doc_id=d["doc_id"],
        title=d["title"],
        source=d.get("source", ""),
        format=d.get("format", ""),
        content_hash=d.get("content_hash", ""),
    )


@router.put("/{doc_id}", response_model=UploadResponse, dependencies=[Depends(require_admin)])
def update_doc(doc_id: str, body: UploadRequest) -> UploadResponse:
    text = body.text.strip()
    if len(text.encode()) > _MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="Text exceeds 1 MB limit.")

    parsed = ParsedDocument(
        title=body.title,
        text=text,
        source=body.source,
        format=body.format,
        language=body.language,
        metadata={},
    )
    chunks = chunk_document(parsed, strategy=body.chunking_strategy)
    dedupe = DedupeIndex()

    try:
        with get_db() as conn:
            new_id = update_document(conn, doc_id, parsed, chunks, dedupe)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    log.info("document_updated", old_id=doc_id, new_id=new_id, chunks=len(chunks))
    return UploadResponse(doc_id=new_id, chunk_count=len(chunks), message="Updated.")


@router.delete("/{doc_id}", dependencies=[Depends(require_admin)])
def delete_doc(doc_id: str) -> dict[str, str]:
    try:
        with get_db() as conn:
            delete_document(conn, doc_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    log.info("document_deleted", doc_id=doc_id)
    return {"message": "Deleted."}
