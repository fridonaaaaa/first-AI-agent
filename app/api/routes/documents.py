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

import io
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.api.auth import require_admin
from app.core.db import get_db
from app.core.errors import DocumentNotFoundError, DuplicateDocumentError, IngestionError
from app.core.logging import get_logger
from app.ingestion.chunkers import chunk_document
from app.ingestion.dedupe import DedupeIndex
from app.ingestion.parsers import ParsedDocument, parse_html, parse_markdown, parse_pdf, parse_text
from app.ingestion.registry import (
    add_document,
    delete_document,
    get_document,
    list_documents,
    update_document,
)
from app.retrieval.vector_store import embed_new_chunks

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
            if doc_id is not None:
                embed_new_chunks(conn, doc_id)
    except Exception as exc:
        log.error("upload_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="internal_error")

    if doc_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="duplicate")

    log.info("document_uploaded", doc_id=doc_id, chunks=len(chunks))
    return UploadResponse(doc_id=doc_id, chunk_count=len(chunks), message="Uploaded.")


_ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
_MAX_FILE_BYTES = 52_428_800  # 50 MB


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_admin)])
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a file upload (PDF, DOCX, TXT, MD, HTML) and ingest it."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="unsupported_type",
        )

    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    title = Path(file.filename).stem.replace("_", " ").replace("-", " ")

    try:
        if ext == ".pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            parsed = parse_pdf(tmp_path, source=file.filename)
            tmp_path.unlink(missing_ok=True)

        elif ext == ".docx":
            try:
                import docx as python_docx  # type: ignore
            except ImportError:
                raise HTTPException(status_code=500, detail="python-docx not installed on server.")
            doc = python_docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            parsed = ParsedDocument(title=title, text=text, source=file.filename, format="txt")

        elif ext in (".html", ".htm"):
            parsed = parse_html(data, source=file.filename, title_hint=title)

        elif ext == ".md":
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            parsed = parse_markdown(tmp_path, source=file.filename)
            parsed.title = title
            tmp_path.unlink(missing_ok=True)

        else:  # .txt
            text = data.decode("utf-8", errors="replace")
            parsed = ParsedDocument(title=title, text=text, source=file.filename, format="txt")

    except HTTPException:
        raise
    except Exception as exc:
        log.error("file_parse_failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=422, detail="parse_error")

    if not parsed.text.strip():
        raise HTTPException(status_code=422, detail="File contained no readable text.")

    chunks = chunk_document(parsed, strategy="sentence_window")
    dedupe = DedupeIndex()

    try:
        with get_db() as conn:
            doc_id = add_document(conn, parsed, chunks, dedupe)
            if doc_id is not None:
                embed_new_chunks(conn, doc_id)
    except Exception as exc:
        log.error("db_error", error=str(exc))
        raise HTTPException(status_code=500, detail="internal_error")

    if doc_id is None:
        raise HTTPException(status_code=409, detail="duplicate")

    log.info("file_uploaded", doc_id=doc_id, filename=file.filename, chunks=len(chunks))
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
        log.error("db_error", error=str(exc))
        raise HTTPException(status_code=500, detail="internal_error")

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
        log.error("db_error", error=str(exc))
        raise HTTPException(status_code=500, detail="internal_error")
    log.info("document_deleted", doc_id=doc_id)
    return {"message": "Deleted."}
