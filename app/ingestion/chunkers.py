"""
Two chunking strategies behind one interface.

Strategy A — fixed_token: split by approximate token count with overlap.
Strategy B — sentence_window: group sentences into windows with stride.

Both preserve title and source metadata on every chunk.
"""

import re
from dataclasses import dataclass, field

from app.ingestion.parsers import ParsedDocument


@dataclass
class Chunk:
    text: str
    position: int        # 0-based index within the document
    strategy: str        # fixed_token | sentence_window
    doc_title: str
    doc_source: str
    doc_format: str
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _approx_tokens(text: str) -> int:
    return len(text.split())


def chunk_fixed_token(
    doc: ParsedDocument,
    window: int = 256,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Split document text into fixed-token windows with overlap.
    'Tokens' are approximated as whitespace-separated words.
    """
    words = doc.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    position = 0

    while start < len(words):
        end = min(start + window, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Chunk(
                text=chunk_text,
                position=position,
                strategy="fixed_token",
                doc_title=doc.title,
                doc_source=doc.source,
                doc_format=doc.format,
                metadata={
                    "title": doc.title,
                    "source": doc.source,
                    "format": doc.format,
                    "word_start": start,
                    "word_end": end,
                    **doc.metadata,
                },
            )
        )
        position += 1
        if end == len(words):
            break
        start += window - overlap

    return chunks


def chunk_sentence_window(
    doc: ParsedDocument,
    window: int = 3,
    stride: int = 1,
) -> list[Chunk]:
    """
    Group sentences into overlapping windows.
    Each chunk contains `window` consecutive sentences.
    The window slides forward by `stride` sentences.
    """
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    position = 0
    start = 0

    while start < len(sentences):
        end = min(start + window, len(sentences))
        chunk_text = " ".join(sentences[start:end])
        chunks.append(
            Chunk(
                text=chunk_text,
                position=position,
                strategy="sentence_window",
                doc_title=doc.title,
                doc_source=doc.source,
                doc_format=doc.format,
                metadata={
                    "title": doc.title,
                    "source": doc.source,
                    "format": doc.format,
                    "sent_start": start,
                    "sent_end": end,
                    **doc.metadata,
                },
            )
        )
        position += 1
        if end == len(sentences):
            break
        start += stride

    return chunks


def chunk_document(
    doc: ParsedDocument,
    strategy: str = "sentence_window",
    fixed_window: int = 256,
    fixed_overlap: int = 64,
    sent_window: int = 3,
    sent_stride: int = 1,
) -> list[Chunk]:
    """Single entry point — dispatches to the chosen strategy."""
    if strategy == "fixed_token":
        return chunk_fixed_token(doc, window=fixed_window, overlap=fixed_overlap)
    if strategy == "sentence_window":
        return chunk_sentence_window(doc, window=sent_window, stride=sent_stride)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
