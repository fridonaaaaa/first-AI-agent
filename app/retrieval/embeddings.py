"""
Embedding interface — single entry point for turning text into vectors.

Wraps sentence-transformers so the rest of the codebase never imports it
directly. The model name and dimension are set by Settings and must match
what was recorded in bm25_corpus.embedding_model at index time.

Design choices:
- Lazy load: the transformer is heavy (~90 MB). It is loaded once on first
  call and reused. Startup stays fast; first query pays the load cost.
- Batch encode: encode() accepts a list so callers can batch for efficiency.
  Single-string callers wrap it in a list themselves (or use encode_one).
- Pure-Python interface: the returned type is list[float] (not numpy) so
  psycopg2 / pgvector can accept it without extra conversion at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.errors import EmbeddingError
from app.core.settings import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        settings = get_settings()
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """
    Encode a batch of texts into embedding vectors.

    Returns one vector per input text, each of length settings.EMBEDDING_DIM.
    Raises EmbeddingError on any failure.
    """
    if not texts:
        return []
    try:
        model = _get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Encoding failed: {exc}") from exc


def encode_one(text: str) -> list[float]:
    """Encode a single string. Convenience wrapper around encode()."""
    return encode([text])[0]
