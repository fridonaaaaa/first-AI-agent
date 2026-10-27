"""
Embedding interface — single entry point for turning text into vectors.

Uses fastembed (ONNX runtime) instead of sentence-transformers (PyTorch).
fastembed uses ~100 MB RAM vs ~470 MB for PyTorch, making it viable on
Render's 512 MB free tier. The ONNX model produces vectors compatible
with those computed by sentence-transformers for the same model.

Design choices:
- Lazy load: the model is heavy. Loaded once on first call and reused.
- Batch encode: encode() accepts a list so callers can batch for efficiency.
- Pure-Python interface: returns list[float] so psycopg2/pgvector can accept
  it without extra conversion at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.errors import EmbeddingError
from app.core.settings import get_settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_model: "TextEmbedding | None" = None


def _get_model() -> "TextEmbedding":
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import]
        except ImportError as exc:
            raise EmbeddingError(
                "fastembed is not installed. Run: pip install fastembed"
            ) from exc
        settings = get_settings()
        _model = TextEmbedding(model_name=settings.embedding_model)
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
        return [v.tolist() for v in model.embed(texts)]
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Encoding failed: {exc}") from exc


def encode_one(text: str) -> list[float]:
    """Encode a single string. Convenience wrapper around encode()."""
    return encode([text])[0]
