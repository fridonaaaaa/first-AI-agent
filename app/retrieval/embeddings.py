"""
Embedding interface — single entry point for turning text into vectors.

Two backends, selected by the EMBEDDING_BACKEND env var:

  "local" (default) — fastembed ONNX runtime, no network calls.
                       Use for local development and scripts.

  "hf_api"          — HuggingFace Inference API, no local model loaded.
                       Use for production on memory-constrained hosts
                       (e.g. Render free tier 512 MB). Requires HF_TOKEN
                       env var for rate-limit headroom; works without it
                       but may be throttled.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx

from app.core.errors import EmbeddingError
from app.core.settings import get_settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_model: "TextEmbedding | None" = None

_HF_API_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
)


def _backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "local").lower()


# ── local backend ──────────────────────────────────────────────────────────────

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


def _encode_local(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return [v.tolist() for v in model.embed(texts)]


# ── HF Inference API backend ───────────────────────────────────────────────────

def _encode_hf_api(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    url = _HF_API_URL.format(model=settings.embedding_model)
    headers: dict[str, str] = {}
    token = os.getenv("HF_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = httpx.post(
            url,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingError(f"HF API returned {exc.response.status_code}") from exc
    except httpx.TimeoutException as exc:
        raise EmbeddingError("HF API embedding timed out") from exc

    data: list = resp.json()
    # API returns list[list[float]] for batch; list[float] for single input
    if data and isinstance(data[0], float):
        data = [data]
    return data


# ── public interface ───────────────────────────────────────────────────────────

def encode(texts: list[str]) -> list[list[float]]:
    """
    Encode a batch of texts into embedding vectors.

    Returns one vector per input text, each of length settings.EMBEDDING_DIM.
    Raises EmbeddingError on any failure.
    """
    if not texts:
        return []
    try:
        if _backend() == "hf_api":
            return _encode_hf_api(texts)
        return _encode_local(texts)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Encoding failed: {exc}") from exc


def encode_one(text: str) -> list[float]:
    """Encode a single string. Convenience wrapper around encode()."""
    return encode([text])[0]
