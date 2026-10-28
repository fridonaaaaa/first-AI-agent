"""
Embedding interface — single entry point for turning text into vectors.

Three backends, selected by the EMBEDDING_BACKEND env var:

  "local"   — fastembed ONNX runtime, no network calls.
               Use for local development. Requires: pip install fastembed

  "hf_api"  — HuggingFace Inference API.
               Requires HF_TOKEN env var. May fail if HF is unreachable.

  "gemini"  — Google Gemini gemini-embedding-001, outputDimensionality=384.
               Requires GEMINI_API_KEY. Recommended for Render production.

Auto-selection order when EMBEDDING_BACKEND is not set:
  1. Try local (fastembed). If not installed →
  2. Try gemini (if GEMINI_API_KEY set). Else →
  3. Fall back to hf_api.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2
    from fastembed import TextEmbedding

import httpx

from app.core.errors import EmbeddingError
from app.core.settings import get_settings

_model: "TextEmbedding | None" = None

_HF_API_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
)
_GEMINI_MODEL = "gemini-embedding-001"
_GEMINI_BATCH_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    f"/models/{_GEMINI_MODEL}:batchEmbedContents?key={{api_key}}"
)
_GEMINI_OUTPUT_DIM = 384  # must match EMBEDDING_DIM and the pgvector column size


def _backend() -> str:
    backend = os.getenv("EMBEDDING_BACKEND", "local").lower()
    if backend == "local":
        try:
            import fastembed  # noqa: F401
        except ImportError:
            # fastembed not installed — prefer gemini if key is available
            has_key = get_settings().gemini_api_key or os.getenv("GEMINI_API_KEY")
            return "gemini" if has_key else "hf_api"
    return backend


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
    if data and isinstance(data[0], float):
        data = [data]
    return data


# ── Gemini backend ─────────────────────────────────────────────────────────────

def _encode_gemini(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using Google Gemini gemini-embedding-001.
    outputDimensionality=384 matches the existing pgvector column size.
    Batches all texts in a single API call.
    """
    api_key = (get_settings().gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        raise EmbeddingError("GEMINI_API_KEY not set; cannot use gemini embedding backend")

    url = _GEMINI_BATCH_URL.format(api_key=api_key)
    payload = {
        "requests": [
            {
                "model": f"models/{_GEMINI_MODEL}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": _GEMINI_OUTPUT_DIM,
            }
            for t in texts
        ]
    }

    for attempt in range(5):
        try:
            resp = httpx.post(url, json=payload, timeout=60.0)
        except httpx.TimeoutException as exc:
            raise EmbeddingError("Gemini embedding API timed out") from exc

        if resp.status_code == 429:
            time.sleep(10.0)
            continue
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Gemini embedding API returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        return [item["values"] for item in resp.json()["embeddings"]]

    raise EmbeddingError("Gemini embedding API rate limit exceeded after retries")


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
        backend = _backend()
        if backend == "gemini":
            return _encode_gemini(texts)
        if backend == "hf_api":
            return _encode_hf_api(texts)
        return _encode_local(texts)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Encoding failed: {exc}") from exc


def encode_one(text: str) -> list[float]:
    """Encode a single string. Convenience wrapper around encode()."""
    return encode([text])[0]


def get_stored_model_name(conn: "psycopg2.extensions.connection") -> str | None:
    """
    Read the embedding model name that was used when the corpus was indexed.

    Returns None if bm25_corpus has no record yet (e.g. before the first embed job).
    Any DB error is swallowed and returns None so callers can decide how to handle it.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding_model FROM bm25_corpus WHERE id = 1")
            row = cur.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None
