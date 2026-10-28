"""
Gemini provider — wraps the Google Generative AI REST API via httpx.

Uses gemini-3.1-flash-lite (verified working 2026-10-27 via models list API).
Rate limits on the free tier: check ai.google.dev/gemini-api/docs/rate-limits.

Retry strategy: exponential backoff on 429 (rate limit) and 5xx errors.
Timeout: 60 s per request.
Disk cache: responses are cached by SHA-256(provider+model+prompt+params) so
eval reruns are free and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

import httpx

from app.core.errors import LLMError, LLMRateLimitError, LLMTimeoutError
from app.core.settings import get_settings
from app.rag.llm.base import LLMClient, LLMResponse

_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)
_DEFAULT_MODEL = "gemini-3.1-flash-lite"
_MAX_RETRIES = 4
_BASE_DELAY = 5.0   # seconds; doubles each retry (5, 10, 20, 40 s)


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.llm_model or _DEFAULT_MODEL
        cache_path = pathlib.Path(cache_dir or settings.llm_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_path

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        cache_key = self._cache_key(prompt, system, temperature, max_tokens)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        body = self._build_body(prompt, system, temperature, max_tokens)
        url = _API_URL.format(model=self._model, key=self._api_key)

        resp = self._post_with_retry(url, body)
        result = self._parse_response(resp)
        result.cached = False
        self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_body(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> dict:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _post_with_retry(self, url: str, body: dict) -> dict:
        delay = _BASE_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=60.0) as client:
                    r = client.post(url, json=body)
            except httpx.TimeoutException as exc:
                if attempt == _MAX_RETRIES:
                    raise LLMTimeoutError("Gemini request timed out") from exc
                time.sleep(delay)
                delay *= 2
                continue
            except httpx.RequestError as exc:
                raise LLMError(f"Gemini network error: {exc}") from exc

            if r.status_code == 429:
                if attempt == _MAX_RETRIES:
                    raise LLMRateLimitError("Gemini rate limit exceeded")
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code >= 500:
                if attempt == _MAX_RETRIES:
                    raise LLMError(f"Gemini server error {r.status_code}")
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                raise LLMError(f"Gemini error {r.status_code}: {r.text[:200]}")

            return r.json()

        raise LLMError("Gemini: max retries exceeded")  # unreachable but satisfies type checkers

    def _parse_response(self, data: dict) -> LLMResponse:
        try:
            candidate = data["candidates"][0]
            text = candidate["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return LLMResponse(
                text=text,
                model=self._model,
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
            )
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from exc

    # ------------------------------------------------------------------
    # Disk cache helpers
    # ------------------------------------------------------------------

    def _cache_key(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> str:
        payload = json.dumps(
            {
                "provider": "gemini",
                "model": self._model,
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> pathlib.Path:
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> LLMResponse | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return LLMResponse(
                text=data["text"],
                model=data["model"],
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                cached=True,
            )
        except Exception:
            return None

    def _save_cache(self, key: str, resp: LLMResponse) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(
                    {
                        "text": resp.text,
                        "model": resp.model,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                    }
                )
            )
        except OSError:
            pass  # cache write failure is non-fatal
