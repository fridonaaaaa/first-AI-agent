"""
LLMClient interface — every provider implements exactly these two methods.

All LLM calls in the RAG pipeline go through an LLMClient. The concrete
provider is chosen by the LLM_PROVIDER env var and never imported directly
outside of app/rag/llm/.

Design rules:
- complete() is for single-turn text generation (rewrite, decompose, answer).
- complete_json() wraps complete() and parses the JSON; callers get a dict,
  never a string.
- Both methods raise typed errors from app.core.errors, never raw exceptions.
- Subclasses add retry/backoff and timeout internally; callers assume one call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class LLMClient(ABC):
    """Abstract base for all LLM provider clients."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Send a prompt and return the model's text response.

        Args:
            prompt:      The user turn.
            system:      Optional system instruction.
            temperature: 0.0 = deterministic (best for structured outputs).
            max_tokens:  Hard cap on output length.

        Returns:
            LLMResponse with the generated text and token counts.

        Raises:
            LLMError (or subclass) on any failure.
        """

    def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Call complete() and parse the result as JSON.

        The model is expected to return a JSON object (or array). The prompt
        should explicitly instruct it to do so. Any text outside the JSON
        block is stripped before parsing.

        Raises:
            MalformedResponseError if the response cannot be parsed as JSON.
        """
        import json
        import re

        from app.core.errors import LLMMalformedResponseError

        resp = self.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        raw = resp.text.strip()

        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if match:
            raw = match.group(1)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMMalformedResponseError(
                f"LLM returned non-JSON: {raw[:200]!r}"
            ) from exc
