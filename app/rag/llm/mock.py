"""
Mock LLM client for unit tests.

Returns pre-scripted responses keyed by substring match on the prompt.
If no key matches, returns the fallback. Never makes a network call.

Usage in tests:
    client = MockLLMClient({
        "decompose": '{"sub_queries": ["q1", "q2"]}',
        "answer":    "The answer is 42. [chunk-1]",
    })
"""

from __future__ import annotations

from app.rag.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        fallback: str = "mock response",
    ) -> None:
        self._responses = responses or {}
        self._fallback = fallback
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(prompt)
        for key, reply in self._responses.items():
            if key in prompt:
                return LLMResponse(text=reply, model="mock", cached=False)
        return LLMResponse(text=self._fallback, model="mock", cached=False)
