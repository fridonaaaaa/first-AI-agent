"""
Mock LLM client for unit tests.

Returns pre-scripted responses keyed by substring match on the prompt.
If no key matches, returns the fallback. Never makes a network call.

Usage in tests:
    client = MockLLMClient({
        "decompose": '{"sub_queries": ["q1", "q2"]}',
        "answer":    "The answer is 42. [chunk-1]",
    })

    # To script tool calls (for tool-loop tests):
    client = MockLLMClient(
        tool_calls=[
            [{"name": "search", "args": {"query": "capitals"}}],  # first call
            [],                                                    # second call → text
        ],
        fallback="The answer is 42.",
    )
"""

from __future__ import annotations

from app.rag.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        fallback: str = "mock response",
        tool_calls: "list[list[dict]] | None" = None,
    ) -> None:
        self._responses = responses or {}
        self._fallback = fallback
        self.calls: list[str] = []
        # Scripted tool-call sequences for complete_with_tools().
        # Each element is the list of tool calls to return on that invocation.
        # An empty list means "return text (no tool calls)" on that turn.
        self._tool_calls: list[list[dict]] = list(tool_calls or [])
        self._tool_call_index: int = 0

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

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> "tuple[str | None, list[dict]]":
        """Return scripted tool calls or fallback text."""
        # Reconstruct the last user message for call tracking.
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break
        self.calls.append(last_user)

        if self._tool_call_index < len(self._tool_calls):
            tc = self._tool_calls[self._tool_call_index]
            self._tool_call_index += 1
            if tc:
                return None, tc
            # Empty list → text response
            for key, reply in self._responses.items():
                if key in last_user:
                    return reply, []
            return self._fallback, []

        # No more scripted calls — match on prompt text or return fallback.
        for key, reply in self._responses.items():
            if key in last_user:
                return reply, []
        return self._fallback, []
