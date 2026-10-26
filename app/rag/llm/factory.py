"""
LLM client factory — returns the right client based on LLM_PROVIDER env var.

Supported providers: gemini, mock.
Adding a new provider: implement LLMClient, add a branch here.
"""

from __future__ import annotations

from app.core.settings import get_settings
from app.rag.llm.base import LLMClient


def get_llm_client(provider: str | None = None) -> LLMClient:
    """
    Return an LLMClient for the configured provider.

    Args:
        provider: Override the LLM_PROVIDER setting (useful in tests).
    """
    settings = get_settings()
    name = (provider or settings.llm_provider).lower()

    if name == "gemini":
        from app.rag.llm.gemini import GeminiClient
        return GeminiClient()

    if name == "mock":
        from app.rag.llm.mock import MockLLMClient
        return MockLLMClient()

    raise ValueError(
        f"Unknown LLM provider {name!r}. "
        "Set LLM_PROVIDER=gemini or LLM_PROVIDER=mock."
    )
