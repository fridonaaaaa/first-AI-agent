"""
Unit tests for the embedding interface.

We do NOT call the real model here — unit tests must not download weights.
Instead we patch app.retrieval.embeddings._get_model so the library import
is never reached. This works whether or not sentence-transformers is installed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _fake_model(dim: int = 4) -> MagicMock:
    """Return a mock that behaves like SentenceTransformer.encode."""
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.ones((len(texts), dim))
    return model


def _reset():
    """Clear the lazy singleton so each test starts fresh."""
    import app.retrieval.embeddings as mod
    mod._model = None
    return mod


def test_encode_returns_list_of_lists():
    mod = _reset()
    with patch.object(mod, "_get_model", return_value=_fake_model()):
        result = mod.encode(["hello", "world"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)


def test_encode_one_returns_single_vector():
    mod = _reset()
    with patch.object(mod, "_get_model", return_value=_fake_model()):
        vec = mod.encode_one("test")
    assert isinstance(vec, list)
    assert len(vec) == 4


def test_encode_empty_returns_empty():
    mod = _reset()
    assert mod.encode([]) == []


def test_encode_returns_floats():
    mod = _reset()
    with patch.object(mod, "_get_model", return_value=_fake_model()):
        vec = mod.encode_one("floats")
    assert all(isinstance(v, float) for v in vec)


def test_encode_raises_embedding_error_on_failure():
    from app.core.errors import EmbeddingError

    bad_model = MagicMock()
    bad_model.encode.side_effect = RuntimeError("model exploded")

    mod = _reset()
    with patch.object(mod, "_get_model", return_value=bad_model):
        with pytest.raises(EmbeddingError):
            mod.encode_one("boom")


def test_model_loaded_once():
    """Lazy singleton: SentenceTransformer constructor is called once, not twice."""
    import sys

    mod = _reset()
    call_count = {"n": 0}

    fake_module = MagicMock()

    def counting_constructor(*args, **kwargs):
        call_count["n"] += 1
        return _fake_model()

    fake_module.SentenceTransformer = counting_constructor
    old = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_module

    # Also patch get_settings so the test is independent of the .env file.
    fake_settings = MagicMock()
    fake_settings.embedding_model = "fake-model"

    try:
        mod._model = None
        with patch("app.retrieval.embeddings.get_settings", return_value=fake_settings):
            mod.encode_one("first")
            mod.encode_one("second")
    finally:
        if old is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = old

    assert call_count["n"] == 1
