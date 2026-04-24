"""Tests for embedding provider registry + batch embedding."""
import pytest

from mnemostack.embeddings import EmbeddingProvider, get_provider, list_providers


def test_list_providers_has_builtins():
    names = list_providers()
    assert "gemini" in names
    assert "ollama" in names


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider("nonexistent")


def test_ollama_provider_interface():
    """Instantiate without actually hitting Ollama — just check interface."""
    provider = get_provider("ollama", host="http://localhost:11434")
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 768  # default for nomic-embed-text
    assert "ollama" in provider.name


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_provider("gemini")


def test_custom_provider_registration():
    from mnemostack.embeddings import register_provider

    class FakeProvider(EmbeddingProvider):
        def embed(self, text):
            return [0.1, 0.2, 0.3]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

        @property
        def dimension(self):
            return 3

        @property
        def name(self):
            return "fake"

    register_provider("fake", FakeProvider)
    p = get_provider("fake")
    assert p.embed("hello") == [0.1, 0.2, 0.3]
    assert p.dimension == 3


def test_batch_fake_provider():
    """Default embed_batch implementation should work for any subclass."""

    calls = []

    class CountingProvider(EmbeddingProvider):
        def embed(self, text):
            calls.append(text)
            return [0.1] * 4

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

        @property
        def dimension(self):
            return 4

        @property
        def name(self):
            return "counting"

    p = CountingProvider()
    result = p.embed_batch(["a", "b", "c"])
    assert len(result) == 3
    assert calls == ["a", "b", "c"]


def test_gemini_batch_empty_returns_empty(monkeypatch):
    """embed_batch([]) should return [] without any API call."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from mnemostack.embeddings import get_provider

    provider = get_provider("gemini")
    assert provider.embed_batch([]) == []


def test_ollama_batch_empty_returns_empty():
    provider = get_provider("ollama")
    assert provider.embed_batch([]) == []


def test_ollama_batch_single_item_uses_embed(monkeypatch):
    """Single-item batch should just call embed once, not use pool."""
    provider = get_provider("ollama")
    called = []

    def fake_embed(self, text):
        called.append(text)
        return [0.0] * 768

    monkeypatch.setattr(type(provider), "embed", fake_embed)
    result = provider.embed_batch(["just one"])
    assert len(result) == 1
    assert called == ["just one"]
