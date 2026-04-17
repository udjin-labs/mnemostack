"""Tests for embedding provider registry."""
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
