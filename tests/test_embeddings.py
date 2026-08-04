"""Tests for embedding provider registry + batch embedding."""

import json

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


def test_ollama_batch_uses_native_api_embed(monkeypatch):
    """A batch is ONE /api/embed request with array input, not N calls."""
    import json as jsonlib
    from io import BytesIO

    provider = get_provider("ollama")
    requests: list[tuple[str, dict]] = []

    class _Resp(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        payload = jsonlib.loads(req.data.decode())
        requests.append((req.full_url, payload))
        vecs = [[0.1, 0.2] for _ in payload["input"]]
        return _Resp(jsonlib.dumps({"embeddings": vecs}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = provider.embed_batch(["a", "b", "c"])
    assert result == [[0.1, 0.2]] * 3
    assert len(requests) == 1
    url, payload = requests[0]
    assert url.endswith("/api/embed")
    assert payload["input"] == ["a", "b", "c"]
    assert provider.endpoint == "api/embed"


def test_ollama_falls_back_to_legacy_endpoint_only_on_404(monkeypatch):
    """Legacy /api/embeddings engages on PROVEN absence (404) — once — while
    a 500 propagates instead of being masked as an old server."""
    import json as jsonlib
    import urllib.error
    from io import BytesIO

    provider = get_provider("ollama")
    calls: list[str] = []

    class _Resp(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/api/embed"):
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
        payload = jsonlib.loads(req.data.decode())
        return _Resp(jsonlib.dumps({"embedding": [0.5, 0.5, len(payload["prompt"])]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert provider.embed_batch(["x", "y"]) == [[0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]
    assert provider.endpoint == "api/embeddings"
    # Detection is remembered: no second /api/embed attempt.
    assert provider.embed("z") == [0.5, 0.5, 1.0]
    assert calls.count(f"{provider.host}/api/embed") == 1

    fresh = get_provider("ollama")

    def failing_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    # 5xx is NOT an old server: public embed contract degrades to [] but the
    # endpoint selection must stay modern (no silent downgrade).
    assert fresh.embed_batch(["x"]) == [[]]
    assert fresh.endpoint == "api/embed"


def test_ollama_batch_cardinality_mismatch_is_not_partial_success(monkeypatch):
    import json as jsonlib
    from io import BytesIO

    provider = get_provider("ollama")

    class _Resp(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return _Resp(jsonlib.dumps({"embeddings": [[0.1]]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    # 1 vector for 2 inputs → every item fails, nothing partial.
    assert provider.embed_batch(["a", "b"]) == [[], []]


def test_ollama_unknown_dimension_is_probed_not_guessed(monkeypatch):
    """No blind 768: unknown models discover their real dimension from the
    live model, and an unreachable host fails LOUD before any collection
    could be created with a wrong size."""
    import json as jsonlib
    from io import BytesIO

    from mnemostack.embeddings import ProviderProbeError
    from mnemostack.embeddings.ollama import OllamaProvider

    class _Resp(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return _Resp(jsonlib.dumps({"embeddings": [[0.1] * 4096]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert OllamaProvider(model="brand-new-embedder").dimension == 4096

    def refusing_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refusing_urlopen)
    with pytest.raises(ProviderProbeError, match="no fallback dimension"):
        _ = OllamaProvider(model="brand-new-embedder").dimension


def test_gemini_uses_api_key_header_not_query_string(monkeypatch):
    from mnemostack.embeddings.gemini import GeminiProvider

    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"embedding": {"values": [0.1, 0.2]}}).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = GeminiProvider(api_key="secret-key", timeout=7)
    assert provider.embed("hello") == [0.1, 0.2]

    assert "key=" not in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "secret-key"
    assert seen["timeout"] == 7


def test_gemini_sequential_fallback_probe_guard(monkeypatch):
    """When the batch endpoint AND the probe item fail, the fallback must not
    hammer the API once per remaining item."""
    import urllib.error

    from mnemostack.embeddings.gemini import GeminiProvider

    calls = {"n": 0}

    def always_503(req, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 503, "boom", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", always_503)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = GeminiProvider(api_key="test-key", max_retries=2)

    out = provider.embed_batch([f"text {i}" for i in range(50)])

    assert out == [[] for _ in range(50)]
    # batch retries (2) + probe retries (2) — NOT 2 + 50*2
    assert calls["n"] == 4


def test_gemini_sequential_fallback_continues_after_probe_ok(monkeypatch):
    import json as _json

    from mnemostack.embeddings.gemini import GeminiProvider

    calls = {"n": 0}

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        calls["n"] += 1
        if "batchEmbedContents" in req.full_url:
            import urllib.error

            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
        return _Resp(_json.dumps({"embedding": {"values": [0.1, 0.2]}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = GeminiProvider(api_key="test-key", max_retries=1)

    out = provider.embed_batch(["a", "b", "c"])

    assert out == [[0.1, 0.2]] * 3


def test_gemini_sequential_fallback_content_failure_does_not_skip_rest(monkeypatch):
    """A content-specific rejection (400 on one oversized chunk) must fail only
    that item — the storm guard is for provider-wide failures, not bad input."""
    import json as _json
    import urllib.error

    from mnemostack.embeddings.gemini import GeminiProvider

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        if "batchEmbedContents" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
        body = _json.loads(req.data)
        text = body["content"]["parts"][0]["text"]
        if text == "oversized":
            raise urllib.error.HTTPError(req.full_url, 400, "too large", {}, None)
        return _Resp(_json.dumps({"embedding": {"values": [0.1, 0.2]}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = GeminiProvider(api_key="test-key", max_retries=1)

    out = provider.embed_batch(["oversized", "b", "c"])

    assert out == [[], [0.1, 0.2], [0.1, 0.2]]


def test_gemini_sequential_fallback_stops_on_midbatch_provider_failure(monkeypatch):
    """If the provider goes down mid-batch, the remaining items are failed
    without further calls."""
    import json as _json
    import urllib.error

    from mnemostack.embeddings.gemini import GeminiProvider

    calls = {"n": 0}

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        if "batchEmbedContents" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
        calls["n"] += 1
        if calls["n"] >= 2:
            raise urllib.error.HTTPError(req.full_url, 503, "down", {}, None)
        return _Resp(_json.dumps({"embedding": {"values": [0.1, 0.2]}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = GeminiProvider(api_key="test-key", max_retries=1)

    out = provider.embed_batch(["a", "b", "c", "d", "e"])

    assert out == [[0.1, 0.2], [], [], [], []]
    assert calls["n"] == 2  # item 2 failed provider-wide; items 3-5 never tried
