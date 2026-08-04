"""Provider configuration plumbing: host/timeout reach the provider on every surface."""

from __future__ import annotations

import json as jsonlib
from io import BytesIO

import pytest

from mnemostack.config import Config, provider_kwargs
from mnemostack.embeddings import ProviderProbeError, get_provider
from mnemostack.embeddings.base import EmbeddingCapabilities, EmbeddingProvider
from mnemostack.embeddings.ollama import OllamaProvider


class _Resp(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------ provider_kwargs


def test_provider_kwargs_routes_knobs_only_to_known_providers():
    kw = provider_kwargs("ollama", model="m", ollama_host="http://h:1", timeout=42)
    assert kw == {"model": "m", "host": "http://h:1", "timeout": 42}
    # Gemini takes a timeout but never an Ollama host.
    assert provider_kwargs("gemini", model="g", timeout=9) == {"model": "g", "timeout": 9}
    assert provider_kwargs("gemini", ollama_host="http://h:1") == {}
    # A custom registered provider keeps the historical model-only contract —
    # an unexpected keyword must not break its constructor.
    assert provider_kwargs("my-provider", model="x", ollama_host="http://h:1", timeout=5) == {
        "model": "x"
    }
    assert provider_kwargs("huggingface", timeout=5) == {}


def test_ollama_kwargs_reach_the_provider():
    p = get_provider("ollama", **provider_kwargs("ollama", ollama_host="http://198.51.100.7:11434", timeout=77))
    assert p.host == "http://198.51.100.7:11434"
    assert p.timeout == 77


# ------------------------------------------------------------ env precedence


def test_embedding_env_vars_reach_config(monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_OLLAMA_HOST", "http://192.0.2.10:11434")
    monkeypatch.setenv("MNEMOSTACK_EMBEDDING_TIMEOUT", "240")
    cfg = Config.load(path=None)
    assert cfg.embedding.ollama_host == "http://192.0.2.10:11434"
    assert cfg.embedding.timeout == 240


def test_malformed_embedding_timeout_env_fails_loud(monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_EMBEDDING_TIMEOUT", "soon")
    with pytest.raises(ValueError):
        Config.load(path=None)


# ------------------------------------------------------------ host resolution


def test_ollama_host_resolution_precedence(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaProvider().host == "http://localhost:11434"
    # Native OLLAMA_HOST is honored (scheme added when missing)…
    monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
    assert OllamaProvider().host == "http://192.0.2.10:11434"
    # …but an explicit argument (CLI/config plumbing) is strongest.
    assert OllamaProvider(host="http://explicit:11434").host == "http://explicit:11434"


def test_ollama_cold_start_default_timeout_is_generous():
    # 30s was too short for cold loads of larger local models.
    assert OllamaProvider().timeout == 180


# --------------------------------------------------------- probe_capabilities


def test_default_probe_capabilities_validates_or_raises():
    class _Good(EmbeddingProvider):
        def embed(self, text):
            return [0.1, 0.2, 0.3]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

        @property
        def dimension(self):
            return 3

        @property
        def name(self):
            return "custom:good"

    caps = _Good().probe_capabilities()
    assert caps == EmbeddingCapabilities(dimension=3)

    class _Empty(_Good):
        def embed(self, text):
            return []

    with pytest.raises(ProviderProbeError, match="empty or non-numeric"):
        _Empty().probe_capabilities()

    class _Broken(_Good):
        def embed(self, text):
            raise ConnectionError("down")

    with pytest.raises(ProviderProbeError, match="probe failed"):
        _Broken().probe_capabilities()


def test_ollama_probe_capabilities_reports_endpoint_and_batch(monkeypatch):
    def fake_urlopen(req, timeout=None):
        payload = jsonlib.loads(req.data.decode())
        vecs = [[0.1] * 1024 for _ in payload["input"]]
        return _Resp(jsonlib.dumps({"embeddings": vecs}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = OllamaProvider(model="brand-new-embedder")
    caps = p.probe_capabilities()
    assert caps.dimension == 1024
    assert caps.batch is True
    assert caps.endpoint == "api/embed"
    # The probe doubles as dimension discovery.
    assert p.dimension == 1024


def test_yaml_embedding_timeout_is_validated_at_load(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMOSTACK_EMBEDDING_TIMEOUT", raising=False)
    for bad in ("soon", "0", "-5", "true"):
        cfg_file = tmp_path / f"cfg-{bad}.yaml"
        cfg_file.write_text(f"embedding:\n  timeout: {bad}\n")
        with pytest.raises(ValueError):
            Config.load(path=cfg_file)
    ok_file = tmp_path / "ok.yaml"
    ok_file.write_text("embedding:\n  timeout: 240\n")
    assert Config.load(path=ok_file).embedding.timeout == 240


def test_ollama_model_404_with_json_body_does_not_trigger_legacy_fallback(monkeypatch):
    """A missing MODEL answers 404 WITH a JSON error body — the route exists,
    so the instance must not downgrade to the legacy endpoint."""
    import urllib.error

    provider = get_provider("ollama")

    def fake_urlopen(req, timeout=None):
        body = BytesIO(jsonlib.dumps({"error": "model not found, pull it first"}).encode())
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert provider.embed_batch(["a"]) == [[]]  # graceful public contract
    assert provider.endpoint == "api/embed"  # no silent downgrade


def test_ollama_legacy_empty_vector_fails_the_whole_batch(monkeypatch):
    """The all-or-nothing contract holds on the legacy path too."""
    import urllib.error

    provider = get_provider("ollama")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/embed"):
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        calls["n"] += 1
        vec = [0.1] if calls["n"] == 1 else []
        return _Resp(jsonlib.dumps({"embedding": vec}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert provider.embed_batch(["a", "b"]) == [[], []]  # no mixed partials


def test_env_timeout_zero_is_rejected_not_clamped(monkeypatch):
    # All three sources (flag, env, file) share one contract: 0/negative is
    # a startup rejection, never a silent clamp to 1.
    monkeypatch.setenv("MNEMOSTACK_EMBEDDING_TIMEOUT", "0")
    with pytest.raises(ValueError):
        Config.load(path=None)


def test_cli_rejects_nonpositive_embedding_timeout():
    import argparse

    from mnemostack.cli import _positive_int

    assert _positive_int("180") == 180
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    with pytest.raises(ValueError):
        _positive_int("soon")


def test_cli_main_reports_probe_errors_cleanly(monkeypatch, capsys):
    # A typed refusal must follow the CLI convention (message + exit 2),
    # never a raw traceback.
    import mnemostack.cli as cli

    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    rc = cli.main(
        [
            "search",
            "anything",
            "--provider",
            "ollama",
            "--embedding-model",
            "brand-new-embedder",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err and "no fallback dimension" in err


def test_unknown_model_probe_on_legacy_server_does_not_deadlock(monkeypatch):
    """The dimension probe holds the detection lock while _embed_api may take
    the legacy-fallback branch that re-acquires it — must complete, not hang."""
    import urllib.error

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/embed"):
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        return _Resp(jsonlib.dumps({"embedding": [0.1, 0.2]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = OllamaProvider(model="brand-new-embedder")
    assert p.dimension == 2
    assert p.endpoint == "api/embeddings"


def test_new_knobs_stay_at_the_positional_tail():
    """ServerConfig and build_server are documented stable and may be used
    positionally — the new provider knobs must never shift older arguments."""
    import dataclasses
    import inspect

    pytest.importorskip("fastapi")
    from mnemostack.server import ServerConfig

    field_names = [f.name for f in dataclasses.fields(ServerConfig)]
    assert field_names[-2:] == ["ollama_host", "embedding_timeout"]

    pytest.importorskip("fastmcp")
    from mnemostack.mcp import build_server

    params = list(inspect.signature(build_server).parameters)
    assert params[-2:] == ["ollama_host", "embedding_timeout"]


def test_config_sdk_helper_carries_provider_knobs(monkeypatch):
    monkeypatch.setenv("MNEMOSTACK_PROVIDER", "ollama")
    monkeypatch.setenv("MNEMOSTACK_OLLAMA_HOST", "http://192.0.2.10:11434")
    monkeypatch.setenv("MNEMOSTACK_EMBEDDING_TIMEOUT", "240")
    cfg = Config.load(path=None)
    kw = cfg.embedding_provider_kwargs()
    assert kw["host"] == "http://192.0.2.10:11434"
    assert kw["timeout"] == 240


# ------------------------------------------------------- bounded ingestion


class _BatchCountingProvider(EmbeddingProvider):
    def __init__(self):
        self.batch_calls: list[int] = []

    def embed(self, text):
        return [0.1, 0.2]

    def embed_batch(self, texts):
        self.batch_calls.append(len(texts))
        return [[0.1, 0.2] for _ in texts]

    @property
    def dimension(self):
        return 2

    @property
    def name(self):
        return "custom:batch-counter"


class _BatchStore:
    def __init__(self):
        self.upsert_batches: list[int] = []
        self.single_upserts = 0

    def upsert(self, cid, vec, payload, **kw):
        self.single_upserts += 1

    def upsert_batch(self, points, **kw):
        self.upsert_batches.append(len(points))

    def set_payload(self, cid, payload, **kw):
        pass

    def delete_payload_keys(self, cid, keys, **kw):
        pass


def _chunks(n):
    return [(f"id{i}", f"text {i}", {"text": f"text {i}", "source": "a.md"}) for i in range(n)]


def test_markdown_sync_embeds_and_upserts_in_bounded_groups():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    provider = _BatchCountingProvider()
    store = _BatchStore()
    res = upsert_markdown_chunks(
        store, provider, _chunks(5), existing_payloads={}, embedding_batch_size=2
    )
    assert res.inserted == 5
    # ONE provider batch call per group, never the whole corpus at once…
    assert provider.batch_calls == [2, 2, 1]
    # …and one store round-trip per group (no per-point fallback used).
    assert store.upsert_batches == [2, 2, 1]
    assert store.single_upserts == 0


def test_markdown_sync_quota_path_batches_but_checks_totals_once():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    provider = _BatchCountingProvider()
    store = _BatchStore()
    hook_calls: list[int] = []

    def hook(inserts, failed_sources):
        hook_calls.append(inserts)

    res = upsert_markdown_chunks(
        store,
        provider,
        _chunks(5),
        existing_payloads={},
        before_upsert=hook,
        embedding_batch_size=2,
    )
    assert res.inserted == 5
    assert provider.batch_calls == [2, 2, 1]  # bounded provider groups
    assert hook_calls == [5]  # the quota check still sees ONE total


def test_resilient_ladder_isolates_poisoned_items():
    from mnemostack.embeddings.roles import embed_documents_resilient

    class _PoisonedBatch(_BatchCountingProvider):
        def embed_batch(self, texts):
            raise RuntimeError("batch endpoint exploded")

        def embed(self, text):
            if "bad" in text:
                raise RuntimeError("poisoned item")
            return [0.5]

    vectors = embed_documents_resilient(_PoisonedBatch(), ["ok one", "bad apple", "ok two"])
    assert vectors == [[0.5], [], [0.5]]


def test_embedding_batch_size_config_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_EMBEDDING_BATCH_SIZE", "16")
    assert Config.load(path=None).embedding.batch_size == 16
    monkeypatch.setenv("MNEMOSTACK_EMBEDDING_BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        Config.load(path=None)
    monkeypatch.delenv("MNEMOSTACK_EMBEDDING_BATCH_SIZE")
    bad = tmp_path / "cfg.yaml"
    bad.write_text("embedding:\n  batch_size: -3\n")
    with pytest.raises(ValueError):
        Config.load(path=bad)


def test_resilient_ladder_normalizes_wrong_cardinality():
    # A provider answering with the wrong vector count must not let trailing
    # items vanish uncounted — missing entries are explicit failures.
    from mnemostack.embeddings.roles import embed_documents_resilient

    class _ShortBatch(_BatchCountingProvider):
        def embed_batch(self, texts):
            return [[0.1]]  # one vector for many inputs

    assert embed_documents_resilient(_ShortBatch(), ["a", "b", "c"]) == [[0.1], [], []]

    class _LongBatch(_BatchCountingProvider):
        def embed_batch(self, texts):
            return [[0.1]] * (len(texts) + 2)

    assert embed_documents_resilient(_LongBatch(), ["a"]) == [[0.1]]


def test_resilient_ladder_degrades_without_batch_api():
    from mnemostack.embeddings.roles import embed_documents_resilient

    class _NoBatch:
        def __init__(self):
            self.calls: list[str] = []

        def embed(self, text):
            self.calls.append(text)
            return [0.7]

    duck = _NoBatch()
    assert embed_documents_resilient(duck, ["a", "b"]) == [[0.7], [0.7]]
    assert duck.calls == ["a", "b"]  # per-item, raw text (duck contract)


def test_cmd_index_embeds_and_upserts_in_bounded_groups(tmp_path, monkeypatch, capsys):
    import mnemostack.cli as cli

    for i in range(3):
        (tmp_path / f"doc{i}.md").write_text(f"content number {i}")

    provider = _BatchCountingProvider()

    class _Store(_BatchStore):
        def __init__(self, **kw):
            super().__init__()

        def collection_exists(self):
            return False

        def ensure_collection(self, recreate=False):
            pass

        def iter_ids(self):
            return iter(())

        def scroll(self, *a, **kw):
            return iter(())

        def count(self):
            return 0

    store_box: list[_Store] = []

    def make_store(**kw):
        store_box.append(_Store())
        return store_box[-1]

    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: provider)
    monkeypatch.setattr(cli, "VectorStore", make_store)
    rc = cli.main(
        [
            "index",
            str(tmp_path),
            "--chunk-size",
            "2000",
            "--embedding-batch-size",
            "2",
        ]
    )
    assert rc == 0, capsys.readouterr()
    # 3 one-chunk files, batch size 2 → provider batches [2, 1]…
    assert provider.batch_calls == [2, 1]
    # …and one store round-trip per group.
    assert store_box[-1].upsert_batches == [2, 1]
    assert store_box[-1].single_upserts == 0
