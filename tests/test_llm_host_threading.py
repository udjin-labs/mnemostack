"""The configured Ollama host must reach the LLM on every surface (issue #180).

The embedding path resolves its host through ``provider_kwargs`` — the shared
resolution point whose docstring promises a configured host can never be
accepted by the config schema yet silently dropped. The LLM path used to be
built through ``model_kwargs`` (model only) at every call site, so
``--ollama-host`` pointed embeddings at the GPU box while every LLM call
dialed localhost. ``llm_kwargs`` is the sibling resolution point; these tests
pin its rules and the two ends of the chain around it.
"""

from __future__ import annotations

import argparse
import os

import pytest

from mnemostack.config import Config, llm_kwargs

# --- the resolution point itself --------------------------------------------


def test_ollama_llm_inherits_embedding_host():
    kw = llm_kwargs("ollama", model="m", embedding_ollama_host="http://gpu:11434")
    assert kw == {"model": "m", "host": "http://gpu:11434"}


def test_explicit_llm_host_beats_inheritance():
    kw = llm_kwargs(
        "ollama",
        llm_host="http://llm-box:11434",
        embedding_ollama_host="http://gpu:11434",
    )
    assert kw["host"] == "http://llm-box:11434"


def test_timeout_only_when_configured():
    assert "timeout" not in llm_kwargs("ollama", embedding_ollama_host="http://x")
    assert llm_kwargs("ollama", timeout=120)["timeout"] == 120


@pytest.mark.parametrize("name", ["gemini", "gemini-flash", "Gemini"])
def test_gemini_gets_timeout_but_never_a_host(name):
    kw = llm_kwargs(name, model="m", llm_host="http://x", timeout=45)
    assert kw == {"model": "m", "timeout": 45}


def test_custom_provider_keeps_model_only_contract():
    # Same gating as provider_kwargs: an unknown registered provider must not
    # receive keywords its constructor never promised to take.
    kw = llm_kwargs("my-custom", model="m", llm_host="http://x", timeout=9)
    assert kw == {"model": "m"}


# --- OllamaLLM's own chain: explicit -> OLLAMA_HOST -> localhost -------------


def test_ollama_llm_honours_native_env(monkeypatch):
    pytest.importorskip("httpx")
    from mnemostack.llm.ollama import OllamaLLM

    monkeypatch.setenv("OLLAMA_HOST", "http://native:11434")
    assert OllamaLLM().host == "http://native:11434"
    # The embedding provider documents the same chain; the two clients used
    # to disagree, which was the container half of #180.
    monkeypatch.setenv("OLLAMA_HOST", "http://native:11434/")
    assert OllamaLLM(host="http://explicit:11434").host == "http://explicit:11434"


def test_ollama_llm_defaults_to_localhost(monkeypatch):
    pytest.importorskip("httpx")
    from mnemostack.llm.ollama import OllamaLLM

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaLLM().host == "http://localhost:11434"


# --- the server surface ------------------------------------------------------


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    for key in list(os.environ.keys()):
        if key.startswith("MNEMOSTACK_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield monkeypatch


def test_server_config_carries_llm_host_and_timeout(isolated_env):
    from mnemostack.server import ServerConfig

    isolated_env.setenv("MNEMOSTACK_LLM_HOST", "http://llm-box:11434")
    isolated_env.setenv("MNEMOSTACK_LLM_TIMEOUT", "180")
    cfg = ServerConfig.from_env()
    assert cfg.llm_host == "http://llm-box:11434"
    assert cfg.llm_timeout == 180


def test_serve_repro_from_the_issue(isolated_env):
    """`--ollama-host` for embeddings must reach an ollama LLM too."""
    from mnemostack.server import ServerConfig

    isolated_env.setenv("MNEMOSTACK_PROVIDER", "ollama")
    isolated_env.setenv("MNEMOSTACK_OLLAMA_HOST", "http://gpu-node:11434")
    isolated_env.setenv("MNEMOSTACK_LLM", "ollama")
    cfg = ServerConfig.from_env()
    kw = llm_kwargs(
        cfg.llm_name,
        model=cfg.llm_model,
        llm_host=cfg.llm_host,
        embedding_ollama_host=cfg.ollama_host,
        timeout=cfg.llm_timeout,
    )
    assert kw.get("host") == "http://gpu-node:11434"


def test_config_file_llm_host(isolated_env, tmp_path):
    cfg_file = tmp_path / "mnemostack.yaml"
    cfg_file.write_text("llm:\n  provider: ollama\n  host: http://llm-box:11434\n  timeout: 90\n")
    isolated_env.setenv("MNEMOSTACK_CONFIG", str(cfg_file))
    cfg = Config.load()
    assert cfg.llm.host == "http://llm-box:11434"
    assert cfg.llm.timeout == 90


# --- the CLI surface ---------------------------------------------------------


def test_cli_helper_reads_one_place():
    from mnemostack import cli

    args = argparse.Namespace(
        llm="ollama",
        llm_model="qwen",
        ollama_host="http://gpu:11434",
        _llm_host=None,
        _llm_timeout=None,
    )
    kw = cli._llm_build_kwargs(args, "ollama")
    assert kw == {"model": "qwen", "host": "http://gpu:11434"}

    args._llm_host = "http://llm-box:11434"
    args._llm_timeout = 240
    kw = cli._llm_build_kwargs(args, "ollama")
    assert kw == {"model": "qwen", "host": "http://llm-box:11434", "timeout": 240}


# --- the SDK surface: the ninth construction site ----------------------------


def test_sdk_helper_carries_llm_host(isolated_env):
    """`get_llm(cfg.llm.provider, **cfg.llm_provider_kwargs())` — the documented
    sibling of the embedding SDK path — must carry the host too. This helper
    builds kwargs rather than calling get_llm, so an enumeration by call sites
    cannot find it; it gets its own pin instead."""
    isolated_env.setenv("MNEMOSTACK_PROVIDER", "ollama")
    isolated_env.setenv("MNEMOSTACK_OLLAMA_HOST", "http://gpu:11434")
    isolated_env.setenv("MNEMOSTACK_LLM", "ollama")
    cfg = Config.load(path=None)
    kw = cfg.llm_provider_kwargs()
    assert kw["host"] == "http://gpu:11434"

    isolated_env.setenv("MNEMOSTACK_LLM_HOST", "http://llm-box:11434")
    isolated_env.setenv("MNEMOSTACK_LLM_TIMEOUT", "120")
    cfg = Config.load(path=None)
    kw = cfg.llm_provider_kwargs()
    assert kw["host"] == "http://llm-box:11434"
    assert kw["timeout"] == 120


def test_ollama_llm_normalizes_schemeless_native_env(monkeypatch):
    """OLLAMA_HOST is commonly bare host:port; the embedding provider
    normalizes it and 'same chain' must include the normalization."""
    pytest.importorskip("httpx")
    from mnemostack.llm.ollama import OllamaLLM

    monkeypatch.setenv("OLLAMA_HOST", "gpu-node:11434")
    assert OllamaLLM().host == "http://gpu-node:11434"


def test_llm_timeout_is_validated_like_its_sibling(isolated_env, tmp_path):
    cfg_file = tmp_path / "mnemostack.yaml"
    cfg_file.write_text("llm:\n  timeout: -5\n")
    isolated_env.setenv("MNEMOSTACK_CONFIG", str(cfg_file))
    with pytest.raises(ValueError, match="llm.timeout must be a positive integer"):
        Config.load()
    isolated_env.delenv("MNEMOSTACK_CONFIG")
    isolated_env.setenv("MNEMOSTACK_LLM_TIMEOUT", "abc")
    with pytest.raises(ValueError):
        Config.load(path=None)
