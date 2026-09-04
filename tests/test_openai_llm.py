"""Tests for the OpenAI-compatible LLM provider (issue #187).

The provider is the universal adapter for LiteLLM proxies, vLLM, llama.cpp
server and OpenAI-compatible cloud endpoints. These tests pin the request
contract (URL joining, auth header, payload shape), the loud construction
errors, and the ``llm_kwargs`` gating that threads host/timeout to it.
"""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from mnemostack.config import llm_kwargs
from mnemostack.llm import get_llm, list_llms
from mnemostack.llm.openai_compat import API_KEY_ENV, OpenAICompatLLM


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _reply(content="Paris.", total_tokens=17):
    return json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"total_tokens": total_tokens},
        }
    ).encode()


def _capture(monkeypatch, body=None):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _FakeResponse(body if body is not None else _reply())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


# ---------------------------------------------------------------- registry


def test_openai_is_registered():
    assert "openai" in list_llms()


def test_get_llm_constructs_openai(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    llm = get_llm("openai", model="m", host="http://gw:4000")
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.name == "openai:m"


# ---------------------------------------------------- loud construction


def test_missing_host_fails_loud_with_the_config_knob_named():
    with pytest.raises(ValueError, match="MNEMOSTACK_LLM_HOST"):
        OpenAICompatLLM(model="m")


def test_missing_model_fails_loud_with_the_flag_named():
    with pytest.raises(ValueError, match="--llm-model"):
        OpenAICompatLLM(host="http://gw:4000")


# ------------------------------------------------------------ URL joining


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("http://gw:4000", "http://gw:4000/v1/chat/completions"),
        ("http://gw:4000/", "http://gw:4000/v1/chat/completions"),
        # OpenAI SDK convention: base_url already ends in /v1 — never double it.
        ("http://gw:4000/v1", "http://gw:4000/v1/chat/completions"),
        ("http://gw:4000/v1/", "http://gw:4000/v1/chat/completions"),
        # Bare host:port gets a scheme, same normalization as the ollama pair.
        ("gw:4000", "http://gw:4000/v1/chat/completions"),
    ],
)
def test_endpoint_url_joining(monkeypatch, host, expected):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    captured = _capture(monkeypatch)
    OpenAICompatLLM(model="m", host=host).generate("q")
    assert captured["url"] == expected


# ------------------------------------------------------------------- auth


def test_bearer_header_from_env(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-test")
    captured = _capture(monkeypatch)
    OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_explicit_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-env")
    captured = _capture(monkeypatch)
    OpenAICompatLLM(model="m", host="http://gw:4000", api_key="sk-arg").generate("q")
    assert captured["headers"]["Authorization"] == "Bearer sk-arg"


@pytest.mark.parametrize("key", [None, "", "none", "NONE"])
def test_keyless_sends_no_authorization_header(monkeypatch, key):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    if key is not None:
        monkeypatch.setenv(API_KEY_ENV, key)
    captured = _capture(monkeypatch)
    OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert "Authorization" not in captured["headers"]


# ---------------------------------------------------------------- payload


def test_generate_payload_shape(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    captured = _capture(monkeypatch)
    resp = OpenAICompatLLM(model="team-llm", host="http://gw:4000", timeout=45).generate(
        "What is the capital of France?", max_tokens=99, temperature=0.3
    )
    body = captured["body"]
    assert body["model"] == "team-llm"
    assert body["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
    assert body["max_tokens"] == 99
    assert body["temperature"] == 0.3
    assert body["stream"] is False
    assert captured["timeout"] == 45
    assert resp.ok and resp.text == "Paris." and resp.tokens_used == 17


# ------------------------------------------------------------ error paths


def test_http_error_surfaces_body_snippet_without_raising(monkeypatch):
    def deny(req, timeout=0):
        raise HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))

    monkeypatch.setattr("urllib.request.urlopen", deny)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok
    assert "401" in resp.error and "bad key" in resp.error


def test_network_error_becomes_response_error(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "connection refused" in resp.error


def test_unexpected_response_shape_is_an_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=json.dumps({"detail": "not a chat response"}).encode())
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "unexpected response shape" in resp.error


def test_empty_content_is_an_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=_reply(content=""))
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "empty content" in resp.error


def test_describe_image_fail_open_default(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").describe_image(b"x")
    assert not resp.ok and "does not support image description" in resp.error


# ------------------------------------------------------- llm_kwargs gating


def test_llm_kwargs_threads_host_and_timeout_to_openai():
    kw = llm_kwargs("openai", model="m", llm_host="http://gw:4000", timeout=120)
    assert kw == {"model": "m", "host": "http://gw:4000", "timeout": 120}


def test_llm_kwargs_never_inherits_embedding_host_for_openai():
    kw = llm_kwargs("openai", model="m", embedding_ollama_host="http://gpu:11434")
    assert "host" not in kw
