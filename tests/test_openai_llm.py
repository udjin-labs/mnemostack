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


class _FakeOpener:
    """Stands in for the module's no-redirect opener (the provider never
    calls bare ``urllib.request.urlopen``)."""

    def __init__(self, handler):
        self._handler = handler

    def open(self, req, timeout=0):
        return self._handler(req, timeout)


def _install(monkeypatch, handler):
    monkeypatch.setattr("mnemostack.llm.openai_compat._OPENER", _FakeOpener(handler))


def _capture(monkeypatch, body=None):
    captured = {}

    def handler(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _FakeResponse(body if body is not None else _reply())

    _install(monkeypatch, handler)
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
    def deny(req, timeout):
        raise HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))

    _install(monkeypatch, deny)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok
    assert "401" in resp.error and "bad key" in resp.error


def test_network_error_becomes_response_error(monkeypatch):
    def boom(req, timeout):
        raise OSError("connection refused")

    _install(monkeypatch, boom)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "connection refused" in resp.error


def test_unexpected_response_shape_is_an_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=json.dumps({"detail": "not a chat response"}).encode())
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "unexpected response shape" in resp.error


def test_whitespace_only_content_is_an_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=_reply(content="\n  \n"))
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "empty content" in resp.error


def test_list_content_names_the_type_not_a_budget(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=_reply(content=[{"type": "text", "text": "hi"}]))
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "unexpected content type: list" in resp.error


@pytest.mark.parametrize("usage", ["n/a", ["tokens"], 7])
def test_malformed_usage_never_raises(monkeypatch, usage):
    body = json.dumps({"choices": [{"message": {"content": "Paris."}}], "usage": usage}).encode()
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=body)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert resp.ok and resp.text == "Paris." and resp.tokens_used is None


def test_empty_content_is_an_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=_reply(content=""))
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "empty content" in resp.error


def test_describe_image_fail_open_default(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").describe_image(b"x")
    assert not resp.ok and "does not support image description" in resp.error


# ----------------------------------------------- redirects and hard limits


def test_redirect_handler_refuses_to_follow():
    """A gateway 3xx must never carry the bearer token to a new origin.
    The real opener is exercised directly: its redirect handler must return
    None (urllib then raises the HTTPError our error path already handles)."""
    import mnemostack.llm.openai_compat as mod

    handlers = [h for h in mod._OPENER.handlers if isinstance(h, mod._NoRedirect)]
    assert handlers, "no-redirect handler not installed on the opener"
    req = mod.urllib.request.Request("http://gw:4000/v1/chat/completions")
    assert handlers[0].redirect_request(req, None, 302, "Found", {}, "http://evil/v1") is None


def test_redirect_surfaces_as_error_response(monkeypatch):
    def redirect(req, timeout):
        raise HTTPError(req.full_url, 302, "Found", {"Location": "http://evil/v1"}, None)

    _install(monkeypatch, redirect)
    monkeypatch.setenv(API_KEY_ENV, "sk-secret")
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "302" in resp.error


def test_http_error_body_read_is_bounded(monkeypatch):
    class _HugeBody(io.BytesIO):
        def read(self, n=-1):
            assert n != -1 and n <= 300, "unbounded read of the error body"
            return b"x" * n

    def deny(req, timeout):
        raise HTTPError(req.full_url, 500, "boom", {}, _HugeBody())

    _install(monkeypatch, deny)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert not resp.ok and "500" in resp.error


@pytest.mark.parametrize("bad", [True, False, -5])
def test_bool_and_negative_token_counts_degrade_to_none(monkeypatch, bad):
    body = json.dumps(
        {"choices": [{"message": {"content": "Paris."}}], "usage": {"total_tokens": bad}}
    ).encode()
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    _capture(monkeypatch, body=body)
    resp = OpenAICompatLLM(model="m", host="http://gw:4000").generate("q")
    assert resp.ok and resp.tokens_used is None


# ------------------------------------------------ reasoning-model knobs


def test_token_param_renames_the_budget_field(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    captured = _capture(monkeypatch)
    OpenAICompatLLM(
        model="o1", host="http://gw:4000", token_param="max_completion_tokens"
    ).generate("q", max_tokens=77)
    assert captured["body"]["max_completion_tokens"] == 77
    assert "max_tokens" not in captured["body"]


def test_options_override_and_none_removes(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    captured = _capture(monkeypatch)
    OpenAICompatLLM(
        model="o1",
        host="http://gw:4000",
        options={"temperature": None, "top_p": 0.9},
    ).generate("q", temperature=0.0)
    assert "temperature" not in captured["body"]
    assert captured["body"]["top_p"] == 0.9


# ------------------------------------------------------- llm_kwargs gating


def test_llm_kwargs_threads_host_and_timeout_to_openai():
    kw = llm_kwargs("openai", model="m", llm_host="http://gw:4000", timeout=120)
    assert kw == {"model": "m", "host": "http://gw:4000", "timeout": 120}


def test_llm_kwargs_never_inherits_embedding_host_for_openai():
    kw = llm_kwargs("openai", model="m", embedding_ollama_host="http://gpu:11434")
    assert "host" not in kw
