"""Tests for OllamaLLM think control and generation-options passthrough."""

from __future__ import annotations

import io
import json

from mnemostack.llm.ollama import OllamaLLM
from mnemostack.llm.registry import get_llm


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _ollama(monkeypatch, reply="ok", **kwargs):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(json.dumps({"response": reply, "eval_count": 7}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return OllamaLLM(model="qwen3:14b", **kwargs), captured


def test_think_false_is_default_and_boolean(monkeypatch):
    """Reasoning models burn the whole num_predict budget on thoughts and
    return empty text — disabling think is the safe default. The value must
    be a JSON boolean: Ollama rejects the strings \"true\"/\"false\"."""
    llm, captured = _ollama(monkeypatch)

    resp = llm.generate("hello")

    assert resp.ok
    assert captured["body"]["think"] is False


def test_think_none_omits_the_field(monkeypatch):
    llm, captured = _ollama(monkeypatch, think=None)

    llm.generate("hello")

    assert "think" not in captured["body"]


def test_think_true_is_sent(monkeypatch):
    llm, captured = _ollama(monkeypatch, think=True)

    llm.generate("hello")

    assert captured["body"]["think"] is True


def test_think_applies_to_describe_image_too(monkeypatch):
    llm, captured = _ollama(monkeypatch)

    llm.describe_image(b"\xff\xd8 fake jpeg")

    assert captured["body"]["think"] is False


def test_options_passthrough_merges_and_overrides(monkeypatch):
    llm, captured = _ollama(monkeypatch, options={"num_ctx": 8192, "temperature": 0.7})

    llm.generate("hello", max_tokens=50, temperature=0.0)

    opts = captured["body"]["options"]
    assert opts["num_ctx"] == 8192
    assert opts["temperature"] == 0.7  # explicit options win over per-call args
    assert opts["num_predict"] == 50


def test_options_default_payload_unchanged(monkeypatch):
    llm, captured = _ollama(monkeypatch)

    llm.generate("hello", max_tokens=120, temperature=0.2)

    assert captured["body"]["options"] == {"temperature": 0.2, "num_predict": 120}


def test_get_llm_threads_think_and_options(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(json.dumps({"response": "ok"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = get_llm("ollama", model="qwen3:14b", think=None, options={"top_p": 0.9})

    llm.generate("hello")

    assert "think" not in captured["body"]
    assert captured["body"]["options"]["top_p"] == 0.9
