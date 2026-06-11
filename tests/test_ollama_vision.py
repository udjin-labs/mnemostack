"""Tests for OllamaLLM.describe_image and the LLMProvider vision contract."""

from __future__ import annotations

import base64
import io
import json

from mnemostack.llm.base import DEFAULT_IMAGE_PROMPT, LLMProvider, LLMResponse
from mnemostack.llm.ollama import OllamaLLM


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _ollama(monkeypatch, reply="a wooden table with a vase of tulips"):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(json.dumps({"response": reply, "eval_count": 42}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return OllamaLLM(model="llava"), captured


def test_describe_image_sends_base64_images_field(monkeypatch):
    llm, captured = _ollama(monkeypatch)
    img = b"\xff\xd8 fake jpeg"

    resp = llm.describe_image(img)

    assert resp.ok and resp.text.startswith("a wooden table")
    body = captured["body"]
    assert body["model"] == "llava"
    assert body["prompt"] == DEFAULT_IMAGE_PROMPT
    assert base64.b64decode(body["images"][0]) == img
    assert body["options"]["num_predict"] == 250


def test_describe_image_custom_prompt(monkeypatch):
    llm, captured = _ollama(monkeypatch)

    llm.describe_image(b"x", prompt="What text is visible?", max_tokens=99)

    assert captured["body"]["prompt"] == "What text is visible?"
    assert captured["body"]["options"]["num_predict"] == 99


def test_describe_image_error_fail_open(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("ollama down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    llm = OllamaLLM(model="llava")

    resp = llm.describe_image(b"x")

    assert not resp.ok and "ollama down" in (resp.error or "")


def test_base_provider_default_reports_unsupported():
    class _TextOnly(LLMProvider):
        @property
        def name(self) -> str:
            return "textonly"

        def generate(self, prompt, max_tokens=200, temperature=0.0) -> LLMResponse:
            return LLMResponse(text="ok")

    resp = _TextOnly().describe_image(b"x")

    assert not resp.ok
    assert "does not support image description" in (resp.error or "")
