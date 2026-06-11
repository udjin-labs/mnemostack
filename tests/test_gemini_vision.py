"""Tests for GeminiLLM.describe_image (multimodal ingest support)."""

from __future__ import annotations

import base64
import io
import json

from mnemostack.llm.gemini import DEFAULT_IMAGE_PROMPT, GeminiLLM


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _gemini(monkeypatch, reply_text="a red bicycle leaning on a wall"):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["headers"] = dict(req.headers)
        return _FakeResponse(
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": reply_text}]}}]}
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return GeminiLLM(api_key="test-key"), captured


def test_describe_image_sends_inline_data(monkeypatch):
    llm, captured = _gemini(monkeypatch)
    img = b"\x89PNG fake bytes"

    resp = llm.describe_image(img, mime_type="image/png")

    assert resp.ok and resp.text == "a red bicycle leaning on a wall"
    parts = captured["body"]["contents"][0]["parts"]
    assert parts[0]["text"] == DEFAULT_IMAGE_PROMPT
    inline = parts[1]["inline_data"]
    assert inline["mime_type"] == "image/png"
    assert base64.b64decode(inline["data"]) == img


def test_describe_image_custom_prompt_and_budget(monkeypatch):
    llm, captured = _gemini(monkeypatch)

    llm.describe_image(b"x", prompt="List the visible text only.", max_tokens=99)

    assert captured["body"]["contents"][0]["parts"][0]["text"] == "List the visible text only."
    assert captured["body"]["generationConfig"]["maxOutputTokens"] == 99


def test_generate_still_text_only(monkeypatch):
    llm, captured = _gemini(monkeypatch, reply_text="pong")

    resp = llm.generate("ping")

    assert resp.text == "pong"
    parts = captured["body"]["contents"][0]["parts"]
    assert parts == [{"text": "ping"}]


def test_describe_image_error_is_fail_open(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    llm = GeminiLLM(api_key="test-key", max_retries=1)

    resp = llm.describe_image(b"x")

    assert not resp.ok and "network down" in (resp.error or "")


def test_extract_text_skips_thought_parts(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse(
            json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "let me reason about this...", "thought": True},
                                    {"text": "21 May "},
                                    {"text": "2023"},
                                ]
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = GeminiLLM(api_key="test-key")

    resp = llm.generate("when?")

    assert resp.ok and resp.text == "21 May 2023"
