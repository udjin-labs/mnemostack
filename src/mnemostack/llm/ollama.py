"""Ollama LLM provider (local)."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any

from .base import DEFAULT_IMAGE_PROMPT, LLMProvider, LLMResponse


class OllamaLLM(LLMProvider):
    """LLM backed by local or remote Ollama server.

    Default model is `llama3.2:3b` — small, fast, good enough for answer
    generation on modest hardware.

    Args:
        think: controls reasoning ("thinking") on models that support it.
            The default `False` disables it: reasoning models otherwise burn
            the whole `num_predict` budget on thoughts and return an empty
            response, which silently degrades reranking, query expansion and
            extraction. `False` is safe on every model (servers that don't
            know the field ignore it; non-thinking models accept it) — only
            `think=True` errors on models without thinking support. Pass
            `None` to omit the field and keep the model's own default.
        options: extra Ollama generation options merged into the request's
            `options` object (e.g. `{"num_ctx": 8192, "top_p": 0.9}`). Keys
            given here override the per-call `temperature`/`num_predict`.
    """

    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        timeout: int = 60,
        think: bool | None = False,
        options: dict[str, Any] | None = None,
    ):
        self.model = model
        # Same resolution chain as the ollama EMBEDDING provider, documented
        # there: an explicit ``host`` argument -> the native ``OLLAMA_HOST``
        # env var -> localhost. The two clients used to disagree — the
        # embedding provider honored ``OLLAMA_HOST`` while this one pinned
        # localhost — so in a container with ``OLLAMA_HOST`` set, embeddings
        # reached the GPU box and every LLM call died on localhost (#180).
        resolved = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        # OLLAMA_HOST is commonly bare "host:port" — the embedding provider
        # normalizes that, and "same chain" must include the normalization,
        # or a scheme-less value builds a malformed urlopen URL here.
        if "://" not in resolved:
            resolved = f"http://{resolved}"
        self.host = resolved.rstrip("/")
        self.timeout = timeout
        self.think = think
        self.options = dict(options or {})

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **self.options,
            },
        }
        return self._post_generate(payload)

    def describe_image(
        self,
        image: bytes,
        mime_type: str = "image/jpeg",
        prompt: str = DEFAULT_IMAGE_PROMPT,
        max_tokens: int = 250,
    ) -> LLMResponse:
        """Describe an image for indexing via an Ollama vision model.

        Requires a vision-capable model (llava, llama3.2-vision, qwen2.5-vl
        and similar). Ollama infers the image format from the bytes, so
        `mime_type` is accepted only for interface parity. Text-only models
        surface the server's error via `.error` — never raised.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(image).decode()],
            "stream": False,
            "options": {"num_predict": max_tokens, **self.options},
        }
        return self._post_generate(payload)

    def _post_generate(self, payload: dict) -> LLMResponse:
        if self.think is not None:
            # Top-level field, not a generation option. Must be a boolean:
            # Ollama rejects the strings "true"/"false".
            payload["think"] = self.think
        url = f"{self.host}/api/generate"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            return LLMResponse(
                text=(data.get("response") or "").strip(),
                tokens_used=data.get("eval_count"),
            )
        except Exception as e:  # noqa: BLE001
            return LLMResponse(text="", error=str(e))
