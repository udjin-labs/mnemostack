"""Ollama LLM provider (local)."""

from __future__ import annotations

import base64
import json
import urllib.request

from .base import DEFAULT_IMAGE_PROMPT, LLMProvider, LLMResponse


class OllamaLLM(LLMProvider):
    """LLM backed by local or remote Ollama server.

    Default model is `llama3.2:3b` — small, fast, good enough for answer
    generation on modest hardware.
    """

    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

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
            "options": {"num_predict": max_tokens},
        }
        return self._post_generate(payload)

    def _post_generate(self, payload: dict) -> LLMResponse:
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
