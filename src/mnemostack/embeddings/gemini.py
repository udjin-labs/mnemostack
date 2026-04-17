"""Gemini embedding provider."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from urllib.error import HTTPError

from .base import EmbeddingProvider


class GeminiProvider(EmbeddingProvider):
    """Embedding provider backed by Google Generative Language API.

    Default model is `gemini-embedding-001` with 3072-dim output.
    Set `GEMINI_API_KEY` env var or pass `api_key` directly.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-embedding-001"
    MODEL_DIMS = {
        "gemini-embedding-001": 3072,
        "embedding-001": 768,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set (env var or explicit api_key param)")
        self.timeout = timeout
        self.max_retries = max_retries
        self._dim = self.MODEL_DIMS.get(model, 3072)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def embed(self, text: str) -> list[float]:
        url = f"{self.BASE_URL}/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data.get("embedding", {}).get("values", [])
            except HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return []
            except Exception:  # noqa: BLE001
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return []
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Sequential embedding — Gemini v1beta embedContent API is single-input only.

        For batch throughput, use :embedBatch endpoint (TODO: add when stable).
        """
        return [self.embed(t) for t in texts]
