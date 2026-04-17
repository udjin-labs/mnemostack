"""Ollama embedding provider (local, no API key)."""
from __future__ import annotations

import json
import urllib.request

from .base import EmbeddingProvider


class OllamaProvider(EmbeddingProvider):
    """Embedding provider backed by a local or remote Ollama server.

    Default model is `nomic-embed-text` (768-dim).
    Override `host` to point to a different Ollama instance.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    MODEL_DIMS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
        timeout: int = 30,
        dimension: int | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._dim = dimension or self.MODEL_DIMS.get(model, 768)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, text: str) -> list[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            return data.get("embedding", [])
        except Exception:  # noqa: BLE001
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
