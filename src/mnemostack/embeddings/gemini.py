"""Gemini embedding provider."""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.request
from urllib.error import HTTPError

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

def _backoff(attempt: int, base: float = 1.0) -> float:
    """Exponential backoff with jitter so concurrent retries don't synchronize."""
    return base * (2**attempt) * (0.5 + random.random())



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
        vec, _provider_down = self._embed_one(text)
        return vec

    def _embed_one(self, text: str) -> tuple[list[float], bool]:
        """Embed one text; returns (vector, provider_down).

        `provider_down` is True only when the failure looks provider-wide
        (429/5xx with retries exhausted, or transport errors) — a
        content-specific rejection (other 4xx, e.g. an oversized chunk)
        returns ([], False) so callers can keep going with other texts.
        """
        url = f"{self.BASE_URL}/{self.model}:embedContent"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key or "",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data.get("embedding", {}).get("values", []), False
            except HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    logger.warning(
                        "gemini embed retry (HTTP %d), attempt %d/%d",
                        e.code,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(_backoff(attempt))
                    continue
                logger.error("gemini embed failed: HTTP %d %s", e.code, e.reason)
                return [], e.code in (429, 500, 502, 503, 504)
            except Exception as e:  # noqa: BLE001
                if attempt < self.max_retries - 1:
                    logger.warning("gemini embed retry: %s", e)
                    time.sleep(_backoff(0))
                    continue
                logger.error("gemini embed exhausted retries: %s", e)
                return [], True
        return [], True

    def _sequential_fallback(self, texts: list[str]) -> list[list[float]]:
        """Per-item fallback with a storm guard.

        When the batch endpoint dies while the API itself is down, falling
        back to N sequential calls multiplies the retry traffic by N against
        an already-failing service. As soon as any item fails in a way that
        looks provider-wide (429/5xx exhausted, transport error), the rest
        of the batch is reported failed instead of hammering. A
        content-specific rejection (e.g. one oversized chunk) only fails
        that item — the remaining texts are still tried.
        """
        out: list[list[float]] = []
        for i, text in enumerate(texts):
            vec, provider_down = self._embed_one(text)
            out.append(vec)
            if not vec and provider_down:
                logger.error(
                    "gemini sequential fallback hit a provider-wide failure — "
                    "skipping %d remaining items",
                    len(texts) - i - 1,
                )
                out.extend([] for _ in texts[i + 1 :])
                break
        return out

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding via :batchEmbedContents endpoint.

        Much faster than sequential embed() calls for large corpora.
        Falls back to sequential if batch fails or items exceed a safe size.
        """
        if not texts:
            return []

        # Gemini batchEmbedContents has a limit on requests per call
        # (100 per call is safe as of 2026-04). Chunk larger batches.
        BATCH_SIZE = 100
        if len(texts) > BATCH_SIZE:
            results: list[list[float]] = []
            for i in range(0, len(texts), BATCH_SIZE):
                results.extend(self.embed_batch(texts[i : i + BATCH_SIZE]))
            return results

        url = f"{self.BASE_URL}/{self.model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in texts
            ]
        }
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key or "",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    # Fallback to sequential if batch mis-aligned
                    return self._sequential_fallback(texts)
                return [e.get("values", []) for e in embeddings]
            except HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    logger.warning(
                        "gemini batch retry (HTTP %d), attempt %d/%d",
                        e.code,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(_backoff(attempt))
                    continue
                logger.warning(
                    "gemini batch endpoint failed (HTTP %d), falling back to sequential",
                    e.code,
                )
                return self._sequential_fallback(texts)
            except Exception as e:  # noqa: BLE001
                if attempt < self.max_retries - 1:
                    logger.warning("gemini batch retry: %s", e)
                    time.sleep(_backoff(0))
                    continue
                logger.warning("gemini batch failed, falling back to sequential: %s", e)
                return self._sequential_fallback(texts)
        return self._sequential_fallback(texts)
