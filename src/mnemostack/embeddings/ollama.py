"""Ollama embedding provider (local, no API key)."""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .base import EmbeddingProvider
from .profiles import known_dimension


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
        # Profile tables know newer model families (e.g. qwen3-embedding);
        # the blind 768 fallback for a fully unknown model is a known trap
        # slated for replacement by capability discovery.
        self._dim = dimension or self._lookup_dimension(model) or 768

    @classmethod
    def _lookup_dimension(cls, model: str) -> int | None:
        """Known dimension for an Ollama tag, quantization-suffix aware.

        Quantized tags carry the size first (`qwen3-embedding:0.6b-q8_0`,
        `...:8b-q4_K_M`); the tables key on the bare size, so retry with the
        suffix stripped — a silent 768 fallback for a 1024/4096 model would
        create a collection its own vectors can never enter.
        """
        candidates = [model]
        name, _, tag = model.partition(":")
        if tag and "-" in tag:
            candidates.append(f"{name}:{tag.split('-', 1)[0]}")
        if tag:
            # Bare-name last resort: `mxbai-embed-large:latest` /
            # `:335m-v1-fp16` must find the family entry. Sized tags hit
            # their exact/stripped entries first, so this can't shadow them.
            candidates.append(name)
        for cand in candidates:
            dim = cls.MODEL_DIMS.get(cand) or known_dimension("ollama", cand)
            if dim:
                return dim
        return None

    def _fingerprint_extras(self) -> dict[str, str]:
        """Pin the space to the pulled weights, not the mutable tag.

        A repointed tag (`:latest`, re-pushed size tag) can serve different
        weights under the same model string; the digest from `/api/tags`
        disambiguates. Re-resolved on EVERY call on purpose — a long-lived
        process (watch daemon, server) must fingerprint the weights CURRENTLY
        pulled, not ones cached at construction; callers bound the frequency
        (per index batch / guard revalidation). REQUIRED, never optional —
        an optional digest would give one configuration two different
        fingerprints depending on reachability. Residual: a tag repointed
        between this lookup and the embedding call itself is a race no
        Ollama API can close atomically; the next batch/revalidation
        detects it.
        """
        url = f"{self.host}/api/tags"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        wanted = {self.model, f"{self.model}:latest"}
        for entry in data.get("models", []):
            if entry.get("name") in wanted and entry.get("digest"):
                return {"digest": str(entry["digest"])}
        raise RuntimeError(
            f"model {self.model!r} not found on the Ollama host — "
            "pull it before indexing/resolving its embedding space"
        )

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

    def embed_batch(self, texts: list[str], max_workers: int = 4) -> list[list[float]]:
        """Parallel embedding via thread pool.

        Ollama has no batch endpoint, but the local server handles concurrent
        requests well on multi-core hardware. 4 workers is a safe default.
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [self.embed(texts[0])]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self.embed, texts))
