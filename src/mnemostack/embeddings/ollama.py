"""Ollama embedding provider (local, no API key)."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any

from .base import EmbeddingCapabilities, EmbeddingProvider, ProviderProbeError
from .profiles import known_dimension

logger = logging.getLogger(__name__)


def _redacted(host: str) -> str:
    """Host for logs with userinfo credentials stripped."""
    if "@" not in host:
        return host
    scheme, sep, rest = host.partition("://")
    if sep and "@" in rest:
        return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
    return f"***@{host.rsplit('@', 1)[1]}"


def _api_level_error(exc: urllib.error.HTTPError) -> bool:
    """Whether a 404/405 came from the API rather than the router.

    Ollama answers an existing route with a JSON error object (e.g. a
    not-pulled model on /api/embed is a 404 WITH a JSON body) — that must
    surface as the model error it is. Only the router's plain-text
    "404 page not found" proves the endpoint itself is absent (old server)
    and justifies the legacy fallback.
    """
    try:
        body = exc.read()
    except Exception:  # noqa: BLE001 — unreadable body: treat as router-level
        return False
    try:
        return isinstance(json.loads(body.decode("utf-8", "replace")), dict)
    except Exception:  # noqa: BLE001 — non-JSON body: router-level
        return False


class OllamaProvider(EmbeddingProvider):
    """Embedding provider backed by a local or remote Ollama server.

    Default model is `nomic-embed-text` (768-dim).

    Host resolution (weakest to strongest): built-in localhost default →
    the native ``OLLAMA_HOST`` env var → an explicit ``host`` argument
    (which the CLI/config plumbing supplies from ``--ollama-host`` /
    ``embedding.ollama_host`` / ``MNEMOSTACK_OLLAMA_HOST``).

    Endpoint: uses the current batch endpoint ``POST /api/embed`` (string or
    array input). The legacy per-item ``/api/embeddings`` is kept ONLY as a
    fallback for old servers, selected once per instance on PROVEN
    incompatibility (HTTP 404/405) — never on timeouts, 5xx or model errors,
    which must surface instead of being masked by a silent downgrade. The
    fallback is logged loudly and visible via ``endpoint``/doctor.

    Dimension: explicit argument → built-in/profile tables (quantized-tag
    aware) → a one-shot PROBE of the live model. There is no arbitrary
    fallback dimension: an unknown model with an unreachable host raises
    :class:`ProviderProbeError` before any collection can be created with a
    wrong size.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    #: Cold loads of larger local models legitimately exceed the old 30s.
    DEFAULT_TIMEOUT = 180
    MODEL_DIMS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        dimension: int | None = None,
    ):
        self.model = model
        resolved = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if "://" not in resolved:
            # OLLAMA_HOST is commonly "host:port" without a scheme.
            resolved = f"http://{resolved}"
        self.host = resolved.rstrip("/")
        self.timeout = timeout
        # None until detection: "api/embed" | "api/embeddings". The lock
        # serializes first-use probing/detection so concurrent recall arms
        # can't fire duplicate probes or duplicate fallback warnings — the
        # "once per instance" claim holds under threads.
        self._endpoint: str | None = None
        # RLock: the dimension probe holds it while calling _embed_api,
        # whose legacy-fallback branch re-acquires it — a plain Lock
        # would deadlock unknown-model probes against old servers.
        self._detect_lock = threading.RLock()
        self._dim: int | None = dimension or self._lookup_dimension(model)

    @classmethod
    def _lookup_dimension(cls, model: str) -> int | None:
        """Known dimension for an Ollama tag, quantization-suffix aware.

        Quantized tags carry the size first (`qwen3-embedding:0.6b-q8_0`,
        `...:8b-q4_K_M`); the tables key on the bare size, so retry with the
        suffix stripped, then the bare family name (`mxbai-embed-large:latest`).
        """
        candidates = [model]
        name, _, tag = model.partition(":")
        if tag and "-" in tag:
            candidates.append(f"{name}:{tag.split('-', 1)[0]}")
        if tag:
            candidates.append(name)
        for cand in candidates:
            dim = cls.MODEL_DIMS.get(cand) or known_dimension("ollama", cand)
            if dim:
                return dim
        return None

    @property
    def dimension(self) -> int:
        """Vector dimension — discovered from the live model when unknown.

        Raises :class:`ProviderProbeError` when the model is unknown to the
        tables AND cannot be probed: a loud startup error beats creating a
        collection that can never accept the returned vectors (the old
        behavior was a blind 768 fallback).
        """
        if self._dim is None:
            with self._detect_lock:
                if self._dim is None:
                    self._dim = self._probe_dimension()
        return self._dim

    def _probe_dimension(self) -> int:
        try:
            vectors = self._embed_api(["dimension probe"])
        except Exception as exc:  # noqa: BLE001 — typed, loud, actionable
            raise ProviderProbeError(
                f"cannot discover the vector dimension of {self.model!r}: {exc} — "
                "no fallback dimension is assumed; make the Ollama host reachable "
                "(or pass dimension= explicitly)"
            ) from exc
        vec = vectors[0] if vectors else []
        if not vec or not all(isinstance(x, (int, float)) for x in vec):
            raise ProviderProbeError(
                f"dimension probe of {self.model!r} returned an empty or "
                "non-numeric vector"
            )
        return len(vec)

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    @property
    def endpoint(self) -> str:
        """The selected API surface ('api/embed' until proven unsupported)."""
        return self._endpoint or "api/embed"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.host}/{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        if not isinstance(data, dict):
            raise ValueError(f"{path} returned non-object JSON")
        return data

    def _embed_legacy(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            data = self._post("api/embeddings", {"model": self.model, "prompt": text})
            vec = list(data.get("embedding") or [])
            if not vec:
                # Same all-or-nothing contract as the modern endpoint: a
                # "successful" response with an empty vector must fail the
                # whole batch, not produce a mixed partial result.
                raise ValueError(
                    "api/embeddings returned an empty vector — refusing partial results"
                )
            out.append(vec)
        return out

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Embed via the selected endpoint. Raises on failure.

        Response cardinality is validated: anything but exactly one
        non-empty vector per input is a PROVIDER failure, not partial
        success. The legacy fallback engages only on proven endpoint
        absence (404/405) — a timeout, 5xx or model error propagates, so a
        real problem is never masked as "old server".
        """
        if not texts:
            return []
        if self._endpoint == "api/embeddings":
            return self._embed_legacy(texts)
        try:
            data = self._post("api/embed", {"model": self.model, "input": texts})
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405) and not _api_level_error(exc):
                # Proven incompatibility: this server predates /api/embed.
                with self._detect_lock:
                    if self._endpoint != "api/embeddings":
                        logger.warning(
                            "Ollama host %s has no /api/embed (HTTP %d) — "
                            "falling back to the legacy per-item "
                            "/api/embeddings endpoint; upgrade Ollama to "
                            "batch natively",
                            _redacted(self.host),
                            exc.code,
                        )
                        self._endpoint = "api/embeddings"
                return self._embed_legacy(texts)
            raise
        self._endpoint = "api/embed"
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError(
                f"api/embed returned {0 if not isinstance(vectors, list) else len(vectors)} "
                f"vector(s) for {len(texts)} input(s) — refusing partial results"
            )
        out = [list(v or []) for v in vectors]
        if any(not v for v in out):
            raise ValueError("api/embed returned an empty vector — refusing partial results")
        return out

    def _fingerprint_extras(self) -> dict[str, str]:
        """Pin the space to the pulled weights, not the mutable tag.

        A repointed tag (`:latest`, re-pushed size tag) can serve different
        weights under the same model string; the digest from `/api/tags`
        disambiguates. Re-resolved on EVERY call on purpose — a long-lived
        process must fingerprint the weights CURRENTLY pulled; callers bound
        the frequency (per index batch / guard revalidation). REQUIRED,
        never optional — an optional digest would give one configuration two
        different fingerprints depending on reachability. Residual: a tag
        repointed between this lookup and the embedding call itself is a
        race no Ollama API can close atomically; the next
        batch/revalidation detects it.
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

    def probe_capabilities(self) -> EmbeddingCapabilities:
        """One live probe: validated dimension + endpoint/batch support."""
        try:
            vectors = self._embed_api(["capability probe"])
        except Exception as exc:  # noqa: BLE001 — typed, loud, actionable
            raise ProviderProbeError(f"embedding probe failed: {exc}") from exc
        vec = vectors[0] if vectors else []
        if not vec or not all(isinstance(x, (int, float)) for x in vec):
            raise ProviderProbeError("embedding probe returned an empty/non-numeric vector")
        if self._dim is None:
            self._dim = len(vec)
        return EmbeddingCapabilities(
            dimension=len(vec),
            batch=self.endpoint == "api/embed",
            endpoint=self.endpoint,
        )

    def embed(self, text: str) -> list[float]:
        try:
            vectors = self._embed_api([text])
            return vectors[0] if vectors else []
        except Exception as exc:  # noqa: BLE001 — public contract: [] on failure
            logger.debug("ollama embed failed: %s", exc)
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Native batch via /api/embed (single request per batch).

        On failure every item reports failed (``[]``) — cardinality
        mismatches are provider failures, never partial success. The
        legacy-server path degrades to per-item requests, observable via
        ``endpoint`` and the fallback warning.
        """
        if not texts:
            return []
        try:
            return self._embed_api(texts)
        except Exception as exc:  # noqa: BLE001 — public contract: [] per failed item
            logger.warning("ollama embed_batch failed (%s) — all items failed", exc)
            return [[] for _ in texts]
