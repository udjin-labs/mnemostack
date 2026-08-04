"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from . import profiles as _profiles
from .profiles import EmbeddingProfile


class EmbeddingProvider(ABC):
    """Base interface for embedding providers.

    Subclasses must implement `embed`, `embed_batch`, and the `dimension` / `name` properties.
    Providers should handle their own errors gracefully and return an empty list on failure.

    `embed`/`embed_batch` are the neutral primitives: they receive input
    verbatim. The role methods (`embed_query`/`embed_document` and their
    batch forms) apply the resolved profile transform exactly once and then
    delegate to the primitives — callers must always pass RAW text to role
    methods and must not pre-apply transforms themselves. For symmetric
    models (identity profile) the role methods are bit-identical to the
    primitives.
    """

    # Resolved lazily from the provider `name`; class-level default so
    # third-party subclasses that never call a base __init__ still work.
    _profile: EmbeddingProfile | None = None

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return embedding vector for a single text. Empty list on failure."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts. Empty list for failed items."""

    def _provider_model(self) -> tuple[str, str]:
        """Split `name` ('ollama:qwen3-embedding:8b') into (provider, model)."""
        provider, _, model = self.name.partition(":")
        return provider, model

    @property
    def profile(self) -> EmbeddingProfile:
        """The embedding profile resolved for this provider/model pair."""
        if self._profile is None:
            self._profile = _profiles.resolve_profile(*self._provider_model())
        return self._profile

    @profile.setter
    def profile(self, value: EmbeddingProfile) -> None:
        self._profile = value

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query (applies the profile query transform once).

        Dispatches through an overridden ``embed_queries`` so a provider
        implementing only the BATCH native role keeps it on singular paths
        too (mirror of the batch→singular dispatch below). No recursion is
        possible: each default delegates only to an OVERRIDDEN counterpart.
        """
        if type(self).embed_queries is not EmbeddingProvider.embed_queries:
            vecs = self.embed_queries([text])
            return vecs[0] if vecs else []
        return self.embed(self.profile.apply_query(text))

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed retrieval queries (applies the profile query transform once).

        Dispatches through an overridden ``embed_query`` so a provider that
        implements only the SINGULAR native role keeps it on batch paths too
        — otherwise expansion retries would silently fall back to neutral
        vectors. Providers wanting batched native calls override this too.
        """
        if type(self).embed_query is not EmbeddingProvider.embed_query:
            return [self.embed_query(t) for t in texts]
        return self.embed_batch([self.profile.apply_query(t) for t in texts])

    def embed_document(self, text: str) -> list[float]:
        """Embed a document/chunk for indexing (applies the document transform once).

        Dispatches through an overridden ``embed_documents`` so a provider
        implementing only the BATCH native role keeps it on singular paths
        (CLI/markdown indexing) too — otherwise those routes would produce
        neutral vectors under the same native-role fingerprint.
        """
        if type(self).embed_documents is not EmbeddingProvider.embed_documents:
            vecs = self.embed_documents([text])
            return vecs[0] if vecs else []
        return self.embed(self.profile.apply_document(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents/chunks for indexing (applies the document transform once).

        Dispatches through an overridden ``embed_document`` so a provider
        that implements only the SINGULAR native role keeps it on batch
        paths too — otherwise `Ingestor` would silently index neutral
        vectors while markdown indexing used the native ones.
        """
        if type(self).embed_document is not EmbeddingProvider.embed_document:
            return [self.embed_document(t) for t in texts]
        return self.embed_batch([self.profile.apply_document(t) for t in texts])

    def _legacy_space_compatible(self) -> bool:
        """Whether this provider's ACTIVE configuration reproduces the vectors
        an older (pre-fingerprint) mnemostack would have produced.

        The legacy-adoption path assumes unstamped points are byte-compatible
        with what the current configuration embeds. Pre-fingerprint points
        were always created through the NEUTRAL primitives, so a provider
        that overrides a document role method (backend-native document task
        types) does not reproduce them — the default answers False for such
        providers; they may override this to attest compatibility explicitly.
        Providers should also override when a config default changed the
        output for the same model (e.g. HuggingFace auto-selecting
        last-token pooling where the old default was mean).
        """
        cls = type(self)
        for name in ("embed_document", "embed_documents"):
            if getattr(cls, name, None) is not getattr(EmbeddingProvider, name):
                return False
        return True

    def _fingerprint_extras(self) -> Mapping[str, str]:
        """Provider-configurable knobs that change the vector space.

        Override when a constructor option alters the embedding output for
        the same model (e.g. a pooling mode) so it participates in the space
        fingerprints.
        """
        return {}

    def _role_override_marker(self, names: tuple[str, ...]) -> str | None:
        """Identity marker when any of *names* is overridden by a subclass.

        Native role semantics live server-side and cannot be hashed like a
        declarative transform — but the override identity at least separates
        the space from the inherited declarative default, so neutral and
        native-role workers cannot stamp identical fingerprints during a
        rolling upgrade.
        """
        cls = type(self)
        overridden = [
            n
            for n in names
            if getattr(cls, n, None) is not getattr(EmbeddingProvider, n)
        ]
        if not overridden:
            return None
        return f"{cls.__module__}.{cls.__qualname__}:{'+'.join(overridden)}"

    def document_space_fingerprint(self) -> str:
        """Stable identity of the document embedding space.

        Points embedded under different document fingerprints must never be
        mixed in one collection.
        """
        provider, model = self._provider_model()
        extras = dict(self._fingerprint_extras())
        marker = self._role_override_marker(("embed_document", "embed_documents"))
        if marker:
            extras["native_document_role"] = marker
        return _profiles.document_space_fingerprint(
            provider, model, self.profile, self.dimension, extras
        )

    def query_profile_fingerprint(self) -> str:
        """Stable identity of the query embedding pipeline (gates query caches)."""
        provider, model = self._provider_model()
        extras = dict(self._fingerprint_extras())
        doc_marker = self._role_override_marker(("embed_document", "embed_documents"))
        if doc_marker:
            extras["native_document_role"] = doc_marker
        query_marker = self._role_override_marker(("embed_query", "embed_queries"))
        if query_marker:
            extras["native_query_role"] = query_marker
        return _profiles.query_profile_fingerprint(
            provider, model, self.profile, self.dimension, extras
        )

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension returned by this provider/model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier like 'gemini:embedding-001' — used in logs and config."""

    def health_check(self) -> tuple[bool, str]:
        """Lightweight reachability check. Returns (is_healthy, message)."""
        try:
            v = self.embed("healthcheck")
            if v and len(v) == self.dimension:
                return True, f"ok, dim={len(v)}"
            return False, f"unexpected dim: got {len(v)}, expected {self.dimension}"
        except Exception as e:  # noqa: BLE001
            return False, f"error: {e}"
