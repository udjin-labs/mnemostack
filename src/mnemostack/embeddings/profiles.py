"""Declarative embedding profiles — role-aware input transforms and space fingerprints.

A profile describes the input conventions of an embedding model family:
what to prepend to a *query* and what (if anything) to prepend to a
*document* before inference. Transforms are declarative specs, not
callables, so they can be fingerprinted stably; they are applied exactly
once, inside the provider role methods (`embed_query` / `embed_document`),
and never touch stored text, chunk ids, lexical search input or citations.

Two fingerprints identify the embedding space:

- ``document_space_fingerprint`` — provider, model, profile identity, the
  document transform and the vector dimension. Points embedded under
  different document fingerprints are incompatible; mixing them in one
  collection is forbidden.
- ``query_profile_fingerprint`` — the same inputs plus the query transform.
  A change invalidates cached query vectors but never requires reindexing
  documents.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

# A transform spec is a flat mapping with a "kind" key. Specs are data (not
# callables) so fingerprints can hash them canonically.
TransformSpec = Mapping[str, str]

IDENTITY_TRANSFORM: TransformSpec = {"kind": "identity"}

_TRANSFORM_KINDS = ("identity", "prefix", "instruct")


def apply_transform(spec: TransformSpec, text: str) -> str:
    """Apply a declarative transform spec to inference input.

    Callers must pass raw text and apply a spec at most once — there is no
    content-based detection of an already-applied transform (a real user
    query may legitimately start with ``query: ``).
    """
    kind = spec.get("kind", "identity")
    if kind == "identity":
        return text
    if kind == "prefix":
        return f"{spec['prefix']}{text}"
    if kind == "instruct":
        # Instruction-on-the-query retrieval convention. The separator is
        # part of the spec because families disagree on it: Qwen3-Embedding
        # uses "\nQuery:" (no space), multilingual-E5-instruct "\nQuery: ".
        separator = spec.get("separator", "\nQuery:")
        return f"Instruct: {spec['instruction']}{separator}{text}"
    raise ValueError(f"unknown transform kind: {kind!r} (known: {', '.join(_TRANSFORM_KINDS)})")


def _validate_transform(spec: TransformSpec) -> dict[str, str]:
    kind = spec.get("kind")
    if kind not in _TRANSFORM_KINDS:
        raise ValueError(
            f"unknown transform kind: {kind!r} (known: {', '.join(_TRANSFORM_KINDS)})"
        )
    required = {"identity": (), "prefix": ("prefix",), "instruct": ("instruction",)}[kind]
    optional = {"identity": set(), "prefix": set(), "instruct": {"separator"}}[kind]
    unknown = set(spec) - {"kind"} - set(required) - optional
    if unknown:
        # A typoed field would otherwise silently participate in fingerprints.
        raise ValueError(
            f"transform kind {kind!r} does not accept field(s): {sorted(unknown)}"
        )
    for key in required:
        if not isinstance(spec.get(key), str):
            raise ValueError(f"transform kind {kind!r} requires a string {key!r} field")
    for key in optional:
        if key in spec and not isinstance(spec[key], str):
            raise ValueError(f"transform kind {kind!r} requires a string {key!r} field")
    canonical = {k: str(v) for k, v in sorted(spec.items())}
    if kind == "instruct":
        # Materialize the default so an implicit and an explicit default
        # separator canonicalize (and fingerprint) identically.
        canonical.setdefault("separator", "\nQuery:")
    return canonical


# eq=False on purpose: frozen+eq would auto-generate a field-based __hash__
# that raises at runtime (the transform specs are dicts). Identity semantics
# are what the code actually uses.
@dataclass(frozen=True, eq=False)
class EmbeddingProfile:
    """Input semantics and stable metadata for an embedding model family.

    ``model_patterns`` are matched against the normalized (lowercased) model
    name; a pattern without wildcards is an exact tag and wins over family
    wildcards. ``version`` participates in fingerprints — bump it whenever a
    transform or template changes meaning.
    """

    name: str
    version: int
    model_patterns: tuple[str, ...]
    known_dimensions: Mapping[str, int] = field(default_factory=dict)
    query_transform: TransformSpec = field(default_factory=lambda: dict(IDENTITY_TRANSFORM))
    document_transform: TransformSpec = field(default_factory=lambda: dict(IDENTITY_TRANSFORM))

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_transform", _validate_transform(self.query_transform))
        object.__setattr__(
            self, "document_transform", _validate_transform(self.document_transform)
        )
        object.__setattr__(
            self, "known_dimensions", {k.lower(): int(v) for k, v in self.known_dimensions.items()}
        )

    def apply_query(self, text: str) -> str:
        return apply_transform(self.query_transform, text)

    def apply_document(self, text: str) -> str:
        return apply_transform(self.document_transform, text)


IDENTITY_PROFILE = EmbeddingProfile(name="identity", version=1, model_patterns=())

# provider key -> registered profiles, later registrations win ties so an
# application can override a built-in without patching mnemostack.
_PROFILES: dict[str, list[EmbeddingProfile]] = {}


def register_embedding_profile(provider: str, profile: EmbeddingProfile) -> None:
    """Register a profile for a provider (e.g. ``"ollama"``, ``"huggingface"``)."""
    _PROFILES.setdefault(provider.lower(), []).append(profile)


def resolve_profile(provider: str, model: str) -> EmbeddingProfile:
    """Resolve the profile for a provider/model pair.

    Exact tags win over family wildcard patterns; among equal matches the
    most recently registered profile wins. Unknown models get the identity
    profile (no transforms).
    """
    norm_model = model.lower()
    best: EmbeddingProfile | None = None
    best_rank = 2  # 0 = exact tag, 1 = wildcard family, 2 = no match
    for profile in _PROFILES.get(provider.lower(), []):
        for pattern in profile.model_patterns:
            if any(ch in pattern for ch in "*?["):
                if fnmatch.fnmatchcase(norm_model, pattern):
                    rank = 1
                else:
                    continue
            elif norm_model == pattern:
                rank = 0
            else:
                continue
            # `<=` so later registrations override earlier ones at equal rank.
            if rank <= best_rank:
                best, best_rank = profile, rank
    return best if best is not None else IDENTITY_PROFILE


def known_dimension(provider: str, model: str) -> int | None:
    """Profile-declared native dimension for a model tag, if any."""
    return resolve_profile(provider, model).known_dimensions.get(model.lower())


# Reserved payload key carrying the document-space fingerprint on stored
# points (same pattern as provenance's `_id_scheme`): stamped at index time,
# protected from enrichers and frontmatter, checked before further indexing.
EMBEDDING_SPACE_KEY = "_embedding_space"

_FINGERPRINT_SCHEMA = "es1"


def _fingerprint(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{_FINGERPRINT_SCHEMA}:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


def _base_payload(
    provider: str,
    model: str,
    profile: EmbeddingProfile,
    dimension: int,
    extras: Mapping[str, str] | None,
) -> dict[str, object]:
    return {
        "provider": provider.lower(),
        "model": model.lower(),
        "profile": profile.name,
        "profile_version": profile.version,
        "document_transform": dict(profile.document_transform),
        "dimension": int(dimension),
        # Provider-configurable inference knobs that change the vector space
        # (e.g. HuggingFace pooling mode).
        "extras": dict(sorted((extras or {}).items())),
    }


def document_space_fingerprint(
    provider: str,
    model: str,
    profile: EmbeddingProfile,
    dimension: int,
    extras: Mapping[str, str] | None = None,
) -> str:
    """Identity of the *document* embedding space.

    A mismatch against an existing collection forbids further indexing
    without an explicit recreate/reindex.
    """
    return _fingerprint(_base_payload(provider, model, profile, dimension, extras))


def query_profile_fingerprint(
    provider: str,
    model: str,
    profile: EmbeddingProfile,
    dimension: int,
    extras: Mapping[str, str] | None = None,
) -> str:
    """Identity of the *query* embedding pipeline.

    A mismatch invalidates cached query vectors but never requires
    reindexing documents.
    """
    payload = _base_payload(provider, model, profile, dimension, extras)
    payload["query_transform"] = dict(profile.query_transform)
    return _fingerprint(payload)


def _register_builtins() -> None:
    qwen3 = EmbeddingProfile(
        name="qwen3-embedding",
        version=1,
        model_patterns=("qwen3-embedding", "qwen3-embedding:*", "qwen/qwen3-embedding-*"),
        known_dimensions={
            "qwen3-embedding": 4096,
            "qwen3-embedding:latest": 4096,
            "qwen3-embedding:8b": 4096,
            "qwen3-embedding:4b": 2560,
            "qwen3-embedding:0.6b": 1024,
            "qwen/qwen3-embedding-8b": 4096,
            "qwen/qwen3-embedding-4b": 2560,
            "qwen/qwen3-embedding-0.6b": 1024,
        },
        query_transform={
            "kind": "instruct",
            "instruction": "Given a search query, retrieve relevant passages that answer the query",
        },
    )
    e5 = EmbeddingProfile(
        name="e5",
        version=1,
        model_patterns=(
            "e5-*",
            "multilingual-e5-*",
            "*/e5-*",
            "*/multilingual-e5-*",
        ),
        known_dimensions={
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
            "intfloat/multilingual-e5-small": 384,
            "intfloat/multilingual-e5-base": 768,
            "intfloat/multilingual-e5-large": 1024,
        },
        # The English role tokens are the model-family requirement, including
        # for multilingual E5.
        query_transform={"kind": "prefix", "prefix": "query: "},
        document_transform={"kind": "prefix", "prefix": "passage: "},
    )
    # The instruct variant is NOT plain E5: it expects an instruction-formatted
    # query (with a space after "Query:") and UNPREFIXED documents. Exact tags
    # so it wins over the e5 family wildcards.
    e5_instruct = EmbeddingProfile(
        name="e5-instruct",
        version=1,
        model_patterns=(
            "intfloat/multilingual-e5-large-instruct",
            "multilingual-e5-large-instruct",
        ),
        known_dimensions={
            "intfloat/multilingual-e5-large-instruct": 1024,
            "multilingual-e5-large-instruct": 1024,
        },
        query_transform={
            "kind": "instruct",
            "instruction": "Given a search query, retrieve relevant passages that answer the query",
            "separator": "\nQuery: ",
        },
    )
    for provider in ("ollama", "huggingface"):
        register_embedding_profile(provider, qwen3)
        register_embedding_profile(provider, e5)
        register_embedding_profile(provider, e5_instruct)


_register_builtins()
