"""Embedding profiles: role-aware transforms, resolution and space fingerprints."""

from __future__ import annotations

import pytest

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.embeddings.profiles import (
    IDENTITY_PROFILE,
    EmbeddingProfile,
    apply_transform,
    document_space_fingerprint,
    known_dimension,
    query_profile_fingerprint,
    register_embedding_profile,
    resolve_profile,
)


class RecordingProvider(EmbeddingProvider):
    """Fake provider that records exactly what reaches the neutral primitives."""

    def __init__(self, name: str = "ollama:nomic-embed-text", dimension: int = 3):
        self._name = name
        self._dimension = dimension
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return [float(len(text))] * self._dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------- transforms


def test_identity_transform_returns_text_unchanged():
    assert apply_transform({"kind": "identity"}, "raw text") == "raw text"


def test_prefix_transform_prepends_verbatim():
    assert apply_transform({"kind": "prefix", "prefix": "query: "}, "q") == "query: q"


def test_instruct_transform_uses_qwen_convention():
    out = apply_transform({"kind": "instruct", "instruction": "Do retrieval"}, "find it")
    assert out == "Instruct: Do retrieval\nQuery:find it"


def test_unknown_transform_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown transform kind"):
        apply_transform({"kind": "suffix"}, "text")


def test_profile_validates_transform_specs_at_construction():
    with pytest.raises(ValueError, match="requires a string"):
        EmbeddingProfile(
            name="broken",
            version=1,
            model_patterns=("x",),
            query_transform={"kind": "prefix"},  # missing "prefix" field
        )
    with pytest.raises(ValueError, match="unknown transform kind"):
        EmbeddingProfile(
            name="broken",
            version=1,
            model_patterns=("x",),
            document_transform={"kind": "mystery"},
        )


def test_unknown_transform_fields_are_rejected():
    # A typoed field must fail loudly, not silently participate in fingerprints.
    with pytest.raises(ValueError, match="does not accept field"):
        EmbeddingProfile(
            name="broken",
            version=1,
            model_patterns=("x",),
            query_transform={"kind": "prefix", "prefix": "q: ", "prefx": "typo"},
        )


def test_transform_is_not_idempotent_by_design():
    # Exactly-once is an architectural guarantee, not content detection: a
    # second application MUST visibly double the prefix so tests catch it.
    spec = {"kind": "prefix", "prefix": "query: "}
    once = apply_transform(spec, "query: real user text")
    twice = apply_transform(spec, once)
    assert once == "query: query: real user text"
    assert twice == "query: query: query: real user text"


# ---------------------------------------------------------------- resolution


def test_builtin_qwen3_profile_resolves_for_ollama_tags():
    for model, dim in [
        ("qwen3-embedding:0.6b", 1024),
        ("qwen3-embedding:4b", 2560),
        ("qwen3-embedding:8b", 4096),
        ("qwen3-embedding:latest", 4096),
        ("qwen3-embedding", 4096),
    ]:
        profile = resolve_profile("ollama", model)
        assert profile.name == "qwen3-embedding", model
        assert known_dimension("ollama", model) == dim


def test_builtin_e5_profile_is_asymmetric():
    profile = resolve_profile("huggingface", "intfloat/multilingual-e5-large")
    assert profile.name == "e5"
    assert profile.apply_query("вопрос") == "query: вопрос"
    assert profile.apply_document("текст") == "passage: текст"


def test_multilingual_e5_instruct_gets_its_own_profile():
    # The instruct variant is NOT plain E5: instruction-formatted query
    # (space after "Query:"), unprefixed documents.
    profile = resolve_profile("huggingface", "intfloat/multilingual-e5-large-instruct")
    assert profile.name == "e5-instruct"
    q = profile.apply_query("вопрос")
    assert q.startswith("Instruct: ")
    assert q.endswith("\nQuery: вопрос")
    assert profile.apply_document("текст") == "текст"
    assert known_dimension("huggingface", "intfloat/multilingual-e5-large-instruct") == 1024


def test_e5_mistral_instruct_does_not_fall_through_to_plain_e5():
    # `*/e5-*` would otherwise hand this instruct model the plain
    # query:/passage: convention it was never trained with.
    profile = resolve_profile("huggingface", "intfloat/e5-mistral-7b-instruct")
    assert profile.name == "e5-instruct"
    assert profile.apply_document("текст") == "текст"
    assert profile.apply_query("q").startswith("Instruct: ")
    assert known_dimension("huggingface", "intfloat/e5-mistral-7b-instruct") == 4096


def test_instruct_separator_is_part_of_the_spec():
    qwen_style = apply_transform({"kind": "instruct", "instruction": "I"}, "q")
    e5_style = apply_transform(
        {"kind": "instruct", "instruction": "I", "separator": "\nQuery: "}, "q"
    )
    assert qwen_style == "Instruct: I\nQuery:q"
    assert e5_style == "Instruct: I\nQuery: q"


def test_implicit_and_explicit_default_separator_fingerprint_identically():
    implicit = EmbeddingProfile(
        name="p", version=1, model_patterns=(),
        query_transform={"kind": "instruct", "instruction": "I"},
    )
    explicit = EmbeddingProfile(
        name="p", version=1, model_patterns=(),
        query_transform={"kind": "instruct", "instruction": "I", "separator": "\nQuery:"},
    )
    spaced = EmbeddingProfile(
        name="p", version=1, model_patterns=(),
        query_transform={"kind": "instruct", "instruction": "I", "separator": "\nQuery: "},
    )
    qi = query_profile_fingerprint("h", "m", implicit, 768)
    qe = query_profile_fingerprint("h", "m", explicit, 768)
    qs = query_profile_fingerprint("h", "m", spaced, 768)
    assert qi == qe
    assert qs != qi


def test_unknown_model_resolves_to_identity_profile():
    profile = resolve_profile("ollama", "nomic-embed-text")
    assert profile is IDENTITY_PROFILE
    assert profile.apply_query("q") == "q"
    assert profile.apply_document("d") == "d"


def test_unknown_provider_resolves_to_identity_profile():
    assert resolve_profile("acme", "qwen3-embedding:8b") is IDENTITY_PROFILE


def test_resolution_is_case_insensitive_on_model():
    assert resolve_profile("ollama", "Qwen3-Embedding:8B").name == "qwen3-embedding"
    assert known_dimension("ollama", "Qwen3-Embedding:8B") == 4096


def test_exact_tag_wins_over_family_wildcard():
    special = EmbeddingProfile(
        name="qwen3-special",
        version=1,
        model_patterns=("qwen3-embedding:8b",),  # exact tag only
    )
    register_embedding_profile("testprov-exact", special)
    family = EmbeddingProfile(
        name="qwen3-family",
        version=1,
        model_patterns=("qwen3-embedding:*",),
    )
    register_embedding_profile("testprov-exact", family)
    # Wildcard registered later, but the exact tag still wins.
    assert resolve_profile("testprov-exact", "qwen3-embedding:8b").name == "qwen3-special"
    assert resolve_profile("testprov-exact", "qwen3-embedding:4b").name == "qwen3-family"


def test_later_registration_overrides_at_equal_rank():
    first = EmbeddingProfile(name="first", version=1, model_patterns=("model-x",))
    second = EmbeddingProfile(name="second", version=1, model_patterns=("model-x",))
    register_embedding_profile("testprov-override", first)
    register_embedding_profile("testprov-override", second)
    assert resolve_profile("testprov-override", "model-x").name == "second"


# ------------------------------------------------------------- role methods


def test_role_methods_are_bit_identical_for_symmetric_models():
    provider = RecordingProvider()  # identity profile
    assert provider.embed_query("some question") == provider.embed("some question")
    assert provider.embed_document("some chunk") == provider.embed("some chunk")
    assert provider.embed_queries(["a", "b"]) == provider.embed_batch(["a", "b"])
    assert provider.embed_documents(["a", "b"]) == provider.embed_batch(["a", "b"])
    # And the primitives received the raw text, unmodified.
    assert provider.seen == [
        "some question", "some question",
        "some chunk", "some chunk",
        "a", "b", "a", "b",
        "a", "b", "a", "b",
    ]


def test_query_transform_applies_exactly_once_at_role_boundary():
    provider = RecordingProvider(name="ollama:qwen3-embedding:8b")
    provider.embed_query("find the config")
    assert provider.seen == [
        "Instruct: Given a search query, retrieve relevant passages that answer the query"
        "\nQuery:find the config"
    ]
    provider.seen.clear()
    provider.embed_document("chunk body")
    assert provider.seen == ["chunk body"]  # qwen3 documents are untouched


def test_batch_role_methods_transform_each_item():
    provider = RecordingProvider(name="huggingface:intfloat/e5-base-v2")
    provider.embed_queries(["q1", "q2"])
    assert provider.seen == ["query: q1", "query: q2"]
    provider.seen.clear()
    provider.embed_documents(["d1", "d2"])
    assert provider.seen == ["passage: d1", "passage: d2"]


def test_profile_can_be_overridden_per_instance():
    provider = RecordingProvider()
    provider.profile = EmbeddingProfile(
        name="custom",
        version=1,
        model_patterns=(),
        query_transform={"kind": "prefix", "prefix": "Q>"},
    )
    provider.embed_query("hello")
    assert provider.seen == ["Q>hello"]


# ------------------------------------------------------------- fingerprints


def _qwen():
    return resolve_profile("ollama", "qwen3-embedding:8b")


def test_document_fingerprint_is_deterministic():
    a = document_space_fingerprint("ollama", "qwen3-embedding:8b", _qwen(), 4096)
    b = document_space_fingerprint("ollama", "qwen3-embedding:8b", _qwen(), 4096)
    assert a == b
    assert a.startswith("es1:")


def test_document_fingerprint_changes_with_model_and_dimension():
    base = document_space_fingerprint("ollama", "qwen3-embedding:8b", _qwen(), 4096)
    assert document_space_fingerprint("ollama", "qwen3-embedding:4b", _qwen(), 2560) != base
    assert document_space_fingerprint("ollama", "qwen3-embedding:8b", _qwen(), 2560) != base
    assert document_space_fingerprint("gemini", "qwen3-embedding:8b", _qwen(), 4096) != base


def test_document_fingerprint_is_content_based_not_profile_identity():
    # A vector is fully determined by (provider, model, transformed input,
    # knobs). qwen3 and identity both embed documents unchanged, so their
    # DOCUMENT spaces are the same — a profile rename/introduction must not
    # demand a reindex of byte-identical vectors.
    a = document_space_fingerprint("ollama", "m", _qwen(), 4096)
    b = document_space_fingerprint("ollama", "m", IDENTITY_PROFILE, 4096)
    assert a == b
    # ...while their QUERY pipelines differ (instruct vs none).
    qa = query_profile_fingerprint("ollama", "m", _qwen(), 4096)
    qb = query_profile_fingerprint("ollama", "m", IDENTITY_PROFILE, 4096)
    assert qa != qb


def test_document_fingerprint_ignores_query_transform():
    with_query = EmbeddingProfile(
        name="p", version=1, model_patterns=(),
        query_transform={"kind": "prefix", "prefix": "query: "},
    )
    without_query = EmbeddingProfile(name="p", version=1, model_patterns=())
    d1 = document_space_fingerprint("ollama", "m", with_query, 768)
    d2 = document_space_fingerprint("ollama", "m", without_query, 768)
    assert d1 == d2  # query-side change never forces a reindex
    q1 = query_profile_fingerprint("ollama", "m", with_query, 768)
    q2 = query_profile_fingerprint("ollama", "m", without_query, 768)
    assert q1 != q2  # ...but it does invalidate query caches


def test_document_transform_change_changes_document_fingerprint():
    plain = EmbeddingProfile(name="p", version=1, model_patterns=())
    prefixed = EmbeddingProfile(
        name="p", version=1, model_patterns=(),
        document_transform={"kind": "prefix", "prefix": "passage: "},
    )
    assert document_space_fingerprint("h", "m", plain, 768) != document_space_fingerprint(
        "h", "m", prefixed, 768
    )


def test_profile_name_and_version_never_change_fingerprints():
    # name/version are registry metadata; only transform CONTENT is hashed —
    # a version bump on an unchanged transform must not invalidate anything.
    v1 = EmbeddingProfile(name="p", version=1, model_patterns=())
    v2 = EmbeddingProfile(name="renamed", version=2, model_patterns=())
    assert document_space_fingerprint("h", "m", v1, 768) == document_space_fingerprint(
        "h", "m", v2, 768
    )
    assert query_profile_fingerprint("h", "m", v1, 768) == query_profile_fingerprint(
        "h", "m", v2, 768
    )


def test_fingerprint_extras_participate():
    p = EmbeddingProfile(name="p", version=1, model_patterns=())
    mean = document_space_fingerprint("huggingface", "m", p, 768, {"pooling": "mean"})
    cls = document_space_fingerprint("huggingface", "m", p, 768, {"pooling": "cls"})
    none = document_space_fingerprint("huggingface", "m", p, 768)
    assert len({mean, cls, none}) == 3


def test_provider_fingerprint_methods_use_profile_and_extras():
    class PoolingProvider(RecordingProvider):
        def _fingerprint_extras(self):
            return {"pooling": "cls"}

    plain = RecordingProvider()
    pooled = PoolingProvider()
    assert plain.document_space_fingerprint() != pooled.document_space_fingerprint()
    assert plain.query_profile_fingerprint() != pooled.query_profile_fingerprint()
