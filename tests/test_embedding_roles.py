"""Role dispatch helpers, the document-space guard and fingerprint stamping."""

from __future__ import annotations

from types import SimpleNamespace

from mnemostack.embeddings.base import EmbeddingProvider
from mnemostack.embeddings.profiles import EMBEDDING_SPACE_KEY, EmbeddingProfile
from mnemostack.embeddings.roles import (
    check_document_space,
    document_space_fingerprint_via,
    embed_document_via,
    embed_documents_via,
    embed_queries_via,
    embed_query_via,
)
from mnemostack.ingest import IngestItem, Ingestor


class _AsymmetricProvider(EmbeddingProvider):
    """E5-profiled provider that records what reaches the neutral primitives."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return [1.0, 2.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "huggingface:intfloat/e5-base-v2"


class _DuckProvider:
    """Legacy duck-typed provider: only the neutral primitives, no base class."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, text):
        self.seen.append(text)
        return [0.5]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


# ------------------------------------------------------------------ helpers


def test_helpers_use_role_methods_when_available():
    p = _AsymmetricProvider()
    embed_query_via(p, "q")
    embed_document_via(p, "d")
    embed_queries_via(p, ["q2"])
    embed_documents_via(p, ["d2"])
    assert p.seen == ["query: q", "passage: d", "query: q2", "passage: d2"]


def test_helpers_fall_back_to_neutral_primitives_for_duck_providers():
    duck = _DuckProvider()
    assert embed_query_via(duck, "q") == [0.5]
    assert embed_document_via(duck, "d") == [0.5]
    embed_queries_via(duck, ["a"])
    embed_documents_via(duck, ["b"])
    # Raw text, no transforms — exactly the legacy behavior.
    assert duck.seen == ["q", "d", "a", "b"]


def test_fingerprint_via_returns_none_for_duck_providers():
    assert document_space_fingerprint_via(_DuckProvider()) is None
    assert document_space_fingerprint_via(_AsymmetricProvider()) is not None


# -------------------------------------------------------------------- guard


class _ScrollStore:
    def __init__(self, payloads: list[dict]):
        self._payloads = payloads
        self.scroll_batches = 0

    def scroll(self):
        for payload in self._payloads:
            self.scroll_batches += 1
            yield SimpleNamespace(id="x", payload=payload)


def test_guard_matches_when_fingerprint_agrees():
    p = _AsymmetricProvider()
    fp = p.document_space_fingerprint()
    status, expected, found = check_document_space(_ScrollStore([{EMBEDDING_SPACE_KEY: fp}]), p)
    assert status == "match"
    assert expected == found == fp


def test_guard_flags_mismatched_space():
    p = _AsymmetricProvider()
    store = _ScrollStore([{EMBEDDING_SPACE_KEY: "es1:deadbeef"}])
    status, expected, found = check_document_space(store, p)
    assert status == "mismatch"
    assert found == "es1:deadbeef"
    assert expected == p.document_space_fingerprint()


def test_guard_reports_legacy_and_empty_collections():
    p = _AsymmetricProvider()
    assert check_document_space(_ScrollStore([{"text": "old"}]), p)[0] == "legacy"
    assert check_document_space(_ScrollStore([]), p)[0] == "empty"


def test_guard_is_unknown_without_fingerprint_or_scroll():
    assert check_document_space(_ScrollStore([]), _DuckProvider())[0] == "unknown"
    no_scroll = SimpleNamespace()
    assert check_document_space(no_scroll, _AsymmetricProvider())[0] == "unknown"


def test_guard_samples_lazily():
    p = _AsymmetricProvider()
    fp = p.document_space_fingerprint()
    store = _ScrollStore([{EMBEDDING_SPACE_KEY: fp} for _ in range(1000)])
    check_document_space(store, p, sample_size=4)
    assert store.scroll_batches <= 5  # islice stops the scroll, no full scan


def test_guard_mixed_legacy_and_stamped_sample_is_match():
    p = _AsymmetricProvider()
    fp = p.document_space_fingerprint()
    store = _ScrollStore([{"text": "legacy"}, {EMBEDDING_SPACE_KEY: fp}])
    assert check_document_space(store, p)[0] == "match"


def test_cli_guard_refuses_legacy_collection_under_document_transform():
    # Pre-fingerprint points were embedded from raw text; an active profile
    # that transforms documents (E5's `passage: `) writes into a DIFFERENT
    # space, so adoption must be refused, not noted.
    from mnemostack.cli import _guard_document_space

    provider = _AsymmetricProvider()  # e5 profile: non-identity document transform
    legacy_store = _ScrollStore([{"text": "old point, no fingerprint"}])
    assert _guard_document_space(legacy_store, provider) == 1


def test_cli_guard_allows_legacy_collection_under_identity_document_transform():
    from mnemostack.cli import _guard_document_space

    class _QwenProvider(_AsymmetricProvider):
        @property
        def name(self) -> str:
            return "ollama:qwen3-embedding:8b"  # query instruct, identity documents

    legacy_store = _ScrollStore([{"text": "old point, no fingerprint"}])
    assert _guard_document_space(legacy_store, _QwenProvider()) is None


def test_cli_guard_aborts_on_fingerprint_mismatch():
    from mnemostack.cli import _guard_document_space

    provider = _AsymmetricProvider()
    store = _ScrollStore([{EMBEDDING_SPACE_KEY: "es1:deadbeef"}])
    assert _guard_document_space(store, provider) == 1


# ----------------------------------------------------------------- stamping


class _RecordingStore:
    def __init__(self):
        self.upserts: list[tuple[str, list[float], dict]] = []

    def upsert(self, id, vector, payload, **kw):
        self.upserts.append((id, vector, payload))

    def upsert_batch(self, points, **kw):
        for p in points:
            self.upsert(*p)
        return len(points)


def test_ingestor_stamps_document_space_fingerprint():
    emb = _AsymmetricProvider()
    store = _RecordingStore()
    ingestor = Ingestor(embedding=emb, vector_store=store)
    ingestor.ingest([IngestItem(text="hello", source="a.md")])
    assert len(store.upserts) == 1
    payload = store.upserts[0][2]
    assert payload[EMBEDDING_SPACE_KEY] == emb.document_space_fingerprint()
    # And the vector went through the DOCUMENT role.
    assert emb.seen == ["passage: hello"]


def test_ingestor_skips_stamp_for_duck_provider():
    store = _RecordingStore()
    ingestor = Ingestor(embedding=_DuckProvider(), vector_store=store)
    ingestor.ingest([IngestItem(text="hello", source="a.md")])
    assert EMBEDDING_SPACE_KEY not in store.upserts[0][2]


def test_enricher_cannot_forge_the_space_fingerprint():
    emb = _AsymmetricProvider()
    store = _RecordingStore()
    ingestor = Ingestor(
        embedding=emb,
        vector_store=store,
        enrich=lambda item: {EMBEDDING_SPACE_KEY: "es1:forged", "topic": "ok"},
    )
    ingestor.ingest([IngestItem(text="hello", source="a.md")])
    payload = store.upserts[0][2]
    assert payload[EMBEDDING_SPACE_KEY] == emb.document_space_fingerprint()
    assert payload["topic"] == "ok"


def test_markdown_sync_never_overwrites_a_conflicting_stamp():
    # A point stamped with a DIFFERENT space (possible beyond the guard's
    # sample window) must stay visible — refresh skips it, never launders it.
    from mnemostack.markdown.sync import upsert_markdown_chunks

    emb = _AsymmetricProvider()

    class _SyncStore(_RecordingStore):
        def __init__(self):
            super().__init__()
            self.set_payloads: list[tuple[str, dict]] = []

        def set_payload(self, cid, payload, **kw):
            self.set_payloads.append((cid, payload))

        def delete_payload_keys(self, cid, keys, **kw):
            pass

    store = _SyncStore()
    chunks = [("old-1", "known text", {"text": "known text", "source": "a.md"})]
    res = upsert_markdown_chunks(
        store,
        emb,
        chunks,
        existing_payloads={"old-1": {"text": "known text", EMBEDDING_SPACE_KEY: "es1:other"}},
    )
    assert res.refreshed == 0
    assert res.space_conflicts == 1
    assert store.set_payloads == []


def test_expansion_retry_splits_roles_between_queries_and_hypothetical():
    # [query, rephrase, rephrase] must embed as QUERIES, the hypothetical
    # answer as a DOCUMENT — under an asymmetric profile a swap would put it
    # on the wrong side of the space.
    from unittest.mock import MagicMock

    from mnemostack.llm.base import LLMResponse
    from mnemostack.recall import AnswerGenerator

    llm = MagicMock()
    llm.generate.side_effect = [
        LLMResponse(text="Not in memory."),
        LLMResponse(text="confident retry\nCONFIDENCE: 0.9"),
    ]
    expansion_llm = MagicMock()
    expansion_llm.generate.return_value = LLMResponse(
        text="rephrase one\nrephrase two\nhypothetical answer"
    )
    emb = _AsymmetricProvider()

    class _Recaller:
        embedding = emb

        def search_many(self, vectors, limit, filters=None, **_):
            return []

    gen = AnswerGenerator(
        llm=llm,
        recaller=_Recaller(),
        retry_with_expansion=True,
        expansion_llm=expansion_llm,
        specificity_resolver=False,
        inference_retry=False,
        category_aware_prompts=False,
    )
    memory = type(
        "R", (), {"id": "m", "text": "some memory", "score": 0.5, "payload": {}, "sources": ["vector"]}
    )()
    gen.generate("a question", [memory])

    assert emb.seen == [
        "query: a question",
        "query: rephrase one",
        "query: rephrase two",
        "passage: hypothetical answer",
    ]


def test_markdown_sync_stamps_new_and_refreshed_chunks():
    from mnemostack.markdown.sync import upsert_markdown_chunks

    emb = _AsymmetricProvider()
    fp = emb.document_space_fingerprint()

    class _SyncStore(_RecordingStore):
        def __init__(self):
            super().__init__()
            self.set_payloads: list[tuple[str, dict]] = []

        def set_payload(self, cid, payload, **kw):
            self.set_payloads.append((cid, payload))

        def delete_payload_keys(self, cid, keys, **kw):
            pass

    store = _SyncStore()
    chunks = [
        ("new-1", "fresh text", {"text": "fresh text", "source": "a.md"}),
        ("old-1", "known text", {"text": "known text", "source": "a.md"}),
    ]
    res = upsert_markdown_chunks(
        store, emb, chunks, existing_payloads={"old-1": {"text": "known text"}}
    )
    assert res.inserted == 1 and res.refreshed == 1
    assert store.upserts[0][2][EMBEDDING_SPACE_KEY] == fp
    # Refresh does not re-embed but adopts the fingerprint (sanctioned
    # legacy-migration path — the CLI guard rejects true mismatches earlier).
    assert store.set_payloads[0][1][EMBEDDING_SPACE_KEY] == fp
    assert emb.seen == ["passage: fresh text"]


# ---------------------------------------------------- provider integrations


def test_ollama_provider_uses_profile_known_dimension():
    from mnemostack.embeddings.ollama import OllamaProvider

    assert OllamaProvider(model="qwen3-embedding:8b").dimension == 4096
    assert OllamaProvider(model="qwen3-embedding:0.6b").dimension == 1024
    # Explicit dimension always wins; legacy table and fallback unchanged.
    assert OllamaProvider(model="qwen3-embedding:8b", dimension=7).dimension == 7
    assert OllamaProvider(model="nomic-embed-text").dimension == 768
    assert OllamaProvider(model="completely-unknown").dimension == 768


def test_shared_query_embedding_forwards_profile_and_fingerprints():
    from mnemostack.recall.retrievers import _SharedQueryEmbedding

    inner = _AsymmetricProvider()
    shim = _SharedQueryEmbedding(inner)
    assert shim.profile is inner.profile
    assert shim.document_space_fingerprint() == inner.document_space_fingerprint()
    assert shim.query_profile_fingerprint() == inner.query_profile_fingerprint()
    # Role method inherited from the base class lands in the memoized embed
    # with the transform already applied — exactly once.
    vec = shim.embed_query("надо найти")
    assert vec == [1.0, 2.0]
    assert inner.seen == ["query: надо найти"]
    shim.embed_query("надо найти")
    assert inner.seen == ["query: надо найти"]  # memo hit, no second call


def test_shared_query_embedding_with_instance_profile_override():
    from mnemostack.recall.retrievers import _SharedQueryEmbedding

    inner = _AsymmetricProvider()
    inner.profile = EmbeddingProfile(
        name="custom", version=1, model_patterns=(),
        query_transform={"kind": "prefix", "prefix": "X>"},
    )
    shim = _SharedQueryEmbedding(inner)
    shim.embed_query("q")
    assert inner.seen == ["X>q"]


def test_hyde_embeds_hypothetical_as_document():
    from mnemostack.recall.retrievers import HyDERetriever

    class _Vec:
        def search(self, vec, limit=20, filters=None, hide_invalidated=False):
            return []

    class _LLM:
        def generate(self, prompt, max_tokens=0, temperature=0.0):
            return SimpleNamespace(ok=True, text="a plausible stored memory")

    emb = _AsymmetricProvider()
    retriever = HyDERetriever(embedding=emb, vector_store=_Vec(), llm=_LLM())
    retriever.search("what happened?")
    assert emb.seen == ["passage: a plausible stored memory"]
