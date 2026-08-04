"""Role dispatch helpers, the document-space guard and fingerprint stamping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_guard_mixed_legacy_and_stamped_sample_is_legacy():
    # Legacy dominates match: the unstamped point may be a raw-text vector,
    # so a document-transforming profile must still see "legacy" and refuse —
    # otherwise --refresh-payloads would launder that raw vector.
    p = _AsymmetricProvider()
    fp = p.document_space_fingerprint()
    store = _ScrollStore([{EMBEDDING_SPACE_KEY: fp}, {"text": "legacy"}])
    assert check_document_space(store, p)[0] == "legacy"


def test_cli_guard_refuses_legacy_collection_under_document_transform():
    # Pre-fingerprint points were embedded from raw text; an active profile
    # that transforms documents (E5's `passage: `) writes into a DIFFERENT
    # space, so adoption must be refused, not noted.
    from mnemostack.cli import _guard_document_space

    provider = _AsymmetricProvider()  # e5 profile: non-identity document transform
    legacy_store = _ScrollStore([{"text": "old point, no fingerprint"}])
    assert _guard_document_space(legacy_store, provider)[0] == 1


def test_cli_guard_allows_legacy_collection_under_identity_document_transform():
    from mnemostack.cli import _guard_document_space

    class _QwenProvider(_AsymmetricProvider):
        @property
        def name(self) -> str:
            return "ollama:qwen3-embedding:8b"  # query instruct, identity documents

    legacy_store = _ScrollStore([{"text": "old point, no fingerprint"}])
    rc, fp = _guard_document_space(legacy_store, _QwenProvider())
    assert rc is None
    assert fp is not None  # the validated fingerprint travels to the stamp


def test_cli_guard_aborts_on_fingerprint_mismatch():
    from mnemostack.cli import _guard_document_space

    provider = _AsymmetricProvider()
    store = _ScrollStore([{EMBEDDING_SPACE_KEY: "es1:deadbeef"}])
    assert _guard_document_space(store, provider)[0] == 1


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


def test_item_metadata_cannot_forge_the_space_fingerprint():
    # With a duck-typed provider there is no fingerprint to overwrite a
    # planted value — the reserved key must be dropped unconditionally,
    # same rule as tenant_id.
    store = _RecordingStore()
    ingestor = Ingestor(embedding=_DuckProvider(), vector_store=store)
    ingestor.ingest(
        [
            IngestItem(
                text="hello",
                source="a.md",
                metadata={EMBEDDING_SPACE_KEY: "es1:forged"},
            )
        ]
    )
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


def test_markdown_sync_sandwiches_fingerprint_around_writes():
    # A tag repointed after the invocation guard must be caught before ANY
    # write — the post-embed check sees a different resolution and aborts.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.markdown.sync import upsert_markdown_chunks

    class _Flipping(_AsymmetricProvider):
        def __init__(self):
            super().__init__()
            self.resolutions = 0

        def _fingerprint_extras(self):
            self.resolutions += 1
            return {"rev": str(self.resolutions)}

    store = _RecordingStore()
    with pytest.raises(EmbeddingSpaceError, match="changed mid-sync"):
        upsert_markdown_chunks(
            store,
            _Flipping(),
            [("c1", "text", {"text": "text", "source": "a.md"})],
            existing_payloads={},
        )
    assert store.upserts == []


def test_native_role_overrides_participate_in_fingerprints():
    # A provider switching from the inherited declarative document role to a
    # backend-native one changes the vectors — the fingerprint must change
    # with it, or rolling old/new workers would stamp alike.
    class _NativeDoc(_AsymmetricProvider):
        def embed_document(self, text):
            return [9.9, 9.9]

    class _NativeQuery(_AsymmetricProvider):
        def embed_query(self, text):
            return [7.0, 7.0]

    plain = _AsymmetricProvider()
    native_doc = _NativeDoc()
    native_q = _NativeQuery()
    assert native_doc.document_space_fingerprint() != plain.document_space_fingerprint()
    assert native_doc.query_profile_fingerprint() != plain.query_profile_fingerprint()
    # A query-only native role changes ONLY the query pipeline identity.
    assert native_q.document_space_fingerprint() == plain.document_space_fingerprint()
    assert native_q.query_profile_fingerprint() != plain.query_profile_fingerprint()


def test_markdown_sync_guards_each_invocation():
    # The watch loop calls this per file batch long after the CLI startup
    # guard ran — each invocation must recheck before embedding.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.markdown.sync import upsert_markdown_chunks

    class _Store(_ScrollStore, _RecordingStore):
        def __init__(self, payloads):
            _ScrollStore.__init__(self, payloads)
            _RecordingStore.__init__(self)

    store = _Store([{EMBEDDING_SPACE_KEY: "es1:other"}])
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        upsert_markdown_chunks(
            store,
            _AsymmetricProvider(),
            [("c1", "text", {"text": "text", "source": "a.md"})],
            existing_payloads={},
        )
    assert store.upserts == []


def test_ingestor_refuses_mismatched_collection():
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    class _Store(_ScrollStore, _RecordingStore):
        def __init__(self, payloads):
            _ScrollStore.__init__(self, payloads)
            _RecordingStore.__init__(self)

    store = _Store([{EMBEDDING_SPACE_KEY: "es1:other"}])
    ingestor = Ingestor(embedding=_AsymmetricProvider(), vector_store=store)
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        ingestor.ingest([IngestItem(text="hello", source="a.md")])
    assert store.upserts == []


def test_guard_refuses_legacy_when_provider_defaults_changed():
    # Pre-fingerprint vectors were produced under the provider's THEN-default
    # settings; an active config that no longer reproduces them (last-token
    # pooling vs the old mean default) must refuse the legacy collection.
    from mnemostack.cli import _guard_document_space
    from mnemostack.embeddings.roles import recall_space_error

    class _RepooledProvider(_AsymmetricProvider):
        @property
        def name(self) -> str:
            return "huggingface:qwen/qwen3-embedding-0.6b"  # identity doc transform

        def _legacy_space_compatible(self) -> bool:
            return False

    legacy_store = _ScrollStore([{"text": "old unstamped point"}])
    err = recall_space_error(legacy_store, _RepooledProvider())
    assert err is not None and "does not reproduce" in err
    assert _guard_document_space(legacy_store, _RepooledProvider())[0] == 1
    # Same model with legacy-compatible settings still adopts.
    class _MeanProvider(_RepooledProvider):
        def _legacy_space_compatible(self) -> bool:
            return True

    assert recall_space_error(legacy_store, _MeanProvider()) is None


def test_registered_patterns_are_case_normalized():
    from mnemostack.embeddings.profiles import (
        EmbeddingProfile,
        register_embedding_profile,
        resolve_profile,
    )

    register_embedding_profile(
        "testprov-case",
        EmbeddingProfile(
            name="acme", version=1, model_patterns=("Acme/MyEmbed-*",)
        ),
    )
    assert resolve_profile("testprov-case", "acme/myembed-v2").name == "acme"
    assert resolve_profile("testprov-case", "Acme/MyEmbed-v2").name == "acme"


def test_search_many_is_space_guarded():
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    recaller = Recaller(
        embedding_provider=_AsymmetricProvider(),
        vector_store=_Vec([{EMBEDDING_SPACE_KEY: "es1:other"}]),
    )
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        recaller.search_many([[0.1, 0.2]], limit=5)


def test_unguardable_store_skips_fingerprint_resolution():
    # A store without scroll cannot be guarded at all — the (potentially
    # failing) fingerprint lookup must not run and break the legacy-store
    # fallback.
    class _RaisingFp(_AsymmetricProvider):
        def document_space_fingerprint(self) -> str:
            raise RuntimeError("tags endpoint unsupported")

    assert check_document_space(SimpleNamespace(), _RaisingFp())[0] == "unknown"


def test_ineligible_arm_does_not_veto_tenant_recall():
    # Under a tenant, an arm the retriever loop skips (no accepts_tenant)
    # can never contribute — its mismatched collection must not veto the
    # recall. Unscoped recall still enforces every arm.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller
    from mnemostack.recall.retrievers import VectorRetriever

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    class _NoTenantArm(VectorRetriever):
        accepts_tenant = False

    class _Identity(_AsymmetricProvider):
        @property
        def name(self):
            return "ollama:nomic-embed-text"

    ok_arm = VectorRetriever(embedding=_Identity(), vector_store=_Vec([]))
    bad_arm = _NoTenantArm(
        embedding=_AsymmetricProvider(),
        vector_store=_Vec([{EMBEDDING_SPACE_KEY: "es1:other"}]),
    )
    recaller = Recaller(retrievers=[ok_arm, bad_arm])
    assert recaller.recall("вопрос", tenant="t1") == []  # bad arm skipped
    with pytest.raises(EmbeddingSpaceError):
        recaller.recall("вопрос")  # unscoped: every arm enforced


def test_native_document_override_is_not_legacy_compatible():
    # Pre-fingerprint points came through the NEUTRAL primitives — a
    # provider with a native document role does not reproduce them.
    from mnemostack.recall.retrievers import _SharedQueryEmbedding

    class _NativeDoc(_AsymmetricProvider):
        def embed_document(self, text):
            return [9.9]

    assert _NativeDoc()._legacy_space_compatible() is False
    assert _AsymmetricProvider()._legacy_space_compatible() is True
    # The memo wrapper overrides role methods for caching only — it must
    # forward the INNER provider's answer, not its own overrides.
    assert _SharedQueryEmbedding(_AsymmetricProvider())._legacy_space_compatible() is True
    assert _SharedQueryEmbedding(_NativeDoc())._legacy_space_compatible() is False


def test_ingestor_stamps_the_guard_validated_fingerprint():
    # The stamp must be the fingerprint the guard validated; the only other
    # resolution per flush is the post-embedding SANDWICH comparison — no
    # independent stamp lookup exists to race against the verdict.
    class _Counting(_AsymmetricProvider):
        def __init__(self):
            super().__init__()
            self.resolutions = 0

        def _fingerprint_extras(self):
            self.resolutions += 1
            return {"rev": "stable"}

    class _Store(_ScrollStore, _RecordingStore):
        def __init__(self, payloads):
            _ScrollStore.__init__(self, payloads)
            _RecordingStore.__init__(self)

    emb = _Counting()
    store = _Store([])
    Ingestor(embedding=emb, vector_store=store).ingest(
        [IngestItem(text="hello", source="a.md")]
    )
    # guard + sandwich comparison + post-commit revalidation = 3, никакого
    # отдельного резолва под штамп.
    assert emb.resolutions == 3
    assert EMBEDDING_SPACE_KEY in store.upserts[0][2]


def test_ingestor_flush_sandwich_aborts_on_mid_embed_repoint():
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    class _Flipping(_AsymmetricProvider):
        def __init__(self):
            super().__init__()
            self.resolutions = 0

        def _fingerprint_extras(self):
            self.resolutions += 1
            return {"rev": str(self.resolutions)}

    store = _RecordingStore()
    with pytest.raises(EmbeddingSpaceError, match="changed mid-flush"):
        Ingestor(embedding=_Flipping(), vector_store=store).ingest(
            [IngestItem(text="hello", source="a.md")]
        )
    assert store.upserts == []


def test_concurrent_empty_bootstrap_is_detected_post_commit():
    # No atomic empty-collection claim exists: a foreign writer's stamp that
    # appears while we write must fail the SAME flush, not a later one.
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    class _RacyStore(_RecordingStore):
        def __init__(self):
            super().__init__()
            self.foreign: dict | None = None

        def scroll(self):
            if self.foreign is not None:
                yield SimpleNamespace(id="foreign", payload=self.foreign)
            for cid, _vec, payload in self.upserts:
                yield SimpleNamespace(id=cid, payload=payload)

        def upsert(self, id, vector, payload, **kw):
            super().upsert(id, vector, payload, **kw)
            # Simulate the concurrent bootstrapping writer landing first.
            self.foreign = {EMBEDDING_SPACE_KEY: "es1:other"}

    store = _RacyStore()
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        Ingestor(embedding=_AsymmetricProvider(), vector_store=store).ingest(
            [IngestItem(text="hello", source="a.md")]
        )


def test_singular_role_defaults_dispatch_through_batch_overrides():
    # Mirror case: a provider implementing only the BATCH native role must
    # keep it on singular paths (CLI/markdown indexing) too.
    class _NativeBatchDocs(_AsymmetricProvider):
        def embed_documents(self, texts):
            return [[5.5] for _ in texts]

    class _NativeBatchQueries(_AsymmetricProvider):
        def embed_queries(self, texts):
            return [[6.6] for _ in texts]

    assert _NativeBatchDocs().embed_document("x") == [5.5]
    assert _NativeBatchQueries().embed_query("q") == [6.6]


def test_batch_role_defaults_dispatch_through_singular_overrides():
    # A provider overriding only the SINGULAR native role must keep it on
    # batch paths — not silently fall back to neutral vectors there.
    class _NativeQuery(_AsymmetricProvider):
        def embed_query(self, text):
            self.seen.append(("native_q", text))
            return [7.0]

    p = _NativeQuery()
    assert p.embed_queries(["a", "b"]) == [[7.0], [7.0]]
    assert p.seen == [("native_q", "a"), ("native_q", "b")]

    class _NativeDoc(_AsymmetricProvider):
        def embed_document(self, text):
            self.seen.append(("native_d", text))
            return [8.0]

    d = _NativeDoc()
    assert d.embed_documents(["x"]) == [[8.0]]
    assert d.seen == [("native_d", "x")]
    # No singular override → the batched declarative path is unchanged.
    plain = _AsymmetricProvider()
    plain.embed_queries(["a"])
    assert plain.seen == ["query: a"]


def test_ingestor_rechecks_space_on_every_flush():
    # Write-side staleness is unacceptable even within a TTL: a repoint
    # BETWEEN two ingests must be refused at the very next flush.
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    class _Store(_ScrollStore, _RecordingStore):
        def __init__(self, payloads):
            _ScrollStore.__init__(self, payloads)
            _RecordingStore.__init__(self)

    emb = _AsymmetricProvider()
    store = _Store([])
    ingestor = Ingestor(embedding=emb, vector_store=store)
    ingestor.ingest([IngestItem(text="one", source="a.md")])
    assert len(store.upserts) == 1
    store._payloads = [{EMBEDDING_SPACE_KEY: "es1:other"}]  # repointed meanwhile
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        ingestor.ingest([IngestItem(text="two", source="a.md")])
    assert len(store.upserts) == 1  # nothing written after the repoint


def test_local_weights_signature_detects_same_size_content_swap(tmp_path):
    from mnemostack.embeddings.huggingface import _local_weights_signature

    (tmp_path / "config.json").write_text('{"model_type": "bert"}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"A" * 4096)
    sig_a = _local_weights_signature(tmp_path)
    weights.write_bytes(b"B" * 4096)  # same name, same size, new content
    sig_b = _local_weights_signature(tmp_path)
    assert sig_a != sig_b
    # Tokenizer asset changes count too.
    (tmp_path / "tokenizer.json").write_text('{"version": "x"}')
    assert _local_weights_signature(tmp_path) not in (sig_a, sig_b)


def test_shared_query_embedding_memo_expires():
    from mnemostack.recall.retrievers import _SharedQueryEmbedding

    inner = _AsymmetricProvider()
    shim = _SharedQueryEmbedding(inner)
    shim._MEMO_TTL_S = 0.0  # every hit is expired → recompute
    shim.embed_query("q")
    shim.embed_query("q")
    assert inner.seen == ["query: q", "query: q"]  # no stale reuse


# ---------------------------------------------------- provider integrations


def test_ollama_provider_uses_profile_known_dimension():
    from mnemostack.embeddings.ollama import OllamaProvider

    assert OllamaProvider(model="qwen3-embedding:8b").dimension == 4096
    assert OllamaProvider(model="qwen3-embedding:0.6b").dimension == 1024
    # Explicit dimension always wins; legacy table and fallback unchanged.
    assert OllamaProvider(model="qwen3-embedding:8b", dimension=7).dimension == 7
    assert OllamaProvider(model="nomic-embed-text").dimension == 768
    assert OllamaProvider(model="completely-unknown").dimension == 768


def test_ollama_provider_resolves_quantized_tag_dimensions():
    # Size-first quantized tags must not fall through to the 768 default —
    # the collection would never accept the model's real vectors.
    from mnemostack.embeddings.ollama import OllamaProvider

    assert OllamaProvider(model="qwen3-embedding:0.6b-q8_0").dimension == 1024
    assert OllamaProvider(model="qwen3-embedding:8b-q4_K_M").dimension == 4096
    assert OllamaProvider(model="mxbai-embed-large:335m-v1-fp16").dimension == 1024


def test_ollama_fingerprint_pins_the_pulled_model_digest(monkeypatch):
    import json as jsonlib
    from io import BytesIO

    from mnemostack.embeddings.ollama import OllamaProvider

    class _Resp(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    digests = {"qwen3-embedding:8b": "sha256:aaa"}

    def fake_urlopen(url, timeout=None):
        assert str(url).endswith("/api/tags")
        models = [{"name": n, "digest": d} for n, d in digests.items()]
        return _Resp(jsonlib.dumps({"models": models}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = OllamaProvider(model="qwen3-embedding:8b")
    fp_a = p.document_space_fingerprint()
    assert p._fingerprint_extras() == {"digest": "sha256:aaa"}
    # A repointed tag (new digest) is a DIFFERENT space — noticed by the
    # SAME long-lived instance (no lifetime cache), not only a fresh one.
    digests["qwen3-embedding:8b"] = "sha256:bbb"
    assert p.document_space_fingerprint() != fp_a
    assert OllamaProvider(model="qwen3-embedding:8b").document_space_fingerprint() != fp_a


def test_ollama_fingerprint_failure_fails_closed(monkeypatch):
    # A fingerprint that EXISTS but can't be resolved (tags endpoint down
    # while embeddings still work) must fail closed — writing unstamped
    # vectors in that window could mix repointed weights into one space.
    from mnemostack.embeddings.ollama import OllamaProvider
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    def fake_urlopen(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = OllamaProvider(model="qwen3-embedding:8b")
    with pytest.raises(EmbeddingSpaceError, match="fingerprint unavailable"):
        document_space_fingerprint_via(p)


def test_cli_guard_fails_closed_when_fingerprint_unresolvable():
    from mnemostack.cli import _guard_document_space

    class _FlakyFp(_AsymmetricProvider):
        def document_space_fingerprint(self) -> str:
            raise RuntimeError("tags endpoint down")

    assert _guard_document_space(_ScrollStore([]), _FlakyFp())[0] == 1


def test_recaller_fails_closed_on_unresolvable_fingerprint_and_retries():
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller

    class _FlakyFp(_AsymmetricProvider):
        fail = True

        def document_space_fingerprint(self) -> str:
            if self.fail:
                raise RuntimeError("tags endpoint down")
            return super().document_space_fingerprint()

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    provider = _FlakyFp()
    recaller = Recaller(embedding_provider=provider, vector_store=_Vec([]))
    with pytest.raises(EmbeddingSpaceError, match="fingerprint unavailable"):
        recaller.recall("вопрос")
    # Nothing was cached — after the provider recovers, recall works.
    provider.fail = False
    assert recaller.recall("вопрос") == []


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


def test_shared_query_embedding_delegates_native_role_overrides():
    # An inner provider with NATIVE query/document task types must have its
    # own role methods called (memoized), not be routed around via the base
    # transform + neutral embed.
    from mnemostack.recall.retrievers import _SharedQueryEmbedding

    class _NativeRoles(EmbeddingProvider):
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def embed(self, text):
            self.calls.append(("embed", text))
            return [1.0]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

        def embed_query(self, text):
            self.calls.append(("embed_query", text))
            return [2.0]

        def embed_document(self, text):
            self.calls.append(("embed_document", text))
            return [3.0]

        @property
        def dimension(self):
            return 1

        @property
        def name(self):
            return "custom:native-task-types"

    inner = _NativeRoles()
    shim = _SharedQueryEmbedding(inner)
    assert shim.embed_query("q") == [2.0]
    assert shim.embed_query("q") == [2.0]  # memo hit
    assert inner.calls == [("embed_query", "q")]
    assert shim.embed_document("q") == [3.0]  # role-tagged key: no collision
    assert shim.embed("q") == [1.0]
    assert inner.calls == [
        ("embed_query", "q"),
        ("embed_document", "q"),
        ("embed", "q"),
    ]


def test_recaller_refuses_recall_across_embedding_spaces():
    # A legacy (unstamped) collection under a transforming profile — or a
    # stamped mismatch — must fail loud on recall, not degrade silently.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    provider = _AsymmetricProvider()
    recaller = Recaller(
        embedding_provider=provider,
        vector_store=_Vec([{"text": "legacy raw point"}]),
    )
    with pytest.raises(EmbeddingSpaceError, match="transforms documents"):
        recaller.recall("вопрос")
    # The determined incompatibility is re-raised on every recall.
    with pytest.raises(EmbeddingSpaceError):
        recaller.recall("вопрос")

    mismatch = Recaller(
        embedding_provider=provider,
        vector_store=_Vec([{EMBEDDING_SPACE_KEY: "es1:other"}]),
    )
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        mismatch.recall("вопрос")


def test_recaller_guard_covers_retrievers_mode():
    # The primary server/MCP construction passes vector arms as retrievers
    # (often with no Recaller-level embedding/vector at all) — the space
    # check must derive the pair from the arm, not silently skip.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller
    from mnemostack.recall.retrievers import VectorRetriever

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    arm = VectorRetriever(
        embedding=_AsymmetricProvider(),
        vector_store=_Vec([{"text": "legacy raw point"}]),
    )
    recaller = Recaller(retrievers=[arm])
    with pytest.raises(EmbeddingSpaceError, match="transforms documents"):
        recaller.recall("вопрос")


def test_recaller_guard_checks_every_vector_arm():
    # A compatible first arm must not speak for an incompatible later one.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller
    from mnemostack.recall.retrievers import VectorRetriever

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    class _Identity(_AsymmetricProvider):
        @property
        def name(self):
            return "ollama:nomic-embed-text"

    ok_arm = VectorRetriever(embedding=_Identity(), vector_store=_Vec([]))
    bad_arm = VectorRetriever(
        embedding=_AsymmetricProvider(),
        vector_store=_Vec([{EMBEDDING_SPACE_KEY: "es1:other"}]),
    )
    recaller = Recaller(retrievers=[ok_arm, bad_arm])
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        recaller.recall("вопрос")


def test_recaller_guard_retries_after_inconclusive_check():
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.recaller import Recaller

    class _FlakyVec:
        def __init__(self, payloads):
            self._payloads = payloads
            self.scroll_calls = 0

        def scroll(self):
            self.scroll_calls += 1
            if self.scroll_calls == 1:
                raise ConnectionError("store hiccup")
            for p in self._payloads:
                yield SimpleNamespace(id="x", payload=p)

        def search(self, *a, **kw):
            return []

    store = _FlakyVec([{EMBEDDING_SPACE_KEY: "es1:other"}])
    recaller = Recaller(embedding_provider=_AsymmetricProvider(), vector_store=store)
    # First recall: check inconclusive → fail open, do NOT cache.
    assert recaller.recall("вопрос") == []
    # Store recovered: the retried check finds the mismatch and refuses.
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        recaller.recall("вопрос")


def test_space_guard_revalidates_verdicts_both_directions():
    # Verdicts age out: a live process must notice both a HEALED collection
    # (operator reindexed) and a freshly-recreated incompatible one.
    from mnemostack.embeddings.roles import EmbeddingSpaceError, SpaceGuard

    p = _AsymmetricProvider()
    store = _ScrollStore([{EMBEDDING_SPACE_KEY: "es1:other"}])
    guard = SpaceGuard(store, p, recheck_seconds=0.0)
    with pytest.raises(EmbeddingSpaceError):
        guard.ensure()
    store._payloads = [{EMBEDDING_SPACE_KEY: p.document_space_fingerprint()}]
    guard.ensure()  # healed without a restart
    store._payloads = [{EMBEDDING_SPACE_KEY: "es1:other"}]
    with pytest.raises(EmbeddingSpaceError):
        guard.ensure()  # recreation under a different config is caught


def test_vector_retriever_guards_direct_search():
    # synthesis/library callers hit retrievers WITHOUT a Recaller — the
    # guard must live at the retriever boundary too.
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.recall.retrievers import VectorRetriever

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    arm = VectorRetriever(
        embedding=_AsymmetricProvider(),
        vector_store=_Vec([{EMBEDDING_SPACE_KEY: "es1:other"}]),
    )
    with pytest.raises(EmbeddingSpaceError, match="different spaces"):
        arm.search("вопрос")


def test_ingestor_stamps_fresh_fingerprint_per_flush():
    # A long-lived ingestor must stamp the space of the weights CURRENTLY
    # served, not one resolved at construction (mutable tags repoint).
    class _MutatingFp(_AsymmetricProvider):
        rev = "a"

        def _fingerprint_extras(self):
            return {"rev": self.rev}

    emb = _MutatingFp()
    store = _RecordingStore()
    ingestor = Ingestor(embedding=emb, vector_store=store)
    ingestor.ingest([IngestItem(text="one", source="a.md")])
    emb.rev = "b"
    ingestor.ingest([IngestItem(text="two", source="a.md")])
    fp_first = store.upserts[0][2][EMBEDDING_SPACE_KEY]
    fp_second = store.upserts[1][2][EMBEDDING_SPACE_KEY]
    assert fp_first != fp_second


def test_recaller_allows_matching_and_identity_legacy_spaces():
    from mnemostack.recall.recaller import Recaller

    class _Vec(_ScrollStore):
        def search(self, *a, **kw):
            return []

    provider = _AsymmetricProvider()
    ok = Recaller(
        embedding_provider=provider,
        vector_store=_Vec([{EMBEDDING_SPACE_KEY: provider.document_space_fingerprint()}]),
    )
    assert ok.recall("вопрос") == []

    # Identity profile against a legacy collection: raw == transformed.
    class _Identity(_AsymmetricProvider):
        @property
        def name(self):
            return "ollama:nomic-embed-text"

    legacy_ok = Recaller(
        embedding_provider=_Identity(),
        vector_store=_Vec([{"text": "legacy raw point"}]),
    )
    assert legacy_ok.recall("вопрос") == []

    # Query-only transform (qwen3) against a legacy collection is the
    # contract's "adding a query transform never requires reindexing" —
    # raw document vectors are exactly what this profile produces.
    class _Qwen(_AsymmetricProvider):
        @property
        def name(self):
            return "ollama:qwen3-embedding:8b"

    qwen_legacy_ok = Recaller(
        embedding_provider=_Qwen(),
        vector_store=_Vec([{"text": "legacy raw point"}]),
    )
    assert qwen_legacy_ok.recall("вопрос") == []


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
