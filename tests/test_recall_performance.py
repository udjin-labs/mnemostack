"""Regression coverage for recall search work reduction."""

import math
import random
from collections import Counter
from difflib import SequenceMatcher

import pytest

from mnemostack.recall.bm25 import BM25, BM25Doc
from mnemostack.synthesis import _similar


def test_bm25_scores_only_matching_docs(monkeypatch):
    index = BM25([BM25Doc(i, 'match' if i == 4 else 'other') for i in range(100)])
    scored = []
    original = index._score

    def score(tokens, doc):
        scored.append(doc.id)
        return original(tokens, doc)

    monkeypatch.setattr(index, '_score', score)
    assert [d.id for d, _ in index.search('match')] == [4]
    assert scored == [4]


def test_bm25_refuses_a_one_shot_corpus():
    with pytest.raises(TypeError):
        BM25(BM25Doc(i, 'text') for i in range(3))


def test_bm25_random_scores_are_bit_identical():
    rng = random.Random(733)
    vocabulary = ['alpha', 'beta', 'gamma', 'שלום', 'мир', '日本語']
    for _ in range(50):
        docs = [BM25Doc(i, ' '.join(rng.choices(vocabulary, k=rng.randrange(20))))
                for i in range(40)]
        index = BM25(docs)
        query = rng.choices(vocabulary + ['missing'], k=5)
        expected = []
        for doc in docs:
            tf = Counter(doc.tokens)
            score = 0.0
            for term in query:
                df = sum(term in (d.tokens or []) for d in docs)
                if not df or not tf[term]:
                    continue
                idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                num = tf[term] * (index.k1 + 1)
                denom = tf[term] + index.k1 * (
                    1 - index.b + index.b * len(doc.tokens or []) / max(index.avgdl, 1)
                )
                score += idf * num / denom
            if score > 0 and doc.id % 3:
                expected.append((doc, score))
        expected.sort(key=lambda pair: -pair[1])
        assert index.search(' '.join(query), limit=13, predicate=lambda d: bool(d.id % 3)) == expected[:13]


def test_similarity_bounds_skip_ratio(monkeypatch):
    def unexpected_ratio(self):
        raise AssertionError('ratio called despite rejecting upper bound')

    monkeypatch.setattr(SequenceMatcher, 'ratio', unexpected_ratio)
    assert not _similar('abcdef', 'uvwxyz')
    assert not _similar('abc', 'x' * 100)


def test_similarity_bounds_preserve_multilingual_decisions():
    rng = random.Random(94)
    alphabet = 'abc שלום мир 日本語🙂'
    for _ in range(1000):
        a = ''.join(rng.choices(alphabet, k=rng.randrange(1, 70)))
        b = ''.join(rng.choices(alphabet, k=rng.randrange(1, 70)))
        if a in b or b in a:
            expected = min(len(a), len(b)) / max(len(a), len(b)) > 0.75
        else:
            expected = SequenceMatcher(None, a, b).ratio() >= 0.92
        assert _similar(a, b) == expected


class CountingEmbedding:
    dimension = 3

    def __init__(self):
        self.queries = []
        self.documents = []
        self.fail = False

    def embed_query(self, text):
        self.queries.append(text)
        if self.fail:
            raise RuntimeError('embedding failed')
        return [1.0, 0.0, 0.0]

    def embed_document(self, text):
        self.documents.append(text)
        return [0.0, 1.0, 0.0]


class EmptyVectorStore:
    def __init__(self):
        self.vectors = []

    def search(self, vector, **kwargs):
        self.vectors.append(list(vector))
        return []


def test_query_vector_shared_by_arms_and_fallback_only_within_recall():
    from mnemostack.recall import Recaller
    from mnemostack.recall.retrievers import TemporalRetriever, VectorRetriever

    provider = CountingEmbedding()
    store = EmptyVectorStore()
    recaller = Recaller(
        embedding_provider=provider,
        vector_store=store,
        retrievers=[VectorRetriever(provider, store), TemporalRetriever(
            provider, store, extractor=lambda _: ('2026-01-01', '2026-02-01')
        )],
        allow_cross_tenant=True,
    )
    for tenant in ('one', 'two'):
        assert recaller.recall('query', tenant=tenant) == []
    assert provider.queries == ['query', 'query']
    assert store.vectors == [[1.0, 0.0, 0.0]] * 6
    assert provider.documents == []


def test_query_vectors_separate_providers_texts_roles_and_failed_recalls():
    from mnemostack.embeddings.roles import embed_document_via
    from mnemostack.recall import Recaller
    from mnemostack.recall.retrievers import VectorRetriever

    first, second = CountingEmbedding(), CountingEmbedding()
    store = EmptyVectorStore()
    recaller = Recaller(
        embedding_provider=first, vector_store=store,
        retrievers=[VectorRetriever(first, store), VectorRetriever(second, store)],
        allow_cross_tenant=True,
    )
    first.fail = True
    assert recaller.recall('query') == []
    first.fail = False
    first.queries.clear()
    assert recaller.recall('query') == []
    assert first.queries == ['query']
    assert second.queries == ['query', 'query']
    assert recaller.recall('variant') == []
    assert first.queries == ['query', 'variant']
    assert embed_document_via(first, 'query') == [0.0, 1.0, 0.0]
    assert first.documents == ['query']


def test_expanded_query_texts_share_vectors_only_when_identical(monkeypatch):
    from mnemostack.recall import Recaller
    from mnemostack.recall.retrievers import VectorRetriever

    provider = CountingEmbedding()
    store = EmptyVectorStore()
    recaller = Recaller(
        embedding_provider=provider, vector_store=store,
        retrievers=[VectorRetriever(provider, store)],
        query_expansion=True, allow_cross_tenant=True,
    )
    monkeypatch.setattr(recaller, '_expanded_queries', lambda _: ['query', 'variant', 'query'])
    assert recaller.recall('query') == []
    assert provider.queries == ['query', 'variant']


@pytest.mark.parametrize('raises', [False, True])
@pytest.mark.parametrize('deadline', [None, 30.0])
def test_transient_query_embedding_failure_retries_and_preserves_results(raises, deadline):
    from threading import Barrier, Lock
    from types import SimpleNamespace

    from mnemostack.recall import Recaller
    from mnemostack.recall.retrievers import TemporalRetriever, VectorRetriever
    from mnemostack.recall.trace import RecallTrace

    class Store:
        def search(self, vector, **kwargs):
            return [SimpleNamespace(id=i, score=1.0 - i / 100, payload={
                'text': f'doc {i}', 'timestamp': '2026-03-01T00:00:00+00:00',
            }) for i in range(1, 6)]

    overlap = Barrier(2)
    lock = Lock()

    class Flaky(CountingEmbedding):
        def embed_query(self, text):
            with lock:
                vector = super().embed_query(text)
                first = len(self.queries) == 1
            if deadline is not None:
                overlap.wait(timeout=5)
            if first:
                if raises:
                    raise RuntimeError('transient')
                return []
            return vector

    provider, store = Flaky(), Store()
    recaller = Recaller(
        embedding_provider=provider, vector_store=store,
        retrievers=[VectorRetriever(provider, store), TemporalRetriever(
            provider, store, extractor=lambda _: ('2026-02-01', '2026-04-01')
        )], fallback_threshold=0.99, allow_cross_tenant=True, overall_deadline=deadline,
    )
    trace = RecallTrace()
    assert [hit.id for hit in recaller.recall('query', trace=trace)] == [1, 2, 3, 4, 5]
    assert provider.queries == ['query', 'query']
    if raises and deadline is not None:
        assert trace.degraded in (['retriever:vector:failed'], ['retriever:temporal:failed'])
    else:
        assert trace.degraded == (['retriever:vector:failed'] if raises else [])


@pytest.mark.parametrize('raises', [False, True])
def test_failed_query_embedding_retries_overlap_and_later_success_is_shared(raises):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event

    import mnemostack.recall.query_vectors as module

    waiting = Barrier(5)
    first_started = Event()
    release = Event()

    cache = module.QueryVectors()
    provider = CountingEmbedding()
    calls = []
    overlap = Barrier(3)

    def embed(provider, query):
        calls.append(query)
        if len(calls) == 1:
            first_started.set()
            assert release.wait(timeout=5)
        else:
            overlap.wait(timeout=5)
        if raises:
            raise RuntimeError('failed')
        return []

    def attempt():
        waiting.wait(timeout=5)
        if raises:
            with pytest.raises(RuntimeError, match='failed'):
                cache.embed(provider, 'query', embed)
        else:
            assert cache.embed(provider, 'query', embed) == []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(attempt) for _ in range(4)]
        try:
            waiting.wait(timeout=5)
            assert first_started.wait(timeout=5)
        finally:
            release.set()
        for future in futures:
            future.result(timeout=10)
    assert calls == ['query'] * 4
    assert not overlap.broken
    assert cache.embed(provider, 'query', lambda *_: [1.0]) == [1.0]
    assert cache.embed(provider, 'query', lambda *_: pytest.fail('cached success lost')) == [1.0]


def test_deadline_query_vectors_overlap_and_fallback_reuses_success():
    from threading import Barrier

    from mnemostack.recall import Recaller
    from mnemostack.recall.retrievers import TemporalRetriever, VectorRetriever

    overlap = Barrier(2)

    class ConcurrentEmbedding(CountingEmbedding):
        def embed_query(self, text):
            vector = super().embed_query(text)
            overlap.wait(timeout=5)
            return vector

    provider, store = ConcurrentEmbedding(), EmptyVectorStore()
    recaller = Recaller(
        embedding_provider=provider, vector_store=store,
        retrievers=[VectorRetriever(provider, store), TemporalRetriever(
            provider, store, extractor=lambda _: ('2026-01-01', '2026-02-01')
        )], overall_deadline=30.0, allow_cross_tenant=True,
    )
    assert recaller.recall('query') == []
    assert provider.queries == ['query', 'query']
    assert store.vectors == [[1.0, 0.0, 0.0]] * 3


@pytest.mark.parametrize('base_name', ['VectorRetriever', 'TemporalRetriever'])
def test_retriever_subclass_with_old_search_signature_still_runs(base_name):
    from types import SimpleNamespace

    import mnemostack.recall.retrievers as retrievers
    from mnemostack.recall import Recaller, RecallTrace

    class Embedding:
        dimension = 3

        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    class Store:
        def search(self, vector, **kwargs):
            return [SimpleNamespace(id=i, score=1.0 - i / 10, payload={'text': f'd{i}'}) for i in (1, 2, 3)]

    if base_name == 'VectorRetriever':
        class Old(retrievers.VectorRetriever):
            def search(self, query, limit=20, filters=None, as_of=None, include_invalidated=False, tenant=None):
                return super().search(
                    query, limit=limit, filters=filters, as_of=as_of,
                    include_invalidated=include_invalidated, tenant=tenant,
                )
    else:
        class Old(retrievers.TemporalRetriever):
            def search(self, query, limit=10, filters=None, tenant=None):
                return super().search(query, limit=limit, filters=filters, tenant=tenant)

    extra = {'extractor': lambda _: ('2026-01-01', '2026-02-01')} if base_name == 'TemporalRetriever' else {}
    trace = RecallTrace()
    recaller = Recaller(retrievers=[Old(Embedding(), Store(), **extra)], allow_cross_tenant=True)
    assert [r.id for r in recaller.recall('query', trace=trace)] == [1, 2, 3]
    assert trace.degraded == []


def test_bm25_index_is_a_snapshot_of_the_document_list():
    docs = [BM25Doc(0, 'alpha'), BM25Doc(1, 'beta')]
    index = BM25(docs)
    del docs[0]
    assert [d.id for d, _ in index.search('alpha')] == [0]
    assert [d.id for d, _ in index.search('beta')] == [1]


def test_bm25_index_can_be_pickled():
    import pickle

    index = pickle.loads(pickle.dumps(BM25([BM25Doc(0, 'alpha'), BM25Doc(1, 'beta')])))
    assert [d.id for d, _ in index.search('beta')] == [1]
