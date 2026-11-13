"""State mutation safety and request-owned lexical result payloads."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mnemostack.recall.bm25 import BM25, BM25Doc
from mnemostack.recall.mca_prefilter import mca_prefilter
from mnemostack.recall.pipeline.state import FileStateStore
from mnemostack.recall.retrievers import BM25Retriever


@pytest.mark.parametrize('operation', ['set', 'update', 'delete', 'batch'])
@pytest.mark.parametrize('content', [b'{"alpha":', b'\xff', b'[]', b'null', b'42'])
def test_invalid_state_blocks_mutations_without_overwrite(tmp_path, operation, content):
    path = tmp_path/'state.json'
    path.write_bytes(content)
    store = FileStateStore(path)
    invoked = []
    def updater(old):
        invoked.append(old)
        return {'new': True}
    with pytest.raises((ValueError, OSError, TypeError)):
        if operation == 'set':
            store.set('third-tenant', {'new': True})
        elif operation == 'update':
            store.update('third-tenant', updater)
        elif operation == 'batch':
            store.update_many([('third', lambda old: {}), ('alpha', lambda old: {})])
        else:
            store.delete('alpha')
    assert path.read_bytes() == content
    assert invoked == []


@pytest.mark.parametrize('operation', ['set', 'update', 'delete', 'batch'])
def test_unreadable_state_blocks_mutations(tmp_path, monkeypatch, operation):
    path = tmp_path/'state.json'
    original = b'{"alpha": {"value": 1}, "beta": {"value": 2}}'
    path.write_bytes(original)
    store = FileStateStore(path)
    read_text = Path.read_text
    def unreadable(self, *args, **kwargs):
        if self == path:
            raise PermissionError('synthetic read denial')
        return read_text(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', unreadable)
    with pytest.raises(PermissionError):
        if operation == 'set':
            store.set('third', {})
        elif operation == 'update':
            store.update('alpha', lambda old: {})
        elif operation == 'batch':
            store.update_many([('third', lambda old: {}), ('alpha', lambda old: {})])
        else:
            store.delete('alpha')
    assert path.read_bytes() == original


@pytest.mark.parametrize('content', [b'{', b'\xff', b'[]', b'null', b'42'])
def test_best_effort_read_does_not_repair_state(tmp_path, content):
    path = tmp_path/'state.json'
    path.write_bytes(content)
    assert FileStateStore(path).get('missing', 'fallback') == 'fallback'
    assert path.read_bytes() == content


def test_missing_and_valid_state_preserve_other_tenants(tmp_path):
    path = tmp_path/'state.json'
    store = FileStateStore(path)
    assert store.get('missing') is None
    assert store.delete('missing') is False
    store.set('q_table:alpha', {'ключ': 1})
    store.set('q_table:beta', {'value': 2})
    assert store.update('q_table:alpha', lambda old: {'ключ': old['ключ'] + 1}) == {'ключ': 2}
    assert store.get('q_table:beta') == {'value': 2}
    assert store.delete('q_table:alpha') is True
    assert json.loads(path.read_text()) == {'q_table:beta': {'value': 2}}


def lexical_search(kind, docs, filters=None):
    if kind == 'bm25':
        retriever = BM25Retriever(docs, tenant_aware=True)
        return lambda: retriever.search('ISSUE-42', filters=filters)
    index = BM25(docs)
    return lambda: mca_prefilter('ISSUE-42', index, filters=filters)


@pytest.mark.parametrize('kind', ['bm25', 'mca'])
def test_lexical_hits_own_nested_payload(kind):
    payload = {'flags': ['client'], 'tenant_id': 'alpha', 'source': 'synthetic.md',
               'nested': {'roles': ['user'], 'provenance': {'offset': 12}}}
    before = copy.deepcopy(payload)
    doc = BM25Doc('stable-id', 'ISSUE-42 привет', payload=payload)
    search = lexical_search(kind, [doc])
    first = search()[0]
    first.payload['flags'].append('pipeline-only')
    first.payload['nested']['roles'].append('assistant')
    first.payload['nested']['provenance']['offset'] = 99
    first.payload['extra'] = True
    assert payload == before
    second = search()[0]
    assert second.payload == before
    assert second.id == first.id == 'stable-id'
    assert second.text == first.text == doc.text
    assert second.score == first.score
    assert second.sources == first.sources == [kind]


@pytest.mark.parametrize('kind', ['bm25', 'mca'])
def test_parallel_results_do_not_annotate_corpus(kind):
    doc = BM25Doc('stable-id', 'ISSUE-42', payload={'flags': ['client'], 'nested': {'values': []}})
    before = copy.deepcopy(doc.payload)
    search = lexical_search(kind, [doc])
    def worker(number):
        result = search()[0]
        result.payload['flags'].append(str(number))
        result.payload['nested']['values'].append(number)
        return result
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(12)))
    assert doc.payload == before
    assert all(len(r.payload['flags']) == 2 and len(r.payload['nested']['values']) == 1 for r in results)
    assert len({id(r.payload['flags']) for r in results}) == len(results)


@pytest.mark.parametrize('kind', ['bm25', 'mca'])
@pytest.mark.parametrize('payload', [None, {}, {'flags': 'client-string', 'nested': ['尾', '🙂']}])
def test_empty_and_opaque_payloads_roundtrip(kind, payload):
    doc = BM25Doc(42, 'ISSUE-42', payload=copy.deepcopy(payload))
    result = lexical_search(kind, [doc])()[0]
    assert result.id == 42
    assert result.payload == (payload or {})
    result.payload['new'] = ['annotation']
    assert doc.payload == (payload or {})


@pytest.mark.parametrize('kind', ['bm25', 'mca'])
def test_ids_scores_filters_and_order_preserved(kind):
    docs = [BM25Doc('alpha', 'ISSUE-42 ISSUE-42', payload={'tenant_id': 'alpha', 'status': 'active'}),
            BM25Doc('beta', 'ISSUE-42', payload={'tenant_id': 'beta', 'status': 'active'}),
            BM25Doc('inactive', 'ISSUE-42', payload={'tenant_id': 'alpha', 'status': 'inactive'})]
    search = lexical_search(kind, docs, {'tenant_id': 'alpha', 'status': 'active'})
    out = search()
    expected = BM25(docs).search('ISSUE-42', predicate=lambda d: d.payload['tenant_id'] == 'alpha' and d.payload['status'] == 'active')
    assert [(r.id, r.score, r.text) for r in out] == [(d.id, score, d.text) for d, score in expected]
