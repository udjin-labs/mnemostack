"""Batched exposure persistence and per-recall state reads."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest

from mnemostack.feedback import apply_feedback, record_recall_events
from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import (
    CuriosityBoost,
    InhibitionOfReturn,
    Pipeline,
    QLearningReranker,
)
from mnemostack.recall.pipeline.state import FileStateStore, InMemoryStateStore, StateStore


class PublicStore(StateStore):
    def __init__(self, store):
        self.store = store
        self.reads = 0

    def get(self, key, default=None):
        self.reads += 1
        return self.store.get(key, default)

    def set(self, key, value):
        self.store.set(key, value)

    def update(self, key, updater):
        return self.store.update(key, updater)


class FrozenClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 13, tzinfo=timezone.utc)


@pytest.mark.parametrize('keep_last', [0, 1, 7, 500])
@pytest.mark.parametrize('tenant', [None, 'alpha:beta'])
def test_sequence_matches_per_entry_recording(tmp_path, monkeypatch, keep_last, tenant):
    monkeypatch.setattr('mnemostack.recall.pipeline.stages.datetime', FrozenClock)
    stores = [FileStateStore(tmp_path / f'{i}.json') for i in range(2)]
    initial = {'ior_log:other': [{'id': 'foreign', 'timestamp': '2020-01-01'}],
               'ior_log': [{'id': '0', 'timestamp': '2000-01-01'}]}
    for store in stores:
        store.path.write_text(json.dumps(initial))
    # Separate public-interface adapters preserve the original three reads.
    legacy = Pipeline([QLearningReranker(PublicStore(stores[0])),
                       CuriosityBoost(PublicStore(stores[0])),
                       InhibitionOfReturn(PublicStore(stores[0]), keep_last=keep_last)])
    batch = Pipeline([QLearningReranker(stores[1]), CuriosityBoost(stores[1]),
                      InhibitionOfReturn(stores[1], keep_last=keep_last)])
    for turn in range(4):
        hits = [RecallResult(id=str(i), text='memory', score=1 - i / 100,
                             sources=['vector'], payload={'timestamp': '2020-01-01'})
                for i in range(15)]
        expected = legacy.apply('query', copy.deepcopy(hits), tenant=tenant)
        actual = batch.apply('query', copy.deepcopy(hits), tenant=tenant)
        assert actual == expected
        for stage in legacy:
            if hasattr(stage, 'record_recall'):
                for hit in expected:
                    stage.record_recall(str(hit.id), tenant=tenant)
        assert record_recall_events(batch, actual, tenant)
        for pipeline in [legacy, batch]:
            apply_feedback(pipeline, str(turn), 'clicked', source='vector', tenant=tenant)
        assert json.loads(stores[0].path.read_text()) == json.loads(stores[1].path.read_text())
        assert stores[1].get('ior_log:other') == initial['ior_log:other']


def test_reads_shared_only_within_run_and_store():
    store = PublicStore(InMemoryStateStore())
    pipeline = Pipeline([QLearningReranker(store), CuriosityBoost(store), InhibitionOfReturn(store)])
    for tenant in ['a', 'b', 'a']:
        pipeline.apply('query', [RecallResult(id='x', text='x', score=1)], tenant=tenant)
    assert store.reads == 6
    other = PublicStore(InMemoryStateStore())
    Pipeline([CuriosityBoost(store), InhibitionOfReturn(other)]).apply(
        'query', [RecallResult(id='x', text='x', score=1)])
    assert other.reads == 1


def test_concurrent_batches_keep_all_entries(tmp_path):
    path = tmp_path / 'state.json'
    stores = [FileStateStore(path), FileStateStore(path)]
    barrier = Barrier(2)

    def record(store):
        barrier.wait(timeout=10)
        InhibitionOfReturn(store, keep_last=1000).record_recalls(['same'] * 100)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(record, stores))
    assert len(stores[0].get('ior_log')) == 200


@pytest.mark.parametrize('failure', ['updater', 'replace'])
def test_batch_failure_leaves_original_file(tmp_path, monkeypatch, failure):
    store = FileStateStore(tmp_path / 'state.json')
    store.set('original', {'n': 1})
    before = store.path.read_bytes()

    def fail(*args):
        raise RuntimeError('synthetic failure')

    if failure == 'replace':
        monkeypatch.setattr(type(store.path), 'replace', fail)
    updates = [('original', lambda old: {'n': 2}),
               ('new', fail if failure == 'updater' else lambda old: 3)]
    with pytest.raises(RuntimeError):
        store.update_many(updates)
    assert store.path.read_bytes() == before


def test_batch_uses_one_read_and_write(tmp_path, monkeypatch):
    store = FileStateStore(tmp_path / 'state.json')
    calls = []
    for name in ['_read_all', '_write_all']:
        original = getattr(store, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, counted)
    pipeline = Pipeline([QLearningReranker(store), CuriosityBoost(store), InhibitionOfReturn(store)])
    hits = pipeline.apply('query', [RecallResult(id=str(i), text='x', score=1) for i in range(15)])
    record_recall_events(pipeline, hits)
    assert calls.count('_read_all') == 3
    assert calls.count('_write_all') == 1


@pytest.mark.parametrize('custom', [False, True])
def test_batch_falls_back_to_public_store(custom):
    store = InMemoryStateStore()
    target = PublicStore(store) if custom else store
    stage = InhibitionOfReturn(target, keep_last=2)
    stage.record_recalls(['a', 'b', 'c'], tenant='alpha')
    assert [entry['id'] for entry in store.get('ior_log:alpha')] == ['b', 'c']
    assert store.get('ior_log') is None


def test_batch_holds_file_lock_through_all_updates(tmp_path, monkeypatch):
    from contextlib import contextmanager

    store = FileStateStore(tmp_path / 'state.json')
    locked = False
    file_lock = store._file_lock

    @contextmanager
    def tracked_lock():
        nonlocal locked
        with file_lock():
            locked = True
            try:
                yield
            finally:
                locked = False

    monkeypatch.setattr(store, '_file_lock', tracked_lock)
    for name in ['_read_all', '_write_all']:
        original = getattr(store, name)

        def checked(*args, _original=original, **kwargs):
            assert locked
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, checked)

    def increment(old):
        assert locked
        return (old or 0) + 1

    store.update_many([('count', increment)] * 3)
    assert store.get('count') == 3


def test_overridden_record_hook_is_still_called_per_result(tmp_path):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore
    from mnemostack.recall.recaller import RecallResult

    seen = []

    class Filtering(InhibitionOfReturn):
        def record_recall(self, memory_id, **kwargs):
            seen.append(memory_id)
            if not memory_id.startswith("secret"):
                super().record_recall(memory_id, **kwargs)

    store = FileStateStore(tmp_path / "state.json")
    stage = Filtering(state_store=store)
    results = [RecallResult("secret-1", "x", 1.0), RecallResult("b", "y", 0.9)]
    record_recall_events([stage], results)
    assert seen == ["secret-1", "b"]
    assert [entry["id"] for entry in store.get(stage.STATE_KEY)] == ["b"]


def test_reused_context_reads_state_again(tmp_path):
    from mnemostack.recall.pipeline import Pipeline, PipelineContext
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore
    from mnemostack.recall.recaller import RecallResult

    store = FileStateStore(tmp_path / "state.json")
    stage = InhibitionOfReturn(state_store=store)
    pipeline = Pipeline([stage])
    context = PipelineContext(query="q")
    first = pipeline.apply_with_context(context, [RecallResult("a", "x", 1.0)])
    assert "ior_penalty" not in first[0].annotations
    stage.record_recall("a")
    second = pipeline.apply_with_context(context, [RecallResult("a", "x", 1.0)])
    assert "ior_penalty" in second[0].annotations


def _recorded_ids(store, stage):
    return [entry["id"] for entry in store.get(stage.STATE_KEY) or []]


def _two_results():
    from mnemostack.recall.recaller import RecallResult

    return [RecallResult("secret-1", "x", 1.0), RecallResult("b", "y", 0.9)]


def test_instance_level_record_hook_is_still_called_per_result(tmp_path):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    store = FileStateStore(tmp_path / "state.json")
    stage = InhibitionOfReturn(state_store=store)
    original = stage.record_recall
    seen = []

    def wrapped(memory_id, **kwargs):
        seen.append(memory_id)
        if not memory_id.startswith("secret"):
            original(memory_id, **kwargs)

    stage.record_recall = wrapped
    record_recall_events([stage], _two_results())
    assert seen == ["secret-1", "b"]
    assert _recorded_ids(store, stage) == ["b"]


def test_proxy_stage_keeps_the_wrapped_record_hook(tmp_path):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    class Filtering(InhibitionOfReturn):
        def record_recall(self, memory_id, **kwargs):
            if not memory_id.startswith("secret"):
                super().record_recall(memory_id, **kwargs)

    class Proxy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    store = FileStateStore(tmp_path / "state.json")
    stage = Filtering(state_store=store)
    record_recall_events([Proxy(stage)], _two_results())
    assert _recorded_ids(store, stage) == ["b"]


def test_mock_stage_gets_one_call_per_result():
    from unittest.mock import MagicMock, call

    from mnemostack.feedback import record_recall_events

    stage = MagicMock()
    record_recall_events([stage], _two_results(), tenant="t")
    assert stage.record_recall.call_args_list == [call("secret-1", tenant="t"), call("b", tenant="t")]
    assert not stage.record_recalls.called


def test_store_subclass_update_override_sees_every_entry(tmp_path):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    audit = []

    class Audited(FileStateStore):
        def update(self, key, updater):
            audit.append(key)
            return super().update(key, updater)

    store = Audited(tmp_path / "state.json")
    stage = InhibitionOfReturn(state_store=store)
    record_recall_events([stage], _two_results())
    assert audit == [stage.STATE_KEY, stage.STATE_KEY]
    assert _recorded_ids(store, stage) == ["secret-1", "b"]


def test_direct_stage_apply_reads_state_every_time(tmp_path):
    from mnemostack.recall.pipeline import PipelineContext
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore
    from mnemostack.recall.recaller import RecallResult

    stage = InhibitionOfReturn(state_store=FileStateStore(tmp_path / "state.json"))
    context = PipelineContext(query="q")
    first = stage.apply(context, [RecallResult("a", "x", 1.0)])
    assert "ior_penalty" not in first[0].annotations
    stage.record_recall("a")
    second = stage.apply(context, [RecallResult("a", "x", 1.0)])
    assert "ior_penalty" in second[0].annotations


def test_legacy_hook_signature_without_tenant_still_works(tmp_path):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    seen = []

    class Legacy(InhibitionOfReturn):
        def record_recall(self, memory_id):
            seen.append(memory_id)

    stage = Legacy(state_store=FileStateStore(tmp_path / "state.json"))
    record_recall_events([stage], _two_results())
    stage.record_recalls(["c"])
    assert seen == ["secret-1", "b", "c"]


def test_class_level_patch_of_record_hook_is_called(tmp_path, monkeypatch):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    seen = []
    monkeypatch.setattr(
        InhibitionOfReturn, "record_recall", lambda self, memory_id, **kwargs: seen.append(memory_id)
    )
    store = FileStateStore(tmp_path / "state.json")
    stage = InhibitionOfReturn(state_store=store)
    record_recall_events([stage], _two_results())
    assert seen == ["secret-1", "b"]
    assert _recorded_ids(store, stage) == []


@pytest.mark.parametrize("level", ["instance", "class"])
def test_replaced_store_update_sees_every_entry(tmp_path, monkeypatch, level):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    audit = []
    original = FileStateStore.update
    store = FileStateStore(tmp_path / "state.json")
    if level == "instance":
        def instance_update(key, updater):
            audit.append(key)
            return original(store, key, updater)

        store.update = instance_update
    else:
        def class_update(self, key, updater):
            audit.append(key)
            return original(self, key, updater)

        monkeypatch.setattr(FileStateStore, "update", class_update)
    stage = InhibitionOfReturn(state_store=store)
    record_recall_events([stage], _two_results())
    assert audit == [stage.STATE_KEY, stage.STATE_KEY]
    assert _recorded_ids(store, stage) == ["secret-1", "b"]


def test_proxy_stage_with_its_own_record_hook_is_called(tmp_path):
    from mnemostack.feedback import record_recall_events
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore

    class Proxy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def record_recall(self, memory_id, **kwargs):
            if not memory_id.startswith("secret"):
                self._inner.record_recall(memory_id, **kwargs)

    store = FileStateStore(tmp_path / "state.json")
    stage = InhibitionOfReturn(state_store=store)
    record_recall_events([Proxy(stage)], _two_results())
    assert _recorded_ids(store, stage) == ["b"]


def test_direct_stage_apply_after_a_pipeline_run_reads_state_again(tmp_path):
    from mnemostack.recall.pipeline import Pipeline, PipelineContext
    from mnemostack.recall.pipeline.stages import InhibitionOfReturn
    from mnemostack.recall.pipeline.state import FileStateStore
    from mnemostack.recall.recaller import RecallResult

    stage = InhibitionOfReturn(state_store=FileStateStore(tmp_path / "state.json"))
    context = PipelineContext(query="q")
    Pipeline([stage]).apply_with_context(context, [RecallResult("a", "x", 1.0)])
    stage.record_recall("a")
    after = stage.apply(context, [RecallResult("a", "x", 1.0)])
    assert "ior_penalty" in after[0].annotations
