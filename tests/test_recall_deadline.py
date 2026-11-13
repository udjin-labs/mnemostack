"""Shared recall deadlines with synchronous fake backends."""

from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest
import yaml

from mnemostack.config import Config
from mnemostack.llm.base import LLMResponse
from mnemostack.recall import Recaller, RecallResult, RecallTrace
from mnemostack.recall.answer import Answer, AnswerGenerator
from mnemostack.recall.deadline import Deadline
from mnemostack.recall.flow import recall_flow
from mnemostack.recall.pipeline import Pipeline, Stage
from mnemostack.recall.retry import retry_weak_recall


class _Arm:
    def __init__(self, name="vector", delay=0.0):
        self.name = name
        self.delay = delay
        self.calls = []

    def search(self, query, limit=10, filters=None):
        self.calls.append(query)
        time.sleep(self.delay)
        return [RecallResult(id=self.name, text="Alice lives in Paris", score=0.9)]


class _LLM:
    def __init__(self, delay=0.0, text="Paris\nCONFIDENCE: 0.9"):
        self.delay = delay
        self.text = text
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        time.sleep(self.delay)
        return LLMResponse(text=self.text)


class _Stage(Stage):
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0

    def apply(self, context, results):
        self.calls += 1
        time.sleep(self.delay)
        return list(reversed(results))


def test_disabled_keeps_existing_internal_call_arguments(monkeypatch):
    recaller = Recaller(retrievers=[_Arm()])
    trace = RecallTrace()
    # Legacy overrides must not receive new keyword arguments when disabled.
    monkeypatch.setattr(recaller, "_expanded_queries", lambda query: [query])
    recaller.query_expansion = True
    results = recaller.recall("q", trace=trace)
    assert [r.id for r in results] == ["vector"]
    assert trace.deadline is None
    assert "recall:deadline" not in trace.degraded


def test_slow_arm_keeps_fast_results_without_waiting_for_shutdown():
    recaller = Recaller(retrievers=[_Arm(), _Arm("slow", 0.6)], overall_deadline=0.16)
    trace = RecallTrace()
    start = time.monotonic()
    results = recaller.recall("q", trace=trace)
    elapsed = time.monotonic() - start
    assert 0.02 <= elapsed < 0.6
    assert [r.id for r in results] == ["vector"]
    assert trace.degraded == ["recall:deadline"]
    assert trace.notes == []
    assert "deadline=" not in repr(trace)
    assert "deadline" not in trace.to_dict()


def test_shared_budget_stops_pipeline_and_reranker_at_last_finished_stage():
    first, later = _Stage(0.3), _Stage()
    reranker = SimpleNamespace(rerank=lambda *a: pytest.fail("reranker started"))
    recaller = Recaller(retrievers=[_Arm(), _Arm("bm25")], overall_deadline=0.12)
    trace = RecallTrace()
    results = recall_flow(
        recaller, "q", pipeline=Pipeline([first, later]), reranker=reranker, trace=trace
    )
    assert [r.id for r in results] == ["bm25", "vector"]
    assert first.calls == 1 and later.calls == 0
    assert len(trace.stages) == 1
    assert trace.degraded.count("recall:deadline") == 1


def test_reranker_timeout_keeps_unmodified_input_list():
    finished = []

    def rerank(query, results):
        results.reverse()
        time.sleep(0.3)
        results.clear()
        finished.append(True)
        return results

    trace = RecallTrace()
    recaller = Recaller(retrievers=[_Arm(), _Arm("bm25")], overall_deadline=0.1)
    results = recall_flow(recaller, "q", reranker=SimpleNamespace(rerank=rerank), trace=trace)
    assert [r.id for r in results] == ["vector", "bm25"]
    time.sleep(0.3)
    assert finished and [r.id for r in results] == ["vector", "bm25"]
    assert trace.degraded.count("recall:deadline") == 1


def test_answer_llm_timeout_starts_neither_retry_nor_specificity(monkeypatch):
    llm = _LLM(0.3)
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=0.1)
    gen = AnswerGenerator(
        llm, recaller=recaller, retry_with_expansion=True, expansion_llm=llm,
        specificity_resolver=True, category_aware_prompts=True,
    )
    monkeypatch.setattr(
        gen, "_retry_with_expansion_answer", lambda **kw: pytest.fail("retry started")
    )
    monkeypatch.setattr(
        "mnemostack.recall.answer.resolve_specificity",
        lambda **kw: pytest.fail("rewrite started"),
    )
    trace = RecallTrace()
    start = time.monotonic()
    answer = gen.generate("where?", _Arm().search("q"), trace=trace)
    assert time.monotonic() - start < 0.4
    assert answer.error == "recall deadline exceeded"
    assert len(llm.calls) == 1 and trace.degraded == ["recall:deadline"]


def test_expired_answer_keeps_finished_draft_without_specificity(monkeypatch):
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=0.1)
    gen = AnswerGenerator(_LLM(), recaller=recaller, category_aware_prompts=True)
    draft = Answer(text="her country", confidence=0.3)

    def generate(**kwargs):
        time.sleep(0.16)
        return draft

    monkeypatch.setattr(gen, "_generate_single_prompt", generate)
    monkeypatch.setattr(
        "mnemostack.recall.answer.resolve_specificity",
        lambda **kw: pytest.fail("rewrite started"),
    )
    trace = RecallTrace()
    assert gen.generate("where?", _Arm().search("q"), trace=trace) is draft
    assert trace.degraded == ["recall:deadline"]


def test_nested_expansion_reuses_budget(monkeypatch):
    from mnemostack.recall.expansion import QueryExpander

    arm = _Arm(delay=0.16)
    recaller = Recaller(retrievers=[arm], overall_deadline=0.28)
    expander = QueryExpander(recaller, _LLM(), apply_to=lambda q: True)
    monkeypatch.setattr(expander, "generate_variants", lambda q: ["second", "third"])
    trace = RecallTrace()
    start = time.monotonic()
    results = recall_flow(expander, "first", trace=trace)
    assert time.monotonic() - start < 0.5
    assert arm.calls == ["first", "second"]
    assert [r.id for r in results] == ["vector"]
    assert trace.degraded.count("recall:deadline") == 1


def test_weak_retry_child_traces_reuse_original_deadline(monkeypatch):
    from mnemostack.recall.expansion import QueryExpander

    recaller = Recaller(retrievers=[_Arm(delay=0.16)], overall_deadline=0.26)
    trace = RecallTrace(deadline=Deadline(0.26))
    original = trace.deadline
    monkeypatch.setattr(QueryExpander, "generate_variants", lambda self, q: ["one", "two"])
    results, retried = retry_weak_recall(
        recaller, "q", 10, llm=_LLM(), results=[], trace=trace
    )
    assert retried and results and trace.deadline is original
    assert trace.degraded.count("recall:deadline") == 1
    assert recaller.retrievers[0].calls == ["one", "two"]


def test_expired_request_runs_guard_before_returning_empty(monkeypatch):
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=0.1)
    trace = RecallTrace(deadline=Deadline(0.001))
    time.sleep(0.005)
    guards = []
    monkeypatch.setattr(recaller, "_guard_tenantless_recall", lambda: guards.append(True))
    assert recaller.recall("q", trace=trace) == []
    assert guards and not recaller.retrievers[0].calls
    assert trace.degraded == ["recall:deadline"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_deadlines_rejected_by_constructor_and_config(value, monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSTACK_RECALL_OVERALL_DEADLINE", raising=False)
    with pytest.raises(ValueError, match="recall.overall_deadline"):
        Recaller(overall_deadline=value)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"recall": {"overall_deadline": value}}))
    with pytest.raises(ValueError, match="recall.overall_deadline"):
        Config.load(config)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "true", "soon"])
def test_invalid_deadline_environment_rejected(value, monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSTACK_RECALL_OVERALL_DEADLINE", value)
    with pytest.raises(ValueError, match="recall.overall_deadline"):
        Config.load(tmp_path / "missing.yaml")


class _Built(Exception):
    pass


@pytest.mark.parametrize("source", ["file", "environment"])
@pytest.mark.parametrize("surface", ["search", "answer", "serve", "http", "mcp-serve", "mcp"])
def test_config_reaches_each_recaller_factory(source, surface, monkeypatch, tmp_path):
    import mnemostack.cli as cli
    import mnemostack.mcp as mcp_package
    import mnemostack.mcp.server as mcp_server
    import mnemostack.server as http_server

    path = tmp_path / "config.yaml"
    path.write_text("recall:\n  overall_deadline: 0.75\n")
    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(path))
    monkeypatch.delenv("MNEMOSTACK_RECALL_OVERALL_DEADLINE", raising=False)
    if source == "environment":
        monkeypatch.setenv("MNEMOSTACK_RECALL_OVERALL_DEADLINE", "0.5")
    expected = 0.5 if source == "environment" else 0.75
    captured = []

    def capture(**kwargs):
        captured.append(Recaller(**kwargs))
        raise _Built

    provider = SimpleNamespace(dimension=3)
    for module in (cli, http_server, mcp_server):
        monkeypatch.setattr(module, "get_provider", lambda *a, **kw: provider)
        monkeypatch.setattr(module, "VectorStore", lambda **kw: SimpleNamespace())
        monkeypatch.setattr(module, "VectorRetriever", lambda **kw: _Arm())
        monkeypatch.setattr(module, "TemporalRetriever", lambda **kw: _Arm("temporal"))
        monkeypatch.setattr(module, "Recaller", capture)
    monkeypatch.setattr(http_server, "_make_probe_client", lambda *a: SimpleNamespace())
    args = cli.build_parser().parse_args(
        [surface if surface in {"search", "answer", "serve", "mcp-serve"} else "search"]
        + (["q"] if surface not in {"serve", "mcp-serve"} else [])
    )

    original_build_mcp = mcp_server.build_server

    def build_mcp(**kwargs):
        server = original_build_mcp(**kwargs)
        tool = asyncio.run(server.get_tool("mnemostack_search"))
        run_recall = inspect.getclosurevars(tool.fn).nonlocals["_run_recall"]
        get_recaller = inspect.getclosurevars(run_recall).nonlocals["_get_recaller"]
        return get_recaller()

    with pytest.raises(_Built):
        if surface in {"search", "answer"}:
            cli._build_recaller(args, provider, SimpleNamespace())
        elif surface == "serve":
            cli.cmd_serve(args)
        elif surface == "http":
            http_server.build_app(http_server.ServerConfig.from_env())
        elif surface == "mcp-serve":
            monkeypatch.setattr(mcp_package, "build_server", build_mcp)
            cli.cmd_mcp_serve(args)
        else:
            monkeypatch.setattr(mcp_server, "build_server", build_mcp)
            mcp_server.main()
    assert len(captured) == 1 and captured[0].overall_deadline == expected


def test_expansion_llm_wait_is_bounded(monkeypatch):
    llm = _LLM(0.3, "one\ntwo")
    arm = _Arm()
    recaller = Recaller(
        retrievers=[arm], query_expansion=True, expansion_llm=llm, overall_deadline=0.1
    )
    trace = RecallTrace()
    start = time.monotonic()
    assert recaller.recall("q", trace=trace) == []
    assert time.monotonic() - start < 0.4
    assert len(llm.calls) == 1 and arm.calls == []
    assert trace.degraded == ["recall:deadline"]


def test_vector_fallback_timeout_keeps_finished_results(monkeypatch):
    recaller = Recaller(retrievers=[_Arm("bm25")], overall_deadline=0.1)
    completed = []

    def fallback(query, results, **kwargs):
        results[0].sources.append("abandoned")
        results[0].payload["raw_vector_score"] = 99
        time.sleep(0.3)
        completed.append(True)
        return []

    monkeypatch.setattr(recaller, "_maybe_apply_fallback", fallback)
    trace = RecallTrace()
    results = recaller.recall("q", trace=trace)
    assert [r.id for r in results] == ["bm25"]
    time.sleep(0.3)
    assert completed and [r.id for r in results] == ["bm25"]
    assert "abandoned" not in results[0].sources
    assert "raw_vector_score" not in results[0].payload
    assert trace.degraded == ["recall:deadline"]


def test_answer_reuses_retrieval_deadline_without_starting_llm():
    llm = _LLM()
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=0.1)
    trace = RecallTrace()
    memories = recall_flow(recaller, "q", trace=trace)
    original = trace.deadline
    time.sleep(0.15)
    answer = AnswerGenerator(llm, recaller=recaller).generate("q", memories, trace=trace)
    assert answer.error == "recall deadline exceeded"
    assert llm.calls == [] and trace.deadline is original
    assert trace.degraded == ["recall:deadline"]


def test_list_extraction_stops_before_next_batch(monkeypatch):
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=0.12)
    llm = _LLM(text='{"items": ["Paris"]}')
    gen = AnswerGenerator(
        llm, recaller=recaller, category_aware_prompts=True,
        list_extract_mode=True, list_extract_batch_size=1,
    )
    parse = gen._parse_extracted_items

    def slow_parse(raw):
        items = parse(raw)
        time.sleep(0.2)
        return items

    monkeypatch.setattr(gen, "_parse_extracted_items", slow_parse)
    memories = [_Arm().search("q")[0], _Arm("other").search("q")[0]]
    trace = RecallTrace()
    answer = gen.generate("which cities?", memories, category="list", trace=trace)
    assert answer.text == "Paris" and answer.error == "list extraction incomplete"
    assert len(llm.calls) == 1 and trace.degraded == ["recall:deadline"]


def test_specificity_rewrite_wait_is_bounded():
    class _RewriteLLM(_LLM):
        def generate(self, prompt, **kwargs):
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return LLMResponse(text="her country\nCONFIDENCE: 0.9")
            time.sleep(0.3)
            return LLMResponse(text="France")

    llm = _RewriteLLM()
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=0.1)
    gen = AnswerGenerator(
        llm, recaller=recaller, category_aware_prompts=True, specificity_resolver=True
    )
    trace = RecallTrace()
    start = time.monotonic()
    answer = gen.generate("where?", _Arm().search("q"), trace=trace)
    assert time.monotonic() - start < 0.4
    assert answer.text == "her country" and len(llm.calls) == 2
    assert trace.degraded == ["recall:deadline"]


@pytest.mark.parametrize("value", [None, True, "1", 0, -1, float("nan"), float("inf")])
def test_mock_recaller_runs_flow_without_deadline(value):
    from unittest.mock import MagicMock

    recaller = MagicMock()
    recaller.apply_vector_floor_after_rerank = None
    if value is not None:
        recaller.overall_deadline = value
    recaller.recall.return_value = _Arm().search("q")
    trace = RecallTrace()
    results = recall_flow(recaller, "q", trace=trace)
    assert [r.id for r in results] == ["vector"]
    assert trace.deadline is None
    assert trace.degraded == []
    assert recaller.recall.call_args.kwargs["trace"] is trace


@pytest.mark.parametrize("seconds", [None, 1.0])
def test_expansion_accepts_trace_without_deadline(seconds):
    from mnemostack.recall.expansion import QueryExpander

    class Trace:
        __slots__ = ("retrievers", "degraded", "notes", "fused")

        def __init__(self):
            self.retrievers, self.degraded, self.notes, self.fused = [], [], [], []

    calls = []

    def recall(query, *, trace, **kwargs):
        calls.append(query)
        trace.retrievers.append(SimpleNamespace(name="fake", query=query))
        return _Arm().search(query)

    recaller = SimpleNamespace(recall=recall, overall_deadline=seconds)
    expander = QueryExpander(
        recaller, _LLM(text="only variant"),
        include_original=False, apply_to=lambda _: True,
    )
    trace = Trace()
    results = expander.recall("original", trace=trace)
    assert [r.id for r in results] == ["vector"]
    assert calls == ["only variant"]
    assert [(r.name, r.query) for r in trace.retrievers] == [
        ("fake:expansion", "only variant")
    ]
    assert not hasattr(trace, "deadline")
    assert trace.degraded == []


def test_mock_trace_does_not_supply_deadline():
    from unittest.mock import MagicMock

    assert Recaller(retrievers=[_Arm()]).recall("q", trace=MagicMock())


def test_deadline_arm_is_daemon_and_keeps_attribution():
    import threading

    seen = []

    class Arm(_Arm):
        def search(self, *args, **kwargs):
            seen.append(threading.current_thread().daemon)
            return super().search(*args, **kwargs)

    trace = RecallTrace()
    results = Recaller(retrievers=[Arm("first"), Arm("second")], overall_deadline=1).recall(
        "q", trace=trace
    )
    assert seen == [True, True]
    assert [r.id for r in results] == ["first", "second"]
    assert [r.name for r in trace.retrievers] == ["first", "second"]


def test_worker_cap_runs_inline_then_releases_slot(monkeypatch):
    import threading

    import mnemostack.recall.deadline as module

    release = threading.Event()
    finished = threading.Event()
    time.sleep(0.35)
    monkeypatch.setattr(module, "MAX_ABANDONED_WORKERS", 1)

    def blocked():
        release.wait(0.5)
        finished.set()

    try:
        assert Deadline(0.1).run(blocked, "fallback") == "fallback"
        trace = RecallTrace()
        arm = _Arm()
        assert [r.id for r in Recaller(retrievers=[arm], overall_deadline=1).recall("q", trace=trace)] == ["vector"]
        assert arm.calls == ["q"] and trace.degraded == []
        caller = threading.get_ident()
        assert Deadline(1).run(threading.get_ident, None) == caller
        assert module._abandoned_workers == 1
    finally:
        release.set()
        assert finished.wait(1)
    time.sleep(0.01)
    assert Deadline(1).run(lambda: "ok", "fallback") == "ok"


def test_reranker_abandoned_trace_cannot_pass_required_gate():
    def rerank(query, results):
        time.sleep(0.3)
        return list(results)

    trace = RecallTrace()
    recall_flow(Recaller(retrievers=[_Arm()], overall_deadline=0.1), "q",
                reranker=SimpleNamespace(rerank=rerank), trace=trace)
    time.sleep(0.1)
    assert trace.post_rerank is None
    assert trace.degraded == ["recall:deadline"]


def test_untraced_legacy_pipeline_accepts_deadline():
    class Legacy(Pipeline):
        def apply(self, query, results, *, as_of=None, include_invalidated=False,
                  tenant=None, recaller=None):
            return results

    assert recall_flow(Recaller(retrievers=[_Arm()], overall_deadline=1), "q",
                       pipeline=Legacy([]))


def test_inference_subrecalls_leave_main_trace_unchanged(monkeypatch):
    monkeypatch.setattr("mnemostack.recall.answer.decompose_query", lambda *a: ["one", "two"])
    recaller = Recaller(retrievers=[_Arm()], overall_deadline=1)
    trace = RecallTrace()
    memories = recaller.recall("q", trace=trace)
    original = list(trace.retrievers), list(trace.fused)
    gen = AnswerGenerator(_LLM(), recaller=recaller)
    gen._retry_inference_answer("q", memories, Answer(text="draft", confidence=0.1),
                                "{context}\n{query}", None, trace=trace)
    assert (trace.retrievers, trace.fused) == original


@pytest.mark.parametrize("slow", ["embedding", "search"])
def test_legacy_network_waits_are_bounded(monkeypatch, slow):
    def embed(*args):
        if slow == "embedding":
            time.sleep(0.3)
        return [1.0]

    def search(*args, **kwargs):
        if slow == "search":
            time.sleep(0.3)
        return []

    recaller = Recaller(embedding_provider=SimpleNamespace(),
                        vector_store=SimpleNamespace(search=search), overall_deadline=0.1)
    monkeypatch.setattr(recaller, "_ensure_space_compat", lambda **kw: None)
    monkeypatch.setattr("mnemostack.recall.recaller.embed_query_via", embed)
    trace = RecallTrace()
    start = time.monotonic()
    assert recaller.recall("q", trace=trace) == []
    assert time.monotonic() - start < 0.4
    assert trace.degraded == ["recall:deadline"]


@pytest.mark.parametrize("slow", ["queries", "documents", "search"])
def test_answer_expansion_network_waits_keep_draft(monkeypatch, slow):
    def embed_queries(*args):
        time.sleep(0.3 if slow == "queries" else 0)
        return [[1.0]]

    def embed_documents(*args):
        time.sleep(0.3 if slow == "documents" else 0)
        return [[1.0]]

    def search(*args, **kwargs):
        time.sleep(0.3 if slow == "search" else 0)
        return _Arm().search("q")

    monkeypatch.setattr("mnemostack.recall.answer.embed_queries_via", embed_queries)
    monkeypatch.setattr("mnemostack.recall.answer.embed_documents_via", embed_documents)
    gen = AnswerGenerator(_LLM(), recaller=SimpleNamespace(embedding=object(), search_many=search),
                          expansion_llm=_LLM(text="one\ntwo\nthree"))
    draft = Answer(text="draft", confidence=0.1)
    trace = RecallTrace(deadline=Deadline(0.1))
    start = time.monotonic()
    answer, _ = gen._retry_with_expansion_answer("q", [], draft, "{context}\n{query}",
                                               None, trace=trace)
    assert answer is draft and time.monotonic() - start < 0.4
    assert trace.degraded == ["recall:deadline"]


def test_source_filtered_recaller_preserves_deadline():
    from mnemostack.synthesis import _filter_recaller

    recaller = Recaller(retrievers=[_Arm(delay=0.3)], overall_deadline=0.1)
    clone = _filter_recaller(recaller, {"vector"})
    assert clone.overall_deadline == 0.1
    trace = RecallTrace()
    assert clone.recall("q", trace=trace) == []
    assert trace.degraded == ["recall:deadline"]


def test_weak_retry_without_trace_shares_one_budget(monkeypatch):
    from mnemostack.recall.expansion import QueryExpander

    deadlines = []
    arm = _Arm(delay=0.16)
    recaller = Recaller(retrievers=[arm], overall_deadline=0.26)
    original = recaller.recall

    def recall(*args, **kwargs):
        deadlines.append(kwargs["trace"].deadline)
        return original(*args, **kwargs)

    monkeypatch.setattr(recaller, "recall", recall)
    monkeypatch.setattr(QueryExpander, "generate_variants", lambda *a: ["one", "two"])
    retry_weak_recall(recaller, "q", 10, llm=_LLM(), results=[])
    assert len(deadlines) == 2 and deadlines[0] is deadlines[1]


def test_completed_slow_stage_only_tags_skipped_work():
    for skip in (False, True):
        trace = RecallTrace()
        stages = [_Stage(0.16)] + ([_Stage()] if skip else [])
        recall_flow(Recaller(retrievers=[_Arm()], overall_deadline=0.1), "q",
                    pipeline=Pipeline(stages), trace=trace)
        assert ("recall:deadline" in trace.degraded) is skip


def test_cli_required_rerank_rejects_timed_out_worker(monkeypatch):
    import argparse

    import mnemostack.cli as cli

    def rerank(query, results):
        time.sleep(0.3)
        return list(results)

    monkeypatch.setattr(cli, "build_full_pipeline", lambda **kw: Pipeline([]))
    monkeypatch.setattr(cli, "get_llm", lambda *a, **kw: _LLM())
    monkeypatch.setattr(cli, "Reranker", lambda *a, **kw: SimpleNamespace(rerank=rerank))
    args = argparse.Namespace(require_rerank=True, raw=False, llm="fake", memgraph_uri=None)
    trace = RecallTrace()
    with pytest.raises(SystemExit) as exc:
        cli._recall_for_cli(args, Recaller(retrievers=[_Arm()], overall_deadline=0.1),
                            "q", 10, trace=trace)
    assert exc.value.code == 2
    time.sleep(0.1)
    assert trace.post_rerank is None


def test_abandoned_reranker_does_not_record_fallback_counter(monkeypatch):
    marks = []
    monkeypatch.setattr("mnemostack.recall.trace.counter", lambda *a, **kw: marks.append(kw))

    def rerank(query, results):
        time.sleep(0.3)
        raise RuntimeError("unavailable")

    trace = RecallTrace()
    recall_flow(Recaller(retrievers=[_Arm()], overall_deadline=0.1), "q",
                reranker=SimpleNamespace(rerank=rerank), trace=trace)
    time.sleep(0.1)
    assert trace.post_rerank is None and trace.degraded == ["recall:deadline"]
    assert marks == [{"labels": {"reason": "recall:deadline"}}]


def test_healthy_concurrent_recalls_do_not_consume_abandoned_cap():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(24)

    def recall(_):
        barrier.wait(timeout=5)
        trace = RecallTrace()
        results = Recaller(
            retrievers=[_Arm(str(i), 0.2) for i in range(4)], overall_deadline=5,
        ).recall("q", trace=trace)
        assert trace.degraded == []
        return len(results)

    with ThreadPoolExecutor(max_workers=24) as executor:
        assert list(executor.map(recall, range(24))) == [4] * 24


def test_abandoned_cap_preserves_healthy_arm_and_drains(monkeypatch):
    from threading import Event

    import mnemostack.recall.deadline as module

    time.sleep(0.2)
    monkeypatch.setattr(module, "MAX_ABANDONED_WORKERS", 2)
    release = Event()
    futures = [module.submit_deadline_work(lambda: release.wait(1)) for _ in range(2)]
    try:
        assert all(future.abandon() for future in futures)
        assert module._abandoned_workers == 2
        recaller = Recaller(retrievers=[_Arm(), _Arm("other")], overall_deadline=1)
        assert {r.id for r in recaller.recall("q")} == {"vector", "other"}
        assert module._abandoned_workers == 2
    finally:
        release.set()
        for future in futures:
            future.result(timeout=1)
    time.sleep(0.01)
    assert module._abandoned_workers == 0


@pytest.mark.parametrize("error", [RuntimeError, BaseException])
def test_abandoned_raising_worker_drains(monkeypatch, error):
    import threading

    import mnemostack.recall.deadline as module

    release = threading.Event()

    def fail():
        release.wait(1)
        raise error("failed")

    time.sleep(0.2)
    assert Deadline(0.01).run(fail, "fallback") == "fallback"
    assert module._abandoned_workers == 1
    release.set()
    time.sleep(0.12)
    assert module._abandoned_workers == 0


@pytest.mark.parametrize("started", [False, True])
def test_thread_start_failure_does_not_change_abandoned_count(monkeypatch, started):
    import mnemostack.recall.deadline as module

    original = module.Thread.start

    def start(thread):
        if started:
            original(thread)
            thread.join()
        raise RuntimeError("start failed")

    time.sleep(0.2)
    monkeypatch.setattr(module.Thread, "start", start)
    with pytest.raises(RuntimeError, match="start failed"):
        Deadline(1).run(lambda: 1, None)
    assert module._abandoned_workers == 0


def test_timely_vector_fallback_preserves_floor_and_identity(monkeypatch):
    outputs = []
    for deadline in (None, 5):
        originals = [RecallResult(id=f"m{i}", text=f"t{i}", score=0.9) for i in range(1, 4)]
        bm25 = SimpleNamespace(name="bm25", search=lambda *a, originals=originals, **kw: originals)
        vector = SimpleNamespace(name="vector", search=lambda *a, **kw: [])
        recaller = Recaller(retrievers=[bm25, vector], vector_floor=1, overall_deadline=deadline)
        monkeypatch.setattr(recaller, "_vector_fallback_hits", lambda *a, **kw: [
            RecallResult(id="m9", text="t9", score=0.95, sources=["vector"], payload={"raw_vector_score": 0.95}),
            RecallResult(id="m3", text="t3", score=0.8, sources=["vector"], payload={"raw_vector_score": 0.8}),
        ])

        original_recall = recaller.recall

        def recall(*args, original_recall=original_recall, originals=originals, **kwargs):
            results = original_recall(*args, **kwargs)
            for result in results:
                if result.id in {"m1", "m2", "m3"}:
                    assert result is originals[int(result.id[1:]) - 1]
            return results

        monkeypatch.setattr(recaller, "recall", recall)

        def rerank(query, results, originals=originals):
            for result in results:
                if result.id in {"m1", "m2", "m3"}:
                    # Reranking isolates its own inputs when the deadline is on.
                    assert result.payload == originals[int(result.id[1:]) - 1].payload
            return [r for r in results if r.id == "m1"]

        results = recall_flow(recaller, "q", limit=3, reranker=SimpleNamespace(rerank=rerank))
        assert "vector" in originals[2].sources
        assert originals[2].payload["raw_vector_score"] == 0.8
        outputs.append([r.id for r in results])
    assert outputs == [["m1", "m3"], ["m1", "m3"]]


@pytest.mark.parametrize("fallback", [False, True])
def test_timely_reranker_trace_matches_disabled(fallback):
    traces = []
    for seconds in (None, 5):
        trace = RecallTrace()
        recaller = Recaller(retrievers=[_Arm()], overall_deadline=seconds)
        reranker = SimpleNamespace(
            rerank=lambda q, results: results if fallback else list(reversed(results)),
            fallback_keeps_input_object=fallback,
        )
        results = recall_flow(recaller, "q", reranker=reranker, trace=trace)
        traces.append(([r.id for r in results], trace.post_rerank, trace.degraded))
    assert traces[0] == traces[1]


def test_timely_inference_sub_recall_trace_and_counter_match_disabled(monkeypatch):
    import mnemostack.recall.trace as trace_module

    monkeypatch.setattr("mnemostack.recall.answer.decompose_query", lambda *a: ["sub"])
    observations = []
    for seconds in (None, 5):
        marks = []
        monkeypatch.setattr(trace_module, "counter", lambda *a, marks=marks, **kw: marks.append((a, kw)))

        class RecallerFake:
            def recall(self, query, **kwargs):
                if kwargs.get("trace") is not None:
                    kwargs["trace"].mark("retriever:bm25:failed")
                return _Arm().search(query)

        trace = RecallTrace(deadline=Deadline(seconds) if seconds else None)
        generator = AnswerGenerator(_LLM(), recaller=RecallerFake())
        answer, results = generator._retry_inference_answer(
            query="q", memories=[], draft=Answer(text="unknown", confidence=0.1),
            prompt_template="{context} {query}", recall_filters=None, trace=trace,
        )
        observations.append((answer.text, [r.id for r in results], trace.to_dict(), marks))
    assert observations[0] == observations[1]


@pytest.mark.parametrize("kind", ["options_off", "confident", "no_placeholder", "strong_recall", "no_llm"])
def test_expired_deadline_does_not_tag_inapplicable_steps(monkeypatch, kind):
    trace = RecallTrace(deadline=Deadline(0.001))
    time.sleep(0.005)
    if kind in {"strong_recall", "no_llm"}:
        results = _Arm().search("q") if kind == "strong_recall" else []
        retry_weak_recall(Recaller(), "q", 10, llm=_LLM() if results else None,
                          results=results, trace=trace)
    else:
        generator = AnswerGenerator(
            _LLM(), recaller=Recaller(), retry_with_expansion=kind == "confident",
            specificity_resolver=kind == "no_placeholder", category_aware_prompts=True,
        )
        monkeypatch.setattr(generator, "_generate_single_prompt", lambda **kw: Answer(text="Paris", confidence=0.9))
        generator.generate("q", _Arm().search("q"), trace=trace)
    assert "recall:deadline" not in trace.degraded


def test_expired_legacy_recall_skips_remote_guards(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("remote work started")

    recaller = Recaller(embedding_provider=SimpleNamespace(),
                        vector_store=SimpleNamespace(search=unexpected))
    trace = RecallTrace(deadline=Deadline(1))
    trace.deadline._end = 0
    guards = []
    monkeypatch.setattr(recaller, "_guard_tenantless_recall", lambda: guards.append(True))
    monkeypatch.setattr(recaller, "_ensure_space_compat", unexpected)
    assert recaller.recall("q", trace=trace) == []
    assert guards == [True]
    assert trace.degraded == ["recall:deadline"]


def test_async_answer_forwards_fresh_trace(monkeypatch):
    recaller = Recaller(overall_deadline=1)
    trace = RecallTrace()
    generator = AnswerGenerator(_LLM(), recaller=recaller)

    def expired_prompt(**kwargs):
        trace.deadline._end = 0
        return Answer(text="unknown", confidence=0.1)

    async def to_thread(call):
        return call()

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    generator.retry_with_expansion = True
    monkeypatch.setattr(generator, "_generate_single_prompt", expired_prompt)
    asyncio.run(generator.generate_async("q", _Arm().search("q"), trace=trace))
    assert trace.degraded == ["recall:deadline"]


def test_nonempty_extraction_does_not_tag_unused_retry(monkeypatch):
    generator = AnswerGenerator(_LLM(), list_extract_mode=True, list_finalize="verbatim")
    trace = RecallTrace(deadline=Deadline(1))

    def extract(*args, **kwargs):
        trace.deadline._end = 0
        return ["Paris"], []

    monkeypatch.setattr(generator, "_extract_items_over_pool", extract)
    assert generator._generate_list_extract("q", [], trace=trace).text == "Paris"
    assert trace.degraded == []


def test_empty_variants_do_not_tag_unused_retry(monkeypatch):
    from mnemostack.recall.expansion import QueryExpander

    trace = RecallTrace(deadline=Deadline(1))

    def variants(*args):
        trace.deadline._end = 0
        return []

    monkeypatch.setattr(QueryExpander, "generate_variants", variants)
    assert retry_weak_recall(Recaller(), "q", 10, llm=_LLM(), results=[], trace=trace) == ([], False)
    assert "recall:deadline" not in trace.degraded
    assert "weak_retry:no_variants" in trace.notes


@pytest.mark.parametrize("raises", [False, True])
def test_refused_arms_run_together(monkeypatch, raises):
    from threading import Barrier

    barrier = Barrier(3)

    class Arm(_Arm):
        def search(self, *args, **kwargs):
            barrier.wait(timeout=0.7)
            if raises and self.name == "second":
                raise RuntimeError("unavailable")
            results = super().search(*args, **kwargs)
            results[0].sources = [self.name]
            return results

    monkeypatch.setattr("mnemostack.recall.recaller.submit_deadline_work", lambda call: None)
    trace = RecallTrace()
    start = time.monotonic()
    results = Recaller(retrievers=[Arm("first"), Arm("second"), Arm("third")],
                       overall_deadline=5).recall("q", trace=trace)
    assert time.monotonic() - start < 0.6
    assert {r.id for r in results} == ({"first", "third"} if raises else {"first", "second", "third"})
    assert [r.name for r in trace.retrievers] == ["first", "second", "third"]
    assert all(r.sources == [r.id] for r in results)
    assert ("retriever:second:failed" in trace.degraded) is raises
    assert "recall:deadline" not in trace.degraded


def test_double_abandon_counts_once_and_drains():
    from threading import Event

    import mnemostack.recall.deadline as module

    time.sleep(0.3)
    release = Event()
    future = module.submit_deadline_work(lambda: release.wait(1))
    try:
        assert future.abandon() is True
        assert future.abandon() is True
        assert module._abandoned_workers == 1
    finally:
        release.set()
        future.result(timeout=1)
    time.sleep(0.01)
    assert future.abandon() is True
    assert module._abandoned_workers == 0


def test_paraphrase_timeout_is_not_no_variants(monkeypatch):
    counters = []
    monkeypatch.setattr("mnemostack.recall.retry.counter", lambda name, *a, **kw: counters.append(name))
    trace = RecallTrace(deadline=Deadline(0.05))
    assert retry_weak_recall(Recaller(), "q", 10, llm=_LLM(delay=0.2),
                             results=[], trace=trace) == ([], False)
    assert trace.degraded == ["recall:deadline"]
    assert "weak_retry:no_variants" not in trace.notes
    assert "mnemostack.recall.weak_retry_no_variants" not in counters


@pytest.mark.parametrize("mode", ["slow", "fast", "space_error", "other_error"])
def test_legacy_space_check_deadline(monkeypatch, mode):
    from mnemostack.embeddings.roles import EmbeddingSpaceError
    from mnemostack.vector.qdrant import Hit

    observations = []
    error = EmbeddingSpaceError("incompatible") if mode == "space_error" else RuntimeError("failed")
    for seconds in (None, 0.05):
        calls = []

        def check(**kwargs):
            if mode == "slow":
                time.sleep(0.3)
            elif mode.endswith("error"):
                raise error

        def embed(*args, calls=calls):
            calls.append("embed")
            return [1.0]

        def search(*args, calls=calls, **kwargs):
            calls.append("search")
            return [Hit(id="m1", score=0.9, payload={"text": "Paris"})]

        recaller = Recaller(embedding_provider=SimpleNamespace(),
                            vector_store=SimpleNamespace(search=search), overall_deadline=seconds)
        monkeypatch.setattr(recaller, "_ensure_space_compat", check)
        monkeypatch.setattr("mnemostack.recall.recaller.embed_query_via", embed)
        trace = RecallTrace()
        if mode.endswith("error"):
            with pytest.raises(type(error)) as exc:
                recaller.recall("q", trace=trace)
            assert exc.value is error
            assert calls == [] and trace.degraded == []
            continue
        start = time.monotonic()
        results = recaller.recall("q", trace=trace)
        if mode == "slow" and seconds is not None:
            assert time.monotonic() - start < 0.2
            assert results == [] and calls == []
            assert trace.degraded == ["recall:deadline"]
        else:
            assert calls == ["embed", "search"] and trace.degraded == []
            observations.append(results)
    if mode == "fast":
        assert observations[0] == observations[1]


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("expired", [False, True])
def test_flow_checks_deadline_before_pipeline(traced, expired):
    calls = []
    traces = []
    original = _Arm().search("q")

    def recall(query, **kwargs):
        trace = kwargs["trace"]
        traces.append(trace)
        if expired:
            trace.deadline._end = 0
        return original

    class Legacy(Pipeline):
        def apply(self, query, results, *, as_of=None, include_invalidated=False,
                  tenant=None, recaller=None):
            calls.append(True)
            return results

    class Traced(Pipeline):
        def apply(self, query, results, **kwargs):
            calls.append(kwargs["trace"])
            return results

    recaller = SimpleNamespace(recall=recall, overall_deadline=1)
    trace = RecallTrace() if traced else None
    results = recall_flow(recaller, "q", pipeline=Traced([]) if traced else Legacy([]), trace=trace)
    assert results == original
    assert bool(calls) is not expired
    assert traces[0].degraded == (["recall:deadline"] if expired else [])
