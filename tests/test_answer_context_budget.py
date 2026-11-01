"""Budgeted evidence must survive the actual provider request, offline."""

import asyncio
import io
import json
import os
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mnemostack.llm.ollama import OllamaLLM
from mnemostack.recall import AnswerGenerator, RecallResult
from mnemostack.recall.tokens import estimate_tokens

WITNESS = "Evidence: violet key."
TEXT = "Background. " * 45 + WITNESS


def memory(text=TEXT, rid=1, **payload):
    return RecallResult(
        id=rid, text=text, score=1.0,
        payload={"source": "document.txt", "timestamp": "2026-01-02T12:30:00Z", **payload},
        sources=["vector"],
    )


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    import dotenv

    for key in tuple(os.environ):
        if key.startswith(("MNEMOSTACK_", "OLLAMA_", "QDRANT_", "GEMINI_", "GOOGLE_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda *a, **kw: "")
    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    state = tmp_path / "state/mnemostack/server-state.json"
    state.parent.mkdir(parents=True)
    state.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("FASTMCP_CHECK_FOR_UPDATES", "off")

    def no_network(*a, **kw):
        pytest.fail("Formatter regressions must not connect to services")

    monkeypatch.setattr(socket.socket, "connect", no_network)


@pytest.fixture
def wire(monkeypatch):
    bodies = []
    replies = []

    def capture(request, **kwargs):
        body = json.loads(request.data.decode("utf-8"))
        bodies.append(body)
        reply = replies.pop(0) if replies else "violet key\nCONFIDENCE: 0.9"
        return io.BytesIO(json.dumps({"response": reply, "eval_count": 5}).encode())

    monkeypatch.setattr("urllib.request.urlopen", capture)
    llm = OllamaLLM(model="fixture", host="http://offline.invalid")
    return llm, bodies, replies


def generator(llm, **kwargs):
    return AnswerGenerator(llm, specificity_resolver=False, **kwargs)


@pytest.mark.parametrize("category", ["general", "temporal", "list", "count", "multihop", "inference", "adversarial"])
def test_budgeted_categories_keep_evidence_in_wire_prompt(wire, category):
    llm, bodies, _ = wire
    m = memory(author="Ada", parent_id="parent", char_start=0, char_end=len(TEXT))
    gen = generator(llm, context_fields=("author",))
    answer = gen.generate("lookup evidence", [m], category=category,
                          token_budget=estimate_tokens(TEXT))
    assert len(bodies) == 1
    assert TEXT in bodies[0]["prompt"]
    assert "[1] [2026-01-02 12:30] (document.txt) author=Ada" in bodies[0]["prompt"]
    assert bodies[0]["options"]["num_predict"] == 200
    assert answer.context_memories == [m]
    assert answer.context_tokens_estimate == estimate_tokens(TEXT)
    assert answer.sources == ["document.txt"]
    assert m.payload["parent_id"] == "parent" and m.payload["char_end"] == len(TEXT)


@pytest.mark.parametrize("offset,expected", [(-1, []), (0, [1]), (1, [1])])
def test_exact_budget_prefix_and_oversized_first_record(wire, offset, expected):
    llm, bodies, _ = wire
    first, second = memory(), memory("next", rid=2, source="second.txt")
    answer = generator(llm).generate("lookup", [first, second],
                                     token_budget=len(TEXT) + offset, token_counter=len)
    assert [m.id for m in answer.context_memories] == expected
    assert first.text == TEXT and second.text == "next"
    if expected:
        assert TEXT in bodies[0]["prompt"] and "second.txt" not in bodies[0]["prompt"]
        assert answer.sources == ["document.txt"]
        assert answer.context_tokens_estimate == len(TEXT)
    else:
        assert not bodies and answer.sources == [] and answer.confidence == 0


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_fails_before_provider(wire, budget):
    llm, bodies, _ = wire
    with pytest.raises(ValueError, match="positive"):
        generator(llm).generate("lookup", [memory()], token_budget=budget)
    assert not bodies


def test_none_keeps_legacy_preview_and_max_memories(wire):
    llm, bodies, _ = wire
    huge = memory("z" * 100_000 + WITNESS)
    generator(llm, max_memories=1).generate("lookup", [huge, memory(rid=2)], token_budget=None)
    assert "z" * 400 in bodies[0]["prompt"]
    assert "z" * 401 not in bodies[0]["prompt"]
    assert WITNESS not in bodies[0]["prompt"] and "[2]" not in bodies[0]["prompt"]
    assert AnswerGenerator._format_context([huge]).endswith("z" * 400)


def test_oversized_budgeted_record_is_not_partially_rendered(wire):
    llm, bodies, _ = wire
    answer = generator(llm).generate("lookup", [memory("x" * 100_000), memory()],
                                     token_budget=4000)
    assert not bodies
    assert answer.context_memories == []  # Stop at oversized first hit, not best-effort fill.


@pytest.mark.parametrize("text", ["", " \n ", "line one\nline two", "界😀 е\u0301 " * 90 + WITNESS])
def test_normalization_and_unicode_with_custom_template(wire, text):
    llm, bodies, _ = wire
    gen = generator(llm, prompt_template="{context}\nQUESTION:{query}", context_fields=("tags", "missing", "blob"))
    gen.generate("lookup", [memory(text, source="path/файл (v2).txt", tags=["a", "b"], blob="x" * 105)],
                 token_budget=max(1, len(text)), token_counter=len)
    assert bodies[0]["prompt"] == (
        "[1] [2026-01-02 12:30] (path/файл (v2).txt) tags=a, b blob=" + "x" * 100 + "… "
        + text.strip().replace("\n", " ") + "\nQUESTION:lookup"
    )


def test_budgeted_order_and_sources_are_unchanged(wire):
    llm, bodies, _ = wire
    rows = [memory(rid=i, source=f"source-{i}") for i in range(7)]
    answer = generator(llm).generate("lookup", rows, token_budget=len(TEXT) * 7, token_counter=len)
    assert answer.context_memories == rows
    assert answer.sources == [f"source-{i}" for i in range(5)]  # Existing source cap.
    assert bodies[0]["prompt"].count(WITNESS) == 7
    assert [bodies[0]["prompt"].index(f"(source-{i})") for i in range(7)] == sorted(
        bodies[0]["prompt"].index(f"(source-{i})") for i in range(7)
    )


@pytest.mark.asyncio
async def test_async_shared_generator_does_not_leak_budget_state(wire):
    llm, bodies, _ = wire
    gen = generator(llm)
    await asyncio.gather(
        gen.generate_async("finite", [memory()], token_budget=4000),
        gen.generate_async("unbounded", [memory()], token_budget=None),
    )
    finite = next(b for b in bodies if "QUESTION: finite" in b["prompt"])
    unbounded = next(b for b in bodies if "QUESTION: unbounded" in b["prompt"])
    assert WITNESS in finite["prompt"] and WITNESS not in unbounded["prompt"]


@pytest.mark.parametrize("mode", ["verbatim", "llm", "empty", "invalid"])
def test_extraction_and_fallback_keep_budgeted_text(wire, mode):
    llm, bodies, replies = wire
    if mode == "empty":
        replies.extend(['{"items": []}', '{"items": []}'])
    elif mode == "invalid":
        replies.append("not JSON")
    else:
        replies.append('{"items": ["violet key"]}')
    gen = generator(llm, list_extract_mode=True, list_finalize="verbatim" if mode == "verbatim" else "llm")
    answer = gen.generate("lookup", [memory()], category="list", token_budget=4000)
    contexts = [b["prompt"] for b in bodies if "MEMORIES:" in b["prompt"]]
    assert contexts and all(WITNESS in p for p in contexts)
    assert answer.sources == ["document.txt"]
    assert len(bodies) == {"verbatim": 1, "llm": 2, "empty": 3, "invalid": 2}[mode]


@pytest.mark.parametrize("budget", [None, 4000])
@pytest.mark.parametrize("retry", ["inference", "expansion"])
def test_retry_preserves_rendering_mode_and_scoped_recall(wire, monkeypatch, retry, budget):
    import mnemostack.recall.answer as module

    llm, bodies, replies = wire
    replies.append("Not in memory.\nCONFIDENCE: 0.1")
    fresh = memory(rid=2)
    calls = []

    def recall(*a, **kw):
        calls.append(kw)
        return [fresh]

    embedding = SimpleNamespace(embed_queries=lambda xs: [[1.0] for _ in xs],
                                embed_documents=lambda xs: [[1.0] for _ in xs])
    recaller = SimpleNamespace(recall=recall, search_many=recall, embedding=embedding)
    gen = generator(llm, recaller=recaller, retry_with_expansion=retry == "expansion", expansion_llm=llm)
    if retry == "inference":
        monkeypatch.setattr(module, "decompose_query", lambda *a: ["subquery"])
    else:
        replies.append("first rephrase\nsecond rephrase\nhypothetical answer")
    gen.generate("lookup", [memory("short draft")], category="inference", token_budget=budget,
                 recall_filters={"project": "P"}, tenant="one", as_of="2026-01-03T00:00:00Z")
    assert calls
    assert all(c["filters"] == {"project": "P"} and c["tenant"] == "one"
               and c["as_of"] == "2026-01-03T00:00:00Z" for c in calls)
    assert (WITNESS in bodies[-1]["prompt"]) is (budget is not None)


@pytest.mark.parametrize("interface", ["cli", "http", "mcp"])
@pytest.mark.parametrize("budget", [None, 1, 4000])
def test_answer_adapters_reach_actual_provider_payload(wire, monkeypatch, tmp_path, capsys, interface, budget):
    llm, bodies, _ = wire
    rows = [memory()]
    if interface == "cli":
        import mnemostack.cli as module
    elif interface == "http":
        pytest.importorskip("fastapi")
        import mnemostack.server as module
    else:
        pytest.importorskip("fastmcp")
        import mnemostack.mcp.server as module

    monkeypatch.setattr(module, "get_provider", lambda *a, **kw: SimpleNamespace(dimension=3))
    monkeypatch.setattr(module, "get_llm", lambda *a, **kw: llm)
    store = MagicMock(dimension=3)
    monkeypatch.setattr(module, "VectorStore", lambda **kw: store)
    if interface == "cli":
        monkeypatch.setattr(module, "_build_recaller", lambda *a: object())
        monkeypatch.setattr(module, "_recall_for_cli", lambda *a: rows)
        argv = ["answer", "lookup evidence", "--json"]
        if budget is not None:
            argv += ["--token-budget", str(budget)]
        assert module.main(argv) == 0
        assert json.loads(capsys.readouterr().out)["sources"] == ([] if budget == 1 else ["document.txt"])
    else:
        monkeypatch.setattr(module, "recall_flow", lambda *a, **kw: rows)
        if interface == "http":
            from fastapi.testclient import TestClient

            monkeypatch.setattr(module, "_make_probe_client", lambda *a: MagicMock())
            app = module.build_app(module.ServerConfig(
                graph_uri=None, text_search="off", record_access=False, retry_on_weak=False,
                state_path=str(tmp_path / "http-state.json"),
            ))
            with TestClient(app) as client:
                response = client.post("/answer", json={"query": "lookup evidence", "token_budget": budget, "full_pipeline": False})
                assert response.status_code == 200, response.text
                data = response.json()
        else:
            mcp = module.build_server(memgraph_uri=None, text_search="off", state_path=str(tmp_path / "mcp-state.json"))
            result = asyncio.run(mcp.call_tool("mnemostack_answer", {"query": "lookup evidence", "token_budget": budget}))
            data = result.structured_content
            assert data["ok"], data
        assert data["sources"] == ([] if budget == 1 else ["document.txt"])
    if budget == 1:
        assert not bodies  # Generator rechecks budgets even with custom retrieval.
    else:
        assert len(bodies) == 1
        assert (WITNESS in bodies[0]["prompt"]) is (budget is not None)
