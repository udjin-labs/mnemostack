"""Tests for the `index --recreate` confirmation prompt."""

from __future__ import annotations

import argparse
import types

import mnemostack.cli as cli


class _FakeProvider:
    dimension = 4

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _FakeStore:
    def __init__(self, **_kwargs):
        self.ensure_calls: list[bool] = []
        self.upserts: list[str] = []

    def collection_exists(self) -> bool:
        return True

    def count(self) -> int:
        return 42

    def ensure_collection(self, recreate: bool = False) -> bool:
        self.ensure_calls.append(recreate)
        return recreate

    def iter_ids(self, filters=None):
        return iter(())

    def upsert(self, cid, vec, payload) -> None:
        self.upserts.append(cid)


def _args(tmp_path, **overrides) -> argparse.Namespace:
    doc = tmp_path / "note.md"
    doc.write_text("hello world", encoding="utf-8")
    defaults = dict(
        path=str(doc),
        provider="fake",
        collection="test",
        qdrant="http://localhost:6333",
        recreate=True,
        yes=False,
        prune=False,
        enrich=None,
        refresh_payloads=False,
        chunk_size=800,
        window_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch_stack(monkeypatch) -> _FakeStore:
    store = _FakeStore()
    monkeypatch.setattr(cli, "get_provider", lambda _name, **_kw: _FakeProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **_kw: store)
    monkeypatch.setattr(cli, "_embedding_model", lambda _args: None, raising=False)
    monkeypatch.setattr(cli, "model_kwargs", lambda _model: {})
    return store


def test_non_tty_without_yes_exits_2(monkeypatch, tmp_path, capsys):
    store = _patch_stack(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))

    rc = cli.cmd_index(_args(tmp_path))

    assert rc == 2
    assert store.ensure_calls == []
    assert "--yes" in capsys.readouterr().err


def test_interactive_decline_aborts(monkeypatch, tmp_path):
    store = _patch_stack(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = cli.cmd_index(_args(tmp_path))

    assert rc == 1
    assert store.ensure_calls == []


def test_interactive_confirm_proceeds(monkeypatch, tmp_path):
    store = _patch_stack(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = cli.cmd_index(_args(tmp_path))

    assert rc == 0
    assert store.ensure_calls == [True]
    assert store.upserts


def test_yes_flag_skips_prompt(monkeypatch, tmp_path):
    store = _patch_stack(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("prompt must not be shown")),
    )

    rc = cli.cmd_index(_args(tmp_path, yes=True))

    assert rc == 0
    assert store.ensure_calls == [True]


def test_no_recreate_never_prompts(monkeypatch, tmp_path):
    store = _patch_stack(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("prompt must not be shown")),
    )

    rc = cli.cmd_index(_args(tmp_path, recreate=False))

    assert rc == 0
    assert store.ensure_calls == [False]
