"""Literal graph names must not be suppressed by unrelated substrings."""
from types import SimpleNamespace

import pytest

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import GraphResurrection, PipelineContext


@pytest.mark.parametrize("name,text,added", [("Rust", "trust is important", True), ("Rust", "Rustacean notes", True), ("Rust", "we use RUST.", False), ("New York", "a New Yorker", True), ("New York", "visited 'New York'", False), ("C++", "uses C++17", True), ("C++", "uses C++.", False), ("Руст", "грусть", True), ("Руст", "Имя: РУСТ.", False), ("北京", "我住在北京。", False), ("京都", "京都に住んでいます。", False), ("เชียงใหม่", "ฉันอยู่เชียงใหม่", False), ("서울", "서울에서삽니다", False), ("北京", "我住在上海。", True)])
def test_graph_name_dedup_uses_literal_boundaries(name, text, added):
    seen = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, cypher, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(data=lambda: [{"name": name, "type": "Entity", "mc": "fact", "rel": "related"}])

    driver = SimpleNamespace(session=lambda **kwargs: Session())
    stage = GraphResurrection(driver=driver)
    original = RecallResult(id="stored", text=text, score=0.9, payload={"text": text, "tenant_id": "alpha"})
    pool = [original]
    result = stage.apply(PipelineContext(query="project", extras={"tenant": "alpha"}), pool)
    assert result is pool and result[0] is original
    assert len(result) == (2 if added else 1)
    assert all(p["tenant"] == "alpha" for p in seen)
    if added:
        assert result[-1].payload["tenant_id"] == "alpha"
        assert result[-1].payload["resurrected"] is True
        assert name in result[-1].text


def test_unspaced_duplicate_does_not_consume_the_only_graph_slot():
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, *args, **kwargs):
            return SimpleNamespace(data=lambda: [{"name": name, "type": "Place", "mc": "fact", "rel": "related"} for name in ("北京", "Paris")])

    stage = GraphResurrection(driver=SimpleNamespace(session=lambda **kwargs: Session()), limit=1)
    original = RecallResult(id="stored", text="我住在北京。", score=0.9, payload={"tenant_id": "alpha"})
    result = stage.apply(PipelineContext(query="project", extras={"tenant": "alpha"}), [original])
    assert len(result) == 2 and result[0] is original
    assert "Paris" in result[-1].text
    assert result[-1].payload["tenant_id"] == "alpha"
