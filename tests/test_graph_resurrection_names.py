"""Literal graph names must not be suppressed by unrelated substrings."""
from types import SimpleNamespace

import pytest

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import GraphResurrection, PipelineContext


@pytest.mark.parametrize("name,text,added", [("Rust", "trust is important", True), ("Rust", "Rustacean notes", True), ("Rust", "we use RUST.", False), ("New York", "a New Yorker", True), ("New York", "visited 'New York'", False), ("C++", "uses C++17", True), ("C++", "uses C++.", False), ("Руст", "грусть", True), ("Руст", "Имя: РУСТ.", False)])
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
