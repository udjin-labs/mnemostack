from __future__ import annotations

import pytest

from mnemostack.retriever_base import Retriever, RetrieverConfig


class DummyRetriever(Retriever):
    name = "dummy"

    def retrieve(self, query: str, **kwargs):
        return []


def test_retriever_instances_do_not_share_config():
    first = DummyRetriever()
    second = DummyRetriever()

    first.config.enabled = False
    first.config.weight = 0.25

    assert first.is_enabled is False
    assert second.is_enabled is True
    assert second.config.weight == 1.0


def test_retriever_accepts_explicit_config():
    retriever = DummyRetriever(RetrieverConfig(enabled=False, weight=0.5))

    assert retriever.is_enabled is False
    assert retriever.config.weight == 0.5


def test_retriever_config_rejects_negative_weight():
    with pytest.raises(ValueError):
        RetrieverConfig(weight=-0.1)
