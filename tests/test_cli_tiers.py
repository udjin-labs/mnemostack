"""Test Progressive Tiers API (--tier {1,2,3}) for mnemostack CLI.

Tier 1: ≤50 tokens (sources list, no text)
Tier 2: ≤200 tokens (short snippets)
Tier 3: ≤500 tokens (fuller detail)
No tier: backward-compatible full output
"""
from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from mnemostack.cli import (
    TIER_PROFILES,
    _apply_tier,
    cmd_search,
)


class FakeResult:
    def __init__(self, rid, score, text, sources, payload=None):
        self.id = rid
        self.score = score
        self.text = text
        self.sources = sources
        self.payload = payload or {}


FAKE_RESULTS = [
    FakeResult(i, 0.9 - i * 0.05, f"example text content {i} " * 30, ["vector", "bm25"], {"k": i})
    for i in range(12)
]


def test_tier_profiles_schema():
    """Each tier has required keys and sensible bounds."""
    for tier, profile in TIER_PROFILES.items():
        assert "limit" in profile
        assert "snippet_chars" in profile
        assert "max_sources" in profile
        assert profile["limit"] > 0
        assert profile["snippet_chars"] >= 0
        assert profile["max_sources"] >= 1
    # Tier progression: 1 < 2 < 3 in output size
    assert TIER_PROFILES[1]["snippet_chars"] < TIER_PROFILES[2]["snippet_chars"]
    assert TIER_PROFILES[2]["snippet_chars"] < TIER_PROFILES[3]["snippet_chars"]
    assert TIER_PROFILES[1]["limit"] <= TIER_PROFILES[3]["limit"]


def test_apply_tier_no_tier_returns_none():
    args = argparse.Namespace(tier=None, limit=10)
    assert _apply_tier(args) is None
    assert args.limit == 10  # unchanged


def test_apply_tier_caps_limit_to_tier():
    args = argparse.Namespace(tier=1, limit=100)
    profile = _apply_tier(args)
    assert profile == TIER_PROFILES[1]
    assert args.limit == TIER_PROFILES[1]["limit"]  # clamped


def _run_search_capture(tier, json_out=False, results=None):
    """Run cmd_search with mocked retrieval, capture stdout."""
    results = results if results is not None else FAKE_RESULTS
    args = argparse.Namespace(
        provider="fake",
        collection="test",
        qdrant="http://localhost:6333",
        query="anything",
        limit=10,
        json=json_out,
        tier=tier,
    )
    mock_provider = MagicMock(dimension=3072)
    mock_store = MagicMock()
    mock_store.collection_exists.return_value = True
    mock_recaller = MagicMock()
    mock_recaller.recall.return_value = results[: args.limit if tier is None else TIER_PROFILES[tier]["limit"]]

    buf = StringIO()
    with patch("mnemostack.cli.get_provider", return_value=mock_provider), \
         patch("mnemostack.cli.VectorStore", return_value=mock_store), \
         patch("mnemostack.cli.Recaller", return_value=mock_recaller), \
         patch("sys.stdout", buf):
        rc = cmd_search(args)
    return rc, buf.getvalue()


def test_tier1_list_view_text_output():
    """Tier 1 plain output: no text preview, just index/score/sources."""
    rc, out = _run_search_capture(tier=1)
    assert rc == 0
    # Should be 5 short lines
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == TIER_PROFILES[1]["limit"]
    # No "score=" detailed format, no text preview
    for line in lines:
        assert "score=" not in line  # short format uses space-separated
        assert "example text content" not in line  # no text


def test_tier1_json_no_text_field():
    """Tier 1 JSON: each entry has id/score/sources but no text/payload."""
    rc, out = _run_search_capture(tier=1, json_out=True)
    assert rc == 0
    payload = json.loads(out)
    assert len(payload) == TIER_PROFILES[1]["limit"]
    for entry in payload:
        assert "id" in entry
        assert "score" in entry
        assert "sources" in entry
        assert "text" not in entry  # tier 1 strips text
        assert "payload" not in entry


def test_tier2_snippet_length_capped():
    """Tier 2 JSON: text present but ≤ snippet_chars."""
    rc, out = _run_search_capture(tier=2, json_out=True)
    assert rc == 0
    payload = json.loads(out)
    max_chars = TIER_PROFILES[2]["snippet_chars"]
    for entry in payload:
        assert "text" in entry
        assert len(entry["text"]) <= max_chars


def test_tier3_more_results_longer_text():
    """Tier 3: up to 10 results, text up to 200 chars."""
    rc, out = _run_search_capture(tier=3, json_out=True)
    assert rc == 0
    payload = json.loads(out)
    assert len(payload) == TIER_PROFILES[3]["limit"]
    max_chars = TIER_PROFILES[3]["snippet_chars"]
    for entry in payload:
        assert len(entry["text"]) <= max_chars


def test_no_tier_backward_compatible():
    """Without --tier, output contains full text and payload (original behavior)."""
    rc, out = _run_search_capture(tier=None, json_out=True)
    assert rc == 0
    payload = json.loads(out)
    # limit defaults to 10
    assert len(payload) <= 10
    for entry in payload:
        assert "text" in entry
        assert "payload" in entry  # full payload only in back-compat mode
        # Full text, not truncated
        assert len(entry["text"]) > TIER_PROFILES[2]["snippet_chars"]


def test_tier_sources_capped():
    """Tier caps the number of source labels per result."""
    rc, out = _run_search_capture(tier=1, json_out=True)
    payload = json.loads(out)
    max_sources = TIER_PROFILES[1]["max_sources"]
    for entry in payload:
        assert len(entry["sources"]) <= max_sources
