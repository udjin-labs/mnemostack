"""Synthetic benchmark metrics include misses and respect measured depth."""

import runpy
from pathlib import Path

import pytest

benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/synthetic_longhorizon.py"))
ProbeResult = benchmark["ProbeResult"]
summarise = benchmark["summarise"]


@pytest.mark.parametrize("ranks,expected", [([1, None], 0.5), ([2, None], 0.25),
                                         ([None, None], 0.0), ([], 0.0)])
def test_mrr_includes_every_probe(ranks, expected):
    report = summarise([ProbeResult(str(i), rank, 1.0) for i, rank in enumerate(ranks)])
    assert report["mrr"] == expected
    assert report["probes"] == len(ranks)
    assert report["metrics_version"] == 2


def test_legacy_found_only_metric_is_named_explicitly():
    report = summarise([ProbeResult("found", 1, 2.0), ProbeResult("miss", None, 4.0)])
    assert report["mrr_found_only"] == 1.0
    assert report["mrr"] == 0.5
    assert report["latency_ms_mean"] == 3.0


@pytest.mark.parametrize("depth", [1, 3, 5, 10, 20, 30])
def test_unmeasured_cutoffs_are_null(depth):
    report = summarise([ProbeResult("found", 1, 1.0)], k=depth)
    assert report["search_k"] == depth
    assert report["recall_at_5"] == (1.0 if depth >= 5 else None)
    assert report["recall_at_20"] == (1.0 if depth >= 20 else None)
    assert report["hit_rate_at_k"] == 1.0


def test_rank_outside_requested_cutoff_counts_as_a_miss():
    report = summarise([ProbeResult("far", 6, 1.0)], k=5)
    assert report["mrr"] == report["hit_rate_at_k"] == 0.0


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_invalid_depth_fails(depth):
    with pytest.raises(ValueError):
        summarise([], k=depth)
