"""
Observability for mnemostack — counters, histograms, tracing hooks.

Design: lightweight, zero-dependency by default. Users who want Prometheus or
OpenTelemetry plug in a recorder via `set_recorder()`.

Usage:
    from mnemostack.observability import counter, histogram, set_recorder

    # In your code
    counter("mnemostack.recall.calls", 1, labels={"provider": "gemini"})
    with histogram("mnemostack.recall.latency_ms", labels={"op": "search"}) as h:
        results = recaller.recall(query)
        # h.value set automatically by context manager

    # Plug in Prometheus
    from prometheus_client import Counter, Histogram
    prom_counter = Counter('mnemostack_recall_calls', 'Recall calls', ['provider'])
    ...

    class PrometheusRecorder:
        def record_counter(self, name, value, labels):
            prom_counter.labels(**labels).inc(value)
        def record_histogram(self, name, value_ms, labels):
            # ...
    set_recorder(PrometheusRecorder())
"""

from .recorder import (
    MetricsRecorder,
    NullRecorder,
    counter,
    get_recorder,
    histogram,
    set_recorder,
    timed,
)

__all__ = [
    "MetricsRecorder",
    "NullRecorder",
    "counter",
    "histogram",
    "timed",
    "get_recorder",
    "set_recorder",
]
