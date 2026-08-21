"""The stack's one answer to "are these the same memory?".

A leaf module on purpose. The rule has to be reachable from both ends of
the recall path — the fusion that merges ranked lists and the recaller
that builds them — and `recaller` already imports `fusion`, so a rule
living in either would make the other import a cycle. Nothing is imported
here, so nothing can.
"""

from __future__ import annotations

from typing import Any


def memory_key(value: Any) -> str:
    """THE identity rule for a memory. Every id this stack keys on comes here.

    One memory can come back as `1` from one retriever and `"1"` from
    another, so anything asking "are these the same memory" has to ask it
    the same way. Six places learned that separately — the weak-retry's
    "found something new" dict, a retry pass's own ranking, the vector
    floor's candidate pool and its dedup against the page, the fusion's
    own deduplication, and the recaller's merge dictionaries — and each
    was found the same way: one memory occupying two slots of a caller's
    page, or a genuinely different memory pushed out of one.

    Two DISTINCT memories cannot collide under it. A string point id must
    be a UUID in Qdrant (`Point id 1 is not a valid UUID`), ingest mints
    ids as `str(uuid.UUID(...))`, and graph hits are namespaced by
    `graph_result_id()`.
    """
    return str(value)
