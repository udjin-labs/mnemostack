"""Synthetic long-horizon retrieval benchmark.

Generates a large synthetic corpus of dialogue-like chunks with deterministic
"needle" facts planted at known offsets, indexes it into mnemostack, and then
probes recall quality as a function of corpus size. The point is to expose
where the stack breaks on long horizons — something the 500-turn LoCoMo samples
can't do.

What we measure:

- recall@k (k=1, 5, 20) — did the exact needle chunk make it into the top-K?
- MRR across needles — how high, on average, does the right chunk rank?
- Latency per query
- Per-corpus-size scaling when --scale-steps is used

Usage:

    python benchmarks/synthetic_longhorizon.py --turns 50000 --needles 50
    python benchmarks/synthetic_longhorizon.py --scale-steps 5000,10000,25000,50000

Requires a running Qdrant and (optionally) Memgraph — see benchmarks/README.md.
Uses a deterministic RNG seed so runs are comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from mnemostack.embeddings import get_provider
from mnemostack.recall import BM25Doc, BM25Retriever, Recaller, VectorRetriever
from mnemostack.vector import VectorStore

# Topics and templates that produce varied but deterministic surface text.
# Needles get planted into the same template so the retrieval test is fair.
TOPICS = [
    "deployed", "deprecated", "migrated", "refactored", "benchmarked",
    "profiled", "documented", "audited", "patched", "rolled-out",
]
SUBJECTS = [
    "auth service", "billing pipeline", "search index", "notification queue",
    "recall stack", "graph store", "vector backend", "telemetry collector",
    "scheduler", "ingest worker",
]
FILLER_SENTENCES = [
    "Nothing blocking on review.",
    "Follow-up scheduled next week.",
    "No regressions detected.",
    "Docs updated accordingly.",
    "Ops are aware and on-call is set.",
    "Cost impact is negligible.",
    "Ran smoke tests in staging.",
    "Added a note to the changelog.",
]


@dataclass
class Needle:
    """A planted fact with a unique, memorable identifier."""
    token: str
    subject: str
    date: str
    text: str
    chunk_id: str | None = None  # set after chunking


def make_dialogue_chunk(turn: int, rnd: random.Random) -> str:
    topic = rnd.choice(TOPICS)
    subject = rnd.choice(SUBJECTS)
    filler = rnd.choice(FILLER_SENTENCES)
    return (
        f"[turn {turn}] engineer: today I {topic} the {subject}. "
        f"{filler}"
    )


def make_needle(rnd: random.Random) -> Needle:
    token = f"N-{rnd.randint(1_000_000, 9_999_999)}"
    subject = rnd.choice(SUBJECTS)
    # Realistic-looking but deterministic dates
    year = rnd.randint(2022, 2026)
    month = rnd.randint(1, 12)
    day = rnd.randint(1, 28)
    date = f"{year:04d}-{month:02d}-{day:02d}"
    text = (
        f"On {date} we picked token {token} as the canonical probe for the "
        f"{subject} subsystem. This is a unique, auditable marker."
    )
    return Needle(token=token, subject=subject, date=date, text=text)


def build_corpus(
    turns: int,
    needles: int,
    seed: int = 1337,
) -> tuple[list[str], list[Needle]]:
    """Return (chunks, planted_needles).

    Needles are placed at pseudo-random positions. Chunks are 1 per turn.
    """
    rnd = random.Random(seed)
    planted: list[Needle] = [make_needle(rnd) for _ in range(needles)]
    # Positions where to splice needles in (distinct, sorted)
    positions = sorted(rnd.sample(range(turns), needles))

    chunks: list[str] = []
    pi = 0
    for turn in range(turns):
        if pi < len(positions) and turn == positions[pi]:
            chunks.append(planted[pi].text)
            planted[pi].chunk_id = f"turn-{turn:06d}"
            pi += 1
        else:
            chunks.append(make_dialogue_chunk(turn, rnd))
    return chunks, planted


def index_chunks(chunks: list[str], provider, store: VectorStore, batch: int = 64) -> None:
    """Embed in batches (to saturate the provider) and upsert in batches.

    Single-item `.embed()` was spending most of its time on HTTP overhead;
    embedding 64 texts per call is an order of magnitude faster on Gemini.
    """
    total = 0
    n = len(chunks)
    for start in range(0, n, batch):
        window = chunks[start : start + batch]
        try:
            vectors = provider.embed_batch(window)
        except AttributeError:
            # Providers without batch API — fall back to single-item
            vectors = [provider.embed(t) for t in window]
        points: list[tuple[str, list[float], dict]] = []
        for offset, (text, vec) in enumerate(zip(window, vectors)):
            if not vec:
                continue
            turn = start + offset
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"synth:{turn}"))
            points.append((point_id, vec, {"text": text, "turn": turn}))
        if points:
            store.upsert_batch(points)
            total += len(points)
        if start % (batch * 10) == 0:
            print(f"  indexed {start + len(window)}/{n}", flush=True)
    print(f"indexed {total} chunks", flush=True)


@dataclass
class ProbeResult:
    token: str
    found_rank: int | None  # 1-based rank in top-K, None if not found within K
    latency_ms: float


def probe_needles(
    recaller: Recaller, needles: list[Needle], k: int = 20
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for n in needles:
        # We query by token + subject — mimics a natural "do you remember X?" ask
        query = f"what is token {n.token} and which subsystem does it probe"
        t0 = time.monotonic()
        hits = recaller.recall(query, limit=k)
        latency_ms = (time.monotonic() - t0) * 1000
        rank: int | None = None
        for i, h in enumerate(hits, 1):
            if n.token in (h.text or ""):
                rank = i
                break
        results.append(ProbeResult(token=n.token, found_rank=rank, latency_ms=latency_ms))
    return results


def summarise(results: list[ProbeResult]) -> dict:
    ranks = [r.found_rank for r in results if r.found_rank is not None]
    def rate_at(k: int) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.found_rank is not None and r.found_rank <= k) / len(results)

    mrr = mean([1.0 / r for r in ranks]) if ranks else 0.0
    latencies = [r.latency_ms for r in results]
    return {
        "probes": len(results),
        "recall_at_1": round(rate_at(1), 4),
        "recall_at_5": round(rate_at(5), 4),
        "recall_at_20": round(rate_at(20), 4),
        "mrr": round(mrr, 4),
        "latency_ms_mean": round(mean(latencies), 1) if latencies else 0.0,
        "latency_ms_p50": round(median(latencies), 1) if latencies else 0.0,
        "latency_ms_max": round(max(latencies), 1) if latencies else 0.0,
    }


def run_one_scale(turns: int, needles: int, args) -> dict:
    provider = get_provider(args.provider)
    client = QdrantClient(host="localhost", port=6333)
    collection = f"{args.collection}_synth_{turns}"
    # Always start from a clean collection so results are comparable
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=provider.dimension, distance=Distance.COSINE),
    )
    store = VectorStore(collection=collection, dimension=provider.dimension)

    print(f"\n=== scale: {turns} turns, {needles} needles ===", flush=True)
    t_build = time.monotonic()
    chunks, planted = build_corpus(turns, needles, seed=args.seed)
    print(f"corpus built in {time.monotonic() - t_build:.1f}s", flush=True)

    t_index = time.monotonic()
    index_chunks(chunks, provider, store, batch=args.batch)
    print(f"indexed in {time.monotonic() - t_index:.1f}s", flush=True)

    bm25_docs = [
        BM25Doc(id=f"bm:{i}", text=c, payload={"turn": i}) for i, c in enumerate(chunks)
    ]
    recaller = Recaller(retrievers=[
        VectorRetriever(embedding=provider, vector_store=store),
        BM25Retriever(docs=bm25_docs),
    ])

    results = probe_needles(recaller, planted, k=args.k)
    summary = summarise(results)
    summary["turns"] = turns
    summary["needles"] = needles
    summary["collection"] = collection
    print(json.dumps(summary, indent=2), flush=True)

    if args.cleanup:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
    return summary


def main():
    ap = argparse.ArgumentParser(description="Long-horizon synthetic retrieval benchmark")
    ap.add_argument("--turns", type=int, default=10000)
    ap.add_argument("--needles", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--k", type=int, default=20, help="top-K for recall@K measurement")
    ap.add_argument("--provider", default=os.environ.get("MNEMOSTACK_PROVIDER", "gemini"))
    ap.add_argument("--collection", default=os.environ.get("MNEMOSTACK_COLLECTION", "synth"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--scale-steps",
        type=str,
        default="",
        help="comma-separated turn counts to measure; overrides --turns if set",
    )
    ap.add_argument("--output", default=None, help="write summary JSON here")
    ap.add_argument("--cleanup", action="store_true", help="drop the collection after the run")
    args = ap.parse_args()

    steps = [int(s) for s in args.scale_steps.split(",") if s.strip()] if args.scale_steps else [args.turns]

    all_summaries = []
    for turns in steps:
        s = run_one_scale(turns, args.needles, args)
        all_summaries.append(s)

    final = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "seed": args.seed,
        "scales": all_summaries,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(final, indent=2))
        print(f"\nWrote summary to {args.output}", flush=True)
    else:
        print("\n=== FINAL SUMMARY ===")
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
