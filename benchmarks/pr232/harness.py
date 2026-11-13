"""One-tree offline/real retrieval evaluation. Select product exclusively via PYTHONPATH."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import socket
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

KS = (1, 3, 5, 10, 25)
BASE_ORDER = [
    "ClassifyQuery",
    "ExactTokenRescue",
    "GravityDampen",
    "CalibrateRelevance",
    "FreshnessBlend",
    "ExactTokenProtection",
]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class CachedProvider:
    """Role-aware vectors, with no provider construction at all on a cache-only run."""

    def __init__(self, args):
        self.args = args
        self.root = Path(args.cache) / digest(
            ("fake:" if args.self_test else "") + args.embedding_model
        )
        self.provider = None
        if not args.cache_readonly:
            self.root.mkdir(parents=True, exist_ok=True)

    def vectors(self, texts, role):
        out = []
        for start in range(0, len(texts), 32):
            part = texts[start : start + 32]
            paths = [self.root / f"{role}-{digest(t)}.json" for t in part]
            missing = [i for i, p in enumerate(paths) if not p.exists()]
            if missing and self.args.cache_readonly:
                raise RuntimeError(
                    f"read-only embedding cache miss: {role} {digest(part[missing[0]])}"
                )
            if missing:
                if self.args.self_test:
                    got = []
                    for i in missing:
                        b = hashlib.shake_256(part[i].encode()).digest(128)
                        v = [(x - 127.5) / 127.5 for x in b]
                        norm = math.sqrt(sum(x * x for x in v))
                        got.append([x / norm for x in v])
                else:
                    from mnemostack.embeddings.ollama import OllamaProvider
                    from mnemostack.embeddings.roles import embed_documents_via, embed_queries_via

                    if self.provider is None:
                        self.provider = OllamaProvider(
                            model=self.args.embedding_model, host=self.args.ollama_host
                        )
                    fn = embed_documents_via if role == "document" else embed_queries_via
                    for attempt in range(5):
                        try:
                            got = fn(self.provider, [part[i] for i in missing])
                            if len(got) == len(missing) and all(
                                v and all(math.isfinite(x) for x in v) for v in got
                            ):
                                break
                        except Exception:
                            pass
                        if attempt == 4:
                            raise RuntimeError("embedding batch failed after five attempts")
                        time.sleep(2 + 2 * attempt)
                for i, v in zip(missing, got, strict=True):
                    temp = paths[i].with_suffix(".tmp")
                    temp.write_text(json.dumps(v))
                    temp.replace(paths[i])
            out.extend(json.loads(p.read_text()) for p in paths)
        return out

    def embed_query(self, text):
        return self.vectors([text], "query")[0]

    def embed_queries(self, texts):
        return self.vectors(texts, "query")

    def embed_documents(self, texts):
        return self.vectors(texts, "document")

    def embed(self, text):
        return self.embed_query(text)


def freeze(iso):
    import mnemostack.recall.pipeline.stages as stages
    import mnemostack.recall.retrievers as retrievers
    from mnemostack.recall.pipeline.base import PipelineContext
    from mnemostack.recall.recaller import RecallResult

    instant = datetime.fromisoformat(iso)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    stages.datetime = Frozen
    retrievers.datetime = Frozen
    # Exercise the actual freshness stage, not just the replaced clock.
    stage = stages.FreshnessBlend()
    age = stage.halflife_days
    r = RecallResult(
        id=1,
        text="clock probe",
        score=0,
        payload={"timestamp": (instant - timedelta(days=age)).isoformat(), "source": "probe"},
        sources=["vector"],
    )
    stage.apply(PipelineContext(query="clock probe"), [r])
    assert r.payload["freshness"] == 0.5, "clock patch failed freshness age assertion"
    assert retrievers._now_date() == instant.date(), "temporal clock patch failed"
    return {
        "freshness_probe_age_days": age,
        "freshness": r.payload["freshness"],
        "temporal_today": str(retrievers._now_date()),
    }


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        for kind in ("overall", r["kind"]):
            groups[(r["configuration"], kind)].append(r)
    result = {}
    for (config, kind), rs in groups.items():
        n = len(rs)
        m = {"n": n, "MRR": sum(1 / r["rank"] if r["rank"] else 0 for r in rs) / n}
        for k in KS:
            if k in (1, 3, 5):
                m[f"hit@{k}"] = sum(bool(r["gold_at"][str(k)]) for r in rs) / n
            if k in (5, 10, 25):
                m[f"recall@{k}"] = sum(len(r["gold_at"][str(k)]) / len(r["gold"]) for r in rs) / n
        result.setdefault(config, {})[kind] = m
    return result


def record(q, config, results, trace, limit):
    ids = [int(r.id) for r in results]
    gold = set(q["gold"])

    def flags(r):
        return {
            k: v
            for k, v in r.payload.items()
            if k not in ("text", "timestamp", "source") and not k.startswith("_")
        }

    return dict(
        query_id=q["id"],
        text=q["text"],
        kind=q["kind"],
        gold=q["gold"],
        configuration=config,
        rank=next((i + 1 for i, x in enumerate(ids) if x in gold), None),
        gold_at={str(k): sorted(set(ids[:k]) & gold) for k in KS},
        ranked=[
            dict(id=int(r.id), score=r.score, flags=flags(r), sources=r.sources)
            for r in results[:limit]
        ],
        gold_flags={str(r.id): flags(r) for r in results if int(r.id) in gold},
        degraded=list(getattr(trace, "degraded", [])),
        notes=list(getattr(trace, "notes", [])),
    )


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--ollama-host")
    ap.add_argument("--embedding-model", default="nomic-embed-text")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--qdrant-url")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--cache-readonly", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    return ap


def run(args, ablation=False):
    if args.limit < 25:
        raise ValueError("--limit must be >=25 for recall@25")
    if not args.self_test and not args.ollama_host and not args.cache_readonly:
        raise ValueError("--ollama-host required to fill cache")
    if args.self_test:
        if args.qdrant_url:
            raise ValueError("self-test requires in-memory Qdrant")

        def blocked(*a, **kw):
            raise RuntimeError("network forbidden in self-test")

        socket.socket.connect = blocked
        socket.create_connection = blocked
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from mnemostack.recall import BM25Doc, Recaller
    from mnemostack.recall.pipeline import Pipeline, presets
    from mnemostack.recall.pipeline.state import InMemoryStateStore
    from mnemostack.recall.retrievers import BM25Retriever, TemporalRetriever, VectorRetriever
    from mnemostack.recall.trace import RecallTrace
    from mnemostack.vector import VectorStore

    data = json.loads(Path(args.corpus).read_text())
    clock = freeze(data["reference_now"])
    provider = CachedProvider(args)
    vectors = provider.embed_documents([m["text"] for m in data["memories"]])
    provider.embed_queries([q["text"] for q in data["queries"]])
    dims = {len(v) for v in vectors}
    assert len(dims) == 1
    client = QdrantClient(url=args.qdrant_url) if args.qdrant_url else QdrantClient(":memory:")
    collection = "rank232_" + uuid.uuid4().hex
    rows = []
    skipped = []
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        store = VectorStore.__new__(VectorStore)
        store.collection = collection
        store.dimension = len(vectors[0])
        store.distance = Distance.COSINE
        store.client = client
        store.upsert_batch(
            [
                (m["id"], v, {k: x for k, x in m.items() if k != "id"})
                for m, v in zip(data["memories"], vectors, strict=True)
            ],
            batch_size=100,
        )
        docs = [
            BM25Doc(id=m["id"], text=m["text"], payload={k: v for k, v in m.items() if k != "id"})
            for m in data["memories"]
            if not m.get("fallback_only")
        ]

        class Primary(VectorRetriever):
            def search(self, query, **kwargs):
                if query in fault_queries:
                    return []
                return super().search(query, **kwargs)

        fault_queries = {q["text"] for q in data["queries"] if q.get("fault")}
        arms = [Primary(provider, store), BM25Retriever(docs), TemporalRetriever(provider, store)]

        class ObservedRecaller(Recaller):
            fallback_calls = 0
            fallback_ids = ()

            def _vector_fallback_hits(self, *a, **kw):
                hits = super()._vector_fallback_hits(*a, **kw)
                self.fallback_calls += 1
                self.fallback_ids = [int(h.id) for h in hits]
                return hits

        recaller = ObservedRecaller(
            embedding_provider=provider, vector_store=store, retrievers=arms
        )
        pipe = presets.build_stateless_pipeline()
        orders = {"stateless": [type(s).__name__ for s in pipe.stages]}
        if ablation:
            by = {type(s).__name__: s for s in pipe.stages}
            pipes = {
                "merged_order": Pipeline(list(pipe.stages)),
                "base_order": Pipeline([by[n] for n in BASE_ORDER]),
            }
            orders = {k: [type(s).__name__ for s in p.stages] for k, p in pipes.items()}
        try:
            from mnemostack.recall.flow import recall_flow
        except ImportError:
            recall_flow = None
            skipped.append("recall_flow unavailable")
        for q in data["queries"]:
            trace = RecallTrace()
            before_fallback = recaller.fallback_calls
            raw = recaller.recall(q["text"], limit=max(3 * args.limit, 30), trace=trace)
            if q.get("fault"):
                assert recaller.fallback_calls > before_fallback, "fallback was not exercised"
                assert not any(t.ranked for t in trace.retrievers if t.name == "vector"), (
                    "fault injection failed"
                )
            configs = (
                list(pipes)
                if ablation
                else ["raw", "stateless", "full_fresh_state"]
                + (["recall_flow"] if recall_flow else [])
            )
            for config in configs:
                tr = trace
                try:
                    if ablation:
                        results = pipes[config].apply(q["text"], copy.deepcopy(raw))
                    elif config == "raw":
                        results = copy.deepcopy(raw)
                    elif config == "stateless":
                        results = pipe.apply(q["text"], copy.deepcopy(raw))
                    elif config == "full_fresh_state":
                        results = presets.build_full_pipeline(
                            state_store=InMemoryStateStore()
                        ).apply(q["text"], copy.deepcopy(raw))
                    else:
                        tr = RecallTrace()
                        results = recall_flow(
                            recaller,
                            q["text"],
                            limit=args.limit,
                            pipeline=presets.build_full_pipeline(state_store=InMemoryStateStore()),
                            trace=tr,
                        )
                    row = record(q, config, results, tr, args.limit)
                    if q.get("fault"):
                        row["fault"] = "empty_primary_vector"
                        row["fallback_called"] = recaller.fallback_calls > before_fallback
                        row["gold_in_fallback_candidates"] = sorted(
                            set(recaller.fallback_ids) & set(q["gold"])
                        )
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        dict(
                            query_id=q["id"],
                            kind=q["kind"],
                            configuration=config,
                            error=type(exc).__name__,
                        )
                    )
                    skipped.append(f"{q['id']} {config}: {type(exc).__name__}")
        result = dict(
            label=args.label,
            corpus_sha256=digest(Path(args.corpus).read_text()),
            model=args.embedding_model,
            self_test=args.self_test,
            limit=args.limit,
            clock=clock,
            orders=orders,
            skipped=skipped,
            rows=rows,
            metrics=summarize(rows),
        )
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            "| configuration | kind | n | MRR | hit@1 | hit@3 | hit@5 | recall@5 | recall@10 | recall@25 |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c, groups in result["metrics"].items():
            for k, m in groups.items():
                print(
                    "| "
                    + c
                    + " | "
                    + k
                    + " | "
                    + " | ".join(str(v) if key == "n" else f"{v:.4f}" for key, v in m.items())
                    + " |"
                )
        print("skipped/errors:", len(skipped))
        if args.self_test:
            assert not skipped, skipped
        return result
    finally:
        client.delete_collection(collection_name=collection)
        client.close()


if __name__ == "__main__":
    run(parser().parse_args())
