"""Read-only, scoped A/B retrieval evaluation using the canonical recall flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from qdrant_client import models

from mnemostack import __version__
from mnemostack.recall import (
    InMemoryStateStore,
    QdrantNativeBM25Retriever,
    Recaller,
    RecallTrace,
    Reranker,
    TemporalRetriever,
    VectorRetriever,
    build_full_pipeline,
    recall_flow,
)
from mnemostack.recall.pipeline import stages
from mnemostack.vector.qdrant import quantization_search_params

DEFAULTS = {"vector": True, "text_search": "native_bm25", "pipeline": "full",
            "rerank": False, "rerank_mode": "relevant_only", "limit": 10,
            "token_budget": None, "rrf_k": 60, "vector_floor": 0,
            "hnsw_ef": None, "freshness_weight": 0.2,
            "access_bonus_max": stages.DEFAULT_ACCESS_BONUS_MAX}


class InputError(ValueError):
    """A fixed, safe-to-display validation message, never a backend response."""


class WarmupError(RuntimeError):
    """A failed-open warm-up is not a successfully warmed sample."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise InputError("unsupported evaluation config fields")
    cfg = {**DEFAULTS, **value}
    for key in ("vector", "rerank"):
        if not isinstance(cfg[key], bool):
            raise InputError("vector and rerank must be booleans")
    if cfg["text_search"] not in ("off", "native_bm25") or cfg["pipeline"] not in ("off", "full"):
        raise InputError("unsupported retrieval mode")
    if not cfg["vector"] and cfg["text_search"] == "off":
        raise InputError("at least one retriever is required")
    if cfg["rerank_mode"] not in ("relevant_only", "full_reorder"):
        raise InputError("invalid rerank mode")
    for key in ("limit", "rrf_k"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise InputError("limit and rrf_k must be positive integers")
    if cfg["limit"] > 100:
        raise InputError("evaluation limit must not exceed 100")
    if isinstance(cfg["vector_floor"], bool) or not isinstance(cfg["vector_floor"], int) or cfg["vector_floor"] < 0:
        raise InputError("vector_floor must be a nonnegative integer")
    for key in ("token_budget", "hnsw_ef"):
        if cfg[key] is not None and (isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1):
            raise InputError("optional integer settings must be positive")
    for key in ("freshness_weight", "access_bonus_max"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 <= cfg[key] <= 1:
            raise InputError("ranking weights must be finite values in [0, 1]")
    quantization_search_params(hnsw_ef=cfg["hnsw_ef"])
    return cfg


def validate_cases(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 1000:
        raise ValueError("provide between 1 and 1000 cases")
    seen = set()
    for case in value:
        if not isinstance(case, dict) or set(case) - {"query_id", "query", "expected_ids", "category", "filters", "as_of", "include_invalidated"}:
            raise ValueError("unsupported query fields")
        for key in ("query_id", "query"):
            if not isinstance(case.get(key), str) or not case[key].strip() or len(case[key]) > 8000:
                raise ValueError("query_id and query must be nonempty strings")
        if case["query_id"] in seen:
            raise ValueError("query IDs must be unique")
        seen.add(case["query_id"])
        if case.get("category") is not None and not isinstance(case["category"], str):
            raise ValueError("category must be a string")
        ids = case.get("expected_ids")
        if not isinstance(ids, list) or any(isinstance(x, bool) or not isinstance(x, (str, int)) or not str(x) for x in ids):
            raise ValueError("expected_ids must be an explicit list of IDs")
        if case.get("filters") is not None and not isinstance(case["filters"], dict):
            raise ValueError("filters must be an object")
        if not isinstance(case.get("include_invalidated", False), bool):
            raise ValueError("include_invalidated must be boolean")
        if case.get("as_of") is not None:
            stamp = datetime.fromisoformat(case["as_of"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("as_of requires a timezone")
    return value


def snapshot(store, tenant):
    """Hash this tenant's actual points; never return foreign IDs or payloads."""
    filt = models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant))])
    hasher, owned, offset = hashlib.sha256(), set(), None
    while True:
        points, offset = store.client.scroll(store.collection, scroll_filter=filt, offset=offset,
                                              limit=128, with_payload=True, with_vectors=True)
        for point in points:
            if (point.payload or {}).get("tenant_id") != tenant:
                raise ValueError("snapshot tenant mismatch")
            owned.add(str(point.id))
            hasher.update(digest(point.model_dump(mode="json", include={"id", "payload", "vector"})).encode())
        if offset is None:
            return hasher.hexdigest(), owned


@contextmanager
def ranking_clock(instant):
    original = stages.datetime

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    stages.datetime = Frozen
    try:
        yield
    finally:
        stages.datetime = original


def metrics(ids, expected):
    gold = set(map(str, expected))
    if not gold:
        return {"hit_rate": None, "recall": None, "mrr": None, "complete": None}
    found = gold.intersection(ids)
    rank = next((i for i, pid in enumerate(ids, 1) if pid in gold), None)
    return {"hit_rate": float(bool(found)), "recall": len(found) / len(gold),
            "mrr": 1 / rank if rank else 0.0, "complete": gold <= set(ids)}


def project_trace(trace, owned):
    def order(ranked):
        return [[pid, score] for pid, score in ranked if pid in owned]

    out = {"retrievers": [{"name": r.name, "ranked": order(r.ranked), "failed": bool(r.error)} for r in trace.retrievers],
           "stages": [{"name": r.name, "ranked": order(r.ranked)} for r in trace.stages],
           "fused": order(trace.fused), "post_rerank": None if trace.post_rerank is None else order(trace.post_rerank)}
    if trace.budget is not None:
        out["budget"] = {"limit": trace.budget.limit, "used": trace.budget.used,
                         "before": order(trace.budget.before), "after": order(trace.budget.after)}
    return out


def components(store, embedding, llm, cfg, tenant):
    retrievers = []
    if cfg["vector"]:
        if embedding is None:
            raise ValueError("vector mode needs an explicitly supplied embedding provider")
        retrievers.append(VectorRetriever(embedding, store))
    if cfg["text_search"] == "native_bm25":
        retrievers.append(QdrantNativeBM25Retriever(store))
    if cfg["vector"]:
        retrievers.append(TemporalRetriever(embedding, store))
    recaller = Recaller(retrievers=retrievers, rrf_k=cfg["rrf_k"], vector_floor=cfg["vector_floor"], default_tenant=tenant)
    pipeline = None
    if cfg["pipeline"] == "full":
        pipeline = build_full_pipeline(state_store=InMemoryStateStore(), freshness_weight=cfg["freshness_weight"],
                                       access_bonus_max=cfg["access_bonus_max"])
    reranker = None
    if cfg["rerank"]:
        if llm is None:
            raise ValueError("reranking needs an explicitly supplied LLM")
        reranker = Reranker(llm, max_items=20, rerank_mode=cfg["rerank_mode"])
    return recaller, pipeline, reranker


def compare(cases, configs, *, store, tenant, embedding=None, llm=None,
            cache_mode="cold", include_text=False, now=None, reverse=False):
    """Read existing data only. State is fresh per query/variant; no exposure writes."""
    cases = validate_cases(cases)
    if not isinstance(tenant, str) or not tenant.strip():
        raise ValueError("an explicit tenant scope is required")
    if set(configs) != {"A", "B"} or cache_mode not in ("cold", "warm"):
        raise ValueError("provide configs A/B and a cold/warm client-cache policy")
    configs = {arm: validate_config(cfg) for arm, cfg in configs.items()}
    if any(c["vector"] for c in configs.values()) and embedding is None:
        raise ValueError("missing embedding provider")
    if any(c["rerank"] for c in configs.values()) and llm is None:
        raise ValueError("missing reranking provider")
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("ranking clock requires a timezone")
    before, owned = snapshot(store, tenant)
    report = {"report_version": 1, "metrics_version": 2, "package_version": __version__,
              "dataset_sha256": digest(cases), "corpus_sha256": before, "configs": configs,
              "config_sha256": digest(configs), "clock": instant.isoformat(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "policy": {"state": "fresh in-memory state per query/arm; no feedback or exposure recording",
                         "cache": cache_mode, "query_vectors": "prepared once and shared across A/B",
                         "corpus": "tenant-scoped pre/post content+vector hash, not a transactional snapshot",
                         "time": "ranking stages frozen; relative-date retrieval and external services are not frozen"},
              "cases": []}
    source_root = Path(stages.__file__).resolve().parents[2]
    source_files = ("recall/flow.py", "recall/pipeline/stages.py", "recall/recaller.py",
                    "recall/retrievers.py", "recall/fusion.py", "vector/qdrant.py")
    report["recall_code_sha256"] = digest({name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in source_files})
    old_params = store.search_params
    old_query = embedding.embed_query if embedding is not None else None
    had_query_override = embedding is not None and "embed_query" in vars(embedding)
    query_override_installed = False
    try:
        if any(c["vector"] for c in configs.values()):
            queries = list(dict.fromkeys(case["query"] for case in cases))
            started = time.perf_counter()
            vectors = embedding.embed_queries(queries)
            if len(vectors) != len(queries) or any(not vector for vector in vectors):
                raise ValueError("query vector preparation failed")
            cached = dict(zip(queries, vectors, strict=True))
            embedding.embed_query = lambda query: list(cached[query])
            query_override_installed = True
            report["query_vectors_sha256"] = digest(cached)
            report["embedding_prepare_ms"] = (time.perf_counter() - started) * 1000
            report["embedding_name"] = embedding.name
        with ranking_clock(instant):
            for index, case in enumerate(cases):
                row = {"query_id": case["query_id"], "category": case.get("category"),
                       "negative_query": not case["expected_ids"],
                       "unresolved_expected_count": len(set(map(str, case["expected_ids"])) - owned), "arms": {}}
                if include_text:
                    row["query"] = case["query"]
                order = ["A", "B"] if (index % 2 == 0) != reverse else ["B", "A"]
                row["order"] = order
                for arm in order:
                    cfg = configs[arm]
                    store.search_params = quantization_search_params(hnsw_ef=cfg["hnsw_ef"])
                    recaller, pipeline, reranker = components(store, embedding, llm, cfg, tenant)
                    def run(trace, recaller=recaller, case=case, cfg=cfg, pipeline=pipeline, reranker=reranker):
                        return recall_flow(recaller, case["query"], cfg["limit"], pipeline=pipeline,
                                           reranker=reranker, tenant=tenant, trace=trace,
                                           filters=case.get("filters"), as_of=case.get("as_of"),
                                           include_invalidated=case.get("include_invalidated", False),
                                           token_budget=cfg["token_budget"])
                    trace = RecallTrace()
                    started = time.perf_counter()
                    warmup_faults, timed = [], False
                    try:
                        if cache_mode == "warm":
                            warmup_trace = RecallTrace()
                            run(warmup_trace)
                            warmup_faults = sorted(set(warmup_trace.degraded) - set(warmup_trace.notes))
                            if warmup_faults:
                                raise WarmupError("warm-up degraded")
                        started = time.perf_counter()
                        timed = True
                        hits = run(trace)
                        if any(str(h.id) not in owned or h.payload.get("tenant_id") != tenant for h in hits):
                            raise ValueError("result outside verified tenant snapshot")
                        result = {"ids": [str(h.id) for h in hits], "error_type": None}
                        if include_text:
                            result["texts"] = [h.text for h in hits]
                    except Exception as exc:
                        result = {"ids": [], "error_type": type(exc).__name__}
                    result["latency_ms"] = (time.perf_counter() - started) * 1000 if timed else None
                    result["warmup_ok"] = timed if cache_mode == "warm" else None
                    result["warmup_degraded"] = warmup_faults
                    result["returned_count"] = len(result["ids"])
                    result["metrics"] = metrics(result["ids"][:cfg["limit"]], case["expected_ids"])
                    result["returned_metrics"] = metrics(result["ids"], case["expected_ids"])
                    result["degraded"] = sorted((set(trace.degraded) - set(trace.notes)) | set(warmup_faults))
                    result["notes"] = list(trace.notes)
                    result["trace"] = project_trace(trace, owned)
                    result["loss_report"] = trace.loss_report([pid for pid in map(str, case["expected_ids"]) if pid in owned], cutoffs=(cfg["limit"],))
                    row["arms"][arm] = result
                row["changed"] = row["arms"]["A"]["ids"] != row["arms"]["B"]["ids"]
                report["cases"].append(row)
    finally:
        store.search_params = old_params
        if query_override_installed:
            if had_query_override:
                embedding.embed_query = old_query
            else:
                del embedding.embed_query
    after, _ = snapshot(store, tenant)
    report["corpus_unchanged"] = before == after
    report["valid_comparison"] = before == after
    report["summary"] = {}
    for arm in ("A", "B"):
        all_rows = [row["arms"][arm] for row in report["cases"]]
        scored = [row["arms"][arm] for row in report["cases"] if not row["negative_query"]]
        latencies = sorted(r["latency_ms"] for r in all_rows if r["latency_ms"] is not None)
        report["summary"][arm] = {"n": len(all_rows), "positive_n": len(scored),
            "errors": sum(r["error_type"] is not None for r in all_rows),
            "degraded_queries": sum(bool(r["degraded"]) for r in all_rows),
            "negative_queries_with_results": sum(bool(row["arms"][arm]["ids"]) for row in report["cases"] if row["negative_query"]),
            "timed_n": len(latencies),
            "latency_ms_median": median(latencies) if latencies else None,
            "latency_ms_p95": latencies[(95 * len(latencies) + 99) // 100 - 1] if latencies else None,
            **{metric: sum(r["metrics"][metric] for r in scored) / len(scored) if scored else None
               for metric in ("hit_rate", "recall", "mrr", "complete")}}
    positive = [row for row in report["cases"] if not row["negative_query"]]
    report["paired"] = {metric: {
        "gains": sum(not row["arms"]["A"]["metrics"][metric] and row["arms"]["B"]["metrics"][metric] for row in positive),
        "losses": sum(row["arms"]["A"]["metrics"][metric] and not row["arms"]["B"]["metrics"][metric] for row in positive),
    } for metric in ("hit_rate", "complete")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("queries", "config-a", "config-b", "collection", "tenant", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()
    from mnemostack.config import model_kwargs, provider_kwargs
    from mnemostack.embeddings import get_provider
    from mnemostack.llm import get_llm
    from mnemostack.vector import VectorStore

    store = None
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        cases = validate_cases(json.loads(Path(args.queries).read_text()))
        configs = {arm: validate_config(json.loads(Path(path).read_text())) for arm, path in (("A", args.config_a), ("B", args.config_b))}
        output = Path(args.output)
        if output.exists() or not output.parent.is_dir():
            raise InputError("choose a new output file in an existing directory")
        needs_models = any(c["vector"] or c["rerank"] for c in configs.values())
        if needs_models and not args.allow_model_calls:
            raise InputError("use --allow-model-calls for vector or reranking providers")
        embedding = get_provider(args.provider, **provider_kwargs(args.provider, model=args.embedding_model, ollama_host=args.ollama_host)) if any(c["vector"] for c in configs.values()) else None
        llm = get_llm("ollama", **model_kwargs(args.llm_model), host=args.ollama_host) if any(c["rerank"] for c in configs.values()) else None
        store = VectorStore(args.collection, dimension=embedding.dimension if embedding is not None else 1, host=args.qdrant)
        report = compare(cases, configs, store=store, tenant=args.tenant, embedding=embedding,
                         llm=llm, cache_mode=args.cache_mode, include_text=args.include_text)
        with output.open("x") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"valid_comparison": report["valid_comparison"], "summary": report["summary"]}))
        return 0 if report["valid_comparison"] and all(not s["errors"] and not s["degraded_queries"] for s in report["summary"].values()) else 1
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "status": "setup_failed"}
        if isinstance(exc, InputError):
            error["message"] = str(exc)
        print(json.dumps(error))
        return 2
    finally:
        logging.disable(previous_logging)
        if store is not None:
            store.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
