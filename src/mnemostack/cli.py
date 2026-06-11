"""mnemostack CLI — command-line interface for hybrid recall.

Commands:
    mnemostack health --provider ollama
    mnemostack search "query" --collection mnemostack --provider gemini
    mnemostack index <file|dir> --collection mnemostack --provider gemini

Most commands need a running Qdrant (default: http://localhost:6333) and a
configured embedding provider (GEMINI_API_KEY for gemini, or a running Ollama).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG_PATHS, Config, generate_example_config, model_kwargs
from .embeddings import get_provider, list_providers
from .llm import get_llm, list_llms
from .recall import (
    RERANK_MODES,
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    Recaller,
    Reranker,
    Retriever,
    TemporalRetriever,
    VectorRetriever,
    build_bm25_docs,
    recall_flow,
)
from .recall.pipeline import FileStateStore, build_full_pipeline, default_state_path
from .synthesis import synthesize
from .vector import VectorStore

# -- Progressive tiers --------------------------------------------------------
# Tiered output budgets let agents pay only for the detail they need.
# Tier 1: enumerate what's in memory (list view, ~50 tokens).
# Tier 2: short snippets around hits (default usable recall, ~200 tokens).
# Tier 3: fuller detail around hits, more results (~500 tokens).
# When --tier is omitted, behavior is unchanged (backward compatible).
TIER_PROFILES: dict[int, dict] = {
    1: {"limit": 5, "snippet_chars": 0, "max_sources": 2},
    2: {"limit": 5, "snippet_chars": 40, "max_sources": 2},
    3: {"limit": 10, "snippet_chars": 200, "max_sources": 3},
}


def _embedding_model(args: argparse.Namespace) -> str | None:
    return getattr(args, "embedding_model", None)


def _llm_model(args: argparse.Namespace) -> str | None:
    return getattr(args, "llm_model", None)


def _apply_tier(args: argparse.Namespace) -> dict | None:
    """Return the tier profile for this call, or None if no --tier was passed.

    Also collapses --limit into the tier-enforced limit so recall doesn't over-fetch.
    """
    tier = getattr(args, "tier", None)
    if tier is None:
        return None
    profile = TIER_PROFILES[tier]
    # Tier caps limit; user-provided --limit is ignored when smaller would waste work,
    # and clamped when larger would bust the token budget.
    args.limit = min(getattr(args, "limit", profile["limit"]), profile["limit"])
    return profile


def cmd_health(args: argparse.Namespace) -> int:
    try:
        provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"Provider: {provider.name} (dim={provider.dimension})")
    ok, msg = provider.health_check()
    status = "OK" if ok else "DOWN"
    print(f"  embedding: {status} — {msg}")

    try:
        store = VectorStore(
            collection=args.collection,
            dimension=provider.dimension,
            host=args.qdrant,
        )
        exists = store.collection_exists()
        count = store.count() if exists else 0
        print(f"  qdrant:    OK — collection '{args.collection}' exists={exists} count={count}")
    except Exception as e:  # noqa: BLE001
        print(f"  qdrant:    DOWN — {e}")
        return 1

    return 0 if ok else 1


def _load_enricher(spec: str):
    """Import a payload enricher from a 'package.module:function' spec."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        print("error: --enrich must look like 'package.module:function'", file=sys.stderr)
        raise SystemExit(2)
    try:
        func = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as e:
        print(f"error: cannot load --enrich {spec!r}: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    if not callable(func):
        print(f"error: --enrich {spec!r} is not callable", file=sys.stderr)
        raise SystemExit(2)
    return func


def _parse_filters(args: argparse.Namespace) -> dict | None:
    """Parse the --filters JSON argument; exits with code 2 on bad input."""
    raw = getattr(args, "filters", None)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"error: --filters must be valid JSON: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    if not isinstance(parsed, dict):
        print("error: --filters must be a JSON object", file=sys.stderr)
        raise SystemExit(2)
    return parsed


def _recall_for_cli(args: argparse.Namespace, recaller, query: str, limit: int):
    """Run the same recall flow as the HTTP and MCP servers.

    Applies the 8-stage pipeline and the fail-open LLM reranker so the CLI
    ranks results identically to the serving surfaces. `--raw` skips both
    and returns plain fused recall (the historical CLI behavior); filters
    apply on both paths.
    """
    filters = _parse_filters(args)
    if getattr(args, "raw", False):
        return recaller.recall(query, limit=limit, filters=filters)
    pipeline = build_full_pipeline(
        state_store=FileStateStore(default_state_path()),
        graph_uri=getattr(args, "memgraph_uri", None) or None,
    )
    reranker = None
    try:
        reranker = Reranker(
            llm=get_llm(getattr(args, "llm", "gemini"), **model_kwargs(_llm_model(args))),
            max_items=20,
            rerank_mode=getattr(args, "rerank_mode", None) or "relevant_only",
        )
    except Exception:  # noqa: BLE001 — no LLM key: search still works, unranked by LLM
        pass
    return recall_flow(
        recaller, query, limit, pipeline=pipeline, reranker=reranker, filters=filters
    )


def cmd_search(args: argparse.Namespace) -> int:
    profile = _apply_tier(args)
    provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    store = VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
    )
    if not store.collection_exists():
        print(
            f"error: collection '{args.collection}' does not exist. Run `mnemostack index` first.",
            file=sys.stderr,
        )
        return 2

    recaller = _build_recaller(args, provider, store)
    results = _recall_for_cli(args, recaller, args.query, args.limit)

    if args.json:
        snippet_chars = profile["snippet_chars"] if profile else None
        max_sources = profile["max_sources"] if profile else None
        payload = []
        for r in results:
            entry: dict = {
                "id": r.id,
                "score": round(r.score, 4),
                "sources": r.sources[:max_sources] if max_sources else r.sources,
            }
            if snippet_chars is None:
                # backward-compatible: full text + payload
                entry["text"] = r.text
                entry["payload"] = {
                    key: value
                    for key, value in r.payload.items()
                    if key != "_vector_floor_candidates"
                }
            elif snippet_chars > 0:
                entry["text"] = r.text[:snippet_chars]
            # tier 1 (snippet_chars == 0) emits no text at all
            payload.append(entry)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if not results:
            print("(no results)")
            return 0
        if profile is not None and profile["snippet_chars"] == 0:
            # Tier 1: list view — sources + score only, one line per hit
            for i, r in enumerate(results, 1):
                sources = ",".join(r.sources[: profile["max_sources"]]) or "?"
                print(f"[{i}] {r.score:.3f} {sources}")
            return 0
        preview_chars = profile["snippet_chars"] if profile else 200
        max_sources = profile["max_sources"] if profile else None
        for i, r in enumerate(results, 1):
            text = r.text or ""
            preview = (text[:preview_chars] + "...") if len(text) > preview_chars else text
            srcs = r.sources[:max_sources] if max_sources else r.sources
            sources = ",".join(srcs) or "?"
            print(f"[{i}] score={r.score:.4f} ({sources})")
            if preview:
                print(f"    {preview}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    sources = list(args.source) if args.source else None
    source_filter = _normalize_source_filter(sources)
    provider = None
    store = None
    if _source_enabled_for_cli("vector", source_filter) or _source_enabled_for_cli(
        "temporal", source_filter
    ):
        provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
        store = VectorStore(
            collection=args.collection,
            dimension=provider.dimension,
            host=args.qdrant,
        )
        if not store.collection_exists():
            print(
                f"error: collection '{args.collection}' does not exist. "
                f"Run `mnemostack index` first.",
                file=sys.stderr,
            )
            return 2

    recaller = _build_recaller(args, provider, store, source_filter=source_filter)
    llm = None
    if args.llm_summarize:
        llm = get_llm(args.llm, **model_kwargs(_llm_model(args)))
    result = synthesize(
        args.entity,
        sources=sources,
        format=args.format,
        max_results=args.limit,
        llm_summarize=args.llm_summarize,
        recaller=recaller,
        llm=llm,
    )
    if args.format == "json":
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    else:
        print(result.markdown(), end="")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    profile = _apply_tier(args)
    provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    store = VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
    )
    if not store.collection_exists():
        print(
            f"error: collection '{args.collection}' does not exist. Run `mnemostack index` first.",
            file=sys.stderr,
        )
        return 2

    recaller = _build_recaller(args, provider, store)
    results = _recall_for_cli(args, recaller, args.query, args.limit)

    llm = get_llm(args.llm, **model_kwargs(_llm_model(args)))
    answer_generator_kwargs = {
        "llm": llm,
        "confidence_threshold": args.min_confidence,
    }
    if getattr(args, "query_expansion", False):
        answer_generator_kwargs.update(
            {
                "recaller": recaller,
                "retry_with_expansion": True,
                "expansion_llm": llm,
            }
        )
    gen = AnswerGenerator(**answer_generator_kwargs)
    # recall_filters keeps retry sub-recalls inside the same filtered scope.
    answer = gen.generate(args.query, results, recall_filters=_parse_filters(args))

    # Tier caps how many sources we emit (answer text itself is model-sized).
    sources_out = answer.sources
    if profile is not None:
        sources_out = answer.sources[: profile["max_sources"]]

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "answer": answer.text,
                    "confidence": round(answer.confidence, 3),
                    "sources": sources_out,
                    "fallback_recommended": gen.should_fallback(answer),
                    "error": answer.error,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if answer.ok else 1

    if not answer.ok:
        print(f"error: {answer.error}", file=sys.stderr)
        return 1

    # Tier 1: answer + confidence only (no SOURCES list)
    if profile is not None and profile["snippet_chars"] == 0:
        print(f"ANSWER: {answer.text}")
        print(f"CONFIDENCE: {answer.confidence:.2f}")
    else:
        print(f"ANSWER: {answer.text}")
        print(f"CONFIDENCE: {answer.confidence:.2f}")
        if sources_out:
            print("SOURCES:")
            for s in sources_out:
                print(f"  - {s}")
    if gen.should_fallback(answer):
        print(
            f"\n⚠ Low confidence ({answer.confidence:.2f}) — consider reviewing raw memories:",
            file=sys.stderr,
        )
        print(
            f'  mnemostack search "{args.query}" --provider {args.provider}',
            file=sys.stderr,
        )
    return 0


def _stable_chunk_id(source: str, offset: int, text: str) -> str:
    """Deterministic UUID for an (source, offset, text) triple.

    Same inputs always produce the same id. That makes `mnemostack index` safe
    to re-run: unchanged chunks upsert onto themselves (no duplicates), and
    edited chunks produce a different id so old content can be cleaned up.
    """
    import hashlib
    import uuid

    digest = hashlib.sha256(f"{source}|{offset}|{text}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _normalize_source_filter(sources: list[str] | None) -> set[str] | None:
    if sources is None:
        return None
    aliases = {"graph": "memgraph"}
    return {aliases.get(source.lower(), source.lower()) for source in sources}


def _source_enabled_for_cli(name: str, source_filter: set[str] | None) -> bool:
    return source_filter is None or name in source_filter


def _build_recaller(
    args: argparse.Namespace,
    provider,
    store,
    source_filter: set[str] | None = None,
) -> Recaller:
    """Build the same retriever-mode Recaller used by the service surfaces."""
    retrievers: list[Retriever] = []
    if (
        provider is not None
        and store is not None
        and _source_enabled_for_cli("vector", source_filter)
    ):
        retrievers.append(VectorRetriever(embedding=provider, vector_store=store))
    if _source_enabled_for_cli("bm25", source_filter):
        bm25_docs = build_bm25_docs(list(getattr(args, "bm25_path", []) or []))
        if bm25_docs:
            retrievers.append(BM25Retriever(docs=bm25_docs))
    memgraph_uri = getattr(args, "memgraph_uri", None)
    if _source_enabled_for_cli("memgraph", source_filter) and memgraph_uri:
        retrievers.append(MemgraphRetriever(uri=memgraph_uri))
    if (
        provider is not None
        and store is not None
        and _source_enabled_for_cli("temporal", source_filter)
    ):
        retrievers.append(TemporalRetriever(embedding=provider, vector_store=store))
    query_expansion = bool(getattr(args, "query_expansion", False))
    expansion_llm = None
    if query_expansion:
        expansion_llm = get_llm(
            getattr(args, "llm", "gemini"),
            **model_kwargs(_llm_model(args)),
        )
    return Recaller(
        embedding_provider=provider,
        vector_store=store,
        retrievers=[r for r in retrievers if r is not None],
        query_expansion=query_expansion,
        expansion_llm=expansion_llm,
        vector_floor=max(0, int(getattr(args, "vector_floor", 0))),
    )


def cmd_index(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if not target.exists():
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 2

    provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    store = VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
    )
    if args.recreate and not args.yes:
        if not sys.stdin.isatty():
            print(
                "error: --recreate drops the collection; pass --yes to confirm "
                "in non-interactive mode",
                file=sys.stderr,
            )
            return 2
        points = store.count() if store.collection_exists() else 0
        reply = input(f"Drop collection '{args.collection}' ({points} points) and recreate? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("aborted")
            return 1
    store.ensure_collection(recreate=args.recreate)

    files = (
        [target]
        if target.is_file()
        else sorted(target.rglob("*.md")) + sorted(target.rglob("*.txt"))
    )
    if not files:
        print(f"error: no .md/.txt files found under {target}", file=sys.stderr)
        return 2

    if args.window_size < 1:
        print("error: --window-size must be >= 1", file=sys.stderr)
        return 2

    from .ingest import IngestItem, apply_enrichment

    enricher = _load_enricher(args.enrich) if args.enrich else None

    chunks: list[tuple[str, str, dict]] = []  # (id, text, payload) triples
    # Sources are stored relative to the indexed root, so different roots can
    # produce the same source name. Record the root in the payload (additive,
    # not part of the chunk id) so --prune can tell those documents apart.
    index_root = str((target if target.is_dir() else target.parent).resolve())
    # Every visited file counts as re-indexed even if it yields zero chunks
    # (emptied / whitespace-only) — its old chunks are exactly what --prune
    # must remove.
    visited_sources: set[str] = set()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        source = str(f.relative_to(target if target.is_dir() else target.parent))
        visited_sources.add(source)
        file_chunks: list[tuple[int, str]] = []
        for i in range(0, len(text), args.chunk_size):
            chunk = text[i : i + args.chunk_size]
            if not chunk.strip():
                continue
            file_chunks.append((i, chunk))
            cid = _stable_chunk_id(source, i, chunk)
            payload: dict[str, Any] = {
                "text": chunk,
                "source": source,
                "offset": i,
                "index_root": index_root,
            }
            if enricher is not None:
                apply_enrichment(enricher, IngestItem(text=chunk, source=source, offset=i), payload)
            chunks.append((cid, chunk, payload))
        if args.window_size > 1:
            for start in range(0, len(file_chunks) - args.window_size + 1):
                window = file_chunks[start : start + args.window_size]
                middle_offset, _middle_text = window[args.window_size // 2]
                chunk = "\n".join(piece for _offset, piece in window)
                cid = _stable_chunk_id(source, middle_offset, chunk)
                payload = {
                    "text": chunk,
                    "source": source,
                    "offset": middle_offset,
                    "index_root": index_root,
                    "chunk_window": args.window_size,
                    "chunk_kind": "sliding_window",
                    "chunk_start_offset": window[0][0],
                    "chunk_end_offset": window[-1][0],
                }
                if enricher is not None:
                    apply_enrichment(
                        enricher,
                        IngestItem(text=chunk, source=source, offset=middle_offset),
                        payload,
                    )
                chunks.append((cid, chunk, payload))

    # Load existing point IDs once so re-runs skip unchanged chunks without
    # re-embedding (saves API quota / local GPU time). When refreshing
    # payloads we also need each point's recorded root: a chunk id carries no
    # root, so an identical (source, offset, text) indexed from another root
    # shares the id — rewriting its index_root would hijack the point and
    # break prune isolation for the original root.
    existing_ids: set[str] = set()
    existing_roots: dict[str, Any] = {}
    if not args.recreate and store.collection_exists():
        if args.refresh_payloads:
            for hit in store.scroll():
                pid = str(hit.id)
                existing_ids.add(pid)
                existing_roots[pid] = hit.payload.get("index_root")
        else:
            existing_ids = {str(pid) for pid in store.iter_ids()}

    to_embed = [c for c in chunks if c[0] not in existing_ids]
    skipped = len(chunks) - len(to_embed)

    print(
        f"Indexing {len(chunks)} chunks from {len(files)} file(s)"
        f" — {len(to_embed)} new, {skipped} already indexed (skipped)."
    )

    inserted = 0
    failed = 0
    failed_sources: set[str] = set()
    for cid, text, payload in to_embed:
        vec = provider.embed(text)
        if not vec:
            failed += 1
            failed_sources.add(payload["source"])
            continue
        store.upsert(cid, vec, payload)
        inserted += 1

    refreshed = 0
    foreign_skipped = 0
    if args.refresh_payloads and existing_ids:
        # Payload-only rewrite of chunks that were skipped as already
        # indexed: applies new payload fields (enrichment output,
        # index_root) to existing points without paying for re-embedding.
        # Only points owned by this root — or unattributed legacy points
        # (no index_root yet; adopting them IS the migration path) — are
        # touched; a point recorded under another root is left alone.
        for cid, _text, payload in chunks:
            if cid not in existing_ids:
                continue
            owner = existing_roots.get(cid)
            if owner is not None and owner != index_root:
                foreign_skipped += 1
                continue
            store.set_payload(cid, payload)
            refreshed += 1
        if foreign_skipped:
            print(
                f"warning: {foreign_skipped} chunk(s) skipped by --refresh-payloads: "
                "owned by another index root",
                file=sys.stderr,
            )

    pruned = 0
    if args.prune and not args.recreate:
        from .ingest import prune_stale_chunks

        fresh_by_source: dict[str, set[str]] = {source: set() for source in visited_sources}
        for cid, _text, payload in chunks:
            fresh_by_source[payload["source"]].add(cid)
        # A source with a failed embedding has a fresh chunk that never landed.
        # Pruning it would delete the previous chunk without a replacement, so
        # leave such sources untouched this run.
        for source in failed_sources:
            fresh_by_source.pop(source, None)
        if failed_sources:
            print(
                f"warning: prune skipped for {len(failed_sources)} source(s) "
                "with failed embeddings; re-run after the provider recovers",
                file=sys.stderr,
            )
        pruned = prune_stale_chunks(store, fresh_by_source, index_root=index_root)

    print(
        f"Done: inserted/updated {inserted}, skipped {skipped},"
        f" failed-embedding {failed}, total chunks seen {len(chunks)}"
        + (f", refreshed {refreshed} payloads" if args.refresh_payloads else "")
        + (f", pruned {pruned} stale" if args.prune else "")
        + f" in collection '{args.collection}'."
    )
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    """Record explicit feedback into the pipeline state store."""
    from .feedback import apply_feedback
    from .recall.pipeline import FileStateStore, build_full_pipeline

    pipeline = build_full_pipeline(
        state_store=FileStateStore(args.state_path),
        graph_uri=None,
    )
    try:
        outcome = apply_feedback(
            pipeline,
            hit_id=args.hit_id,
            signal=args.signal,
            query=args.query,
            query_type=args.query_type,
            source=args.source,
            sources=list(args.source_list or []),
            reward=args.reward,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    payload = outcome.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"Recorded feedback for {payload['hit_id']}: "
            f"signal={payload['signal']} reward={payload['reward']:.3f} "
            f"query_type={payload['query_type']} "
            f"q_updates={payload['q_learning_updates']} "
            f"ior_recorded={payload['ior_recorded']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    cfg = Config.load()
    p = argparse.ArgumentParser(prog="mnemostack", description="Memory stack for AI agents")
    p.add_argument("--version", action="version", version=f"mnemostack {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider",
        default=cfg.embedding.provider,
        choices=list_providers(),
        help=f"Embedding provider (default: {cfg.embedding.provider})",
    )
    common.add_argument(
        "--embedding-model",
        default=cfg.embedding.model,
        help="Embedding model override (default: provider default or config value)",
    )
    common.add_argument(
        "--collection", default=cfg.vector.collection, help="Qdrant collection name"
    )
    common.add_argument("--qdrant", default=cfg.vector.host, help="Qdrant URL")

    p_health = sub.add_parser("health", parents=[common], help="Check stack health")
    p_health.set_defaults(func=cmd_health)

    p_search = sub.add_parser("search", parents=[common], help="Hybrid recall")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--limit", type=int, default=cfg.recall.top_k, help="Max results")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.add_argument(
        "--bm25-path",
        action="append",
        default=list(cfg.recall.bm25_paths),
        help="Directory/file to index for the BM25 retriever (can be given multiple times)",
    )
    p_search.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri,
        help="Memgraph URI to enable graph recall (e.g. bolt://localhost:7687)",
    )
    p_search.add_argument(
        "--raw",
        action="store_true",
        help="Skip the ranking pipeline and LLM reranker (plain fused recall)",
    )
    p_search.add_argument(
        "--rerank-mode",
        choices=sorted(RERANK_MODES),
        default=cfg.recall.rerank_mode,
        help="LLM reranker contract: relevant_only returns a subset, full_reorder ranks all",
    )
    p_search.add_argument(
        "--filters",
        default=None,
        help=(
            "JSON object of payload filters applied inside every retriever, "
            'e.g. \'{"tenant": "a"}\' or \'{"timestamp": {"gte": "2026-01-01"}}\''
        ),
    )
    p_search.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "Progressive output budget: 1=list only (~50 tok), 2=snippets (~200 tok), "
            "3=detail (~500 tok). Omit for full output (back-compat)."
        ),
    )
    p_search.add_argument(
        "--llm",
        default=cfg.llm.provider,
        choices=list_llms(),
        help="LLM provider for query expansion",
    )
    p_search.add_argument(
        "--llm-model",
        default=cfg.llm.model,
        help="LLM model override (default: provider default or config value)",
    )
    p_search.add_argument(
        "--query-expansion",
        action="store_true",
        help="Expand query with an LLM and fuse recall over original + variants",
    )
    p_search.add_argument(
        "--vector-floor",
        type=int,
        default=cfg.recall.vector_floor,
        help="Append missing top-N raw-vector candidates after fusion/rerank",
    )
    p_search.set_defaults(func=cmd_search)

    p_synthesize = sub.add_parser(
        "synthesize",
        parents=[common],
        help="Collect all known information about an entity",
    )
    p_synthesize.add_argument("entity", help="Entity name, handle, or identifier")
    p_synthesize.add_argument("--limit", type=int, default=50, help="Max facts to include")
    p_synthesize.add_argument(
        "--source",
        action="append",
        choices=["vector", "bm25", "graph", "memgraph", "temporal"],
        help="Retriever source to use (can be given multiple times; default: all available)",
    )
    p_synthesize.add_argument(
        "--bm25-path",
        action="append",
        default=list(cfg.recall.bm25_paths),
        help="Directory/file to index for the BM25 retriever (can be given multiple times)",
    )
    p_synthesize.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri,
        help="Memgraph URI to enable graph synthesis (e.g. bolt://localhost:7687)",
    )
    p_synthesize.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    p_synthesize.add_argument(
        "--llm-summarize",
        action="store_true",
        help="Run an optional LLM pass to produce a coherent summary",
    )
    p_synthesize.add_argument(
        "--llm", default=cfg.llm.provider, choices=list_llms(), help="LLM provider for summaries"
    )
    p_synthesize.add_argument(
        "--llm-model",
        default=cfg.llm.model,
        help="LLM model override (default: provider default or config value)",
    )
    p_synthesize.add_argument(
        "--query-expansion",
        action="store_true",
        help="Expand entity query with an LLM and fuse recall over original + variants",
    )
    p_synthesize.add_argument(
        "--vector-floor",
        type=int,
        default=cfg.recall.vector_floor,
        help="Append missing top-N raw-vector candidates after fusion/rerank",
    )
    p_synthesize.set_defaults(func=cmd_synthesize)

    p_answer = sub.add_parser(
        "answer", parents=[common], help="Synthesize concise answer from memories"
    )
    p_answer.add_argument("query", help="Question to answer")
    p_answer.add_argument(
        "--limit", type=int, default=cfg.recall.top_k, help="Max memories to consider"
    )
    p_answer.add_argument(
        "--bm25-path",
        action="append",
        default=list(cfg.recall.bm25_paths),
        help="Directory/file to index for the BM25 retriever (can be given multiple times)",
    )
    p_answer.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri,
        help="Memgraph URI to enable graph recall (e.g. bolt://localhost:7687)",
    )
    p_answer.add_argument(
        "--raw",
        action="store_true",
        help="Skip the ranking pipeline and LLM reranker (plain fused recall)",
    )
    p_answer.add_argument(
        "--rerank-mode",
        choices=sorted(RERANK_MODES),
        default=cfg.recall.rerank_mode,
        help="LLM reranker contract: relevant_only returns a subset, full_reorder ranks all",
    )
    p_answer.add_argument(
        "--filters",
        default=None,
        help=(
            "JSON object of payload filters applied inside every retriever, "
            'e.g. \'{"tenant": "a"}\' or \'{"timestamp": {"gte": "2026-01-01"}}\''
        ),
    )
    p_answer.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "Progressive output budget: 1=answer only (~50 tok), 2=answer+few sources (~200 tok), "
            "3=answer+more sources (~500 tok). Omit for full output (back-compat)."
        ),
    )
    p_answer.add_argument(
        "--llm",
        default=cfg.llm.provider,
        choices=list_llms(),
        help="LLM provider for answer generation",
    )
    p_answer.add_argument(
        "--llm-model",
        default=cfg.llm.model,
        help="LLM model override (default: provider default or config value)",
    )
    p_answer.add_argument(
        "--min-confidence",
        type=float,
        default=cfg.recall.confidence_threshold,
        help="Fallback suggestion threshold",
    )
    p_answer.add_argument(
        "--query-expansion",
        action="store_true",
        help="Expand query with an LLM and fuse recall over original + variants",
    )
    p_answer.add_argument(
        "--vector-floor",
        type=int,
        default=cfg.recall.vector_floor,
        help="Append missing top-N raw-vector candidates after fusion/rerank",
    )
    p_answer.add_argument("--json", action="store_true", help="JSON output")
    p_answer.set_defaults(func=cmd_answer)

    p_index = sub.add_parser("index", parents=[common], help="Index files into vector store")
    p_index.add_argument("path", help="File or directory to index")
    p_index.add_argument(
        "--chunk-size", type=int, default=cfg.vector.chunk_size, help="Chunk size in chars"
    )
    p_index.add_argument(
        "--window-size",
        type=int,
        default=cfg.vector.window_size,
        help="Adjacent chunks to concatenate into overlapping context chunks (1 disables)",
    )
    p_index.add_argument("--recreate", action="store_true", help="Drop existing collection")
    p_index.add_argument(
        "--enrich",
        default=None,
        help=(
            "Dotted path 'package.module:function' to a payload enricher: called "
            "with each chunk as an IngestItem, returns a dict merged into the "
            "chunk payload (fail-open; cannot override text/source/offset)"
        ),
    )
    p_index.add_argument(
        "--refresh-payloads",
        action="store_true",
        help=(
            "Rewrite payloads of already-indexed chunks in place, without "
            "re-embedding — applies --enrich output and other new payload "
            "fields to existing points"
        ),
    )
    p_index.add_argument(
        "--prune",
        action="store_true",
        help=(
            "After indexing, delete stale chunks of the indexed files — points whose "
            "ids the files no longer produce (edits shifted offsets, documents shrank). "
            "Scoped to this indexing root: other sources, same-named files indexed "
            "from other roots, and chunks indexed by versions that did not record a "
            "root are not touched."
        ),
    )
    p_index.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt for --recreate",
    )
    p_index.set_defaults(func=cmd_index)

    p_feedback = sub.add_parser(
        "feedback",
        help="Record explicit feedback for stateful recall learning",
    )
    p_feedback.add_argument("hit_id", help="Memory/result id the feedback refers to")
    p_feedback.add_argument(
        "--signal",
        required=True,
        choices=["useful", "irrelevant", "clicked"],
        help="Feedback signal",
    )
    p_feedback.add_argument("--query", default=None, help="Original query")
    p_feedback.add_argument(
        "--query-type",
        default=None,
        help="Explicit query type override (otherwise inferred from --query)",
    )
    p_feedback.add_argument(
        "--source",
        default=None,
        help="Single retriever/source label",
    )
    p_feedback.add_argument(
        "--source-list",
        action="append",
        default=[],
        help="Retriever/source label; can be given multiple times",
    )
    p_feedback.add_argument(
        "--reward",
        type=float,
        default=None,
        help="Optional reward override in [0, 1]",
    )
    p_feedback.add_argument(
        "--state-path",
        default=default_state_path(),
        help="Pipeline state file path",
    )
    p_feedback.add_argument("--json", action="store_true", help="JSON output")
    p_feedback.set_defaults(func=cmd_feedback)

    p_mcp = sub.add_parser(
        "mcp-serve",
        parents=[common],
        help="Run MCP server (stdio). Requires: pip install 'mnemostack[mcp]'",
    )
    p_mcp.add_argument("--llm", default=cfg.llm.provider, help="LLM provider for answer generation")
    p_mcp.add_argument(
        "--llm-model",
        default=cfg.llm.model,
        help="LLM model override (default: provider default or config value)",
    )
    p_mcp.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri,
        help="Memgraph URI to enable graph tools (e.g. bolt://localhost:7687)",
    )
    p_mcp.add_argument(
        "--graph-timeout",
        type=float,
        default=cfg.graph.timeout,
        help="Memgraph connection timeout in seconds (default 5.0)",
    )
    p_mcp.add_argument(
        "--bm25-path",
        action="append",
        default=list(cfg.recall.bm25_paths),
        help="Directory/file to index for the BM25 retriever (can be given multiple times)",
    )
    p_mcp.add_argument(
        "--state-path",
        default=default_state_path(),
        help="Pipeline state file path for feedback",
    )
    p_mcp.add_argument(
        "--vector-floor",
        type=int,
        default=cfg.recall.vector_floor,
        help="Append missing top-N raw-vector candidates after fusion/rerank",
    )
    p_mcp.add_argument(
        "--rerank-mode",
        choices=sorted(RERANK_MODES),
        default=cfg.recall.rerank_mode,
        help="LLM reranker contract: relevant_only returns a subset, full_reorder ranks all",
    )
    p_mcp.set_defaults(func=cmd_mcp_serve)

    p_init = sub.add_parser(
        "init", help="Create an example config file at ~/.config/mnemostack/config.yaml"
    )
    p_init.add_argument("--path", default=None, help="Custom config path")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")
    p_init.set_defaults(func=cmd_init)

    p_config = sub.add_parser("config", help="Show currently resolved config")
    p_config.add_argument(
        "--config", default=None, help="Explicit config path (overrides defaults)"
    )
    p_config.set_defaults(func=cmd_config_show)

    p_serve = sub.add_parser(
        "serve",
        parents=[common],
        help="Run HTTP API (FastAPI). Requires: pip install 'mnemostack[server]'",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    p_serve.add_argument(
        "--llm",
        default=cfg.llm.provider,
        help="LLM provider for /answer (optional; disables /answer if missing)",
    )
    p_serve.add_argument(
        "--llm-model",
        default=cfg.llm.model,
        help="LLM model override (default: provider default or config value)",
    )
    p_serve.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri or "bolt://localhost:7687",
        help="Memgraph bolt URI for the graph retriever",
    )
    p_serve.add_argument(
        "--graph-timeout",
        type=float,
        default=cfg.graph.timeout,
        help="Memgraph connection timeout in seconds (default 5.0)",
    )
    p_serve.add_argument(
        "--bm25-path",
        action="append",
        default=list(cfg.recall.bm25_paths),
        help="Directory/file to index for the BM25 retriever (can be given multiple times)",
    )
    p_serve.add_argument(
        "--state-path",
        default=default_state_path(),
        help="Pipeline state file path",
    )
    p_serve.add_argument(
        "--auto-record-ior",
        action="store_true",
        help="Record returned recall ids for inhibition-of-return state",
    )
    p_serve.add_argument(
        "--vector-floor",
        type=int,
        default=cfg.recall.vector_floor,
        help="Append missing top-N raw-vector candidates after fusion/rerank",
    )
    p_serve.add_argument(
        "--rerank-mode",
        choices=sorted(RERANK_MODES),
        default=cfg.recall.rerank_mode,
        help="LLM reranker contract: relevant_only returns a subset, full_reorder ranks all",
    )
    p_serve.add_argument(
        "--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)"
    )
    p_serve.set_defaults(func=cmd_serve)

    p_graph_migrate = sub.add_parser(
        "graph-migrate-current",
        help="Backfill legacy NULL graph validity markers to 'current'",
    )
    p_graph_migrate.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri or "bolt://localhost:7687",
        help="Memgraph bolt URI (default bolt://localhost:7687)",
    )
    p_graph_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count graph items that would be updated",
    )
    p_graph_migrate.add_argument(
        "--timeout",
        type=float,
        default=cfg.graph.timeout,
        help="Memgraph connection timeout in seconds (default 5.0)",
    )
    p_graph_migrate.set_defaults(func=cmd_graph_migrate_current)

    return p


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI HTTP server."""
    try:
        import uvicorn

        from mnemostack.server import ServerConfig, build_app
    except ImportError as exc:
        print(
            f"error: server extra not installed ({exc}). Install with: "
            "pip install 'mnemostack[server]'",
            file=sys.stderr,
        )
        return 2

    cfg = ServerConfig(
        provider_name=args.provider,
        embedding_model=_embedding_model(args),
        llm_name=args.llm,
        llm_model=_llm_model(args),
        collection=args.collection,
        qdrant_url=args.qdrant,
        graph_uri=args.memgraph_uri,
        graph_timeout=args.graph_timeout,
        bm25_paths=list(args.bm25_path) if args.bm25_path else None,
        vector_floor=max(0, int(args.vector_floor)),
        rerank_mode=args.rerank_mode,
        state_path=args.state_path,
        auto_record_ior=args.auto_record_ior,
    )
    app = build_app(cfg)

    if args.host == "0.0.0.0":
        print(
            "warning: binding to 0.0.0.0 exposes the unauthenticated API on all interfaces",
            file=sys.stderr,
        )

    print(f"mnemostack serve: http://{args.host}:{args.port}")
    print(f"  provider:   {cfg.provider_name}")
    if cfg.embedding_model:
        print(f"  embed model: {cfg.embedding_model}")
    if cfg.llm_model:
        print(f"  llm model:  {cfg.llm_model}")
    print(f"  collection: {cfg.collection}")
    print(f"  qdrant:     {cfg.qdrant_url}")
    print(f"  memgraph:   {cfg.graph_uri}")
    print(f"  state:      {cfg.state_path}")
    print(f"  auto IoR:   {cfg.auto_record_ior}")
    print(f"  docs:       http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_graph_migrate_current(args: argparse.Namespace) -> int:
    """Backfill legacy graph NULL validity markers to the explicit current marker."""
    from .graph import GraphStore

    store = GraphStore(uri=args.memgraph_uri, timeout=args.timeout)
    try:
        counts = store.backfill_current_markers(dry_run=args.dry_run)
    finally:
        store.close()

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {counts['nodes']} node(s) and {counts['relationships']} relationship(s).")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Create an example config file at the standard location."""
    target = Path(args.path).expanduser() if args.path else DEFAULT_CONFIG_PATHS[0]
    if target.exists() and not args.force:
        print(
            f"error: config already exists at {target} (use --force to overwrite)", file=sys.stderr
        )
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generate_example_config())
    print(f"Config written to {target}")
    print("Edit it and re-run `mnemostack health` to verify.")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    """Print the currently resolved config (file + env overrides)."""
    import yaml

    cfg = Config.load(args.config)
    print(yaml.safe_dump(cfg.to_dict(), default_flow_style=False, sort_keys=False))
    return 0


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    try:
        from .mcp import build_server
    except ImportError as e:
        print(
            f"error: MCP server requires fastmcp. Install with: pip install 'mnemostack[mcp]'\n{e}",
            file=sys.stderr,
        )
        return 2

    mcp = build_server(
        collection=args.collection,
        embedding_provider=args.provider,
        embedding_model=_embedding_model(args),
        llm_provider=args.llm,
        llm_model=_llm_model(args),
        qdrant_host=args.qdrant,
        memgraph_uri=args.memgraph_uri,
        graph_timeout=args.graph_timeout,
        bm25_paths=list(args.bm25_path) if args.bm25_path else None,
        state_path=args.state_path,
        vector_floor=max(0, int(args.vector_floor)),
        rerank_mode=args.rerank_mode,
    )
    mcp.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
