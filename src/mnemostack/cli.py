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
import json
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_PATHS, Config, generate_example_config
from .embeddings import get_provider, list_providers
from .llm import get_llm, list_llms
from .recall import (
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    Recaller,
    TemporalRetriever,
    VectorRetriever,
    build_bm25_docs,
)
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
        provider = get_provider(args.provider)
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


def cmd_search(args: argparse.Namespace) -> int:
    profile = _apply_tier(args)
    provider = get_provider(args.provider)
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

    recaller = _build_recaller(args, provider, store)
    results = recaller.recall(args.query, limit=args.limit)

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
                entry["payload"] = r.payload
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


def cmd_answer(args: argparse.Namespace) -> int:
    profile = _apply_tier(args)
    provider = get_provider(args.provider)
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

    recaller = _build_recaller(args, provider, store)
    results = recaller.recall(args.query, limit=args.limit)

    llm = get_llm(args.llm)
    gen = AnswerGenerator(llm=llm, confidence_threshold=args.min_confidence)
    answer = gen.generate(args.query, results)

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
            f"  mnemostack search \"{args.query}\" --provider {args.provider}",
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


def _build_recaller(args: argparse.Namespace, provider, store) -> Recaller:
    """Build the same retriever-mode Recaller used by the service surfaces."""
    bm25_docs = build_bm25_docs(list(getattr(args, "bm25_path", []) or []))
    retrievers = [
        VectorRetriever(embedding=provider, vector_store=store),
        BM25Retriever(docs=bm25_docs) if bm25_docs else None,
        MemgraphRetriever(uri=getattr(args, "memgraph_uri", None))
        if getattr(args, "memgraph_uri", None)
        else None,
        TemporalRetriever(embedding=provider, vector_store=store),
    ]
    return Recaller(retrievers=[r for r in retrievers if r is not None])


def cmd_index(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if not target.exists():
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 2

    provider = get_provider(args.provider)
    store = VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
    )
    store.ensure_collection(recreate=args.recreate)

    files = (
        [target] if target.is_file()
        else sorted(target.rglob("*.md")) + sorted(target.rglob("*.txt"))
    )
    if not files:
        print(f"error: no .md/.txt files found under {target}", file=sys.stderr)
        return 2

    chunks: list[tuple[str, str, dict]] = []  # (id, text, payload) triples
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        source = str(f.relative_to(target if target.is_dir() else target.parent))
        for i in range(0, len(text), args.chunk_size):
            chunk = text[i : i + args.chunk_size]
            if not chunk.strip():
                continue
            cid = _stable_chunk_id(source, i, chunk)
            chunks.append(
                (
                    cid,
                    chunk,
                    {
                        "text": chunk,
                        "source": source,
                        "offset": i,
                    },
                )
            )

    # Load existing point IDs once so re-runs skip unchanged chunks without
    # re-embedding (saves API quota / local GPU time).
    existing_ids: set[str] = set()
    if not args.recreate and store.collection_exists():
        existing_ids = {str(pid) for pid in store.iter_ids()}

    to_embed = [c for c in chunks if c[0] not in existing_ids]
    skipped = len(chunks) - len(to_embed)

    print(
        f"Indexing {len(chunks)} chunks from {len(files)} file(s)"
        f" — {len(to_embed)} new, {skipped} already indexed (skipped)."
    )

    inserted = 0
    failed = 0
    for cid, text, payload in to_embed:
        vec = provider.embed(text)
        if not vec:
            failed += 1
            continue
        store.upsert(cid, vec, payload)
        inserted += 1

    print(
        f"Done: inserted/updated {inserted}, skipped {skipped},"
        f" failed-embedding {failed}, total chunks seen {len(chunks)}"
        f" in collection '{args.collection}'."
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
        "--collection", default=cfg.vector.collection, help="Qdrant collection name"
    )
    common.add_argument(
        "--qdrant", default=cfg.vector.host, help="Qdrant URL"
    )

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
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "Progressive output budget: 1=list only (~50 tok), 2=snippets (~200 tok), "
            "3=detail (~500 tok). Omit for full output (back-compat)."
        ),
    )
    p_search.set_defaults(func=cmd_search)

    p_answer = sub.add_parser(
        "answer", parents=[common], help="Synthesize concise answer from memories"
    )
    p_answer.add_argument("query", help="Question to answer")
    p_answer.add_argument("--limit", type=int, default=cfg.recall.top_k, help="Max memories to consider")
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
        "--llm", default=cfg.llm.provider, choices=list_llms(), help="LLM provider for answer generation"
    )
    p_answer.add_argument(
        "--min-confidence",
        type=float,
        default=cfg.recall.confidence_threshold,
        help="Fallback suggestion threshold",
    )
    p_answer.add_argument("--json", action="store_true", help="JSON output")
    p_answer.set_defaults(func=cmd_answer)

    p_index = sub.add_parser("index", parents=[common], help="Index files into vector store")
    p_index.add_argument("path", help="File or directory to index")
    p_index.add_argument("--chunk-size", type=int, default=cfg.vector.chunk_size, help="Chunk size in chars")
    p_index.add_argument("--recreate", action="store_true", help="Drop existing collection")
    p_index.set_defaults(func=cmd_index)

    p_mcp = sub.add_parser(
        "mcp-serve",
        parents=[common],
        help="Run MCP server (stdio). Requires: pip install 'mnemostack[mcp]'",
    )
    p_mcp.add_argument(
        "--llm", default=cfg.llm.provider, help="LLM provider for answer generation"
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
        default="gemini",
        help="LLM provider for /answer (optional; disables /answer if missing)",
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
        default="/tmp/mnemostack-server-state.json",
        help="Pipeline state file path",
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
        llm_name=args.llm,
        collection=args.collection,
        qdrant_url=args.qdrant,
        graph_uri=args.memgraph_uri,
        graph_timeout=args.graph_timeout,
        bm25_paths=list(args.bm25_path) if args.bm25_path else None,
        state_path=args.state_path,
    )
    app = build_app(cfg)

    if args.host == "0.0.0.0":
        print(
            "warning: binding to 0.0.0.0 exposes the unauthenticated API on all interfaces",
            file=sys.stderr,
        )

    print(f"mnemostack serve: http://{args.host}:{args.port}")
    print(f"  provider:   {cfg.provider_name}")
    print(f"  collection: {cfg.collection}")
    print(f"  qdrant:     {cfg.qdrant_url}")
    print(f"  memgraph:   {cfg.graph_uri}")
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
    print(
        f"{action} {counts['nodes']} node(s) and "
        f"{counts['relationships']} relationship(s)."
    )
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Create an example config file at the standard location."""
    target = Path(args.path).expanduser() if args.path else DEFAULT_CONFIG_PATHS[0]
    if target.exists() and not args.force:
        print(f"error: config already exists at {target} (use --force to overwrite)", file=sys.stderr)
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
        llm_provider=args.llm,
        qdrant_host=args.qdrant,
        memgraph_uri=args.memgraph_uri,
        graph_timeout=args.graph_timeout,
        bm25_paths=list(args.bm25_path) if args.bm25_path else None,
    )
    mcp.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
