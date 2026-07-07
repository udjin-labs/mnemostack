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
import os
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
    sum_tokens,
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


# ----- doctor: comprehensive deployment diagnostic -----

# Check statuses, ordered by severity. `misconfig` (invalid config) forces exit
# 2, `down` (a core recall dependency — embedding or Qdrant — unreachable) forces
# exit 1; `warn` / `disabled` / `ok` never fail the exit code. The LLM (answer is
# optional) and the graph (optional/fail-soft — recall works without it) never
# exceed `warn`, so a missing LLM or an unreachable graph is reported but does
# not fail a deploy/CI gate wired to the exit code.
_DOCTOR_MARKERS = {
    "ok": "OK",
    "warn": "WARN",
    "down": "DOWN",
    "misconfig": "MISCONFIG",
    "disabled": "OFF",
}


def _doctor_qdrant(add, url: str, collection: str, timeout: int, expected_dim: int | None) -> None:
    """Read-only Qdrant probe: reachability, collection, count, dimension match.

    Never calls ensure_collection — a diagnostic must not create a collection as
    a side effect. Mirrors VectorStore._validate_dimension's read to compare the
    stored vector size against the provider's dimension.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:  # pragma: no cover - qdrant-client is a core dep
        add("qdrant", "down", f"qdrant-client not importable: {e}")
        return
    try:
        client = QdrantClient(url=url, timeout=timeout)
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        add(
            "qdrant",
            "down",
            f"unreachable at {url}: {e}",
            "check the Qdrant URL and that the service is running",
        )
        return
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse

        info = client.get_collection(collection)
    except ValueError as e:
        # Narrow — only a genuine "not found" is the pre-ingest case. Any other
        # ValueError (or the non-404 / non-ValueError errors below) is a real
        # failure and must read as `down`, not be masked as "collection missing".
        if "not found" not in str(e).lower():
            add("qdrant", "down", f"reachable at {url}; collection query failed: {e}")
            return
        add(
            "qdrant",
            "warn",
            f"reachable at {url}; collection '{collection}' does not exist yet",
            "index data (mnemostack index ...) to create the collection",
        )
        return
    except UnexpectedResponse as e:
        if getattr(e, "status_code", None) != 404:
            add(
                "qdrant",
                "down",
                f"reachable at {url}; collection query failed ({e})",
                "check Qdrant auth / permissions for this collection",
            )
            return
        add(
            "qdrant",
            "warn",
            f"reachable at {url}; collection '{collection}' does not exist yet",
            "index data (mnemostack index ...) to create the collection",
        )
        return
    except Exception as e:  # noqa: BLE001
        add("qdrant", "down", f"reachable at {url}; collection query failed: {e}")
        return
    count = getattr(info, "points_count", None) or 0
    size = None
    try:
        # A named-vectors config is a dict with no `.size`; skip the check then,
        # exactly as VectorStore._validate_dimension does.
        size = getattr(info.config.params.vectors, "size", None)
    except Exception:  # noqa: BLE001
        size = None
    detail = f"reachable at {url}; collection '{collection}' exists, points={count}"
    if expected_dim is not None and size is not None and int(size) != int(expected_dim):
        add(
            "qdrant",
            "down",
            detail + f"; VECTOR SIZE MISMATCH: stored={size}, provider={expected_dim}",
            "the collection was built with a different embedding model/dimension; "
            "re-index into a fresh collection or use the original provider/model",
        )
        return
    if size is not None:
        detail += f", dim={size}"
    if count == 0:
        add("qdrant", "warn", detail + " (empty — no data indexed)")
    else:
        add("qdrant", "ok", detail)


def _doctor_llm(add, name: str, model: str | None, live: bool) -> None:
    """LLM reachability. Never exceeds `warn` — /answer is optional."""
    # Case-insensitive: get_llm() lowercases names, so a mixed-case config value
    # is valid.
    if name.lower() not in list_llms():
        add(
            "llm",
            "warn",
            f"unknown provider '{name}' — /answer will be unavailable",
            f"set llm.provider to one of: {', '.join(sorted(list_llms()))}",
        )
        return
    try:
        llm = get_llm(name, **model_kwargs(model))
    except Exception as e:  # noqa: BLE001
        add(
            "llm",
            "warn",
            f"{name} not usable: {e} — /answer will be unavailable",
            "answer is optional; configure the LLM only if you use /answer",
        )
        return
    if not live:
        add("llm", "ok", f"{name} configured (pass --check-llm for a live probe)")
        return
    try:
        ok, msg = llm.health_check()
        add(
            "llm",
            "ok" if ok else "warn",
            f"{name} — {msg}" + ("" if ok else " (/answer would fail)"),
            None if ok else "check the LLM API key / host",
        )
    except Exception as e:  # noqa: BLE001
        add("llm", "warn", f"{name}: {e}")


def _doctor_graph(add, cfg) -> None:
    """Graph reachability. The graph is optional/fail-soft — recall works without
    it — so a configured-but-unreachable graph is a `warn`, not a `down`: it is
    reported prominently but does not fail the exit code (mirrors /readyz, which
    deliberately doesn't gate readiness on the graph). Disabled (uri unset) is
    reported as such, not a failure."""
    if not cfg.graph.uri:
        add("graph", "disabled", "not configured (graph is optional)")
        return
    store = None
    try:
        from .graph.factory import make_graph_store

        store = make_graph_store(
            cfg.graph.uri,
            timeout=cfg.graph.health_timeout,
            user=cfg.graph.user,
            password=cfg.graph.password,
            database=cfg.graph.database,
        )
        ok, msg = store.health_check()
        if ok:
            add(
                "graph",
                "ok",
                f"reachable at {cfg.graph.uri}; nodes={store.count_nodes()} "
                f"edges={store.count_edges()}",
            )
        else:
            add(
                "graph",
                "warn",
                f"configured but unreachable at {cfg.graph.uri}: {msg} "
                "(optional — recall still works; graph recall is skipped)",
                "check the Memgraph/Neo4j URI, auth, and that the service is running",
            )
    except Exception as e:  # noqa: BLE001
        add(
            "graph",
            "warn",
            f"configured but unreachable at {cfg.graph.uri}: {e} "
            "(optional — recall still works; graph recall is skipped)",
            "check the Memgraph/Neo4j URI, auth, and that the service is running",
        )
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass


def _api_key_hint(cfg) -> str | None:
    key_env = getattr(cfg.embedding, "api_key_env", None)
    if key_env and cfg.embedding.provider == "gemini" and not os.environ.get(key_env):
        return f"set {key_env} (required for the gemini provider)"
    return None


def cmd_tenant_migrate(args: argparse.Namespace) -> int:
    """Stamp a tenant_id onto existing points (single-tenant → multi-tenant).

    Read-only until you drop ``--dry-run``: reports how many points would be
    stamped. By default only points that lack a tenant_id are touched, so it's
    safe to re-run — a payload-only merge (ids and vectors untouched). After
    migrating, run the first tenant-scoped re-ingest with ``--prune`` so its
    tenant-scoped ids replace the migrated points instead of duplicating them.
    """
    try:
        store = VectorStore(
            collection=args.collection,
            dimension=1,  # dimension is irrelevant for a payload-only migration
            host=args.qdrant,
        )
        exists = store.collection_exists()  # first network call — inside the guard
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot reach Qdrant at {args.qdrant}: {e}", file=sys.stderr)
        return 1
    if not exists:
        print(f"error: collection '{args.collection}' does not exist", file=sys.stderr)
        return 1

    only_missing = not args.all
    from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

    from mnemostack.vector.qdrant import TENANT_ID_KEY

    # Exact count (client.count), not the approximate CollectionInfo.points_count
    # that VectorStore.count() reads — the --all safety check below relies on
    # (total - missing) to detect pre-stamped points, so an under-report there
    # could let --all silently relabel an existing tenant.
    try:
        total = store.client.count(collection_name=args.collection).count
        missing = store.client.count(
            collection_name=args.collection,
            count_filter=Filter(must=[IsEmptyCondition(is_empty=PayloadField(key=TENANT_ID_KEY))]),
        ).count
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot reach Qdrant at {args.qdrant}: {e}", file=sys.stderr)
        return 1
    # Safety: --all relabels points that already carry a tenant_id (legacy user
    # metadata OR a real other-tenant owner), so require --yes to confirm — this
    # keeps an accidental --all from collapsing a multi-tenant collection into one
    # tenant, while still giving a cleanup path for a legacy collection where
    # tenant_id was an ordinary payload field. (Default only-missing is always safe.)
    if args.all and (total - missing) > 0 and not args.yes:
        print(
            f"error: {total - missing} point(s) already carry a tenant_id; --all would "
            f"relabel them all to '{args.tenant}'. Pass --yes to confirm (use this to "
            "migrate a legacy collection where tenant_id was an ordinary payload field); "
            "run without --all to stamp only unassigned points.",
            file=sys.stderr,
        )
        return 2

    # Preflight the GRAPH --all relabel gate BEFORE mutating Qdrant, so a graph
    # that already carries tenants can't force the command to stamp the vectors
    # and then abort — leaving the collection migrated but the graph untouched.
    if args.all and not args.dry_run and not args.yes:
        rc = _graph_relabel_preflight(args)
        if rc != 0:
            return rc

    pending = missing if only_missing else total
    scope = "without a tenant_id" if only_missing else "in the collection"
    if args.dry_run:
        print(f"dry-run: would stamp tenant_id='{args.tenant}' onto {pending} point(s) {scope}")
        return _graph_stamp_tenant(args, only_missing, dry_run=True)
    try:
        stamped = store.stamp_tenant(args.tenant, only_missing=only_missing)
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot reach Qdrant at {args.qdrant}: {e}", file=sys.stderr)
        return 1
    print(f"stamped tenant_id='{args.tenant}' onto {stamped} point(s) {scope}")
    return _graph_stamp_tenant(args, only_missing, dry_run=False)


def _graph_stamp_tenant(args: argparse.Namespace, only_missing: bool, *, dry_run: bool) -> int:
    """Stamp the tenant onto the graph too, when ``--memgraph-uri`` is given.

    The graph twin of the vector stamp above: adopts tenancy on an existing
    single-tenant graph so tenant-scoped graph recall (which filters on the
    node/edge ``tenant`` property) can see the migrated facts. Opt-in — a
    vector-only deployment passes no ``--memgraph-uri`` and this is a no-op.
    Returns an exit code (0 ok, 1 graph unreachable) so a graph failure surfaces.
    """
    memgraph_uri = getattr(args, "memgraph_uri", None) or None
    if not memgraph_uri:
        return 0
    from .graph.factory import make_graph_store

    try:
        gs = make_graph_store(
            memgraph_uri,
            timeout=getattr(args, "graph_timeout", 5.0),
            **_graph_auth(args),
        )
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot reach graph at {memgraph_uri}: {e}", file=sys.stderr)
        return 1
    try:
        counts = gs.stamp_tenant(args.tenant, only_missing=only_missing, dry_run=dry_run)
    except Exception as e:  # noqa: BLE001
        print(f"error: graph tenant stamp failed: {e}", file=sys.stderr)
        return 1
    finally:
        gs.close()
    action = "would stamp" if dry_run else "stamped"
    print(
        f"graph: {action} tenant='{args.tenant}' onto {counts['nodes']} node(s) "
        f"and {counts['relationships']} relationship(s)"
    )
    return 0


def _graph_relabel_preflight(args: argparse.Namespace) -> int:
    """Gate a graph ``--all`` force-relabel behind ``--yes`` when the graph already
    carries tenants — run *before* any store is mutated (see ``cmd_tenant_migrate``).

    The vector ``--yes`` gate only inspects Qdrant, so a graph that already holds
    tenant values (a real multi-tenant graph) could otherwise be collapsed into one
    tenant. Counts both **nodes and relationships** carrying a tenant, since the
    relabel rewrites edges too. Returns 0 (ok / no graph), 1 (unreachable), or 2
    (refused — needs ``--yes``).
    """
    if getattr(args, "yes", False):
        return 0  # operator confirmed the relabel
    memgraph_uri = getattr(args, "memgraph_uri", None) or None
    if not memgraph_uri:
        return 0
    from .graph.factory import make_graph_store

    try:
        gs = make_graph_store(
            memgraph_uri,
            timeout=getattr(args, "graph_timeout", 5.0),
            **_graph_auth(args),
        )
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot reach graph at {memgraph_uri}: {e}", file=sys.stderr)
        return 1
    try:
        total = gs.stamp_tenant(args.tenant, only_missing=False, dry_run=True)
        missing = gs.stamp_tenant(args.tenant, only_missing=True, dry_run=True)
    except Exception as e:  # noqa: BLE001
        print(f"error: graph tenant check failed: {e}", file=sys.stderr)
        return 1
    finally:
        gs.close()
    already = (total["nodes"] - missing["nodes"]) + (
        total["relationships"] - missing["relationships"]
    )
    if already > 0:
        print(
            f"error: {already} graph record(s) (nodes + edges) already carry a tenant; "
            f"--all would relabel every node/edge to '{args.tenant}'. Pass --yes to "
            "confirm, or run without --all to stamp only untenanted records.",
            file=sys.stderr,
        )
        return 2
    return 0


def _keys_store(args: argparse.Namespace):
    from mnemostack.auth import FileKeyStore

    return FileKeyStore(getattr(args, "keys_file", None) or None)


def cmd_keys_add(args: argparse.Namespace) -> int:
    """Issue a service key for a tenant. Prints the key once (never stored)."""
    from mnemostack.auth import KeyStoreError

    try:
        key_id, key = _keys_store(args).issue(args.tenant, args.scopes, args.label or "")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    scopes = ",".join(sorted({s.strip() for s in args.scopes.split(",") if s.strip()}))
    print(f"key id:  {key_id}")
    print(f"tenant:  {args.tenant}")
    print(f"scopes:  {scopes}")
    print("\nkey (shown once — store it now, it is not recoverable):")
    print(f"  {key}")
    return 0


def cmd_keys_list(args: argparse.Namespace) -> int:
    from mnemostack.auth import KeyStoreError

    try:
        keys = _keys_store(args).list_keys()
    except KeyStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(keys, ensure_ascii=False, indent=2))
        return 0
    if not keys:
        print("(no keys)")
        return 0
    print(f"{'ID':<10} {'TENANT':<20} {'SCOPES':<20} {'CREATED':<22} LABEL")
    for k in keys:
        print(
            f"{k.get('id', ''):<10} {k.get('tenant', ''):<20} "
            f"{','.join(k.get('scopes', [])):<20} {k.get('created_at', ''):<22} {k.get('label', '')}"
        )
    return 0


def cmd_keys_revoke(args: argparse.Namespace) -> int:
    from mnemostack.auth import KeyStoreError

    try:
        removed = _keys_store(args).revoke(args.id)
    except KeyStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if removed:
        print(f"revoked key {args.id}")
        return 0
    print(f"error: no key with id '{args.id}'", file=sys.stderr)
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose a mnemostack deployment: config, dependencies, versions.

    Read-only — never creates or modifies a collection. Exit codes: 0 = healthy,
    1 = a core recall dependency is down, 2 = configuration is invalid.
    """
    import platform

    cfg = Config.load()
    checks: list[dict[str, Any]] = []

    def add(section: str, status: str, detail: str, hint: str | None = None) -> None:
        checks.append({"section": section, "status": status, "detail": detail, "hint": hint})

    add("mnemostack", "ok", f"version {__version__}")
    add("python", "ok", platform.python_version())

    # Config validation. Provider name comes from args (seeded from config, so
    # it reflects the deployed config file / env unless overridden on the CLI).
    # Compare case-insensitively — get_provider() lowercases names, so a
    # mixed-case config value (e.g. MNEMOSTACK_PROVIDER=OLLAMA) is valid.
    provider_name = args.provider
    provider_known = provider_name.lower() in list_providers()
    if provider_known:
        add("config.embedding_provider", "ok", provider_name)
    else:
        add(
            "config.embedding_provider",
            "misconfig",
            f"unknown provider '{provider_name}'",
            f"set embedding.provider to one of: {', '.join(sorted(list_providers()))}",
        )
    if cfg.recall.rerank_mode in RERANK_MODES:
        add("config.rerank_mode", "ok", cfg.recall.rerank_mode)
    else:
        add(
            "config.rerank_mode",
            "misconfig",
            f"invalid rerank_mode '{cfg.recall.rerank_mode}'",
            f"set recall.rerank_mode to one of: {', '.join(sorted(RERANK_MODES))}",
        )

    # Embedding provider (a hard recall dependency).
    provider = None
    if provider_known:
        try:
            provider = get_provider(provider_name, **model_kwargs(_embedding_model(args)))
        except ValueError as e:
            add("embedding", "misconfig", str(e), _api_key_hint(cfg))
        except Exception as e:  # noqa: BLE001
            add("embedding", "down", str(e))
        if provider is not None:
            try:
                ok, msg = provider.health_check()
                add(
                    "embedding",
                    "ok" if ok else "down",
                    f"{provider.name} (dim={provider.dimension}) — {msg}",
                    None if ok else "check the provider is reachable and the API key/host is valid",
                )
            except Exception as e:  # noqa: BLE001
                add("embedding", "down", f"{provider.name}: {e}")

    # Qdrant (a hard recall dependency), read-only.
    expected_dim = provider.dimension if provider is not None else None
    _doctor_qdrant(add, args.qdrant, args.collection, cfg.vector.health_timeout, expected_dim)

    # LLM (optional — answer only) and graph (optional).
    _doctor_llm(add, cfg.llm.provider, cfg.llm.model, bool(getattr(args, "check_llm", False)))
    _doctor_graph(add, cfg)

    statuses = [c["status"] for c in checks]
    exit_code = 2 if "misconfig" in statuses else (1 if "down" in statuses else 0)

    if getattr(args, "json", False):
        summary = {s: statuses.count(s) for s in _DOCTOR_MARKERS}
        print(
            json.dumps(
                {
                    "version": __version__,
                    "checks": checks,
                    "summary": summary,
                    "exit_code": exit_code,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for c in checks:
            marker = _DOCTOR_MARKERS.get(c["status"], c["status"]).ljust(9)
            print(f"[{marker}] {c['section']}: {c['detail']}")
            if c["hint"]:
                print(f"{' ' * 12}↳ {c['hint']}")
        verdict = {
            0: "healthy",
            1: "degraded — a core dependency is down",
            2: "misconfigured — fix the config and re-run",
        }[exit_code]
        print(f"\ndoctor: {verdict}")

    return exit_code


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


def _effective_token_budget(args: argparse.Namespace) -> int | None:
    """Token budget from CLI flags; 0 or negative means "no budget"."""
    budget = getattr(args, "token_budget", None)
    return budget if budget is not None and budget > 0 else None


def _add_validity_recall_flags(parser: argparse.ArgumentParser) -> None:
    """Shared --include-invalidated / --as-of flags for search and answer."""
    parser.add_argument(
        "--include-invalidated",
        action="store_true",
        help="Include facts marked stale (default: invalidated facts are hidden)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        metavar="ISO",
        help="Point-in-time recall: return facts valid at this world-time instant",
    )


def _recall_for_cli(args: argparse.Namespace, recaller, query: str, limit: int):
    """Run the same recall flow as the HTTP and MCP servers.

    Applies the 8-stage pipeline and the fail-open LLM reranker so the CLI
    ranks results identically to the serving surfaces. `--raw` skips both
    and returns plain fused recall (the historical CLI behavior); filters
    apply on both paths.
    """
    filters = _parse_filters(args)
    token_budget = _effective_token_budget(args)
    include_invalidated = bool(getattr(args, "include_invalidated", False))
    as_of = getattr(args, "as_of", None)
    if getattr(args, "raw", False):
        return recaller.recall(
            query,
            limit=limit,
            filters=filters,
            token_budget=token_budget,
            include_invalidated=include_invalidated,
            as_of=as_of,
        )
    pipeline = build_full_pipeline(
        state_store=FileStateStore(default_state_path()),
        graph_uri=getattr(args, "memgraph_uri", None) or None,
        graph_timeout=getattr(args, "graph_timeout", 5.0),
        **{f"graph_{k}": v for k, v in _graph_auth(args).items()},
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
        recaller,
        query,
        limit,
        pipeline=pipeline,
        reranker=reranker,
        filters=filters,
        token_budget=token_budget,
        include_invalidated=include_invalidated,
        as_of=as_of,
    )


def cmd_invalidate(args: argparse.Namespace) -> int:
    """Mark memories stale by id (non-destructive; no re-embedding)."""
    # Invalidation is a payload write — no embeddings needed, so skip provider
    # construction (which for some providers loads a model). The dimension is
    # unused by set_payload/retrieve.
    store = VectorStore(collection=args.collection, dimension=1, host=args.qdrant)
    if not store.collection_exists():
        print(
            f"error: collection '{args.collection}' does not exist.",
            file=sys.stderr,
        )
        return 2
    # argparse hands every id in as a string, but Qdrant integer point ids are
    # distinct from string ids — coerce digit-only args so numeric-id
    # collections can be invalidated (UUIDs contain hyphens, so stay strings).
    ids = [int(x) if x.isdigit() else x for x in args.ids]
    updated = store.invalidate(
        ids,
        invalidated_at=args.invalidated_at,
        valid_until=args.valid_until,
        index_root=args.index_root,
    )
    if args.json:
        print(json.dumps({"invalidated": updated, "requested": len(args.ids)}))
    else:
        print(f"invalidated {updated}/{len(args.ids)} point(s)")
    return 0


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
    # recall_filters, the token budget, and the validity view must all reach
    # the generator too, or its retry paths would run fresh sub-recalls that
    # ignore them (an --as-of retry would answer from current facts).
    answer = gen.generate(
        args.query,
        results,
        recall_filters=_parse_filters(args),
        token_budget=_effective_token_budget(args),
        include_invalidated=bool(getattr(args, "include_invalidated", False)),
        as_of=getattr(args, "as_of", None),
    )

    # Tier caps how many sources we emit (answer text itself is model-sized).
    sources_out = answer.sources
    if profile is not None:
        sources_out = answer.sources[: profile["max_sources"]]

    # Prefer the generator's own estimate: its retry paths can swap in a
    # freshly recalled context pool that differs from the primary results.
    tokens_estimate = getattr(answer, "context_tokens_estimate", None)
    if tokens_estimate is None:
        tokens_estimate = sum_tokens(results)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "answer": answer.text,
                    "confidence": round(answer.confidence, 3),
                    "sources": sources_out,
                    "fallback_recommended": gen.should_fallback(answer),
                    "tokens_estimate": tokens_estimate,
                    "tokens_used": getattr(answer, "tokens_used", None),
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


def _graph_auth(args: argparse.Namespace) -> dict[str, Any]:
    """Graph auth (user/password/database) seeded onto the namespace by build_parser."""
    return {
        "user": getattr(args, "graph_user", "") or "",
        "password": getattr(args, "graph_password", "") or "",
        "database": getattr(args, "graph_database", None),
    }


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
        retrievers.append(MemgraphRetriever(uri=memgraph_uri, **_graph_auth(args)))
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
                    # Window metadata travels on the item so enrichers shared
                    # with Ingestor(enrich=...) can branch on chunk_kind /
                    # window offsets for CLI window chunks too.
                    apply_enrichment(
                        enricher,
                        IngestItem(
                            text=chunk,
                            source=source,
                            offset=middle_offset,
                            metadata={
                                "chunk_window": args.window_size,
                                "chunk_kind": "sliding_window",
                                "chunk_start_offset": window[0][0],
                                "chunk_end_offset": window[-1][0],
                            },
                        ),
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
    existing_payloads: dict[str, dict] = {}
    if not args.recreate and store.collection_exists():
        if args.refresh_payloads:
            for hit in store.scroll():
                pid = str(hit.id)
                existing_ids.add(pid)
                existing_payloads[pid] = hit.payload or {}
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
            old_payload = existing_payloads.get(cid, {})
            owner = old_payload.get("index_root")
            if owner is not None and owner != index_root:
                foreign_skipped += 1
                continue
            # set_payload merges, so enrichment keys the new run no longer
            # produces would survive and keep feeding filters/context_fields
            # stale facts. The _enrich_keys ownership record names exactly
            # what the previous enrichment wrote — delete what it owned and
            # the new payload doesn't claim, leaving foreign fields alone.
            old_enrich = old_payload.get("_enrich_keys") or []
            stale_keys = [k for k in old_enrich if k not in payload]
            if old_enrich and "_enrich_keys" not in payload:
                stale_keys.append("_enrich_keys")
            store.delete_payload_keys(cid, stale_keys)
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


def cmd_index_markdown(args: argparse.Namespace) -> int:
    """Index a markdown folder: frontmatter -> payload, links -> graph edges."""
    from .markdown import collect_markdown

    # Resolve to an absolute path so it's comparable with a resolved --index-root
    # when computing corpus-relative sources (relative_to needs both same-form).
    target = Path(args.path).resolve()
    if not target.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 2
    if args.chunk_size <= 0:
        print(
            f"error: --chunk-size must be a positive integer, got {args.chunk_size}",
            file=sys.stderr,
        )
        return 2
    watching = bool(getattr(args, "watch", False))
    if watching and not target.is_dir():
        print("error: --watch requires a directory to watch", file=sys.stderr)
        return 2
    _raw_tenant = getattr(args, "tenant", None)
    if _raw_tenant is not None and not str(_raw_tenant).strip():
        # An explicitly empty --tenant (e.g. `--tenant "$UNSET_VAR"`) must fail
        # closed, not silently normalize to an unscoped run — otherwise
        # `--tenant "" --recreate` would slip past the guard below and drop the
        # whole shared collection. Omit --tenant entirely for an unscoped index.
        print(
            "error: --tenant was given an empty value; omit --tenant for an unscoped "
            "index, or pass a non-empty tenant id",
            file=sys.stderr,
        )
        return 2
    if (getattr(args, "tenant", None) or None) is not None and args.recreate:
        # --recreate drops and rebuilds the WHOLE Qdrant collection, which in a
        # shared multi-tenant collection would delete every other tenant's points.
        # Refuse the combination — rebuild a single tenant with --prune instead.
        print(
            "error: --recreate drops the entire collection (all tenants); it can't be "
            "scoped to --tenant. Re-index the tenant with --prune, or recreate without "
            "--tenant.",
            file=sys.stderr,
        )
        return 2
    # Snapshot mtimes BEFORE the initial index so the watcher can reconcile any
    # file that changes during the (possibly long) collect/embed pass — closing
    # the gap between "indexed" and "observing".
    watch_baseline = None
    if watching:
        from .markdown.watch import _scan_mtimes

        watch_baseline = _scan_mtimes(target)

    provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    store = VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
    )

    # Collect and validate the file set BEFORE any destructive collection call:
    # --recreate on a mistyped/empty path must not drop the existing collection
    # when there is nothing to index. An explicit --index-root pins the corpus
    # root so a nested single-file refresh (`index-markdown notes/sub/a.md
    # --index-root notes`) updates the same chunks/graph nodes as the parent
    # directory index, instead of inserting a second copy under a narrower root.
    root_dir = None
    if getattr(args, "index_root", None):
        root_path = Path(args.index_root).resolve()
        if not root_path.is_dir():
            print(f"error: --index-root is not a directory: {args.index_root}", file=sys.stderr)
            return 2
        if not target.is_relative_to(root_path):
            print(
                f"error: {args.path} is not under --index-root {args.index_root}",
                file=sys.stderr,
            )
            return 2
        root_dir = str(root_path)
        index_root = root_dir
    else:
        index_root = str((target if target.is_dir() else target.parent).resolve())
    # A "full root walk" is a directory index whose target IS the index_root, so
    # col.sources covers every file under the root. Only then is it safe to
    # reconcile missing sources globally (prune / clear graph links); a nested
    # subtree or single file covers only part of the root and must not.
    full_root_walk = target.is_dir() and str(target) == index_root
    tenant = getattr(args, "tenant", None) or None
    # Forward tenant= to store/graph calls only when set (unscoped = no kwarg,
    # so a store/graph predating the kwarg still works); our own helpers take it.
    mtkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    col = collect_markdown(
        target,
        chunk_size=args.chunk_size,
        index_root=index_root,
        root_dir=root_dir,
        tenant=tenant,
    )
    if col.files == 0:
        print(f"error: no .md files found under {target}", file=sys.stderr)
        return 2

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

    chunks = [(c.id, c.text, c.payload) for c in col.chunks]
    # Fetch existing points (with payloads) for this root so a chunk whose id is
    # unchanged but whose frontmatter changed still has its payload refreshed —
    # the id is derived from (index_root, source, offset, text), not the
    # metadata, so a tag/date edit would otherwise leave stale filters in Qdrant.
    existing_payloads: dict[str, dict] = {}
    if not args.recreate and store.collection_exists():
        for hit in store.scroll(filters={"index_root": index_root}, **mtkw):
            existing_payloads[str(hit.id)] = hit.payload or {}
    from .markdown.sync import build_link_map, upsert_markdown_chunks

    existing_ids = set(existing_payloads)
    skipped = sum(1 for c in chunks if c[0] in existing_ids)
    print(
        f"Indexing {len(chunks)} markdown chunks from {col.files} file(s)"
        f" — {len(chunks) - skipped} new, {skipped} already indexed (skipped)."
    )

    # Embed new chunks and refresh changed payloads (shared with the watcher's
    # per-file path so a full walk and a single-file update behave identically).
    cs = upsert_markdown_chunks(store, provider, chunks, existing_payloads, tenant=tenant)
    inserted, refreshed, failed = cs.inserted, cs.refreshed, cs.failed
    failed_sources = cs.failed_sources

    pruned = 0
    if args.prune and not args.recreate:
        from .ingest import prune_stale_chunks

        # Seed from EVERY visited file, not just files that produced chunks: a
        # file edited to empty / frontmatter-only yields no chunks but must
        # still have its old points pruned.
        fresh_by_source: dict[str, set[str]] = {source: set() for source in col.sources}
        for cid, _text, payload in chunks:
            fresh_by_source[payload["source"]].add(cid)
        # Reconcile deletions: a file removed from the folder is absent from
        # col.sources. Seed each prior source under this index_root (but no
        # longer present) with an empty fresh set so its stale points are pruned
        # instead of lingering searchable. ONLY for a full-root directory walk —
        # a single file or a nested subtree covers only part of the index_root,
        # so reconciling here would wrongly prune untouched siblings.
        if full_root_walk:
            for hit in store.scroll(filters={"index_root": index_root}, **mtkw):
                prior = (hit.payload or {}).get("source")
                if prior and prior not in fresh_by_source:
                    fresh_by_source[prior] = set()
        for source in failed_sources:
            fresh_by_source.pop(source, None)
        if failed_sources:
            print(
                f"warning: prune skipped for {len(failed_sources)} source(s) "
                "with failed embeddings; re-run after the provider recovers",
                file=sys.stderr,
            )
        pruned = prune_stale_chunks(store, fresh_by_source, index_root=index_root, tenant=tenant)

    # Links -> graph edges. Only when a graph is configured; sources whose
    # embeddings failed are skipped so a partial index doesn't rewrite links.
    edges_written = 0
    graph_uri = getattr(args, "memgraph_uri", None) or None
    if graph_uri:
        from .graph.factory import make_graph_store

        # Seed with every indexed source so a file whose links were all removed
        # is still synced (sync_file_links deletes its stale LINKS_TO edges).
        by_source = build_link_map(col, failed_sources)
        try:
            graph = make_graph_store(
                graph_uri,
                timeout=getattr(args, "graph_timeout", 5.0),
                **_graph_auth(args),
            )
        except Exception as exc:  # noqa: BLE001 — graph optional; indexing already succeeded
            print(f"warning: graph unavailable, links not written ({exc})", file=sys.stderr)
        else:
            try:
                # Reconcile deletions: a file removed from the folder still has
                # LINKS_TO edges in the graph. Seed each prior link-source no
                # longer present with an empty target list so sync clears its
                # stale edges. ONLY for a full-root directory walk — a single
                # file or nested subtree covers only part of the index_root, so
                # this would wrongly clear untouched siblings' edges.
                if full_root_walk:
                    for prior in graph.file_link_sources(index_root=index_root, **mtkw):
                        if prior not in by_source and prior not in failed_sources:
                            by_source[prior] = []
                for source, targets in by_source.items():
                    edges_written += graph.sync_file_links(
                        source, targets, index_root=index_root, **mtkw
                    )
            except Exception as exc:  # noqa: BLE001 — graph optional; vectors already upserted
                # A mid-run graph outage (e.g. Memgraph disconnects while
                # writing) must not fail the whole index: the vector upserts
                # already succeeded, so warn and keep the partial link sync.
                print(
                    f"warning: graph write failed, links partially written ({exc})",
                    file=sys.stderr,
                )
            finally:
                graph.close()

    print(
        f"Done: inserted/updated {inserted}, skipped {skipped},"
        f" failed-embedding {failed}, total chunks {len(chunks)}"
        + (f", refreshed {refreshed} payloads" if refreshed else "")
        + (f", pruned {pruned} stale" if args.prune else "")
        + (f", {edges_written} link edges" if graph_uri else "")
        + f" in collection '{args.collection}'."
    )

    if getattr(args, "watch", False):
        return _watch_markdown(
            args, store, provider, index_root, str(target), graph_uri, watch_baseline
        )
    return 0


def _watch_markdown(
    args: argparse.Namespace,
    store: Any,
    provider: Any,
    index_root: str,
    watch_root: str,
    graph_uri: str | None,
    baseline: dict[str, float] | None = None,
) -> int:
    """Run the incremental file-watch loop after the initial index (blocking).

    ``watch_root`` is the folder to observe (what was indexed); ``index_root`` is
    the corpus root the syncer scopes ids/sources/graph nodes to — identical in
    the common case, but distinct when ``--index-root`` pins a nested subtree.
    """
    from .markdown.sync import MarkdownSyncer
    from .markdown.watch import _WATCHDOG, MarkdownWatcher

    # A single graph connection kept open for the whole watch (each event's link
    # sync reuses it). Fail-open: no graph just means links aren't synced.
    graph = None
    if graph_uri:
        from .graph.factory import make_graph_store

        try:
            graph = make_graph_store(
                graph_uri, timeout=getattr(args, "graph_timeout", 5.0), **_graph_auth(args)
            )
        except Exception as exc:  # noqa: BLE001 — graph optional; watch still syncs vectors
            print(f"warning: graph unavailable, links not watched ({exc})", file=sys.stderr)

    syncer = MarkdownSyncer(
        store,
        provider,
        index_root=index_root,
        chunk_size=args.chunk_size,
        graph=graph,
        subtree=watch_root,  # keep referrer re-resolution inside the watched tree
        tenant=getattr(args, "tenant", None) or None,
    )

    def _on_result(res: Any) -> None:
        if res.source is None:
            return
        if res.error:
            return
        # A failed embedding leaves the old chunks searchable (pruning is skipped
        # for that source, like the one-shot path). Surface it as a warning so a
        # transient provider outage isn't reported as a clean sync.
        if res.failed:
            print(
                f"  warning: {res.source} — {res.failed} chunk(s) failed to embed; "
                "stale content kept, re-save after the provider recovers",
                file=sys.stderr,
            )
            return
        verb = "removed" if res.pruned and not res.inserted and not res.refreshed else "synced"
        print(
            f"  {verb} {res.source}"
            f" (+{res.inserted}/~{res.refreshed}/-{res.pruned}"
            + (f", {res.edges} edges" if graph is not None else "")
            + ")"
        )

    def _on_error(path: str, kind: str, exc: Exception) -> None:
        print(f"  error syncing {path} ({kind}): {exc}", file=sys.stderr)

    watcher = MarkdownWatcher(
        syncer,
        watch_root,
        debounce=args.watch_debounce,
        poll_interval=args.watch_poll_interval,
        on_result=_on_result,
        on_error=_on_error,
    )
    backend = "watchdog" if _WATCHDOG else f"polling every {args.watch_poll_interval}s"
    print(f"Watching {watch_root} for changes ({backend}) — Ctrl+C to stop.")
    try:
        watcher.run(baseline=baseline)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        if graph is not None:
            graph.close()
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


def build_parser(config_light: bool = False) -> argparse.ArgumentParser:
    # `config_light` skips loading the stack config (used only to seed arg
    # defaults) so a command that needs no embedding/vector/recall config —
    # notably `keys` — isn't blocked by an unrelated malformed config/env.
    cfg = Config() if config_light else Config.load()
    p = argparse.ArgumentParser(prog="mnemostack", description="Memory stack for AI agents")
    p.add_argument("--version", action="version", version=f"mnemostack {__version__}")
    # Graph auth (user/password/database) is sourced from the config file / env,
    # not typed CLI flags (a password on argv leaks via `ps`). Seed it onto every
    # command's namespace so the graph construction helpers can thread it through.
    p.set_defaults(
        graph_user=cfg.graph.user,
        graph_password=cfg.graph.password,
        graph_database=cfg.graph.database,
    )
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

    p_doctor = sub.add_parser(
        "doctor",
        parents=[common],
        help="Diagnose config + dependencies (read-only; exit 0 ok / 1 down / 2 misconfigured)",
    )
    p_doctor.add_argument(
        "--json", action="store_true", help="Emit the diagnostic report as JSON"
    )
    p_doctor.add_argument(
        "--check-llm",
        action="store_true",
        help="Include a live (billable) LLM generation probe (default: config check only)",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    p_tenant_migrate = sub.add_parser(
        "tenant-migrate",
        parents=[common],
        help="Stamp a tenant_id onto existing points (single-tenant -> multi-tenant)",
    )
    p_tenant_migrate.add_argument("--tenant", required=True, help="tenant_id to assign")
    p_tenant_migrate.add_argument(
        "--all",
        action="store_true",
        help="Stamp every point (default: only points that lack a tenant_id)",
    )
    p_tenant_migrate.add_argument(
        "--dry-run", action="store_true", help="Report the count without writing"
    )
    p_tenant_migrate.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Confirm --all when points already carry a tenant_id (relabels them)",
    )
    p_tenant_migrate.add_argument(
        "--memgraph-uri",
        default=None,
        help="Also stamp the tenant onto the graph at this bolt URI (nodes + edges). "
        "Omit to migrate the vector store only.",
    )
    p_tenant_migrate.add_argument(
        "--graph-timeout",
        type=float,
        default=cfg.graph.timeout,
        help="Memgraph connection timeout in seconds (default 5.0)",
    )
    p_tenant_migrate.set_defaults(func=cmd_tenant_migrate)

    p_keys = sub.add_parser(
        "keys", help="Manage service keys for multi-tenant auth (add / list / revoke)"
    )
    ksub = p_keys.add_subparsers(dest="keys_action")

    def _keys_help(_args: argparse.Namespace) -> int:
        p_keys.print_help()
        return 2

    p_keys.set_defaults(func=_keys_help)
    _keys_file_help = (
        "Key store path (default: $MNEMOSTACK_KEYS_FILE or ~/.config/mnemostack/keys.json)"
    )
    pk_add = ksub.add_parser("add", help="Issue a key for a tenant")
    pk_add.add_argument("--tenant", required=True, help="tenant_id the key is scoped to")
    pk_add.add_argument(
        "--scopes", default="read", help="Comma-separated scopes: read,write,admin (default read)"
    )
    pk_add.add_argument("--label", default="", help="Human label (e.g. app name)")
    pk_add.add_argument("--keys-file", default=None, help=_keys_file_help)
    pk_add.set_defaults(func=cmd_keys_add)
    pk_list = ksub.add_parser("list", help="List keys (hashes never shown)")
    pk_list.add_argument("--json", action="store_true", help="JSON output")
    pk_list.add_argument("--keys-file", default=None, help=_keys_file_help)
    pk_list.set_defaults(func=cmd_keys_list)
    pk_revoke = ksub.add_parser("revoke", help="Revoke a key by its id")
    pk_revoke.add_argument("id", help="key id (from `keys list`)")
    pk_revoke.add_argument("--keys-file", default=None, help=_keys_file_help)
    pk_revoke.set_defaults(func=cmd_keys_revoke)

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
    p_search.add_argument(
        "--token-budget",
        type=int,
        default=cfg.recall.token_budget,
        help=(
            "Hard cap on the total (estimated) text tokens of the returned "
            "results: the final ranking is cut to the prefix that fits"
        ),
    )
    _add_validity_recall_flags(p_search)
    p_search.set_defaults(func=cmd_search)

    p_invalidate = sub.add_parser(
        "invalidate",
        parents=[common],
        help="Mark memories stale by id (non-destructive, no re-embedding)",
    )
    p_invalidate.add_argument("ids", nargs="+", help="Point id(s) to invalidate")
    p_invalidate.add_argument(
        "--invalidated-at",
        default=None,
        help="System-time stamp (ISO-8601); default: now (UTC)",
    )
    p_invalidate.add_argument(
        "--valid-until",
        default=None,
        help="World-time the fact stopped being true (ISO-8601); optional",
    )
    p_invalidate.add_argument(
        "--index-root",
        default=None,
        help="Owner guard: skip points owned by a different index_root",
    )
    p_invalidate.add_argument("--json", action="store_true", help="JSON output")
    p_invalidate.set_defaults(func=cmd_invalidate)

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
    p_answer.add_argument(
        "--token-budget",
        type=int,
        default=cfg.recall.token_budget,
        help=(
            "Hard cap on the total (estimated) text tokens of the memories "
            "fed to the answer LLM"
        ),
    )
    _add_validity_recall_flags(p_answer)
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

    p_index_md = sub.add_parser(
        "index-markdown",
        parents=[common],
        help="Index a markdown folder: frontmatter -> filters, links -> graph edges",
    )
    p_index_md.add_argument("path", help="Markdown file or directory to index")
    p_index_md.add_argument(
        "--tenant",
        default=None,
        metavar="ID",
        help=(
            "Index this corpus under a tenant: chunk ids, payloads, and graph "
            "nodes/edges are scoped to it, so authenticated multi-tenant recall "
            "sees only its own markdown (default: unscoped / single-tenant)"
        ),
    )
    p_index_md.add_argument(
        "--index-root",
        default=None,
        metavar="DIR",
        help=(
            "Corpus root for a single-file refresh; source paths become relative "
            "to it so `index-markdown vault/sub/a.md --index-root vault` updates "
            "the same chunks as the directory index (default: the file's parent)"
        ),
    )
    p_index_md.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Target chars per chunk; the markdown chunker splits on headers (default 1200)",
    )
    p_index_md.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri,
        help=(
            "Memgraph/Neo4j bolt URI to write wikilink/markdown-link edges "
            "(File -[LINKS_TO]-> File); omit to index chunks + frontmatter only"
        ),
    )
    p_index_md.add_argument(
        "--graph-timeout",
        type=float,
        default=cfg.graph.timeout,
        help="Graph connection timeout in seconds (default 5.0)",
    )
    p_index_md.add_argument("--recreate", action="store_true", help="Drop existing collection")
    p_index_md.add_argument(
        "--prune",
        action="store_true",
        help="After indexing, delete stale chunks the files no longer produce (per index root)",
    )
    p_index_md.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt for --recreate"
    )
    p_index_md.add_argument(
        "--watch",
        action="store_true",
        help="After the initial index, keep watching the folder and sync changes "
        "incrementally (add/modify/delete). Requires a directory. Ctrl+C to stop.",
    )
    p_index_md.add_argument(
        "--watch-debounce",
        type=float,
        default=0.5,
        help="Quiet window (seconds) to coalesce rapid file events under --watch (default 0.5)",
    )
    p_index_md.add_argument(
        "--watch-poll-interval",
        type=float,
        default=1.0,
        help="Polling interval (seconds) when the watchdog extra isn't installed (default 1.0)",
    )
    p_index_md.set_defaults(func=cmd_index_markdown)

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
    p_mcp.add_argument(
        "--token-budget",
        type=int,
        default=cfg.recall.token_budget,
        help="Default recall token budget for search/answer tool calls",
    )
    p_mcp.add_argument(
        "--auth",
        action="store_true",
        help=(
            "Run the MCP server as an authenticated tenant: tools require the "
            "key's scope and recall is scoped to its tenant. Provide the key with "
            "--api-key or MNEMOSTACK_API_KEY. Issue keys with `mnemostack keys add`"
        ),
    )
    p_mcp.add_argument(
        "--api-key",
        default=None,
        help="Service key the MCP process runs as (or set MNEMOSTACK_API_KEY)",
    )
    p_mcp.add_argument(
        "--keys-file",
        default=None,
        help="Service-key store path (default: $MNEMOSTACK_KEYS_FILE or ~/.config/mnemostack/keys.json)",
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
        "--qdrant-health-timeout",
        type=int,
        default=cfg.vector.health_timeout,
        help="Qdrant timeout (seconds) for /healthz, /readyz, /status probes (default 2)",
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
        "--auth",
        action="store_true",
        help=(
            "Require a service key on /recall /answer /feedback (default-deny); "
            "the key resolves the tenant + scopes. Issue keys with `mnemostack keys add`"
        ),
    )
    p_serve.add_argument(
        "--keys-file",
        default=None,
        help="Service-key store path (default: $MNEMOSTACK_KEYS_FILE or ~/.config/mnemostack/keys.json)",
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
        "--token-budget",
        type=int,
        default=cfg.recall.token_budget,
        help="Server-wide default recall token budget; per-request token_budget overrides it",
    )
    p_serve.add_argument(
        "--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)"
    )
    p_serve.set_defaults(func=cmd_serve)

    p_inspect = sub.add_parser(
        "inspect",
        parents=[common],
        help="Run the read-only web inspector (operator console). Requires [server]",
    )
    p_inspect.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p_inspect.add_argument("--port", type=int, default=8100, help="Port (default 8100)")
    p_inspect.add_argument(
        "--memgraph-uri",
        default=cfg.graph.uri,  # None when unconfigured — don't probe a graph that isn't set up
        help="Memgraph bolt URI (for graph reachability display; omit for no-graph deployments)",
    )
    p_inspect.add_argument(
        "--graph-timeout", type=float, default=cfg.graph.timeout, help="Graph connect timeout (s)"
    )
    p_inspect.add_argument(
        "--qdrant-health-timeout",
        type=int,
        default=cfg.vector.health_timeout,
        help="Qdrant reachability-probe timeout (s) for the inspector (default from config)",
    )
    p_inspect.set_defaults(func=cmd_inspect)

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

        from mnemostack.server import ServerConfig, _env_bool, build_app
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
        graph_user=_graph_auth(args)["user"],
        graph_password=_graph_auth(args)["password"],
        graph_database=_graph_auth(args)["database"],
        graph_timeout=args.graph_timeout,
        qdrant_health_timeout=args.qdrant_health_timeout,
        bm25_paths=list(args.bm25_path) if args.bm25_path else None,
        vector_floor=max(0, int(args.vector_floor)),
        rerank_mode=args.rerank_mode,
        token_budget=_effective_token_budget(args),
        state_path=args.state_path,
        auto_record_ior=args.auto_record_ior,
        # Honor MNEMOSTACK_AUTH_ENABLED too: cmd_serve builds ServerConfig
        # explicitly (never from_env), so without this the documented env toggle
        # would silently leave the endpoints unauthenticated.
        auth_enabled=args.auth or _env_bool("MNEMOSTACK_AUTH_ENABLED"),
        keys_file=args.keys_file,
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
    from .graph.factory import make_graph_store

    store = make_graph_store(args.memgraph_uri, timeout=args.timeout, **_graph_auth(args))
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


def cmd_inspect(args: argparse.Namespace) -> int:
    """Run the read-only web inspector (a memory operations console)."""
    try:
        import uvicorn

        from mnemostack.inspector import build_inspector_app
        from mnemostack.server import ServerConfig
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
        collection=args.collection,
        qdrant_url=args.qdrant,
        graph_uri=args.memgraph_uri,
        graph_user=_graph_auth(args)["user"],
        graph_password=_graph_auth(args)["password"],
        graph_database=_graph_auth(args)["database"],
        graph_timeout=args.graph_timeout,
        # The inspector's graph reachability probe uses graph_health_timeout, so
        # feed the CLI --graph-timeout into it or a slow/remote graph keeps the 1s
        # default and shows as down.
        graph_health_timeout=args.graph_timeout,
        qdrant_health_timeout=args.qdrant_health_timeout,
    )
    app = build_inspector_app(cfg)
    if args.host == "0.0.0.0":
        print(
            "warning: the inspector exposes read-only memory contents; binding to "
            "0.0.0.0 makes them reachable on all interfaces — keep it behind auth",
            file=sys.stderr,
        )
    print(f"mnemostack inspect: http://{args.host}:{args.port}  (read-only)")
    print(f"  collection: {cfg.collection}")
    print(f"  qdrant:     {cfg.qdrant_url}")
    print(f"  memgraph:   {cfg.graph_uri}")
    uvicorn.run(app, host=args.host, port=args.port)
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
        graph_user=_graph_auth(args)["user"],
        graph_password=_graph_auth(args)["password"],
        graph_database=_graph_auth(args)["database"],
        graph_timeout=args.graph_timeout,
        bm25_paths=list(args.bm25_path) if args.bm25_path else None,
        state_path=args.state_path,
        vector_floor=max(0, int(args.vector_floor)),
        rerank_mode=args.rerank_mode,
        token_budget=_effective_token_budget(args),
        # Honor the env toggles too (parity with serve / the MCP main()).
        auth_enabled=args.auth
        or os.environ.get("MNEMOSTACK_AUTH_ENABLED", "").strip().lower()
        in {"1", "true", "yes", "on"},
        api_key=args.api_key or os.environ.get("MNEMOSTACK_API_KEY") or None,
        keys_file=args.keys_file,
    )
    mcp.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    # `keys` manages the auth store and needs no stack config, so a malformed
    # unrelated config/env must not block adding or revoking a service key.
    subcmd = next((a for a in raw if not a.startswith("-")), None)
    parser = build_parser(config_light=subcmd == "keys")
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
