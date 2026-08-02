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
import functools
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


def _doctor_qdrant(
    add,
    url: str,
    collection: str,
    timeout: int,
    expected_dim: int | None,
    text_search: str = "auto",
) -> None:
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
    # Live readiness for the configured lexical arm — the enum being valid
    # says nothing about the COLLECTION being ready; a green doctor followed
    # by a permanently-degraded lexical arm is exactly the false signal a
    # diagnostic exists to prevent.
    if text_search == "sparse":
        from mnemostack.vector.sparse import SPARSE_TEXT_VECTOR

        sparse_spaces = getattr(info.config.params, "sparse_vectors", None) or {}
        if SPARSE_TEXT_VECTOR in sparse_spaces:
            # The space existing isn't readiness — uncovered points are
            # silently invisible to sparse recall.
            try:
                from qdrant_client.models import Filter as _F
                from qdrant_client.models import HasVectorCondition as _HV

                covered = client.count(
                    collection,
                    count_filter=_F(must=[_HV(has_vector=SPARSE_TEXT_VECTOR)]),
                    exact=True,
                ).count
                gap = max(0, count - covered)
            except Exception:  # noqa: BLE001 — older server: can't verify
                gap = -1
            if gap == 0:
                add("qdrant.sparse_space", "ok", f"'{SPARSE_TEXT_VECTOR}' space present, all points covered")
            elif gap > 0:
                add(
                    "qdrant.sparse_space",
                    "misconfig",
                    f"'{SPARSE_TEXT_VECTOR}' space present but {gap} point(s) have no sparse vector",
                    "run `mnemostack sparse-backfill`",
                )
            else:
                add("qdrant.sparse_space", "warn", f"'{SPARSE_TEXT_VECTOR}' space present (coverage not verifiable on this server)")
        else:
            add(
                "qdrant.sparse_space",
                "misconfig",
                f"text_search=sparse but collection '{collection}' has no "
                f"'{SPARSE_TEXT_VECTOR}' space",
                "re-index under text_search=sparse (new collection) or run "
                "`mnemostack sparse-backfill` after the space is added",
            )
    elif text_search == "lexical":
        schema = getattr(info, "payload_schema", None) or {}
        # Every configured gate field needs its own full-text index — a
        # multi-field layout with only the body field indexed would leave
        # the title arm permanently degraded on a real server.
        missing = []
        for gate_field in _lexical_gate_fields():
            field_info = schema.get(gate_field)
            if field_info is not None and "text" in str(
                getattr(field_info, "data_type", field_info)
            ).lower():
                add("qdrant.text_index", "ok", f"full-text index on '{gate_field}'")
            else:
                missing.append(gate_field)
        for gate_field in missing:
            add(
                "qdrant.text_index",
                "misconfig",
                f"text_search=lexical but no full-text index on '{gate_field}'",
                "run `mnemostack text-index` once against this collection",
            )


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
        # A non-dry-run migration that dies in preflight is a failed ATTEMPT at
        # a mutation — the trail must show it (same rule as tenant-export's
        # early errors). A --dry-run pass stays unaudited (nothing attempted).
        if not args.dry_run:
            _audit("tenant.migrate", tenant=args.tenant, outcome="error", error=str(e))
        print(f"error: cannot reach Qdrant at {args.qdrant}: {e}", file=sys.stderr)
        return 1
    if not exists:
        if not args.dry_run:
            _audit(
                "tenant.migrate", tenant=args.tenant, outcome="error",
                reason="collection_absent",
            )
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
        if not args.dry_run:
            _audit("tenant.migrate", tenant=args.tenant, outcome="error", error=str(e))
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
            # rc 1 = graph unreachable (a failed attempt — audited); rc 2 = the
            # needs---yes refusal (a validation gate, nothing attempted — not).
            if rc == 1:
                _audit(
                    "tenant.migrate", tenant=args.tenant, outcome="error",
                    reason="graph_preflight_failed",
                )
            return rc

    pending = missing if only_missing else total
    scope = "without a tenant_id" if only_missing else "in the collection"
    if args.dry_run:
        print(f"dry-run: would stamp tenant_id='{args.tenant}' onto {pending} point(s) {scope}")
        return _graph_stamp_tenant(args, only_missing, dry_run=True)
    try:
        stamped = store.stamp_tenant(args.tenant, only_missing=only_missing)
    except Exception as e:  # noqa: BLE001
        _audit("tenant.migrate", tenant=args.tenant, outcome="error", error=str(e))
        print(f"error: cannot reach Qdrant at {args.qdrant}: {e}", file=sys.stderr)
        return 1
    print(f"stamped tenant_id='{args.tenant}' onto {stamped} point(s) {scope}")
    rc = _graph_stamp_tenant(args, only_missing, dry_run=False)
    # Vectors are stamped at this point; a failed graph stamp = partial migration.
    _audit(
        "tenant.migrate",
        tenant=args.tenant,
        outcome="success" if rc == 0 else "partial",
        stamped=stamped,
        all=bool(args.all),
    )
    return rc


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


def _require_tenant_arg(args: argparse.Namespace) -> str | None:
    """The tenant id from ``--tenant``, VERBATIM, or None (error printed).

    An empty/whitespace value must fail closed — for export it would silently
    dump nothing, and for rm the stores' own guards would refuse anyway, but the
    operator deserves one clear message instead of five store errors.

    The value is deliberately NOT normalized (no strip): the onboarding paths
    (``keys add``, ``quota set``, stamped ``tenant_id``) store whatever was
    passed, so export/rm must match it byte-for-byte — a silently stripped
    ``" acme "`` would under-delete and then falsely report "fully removed".
    """
    tenant = str(getattr(args, "tenant", "") or "")
    if not tenant.strip():
        print("error: --tenant requires a non-empty tenant id", file=sys.stderr)
        return None
    return tenant


def cmd_tenant_export(args: argparse.Namespace) -> int:
    """Export a tenant's vector points (vectors + payloads) as JSONL.

    The vector points are the source of truth — the graph is derived (file links
    from re-indexing, fact triples from enrichment) and per-tenant learning state
    is behavioral, so a point dump is the honest portable backup. One JSON object
    per line: a ``meta`` header, then ``{"kind": "point", id, vector, payload}``
    rows. NOT included (documented): service keys (secrets — copy the key store
    explicitly if that's intended), quotas, learning state, graph records.
    """
    import json as _json

    tenant = _require_tenant_arg(args)
    if tenant is None:
        return 2
    try:
        store = VectorStore(collection=args.collection, dimension=1, host=args.qdrant)
        if not store.collection_exists():
            _audit("tenant.export", tenant=tenant, outcome="error", reason="collection_absent")
            print(f"error: collection '{args.collection}' does not exist", file=sys.stderr)
            return 1
    except Exception as e:  # noqa: BLE001
        _audit("tenant.export", tenant=tenant, outcome="error", error=str(e))
        print(f"error: cannot reach Qdrant at {args.qdrant}: {e}", file=sys.stderr)
        return 1

    # A file export is written ATOMICALLY (temp in the same dir + replace): a
    # backup command must never truncate an existing dump and leave a partial
    # file at the trusted path when Qdrant drops mid-scroll. The final name
    # (over)writes only after the export completed. mkstemp (unique, O_EXCL,
    # 0600) — a predictable ".tmp" name could truncate/unlink another user's
    # file or follow a planted symlink in a shared directory.
    import tempfile

    tmp_path: Path | None = None
    if args.output == "-":
        out = sys.stdout
    else:
        try:
            target = Path(args.output)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent) or ".", prefix=target.name + ".", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            out = os.fdopen(fd, "w", encoding="utf-8")
        except OSError as e:
            _audit("tenant.export", tenant=tenant, outcome="error", error=str(e))
            print(f"error: cannot write {args.output}: {e}", file=sys.stderr)
            return 1
    count = 0
    try:
        meta = {
            "kind": "meta",
            "format": "mnemostack-tenant-export/1",
            "tenant": tenant,
            "collection": args.collection,
            "vectors": not args.no_vectors,
        }
        out.write(_json.dumps(meta, ensure_ascii=False) + "\n")
        for hit in store.scroll(with_vectors=not args.no_vectors, tenant=tenant):
            row: dict[str, Any] = {"kind": "point", "id": hit.id, "payload": hit.payload}
            if hit.vector is not None:
                row["vector"] = hit.vector
            out.write(_json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        if out is not sys.stdout:
            out.close()
            assert tmp_path is not None
            # mkstemp made the temp 0600; if we're replacing an existing dump,
            # keep ITS mode rather than silently narrowing a 0644 backup (the
            # export carries no plaintext secrets — keys aren't exported).
            try:
                prev_mode = os.stat(args.output).st_mode & 0o777
                os.chmod(tmp_path, prev_mode)
            except FileNotFoundError:
                pass  # new file: leave mkstemp's owner-only default
            except OSError:
                pass  # best-effort; a platform without chmod keeps 0600
            os.replace(tmp_path, args.output)  # success: land the dump atomically
            tmp_path = None
    except Exception as e:  # noqa: BLE001
        _audit("tenant.export", tenant=tenant, outcome="error", error=str(e), points=count)
        print(f"error: export failed after {count} point(s): {e}", file=sys.stderr)
        return 1
    finally:
        if out is not sys.stdout:
            try:
                out.close()
            except Exception:  # noqa: BLE001 — already closed on success
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()  # failed run: drop the partial temp
            except OSError:
                pass
    # A whole-corpus read of one tenant's data — exactly what a trail must show.
    _audit(
        "tenant.export",
        tenant=tenant,
        points=count,
        output="stdout" if args.output == "-" else args.output,
    )
    print(
        f"exported {count} point(s) for tenant '{tenant}'"
        + ("" if args.output == "-" else f" to {args.output}"),
        file=sys.stderr if args.output == "-" else sys.stdout,
    )
    if count == 0:
        # An empty dump is valid (a provisioned-but-empty tenant) but is more
        # often a typo'd tenant id — don't let it pass as a trusted backup.
        print(
            f"warning: 0 points matched tenant '{tenant}' — check the tenant id "
            "before trusting this as a backup",
            file=sys.stderr,
        )
    return 0


def _abort_keys_alive(tenant: str, failed: list[str], gs: Any) -> int:
    """Abort the tenant-rm data sweep because a usable tenant key remains.

    Whether revocation was *refused* (last admin key) or *failed* (unwritable
    store), the outcome is the same: a write-capable key survives, and any store
    swept now could be re-written the moment it's cleaned. Nothing below the key
    step has been deleted at this point, so aborting keeps the offboarding
    atomic for the re-run.
    """
    print(
        "skipping vector/graph/quota/learning-state deletion while the tenant "
        "retains an active key — nothing was deleted; fix the key store (or issue "
        "another admin key) and re-run",
        file=sys.stderr,
    )
    if gs is not None:
        gs.close()
    _audit("tenant.rm", tenant=tenant, outcome="aborted", failed=failed)
    print(f"tenant '{tenant}' NOT removed — FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1


def cmd_tenant_rm(args: argparse.Namespace) -> int:
    """Offboard a tenant: delete its data and config across every store.

    Touches five stores — vector points, graph nodes/edges (when
    ``--memgraph-uri`` is given), service keys, quota, and the per-tenant
    learning-state partitions. Counts everything first; ``--dry-run`` stops
    there, and the actual deletion is gated behind ``--yes`` (there is no
    interactive fallback — an irreversible cross-store wipe deserves an explicit
    flag). Best-effort per store: a failure in one is reported and the rest
    proceed, with a nonzero exit if anything failed — so a partial outage
    doesn't strand the whole offboarding, and the report says what remains.

    Concurrency contract — this is a BEST-EFFORT sweep, not a distributed
    transaction. It revokes the tenant's keys BEFORE deleting data, so once the
    sweep starts no further *authenticated* write can land; each data store is
    then swept from live state (never a stale preflight count), and a final
    key re-scan catches a key that raced in mid-sweep. What one CLI pass CANNOT
    do is lock out key issuance cluster-wide or stop out-of-band/unauthenticated
    writers — for a hard guarantee, quiesce writers (pause the servers / freeze
    issuance) first. The command reports what it swept and never claims more.
    """
    tenant = _require_tenant_arg(args)
    if tenant is None:
        return 2

    # ---- count phase (read-only; a store outage degrades to "unknown") ----
    # Stores that could not even be INSPECTED. They pre-seed the delete phase's
    # failure list: an unreachable store's cleanup can't happen, but it must not
    # strand the other stores (best-effort) — and must never read as success.
    unavailable: dict[str, str] = {}

    store = None
    points: int | None = None
    collection_absent = False
    try:
        store = VectorStore(collection=args.collection, dimension=1, host=args.qdrant)
        if store.collection_exists():
            points = store.count(tenant=tenant)
        else:
            points = 0
            collection_absent = True  # nothing to delete — not a failure
    except Exception as e:  # noqa: BLE001 — a Qdrant outage must not strand the rest
        unavailable["vector points"] = f"cannot reach Qdrant at {args.qdrant}: {e}"

    # --no-graph explicitly opts out of graph cleanup (a vector-only deployment);
    # it wins over the configured/default URI.
    if getattr(args, "no_graph", False):
        raw_graph = None
    else:
        raw_graph = getattr(args, "memgraph_uri", None)
    # An EXPLICIT but empty --memgraph-uri (e.g. `--memgraph-uri "$UNSET"`) signals
    # intent to clean the graph but would coerce to None and silently skip it,
    # then still report "fully removed". Reject it rather than treat it as omission.
    if raw_graph is not None and not str(raw_graph).strip():
        print(
            "error: --memgraph-uri was given an empty value; omit it to skip the "
            "graph, or pass the bolt URI to clean it",
            file=sys.stderr,
        )
        return 2
    memgraph_uri = raw_graph or None
    graph_counts = None
    gs = None
    if memgraph_uri:
        from .graph.factory import make_graph_store

        try:
            gs = make_graph_store(
                memgraph_uri, timeout=getattr(args, "graph_timeout", 5.0), **_graph_auth(args)
            )
            graph_counts = gs.delete_tenant(tenant, dry_run=True)
        except Exception as e:  # noqa: BLE001 — a graph outage must not strand the rest
            unavailable["graph"] = f"cannot reach graph at {memgraph_uri}: {e}"
            if gs is not None:
                gs.close()
                gs = None

    # Mirror the key-store factory's backend set: `file` is handled locally,
    # `openbao` is genuinely external (the store's own tooling revokes) — but a
    # TYPO ("flie") must be a cleanup failure, not a silent skip that lets the
    # tenant's local keys survive an offboarding reported as fully removed.
    _ks_backend = (os.environ.get("MNEMOSTACK_KEYSTORE") or "").strip().lower() or "file"
    external_keys = _ks_backend == "openbao"
    key_ids: list[str] = []
    if _ks_backend == "file":
        try:
            key_ids = [k["id"] for k in _keys_store(args).list_keys() if k["tenant"] == tenant]
        except Exception as e:  # noqa: BLE001 — unreadable keys = incomplete offboarding
            unavailable["service keys"] = f"key store unreadable: {e}"
    elif not external_keys:
        unavailable["service keys"] = (
            f"unknown MNEMOSTACK_KEYSTORE backend {_ks_backend!r} (valid: file, openbao)"
        )

    # Quota: get() fails OPEN (None for a corrupt store), which would silently
    # skip removal and still report success — probe with list_quotas(), which is
    # a management read and raises on a corrupt/unreadable store.
    quota_present = False
    try:
        quota_present = any(q["tenant"] == tenant for q in _quota_store(args).list_quotas())
    except Exception as e:  # noqa: BLE001
        unavailable["quota"] = f"quota store unreadable: {e}"

    from .recall.pipeline import FileStateStore, default_state_path
    from .recall.pipeline.state import tenant_state_key

    # Honor MNEMOSTACK_STATE_PATH like the servers do (the MCP server reads it),
    # or tenant-rm would inspect/clean a DIFFERENT state file than the live
    # server and report full removal while the tenant's partitions survive.
    state_path = (
        getattr(args, "state_path", None)
        or os.environ.get("MNEMOSTACK_STATE_PATH")
        or default_state_path()
    )
    state_keys = [tenant_state_key(base, tenant) for base in ("q_table", "ior_log")]
    state_store = None
    present_state: list[str] = []
    try:
        # Construction can itself fail (mkdir on a bad --state-path); and get()
        # fails OPEN on a corrupt file (recall must survive it), so parse the
        # file strictly here — offboarding must know it couldn't inspect.
        state_store = FileStateStore(state_path)
        import json as _json

        # FileStateStore expands ~; use the SAME expanded path for the strict
        # preflight read, or preview/dry-run inspects a different file than delete.
        spath = Path(state_path).expanduser()
        if spath.exists():
            _json.loads(spath.read_text())
            present_state = [k for k in state_keys if state_store.get(k) is not None]
    except Exception as e:  # noqa: BLE001
        unavailable["learning state"] = f"state store unusable: {e}"

    def _line(label: str, value: str, unavailable_as: str | None = None) -> None:
        if unavailable_as is not None and unavailable_as in unavailable:
            print(f"  {label} unknown — store unavailable")
        else:
            print(f"  {label} {value}")

    print(f"tenant '{tenant}':")
    _line("vector points:   ", f"{points}" + (" (collection absent)" if collection_absent else ""),
          "vector points")
    if "graph" in unavailable:
        print("  graph:            unknown — store unavailable")
    elif graph_counts is not None:
        print(f"  graph nodes:      {graph_counts['nodes']}")
        print(f"  graph edges:      {graph_counts['relationships']}")
        if graph_counts.get("detached"):
            print(
                f"  graph collateral: {graph_counts['detached']} unowned edge(s) "
                "attached to the tenant's nodes (removed with them)"
            )
    else:
        print("  graph:            (skipped — no --memgraph-uri)")
    if external_keys:
        confirmed = " — confirmed revoked" if args.external_keys_revoked else ""
        print(
            "  service keys:     managed externally (MNEMOSTACK_KEYSTORE); "
            f"revoke them in that store{confirmed or ' + pass --external-keys-revoked'}"
        )
    else:
        _line("service keys:    ", f"{len(key_ids)}", "service keys")
    _line("quota:           ", "set" if quota_present else "none", "quota")
    _line("learning state:  ", f"{len(present_state)} partition(s)", "learning state")
    for name, reason in unavailable.items():
        print(f"warning: {name}: {reason}", file=sys.stderr)

    if args.dry_run:
        if gs is not None:
            gs.close()
        print("dry-run: nothing deleted")
        if unavailable:
            print(
                "warning: counts are incomplete — "
                + ", ".join(unavailable)
                + " could not be inspected",
                file=sys.stderr,
            )
            return 1
        return 0
    if not args.yes:
        if gs is not None:
            gs.close()
        print(
            "error: deleting a tenant is irreversible and spans every store; "
            "re-run with --yes to confirm (use --dry-run to preview)",
            file=sys.stderr,
        )
        return 2

    # ---- delete phase (best-effort per store, loud report) ----
    # Unreachable stores are failures up front: their cleanup did NOT happen.
    failed: list[str] = list(unavailable)
    # Keys are handled FIRST: on a live authenticated deployment the tenant's
    # key could otherwise keep writing (feedback, graph facts) into stores this
    # command has already cleaned, and "fully removed" would be stale the moment
    # it printed. Auth re-verifies per request/tool-call, so cutting the key off
    # before the data stores are swept closes that window.
    if "service keys" in unavailable:
        # We couldn't even INSPECT the local keys (no read access, corrupt file,
        # unknown backend). The servers may still verify those keys, so a live
        # write/admin key could re-write swept data — abort like the other
        # active-key cases rather than sweep with keys of unknown state.
        print(
            f"error: tenant '{tenant}' service keys could not be inspected — "
            "cannot confirm they're revoked. Fix key-store access and re-run.",
            file=sys.stderr,
        )
        return _abort_keys_alive(tenant, failed, gs)
    if external_keys:
        # tenant-rm can't revoke a key it only verifies (it lives in the
        # external store). Refuse to sweep unless the operator confirms they
        # revoked it out-of-band — otherwise a live key re-writes the data back.
        if not args.external_keys_revoked:
            failed.append("service keys (external store — not confirmed revoked)")
            print(
                f"error: tenant '{tenant}' keys are in an external store "
                "(MNEMOSTACK_KEYSTORE) — tenant-rm can't revoke them. Revoke the "
                "tenant's key(s) there first (e.g. `bao kv delete ...`), then re-run "
                "with --external-keys-revoked to sweep the data stores.",
                file=sys.stderr,
            )
            return _abort_keys_alive(tenant, failed, gs)
        print("service keys: external store — treated as revoked (--external-keys-revoked)")
        print(
            "note: a running server may still accept a just-revoked external key "
            "until its verify cache expires (MNEMOSTACK_OPENBAO_CACHE_TTL) — ensure "
            "that elapsed (or restart the servers) before trusting this removal",
            file=sys.stderr,
        )
    if not external_keys and "service keys" not in unavailable:
        try:
            # Atomically revoke ALL of the tenant's keys under ONE store lock,
            # re-read at this moment — not a count-phase snapshot. This catches a
            # key issued for the tenant after the earlier count (the concurrent-
            # offboarding race) and any malformed record, so no usable key can
            # survive into the data sweep below.
            result = _keys_store(args).revoke_tenant(tenant, protect_last_admin=True)
            print(f"revoked {result['revoked']} service key(s)")
            if result["last_admin_kept"]:
                failed.append("service keys (last admin key)")
                print(
                    f"error: a key of tenant '{tenant}' is the LAST usable admin key — "
                    "refusing to revoke it (that would lock out key management). Issue "
                    "an admin key for a DIFFERENT tenant (`mnemostack keys add --tenant "
                    "<other> --scopes admin`) — one for this tenant would just be kept "
                    "again — then re-run.",
                    file=sys.stderr,
                )
                # The surviving key is write-capable (admin implies write), so it
                # could re-write into any store swept below the moment it's cleaned.
                return _abort_keys_alive(tenant, failed, gs)
        except Exception as e:  # noqa: BLE001
            # A failed revocation WRITE (unwritable dir, full disk) leaves the
            # tenant's keys just as alive as the last-admin case — same abort:
            # sweeping data now would hand the still-active key a clean store
            # to re-write into.
            failed.append("service keys")
            print(f"error: key revocation failed: {e}", file=sys.stderr)
            return _abort_keys_alive(tenant, failed, gs)
    if "vector points" not in unavailable:
        assert store is not None
        try:
            # If the collection was absent at preflight, RE-CHECK it now — a racer
            # may have created the tenant's first point (and the collection)
            # between the count phase and key revocation. Keys are now revoked, so
            # no further authenticated write can land; one idempotent tenant delete
            # closes the collection-creation race. `collection_exists()` runs only
            # in this already-absent branch, so the common path keeps its single
            # round-trip — and the whole block is guarded, so Qdrant dropping
            # between preflight and here is a reported failure, not a crash.
            if collection_absent and not store.collection_exists():
                print("vector points: collection absent — nothing to delete")
            else:
                deleted = store.delete_tenant(tenant)
                print(f"deleted {deleted} vector point(s)")
        except Exception as e:  # noqa: BLE001
            failed.append("vector points")
            print(f"error: vector delete failed: {e}", file=sys.stderr)
    if gs is not None:
        try:
            gc = gs.delete_tenant(tenant)
            extra = (
                f" (+{gc['detached']} attached unowned edge(s))" if gc.get("detached") else ""
            )
            print(f"deleted {gc['nodes']} graph node(s), {gc['relationships']} edge(s){extra}")
        except Exception as e:  # noqa: BLE001
            failed.append("graph")
            print(f"error: graph delete failed: {e}", file=sys.stderr)
        finally:
            gs.close()
    if "quota" not in unavailable:
        # Call remove() unconditionally (it's idempotent), NOT gated on the
        # count-phase quota_present snapshot: a quota created for the tenant after
        # the count (a concurrent admin/inspector session) would otherwise survive
        # under a "fully removed" report.
        try:
            print("removed quota" if _quota_store(args).remove(tenant) else "quota: none")
        except Exception as e:  # noqa: BLE001
            failed.append("quota")
            print(f"error: quota removal failed: {e}", file=sys.stderr)
    if state_store is not None and "learning state" not in unavailable:
        removed_state = 0
        try:
            for k in state_keys:
                if state_store.delete(k):
                    removed_state += 1
            print(f"removed {removed_state} learning-state partition(s)")
        except Exception as e:  # noqa: BLE001
            failed.append("learning state")
            print(f"error: learning-state cleanup failed: {e}", file=sys.stderr)

    # Final key re-scan: a key issued for the tenant DURING the sweep (a racer
    # concurrently onboarding it) is missed by the pre-sweep revocation and could
    # have re-written data into a store we just cleaned. Re-revoke; if any turned
    # up, report a partial removal rather than claim success — the operator must
    # stop issuing keys for an offboarding tenant (or pause the servers) for a
    # hard guarantee; one CLI pass can't lock out issuance cluster-wide.
    if not external_keys and "service keys" not in unavailable:
        try:
            late = _keys_store(args).revoke_tenant(tenant, protect_last_admin=True)
            if late["revoked"] or late["last_admin_kept"]:
                failed.append("service keys (issued during sweep)")
                print(
                    f"warning: key(s) for '{tenant}' appeared DURING the sweep "
                    f"({late['revoked']} revoked"
                    + (
                        ", 1 kept as the last usable admin and still active"
                        if late["last_admin_kept"]
                        else ""
                    )
                    + ") — data may have been re-created; re-run tenant-rm (and don't "
                    "issue keys for a tenant being removed).",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001 — best-effort second pass
            failed.append("service keys (post-sweep re-scan)")
            print(f"error: post-sweep key re-scan failed: {e}", file=sys.stderr)

    if failed:
        _audit("tenant.rm", tenant=tenant, outcome="partial", failed=failed)
        print(
            f"tenant '{tenant}' partially removed — FAILED: {', '.join(failed)}; "
            "re-run after fixing the store(s) above",
            file=sys.stderr,
        )
        return 1
    _audit("tenant.rm", tenant=tenant)
    print(
        f"tenant '{tenant}' swept from all reachable stores (best-effort). Keys "
        "were revoked before the sweep, so no further authenticated write can "
        "land; completeness still assumes no out-of-band writers — quiesce "
        "writers (pause servers / freeze issuance) first for a hard guarantee."
    )
    return 0


def _audit(action: str, *, tenant: str | None = None, outcome: str = "success", **details: Any) -> None:
    """Best-effort audit of a control-plane operation (see ``mnemostack.audit``:
    never raises, and a no-op unless ``MNEMOSTACK_AUDIT_FILE`` is set).

    Convention across the CLI: **mutations** (and attempted mutations that fail
    against a store) are audited; pure input-validation rejections and
    ``--dry-run`` passes — where nothing was attempted — are not.
    """
    from mnemostack.audit import audit_log_from_env

    audit_log_from_env().record(action, tenant=tenant, outcome=outcome, details=details or None)


def _keys_store(args: argparse.Namespace):
    from mnemostack.auth import FileKeyStore

    backend = (os.environ.get("MNEMOSTACK_KEYSTORE") or "file").strip().lower()
    if backend != "file":
        # `mnemostack keys` manages the LOCAL FILE store only. With an external
        # backend selected the servers verify against that store instead, so a
        # key added here would silently not authenticate — warn, loudly.
        print(
            f"warning: MNEMOSTACK_KEYSTORE={backend} is set — servers verify keys "
            "against that external store, but this command manages only the local "
            "file store. Manage keys with the external store's own tooling.",
            file=sys.stderr,
        )
    return FileKeyStore(getattr(args, "keys_file", None) or None)


def _quota_store(args: argparse.Namespace):
    from mnemostack.quotas import FileQuotaStore

    return FileQuotaStore(getattr(args, "quotas_file", None) or None)


def _int_or_none(s: str) -> int | None:
    """argparse type: an int, or None for the literal 'none'/'unlimited' (clears)."""
    if s.strip().lower() in ("none", "unlimited"):
        return None
    return int(s)


def _float_or_none(s: str) -> float | None:
    """argparse type: a float, or None for the literal 'none'/'unlimited' (clears)."""
    if s.strip().lower() in ("none", "unlimited"):
        return None
    return float(s)


def _resolve_max_points(args: argparse.Namespace, tenant: str | None) -> int | None:
    """The tenant's storage limit from the quota store, or None (no limit).

    Unscoped ingest has no tenant and no quota. A corrupt/unreadable quota store
    fails open (get() returns None) rather than blocking ingest.
    """
    if tenant is None:
        return None
    quota = _quota_store(args).get(tenant)
    return quota.max_points if quota else None


def cmd_keys_add(args: argparse.Namespace) -> int:
    """Issue a service key for a tenant. Prints the key once (never stored)."""
    from mnemostack.auth import KeyStoreError

    try:
        key_id, key = _keys_store(args).issue(args.tenant, args.scopes, args.label or "")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyStoreError as e:
        _audit("keys.issue", tenant=args.tenant, outcome="error", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 1
    scopes_list = sorted({s.strip() for s in args.scopes.split(",") if s.strip()})
    # key_id only — never the key or its hash (audit module contract). Scopes
    # go into the event as a LIST — the same shape the inspector's keys.issue
    # writes — so trail consumers see one schema regardless of surface.
    _audit(
        "keys.issue",
        tenant=args.tenant,
        key_id=key_id,
        scopes=scopes_list,
        label=args.label or "",
    )
    print(f"key id:  {key_id}")
    print(f"tenant:  {args.tenant}")
    print(f"scopes:  {','.join(scopes_list)}")
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

    ks = _keys_store(args)
    # Attribution lookup BEFORE deletion, so the revoke event can say whose
    # credential was removed without joining to an issue event that may have
    # rotated out of the retained trail. Best-effort: a lookup failure must
    # never block the revocation itself.
    tenant: str | None = None
    try:
        tenant = next((k.get("tenant") for k in ks.list_keys() if k.get("id") == args.id), None)
    except Exception:  # noqa: BLE001 — attribution only; revoke proceeds
        pass
    try:
        removed = ks.revoke(args.id)
    except KeyStoreError as e:
        _audit("keys.revoke", tenant=tenant, outcome="error", key_id=args.id, error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 1
    if removed:
        _audit("keys.revoke", tenant=tenant, key_id=args.id)
        print(f"revoked key {args.id}")
        return 0
    _audit("keys.revoke", tenant=tenant, outcome="error", key_id=args.id, reason="not_found")
    print(f"error: no key with id '{args.id}'", file=sys.stderr)
    return 1


def cmd_quota_set(args: argparse.Namespace) -> int:
    """Set a tenant's resource quota (storage size and/or request rate).

    Only the fields passed on the command line change; the rest are left as they
    were (so setting a rate limit doesn't wipe a size cap). Pass ``none`` to clear.
    """
    from mnemostack.quotas import _UNSET, QuotaStoreError

    try:
        quota = _quota_store(args).set(
            args.tenant,
            max_points=getattr(args, "max_points", _UNSET),
            max_rps=getattr(args, "max_rps", _UNSET),
            burst=getattr(args, "burst", _UNSET),
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except QuotaStoreError as e:
        _audit("quota.set", tenant=args.tenant, outcome="error", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 1
    # The RESULTING quota (partial update), so the trail shows what now applies.
    _audit(
        "quota.set",
        tenant=args.tenant,
        max_points=quota.max_points,
        max_rps=quota.max_rps,
        burst=quota.effective_burst(),
    )
    mp = "unlimited" if quota.max_points is None else str(quota.max_points)
    rps = "unlimited" if quota.max_rps is None else f"{quota.max_rps:g}"
    burst = "-" if quota.effective_burst() is None else str(quota.effective_burst())
    print(
        f"quota for tenant '{args.tenant}': "
        f"max_points={mp}, max_rps={rps}, burst={burst}"
    )
    return 0


def cmd_quota_list(args: argparse.Namespace) -> int:
    from mnemostack.quotas import QuotaStoreError

    try:
        quotas = _quota_store(args).list_quotas()
    except QuotaStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(quotas, ensure_ascii=False, indent=2))
        return 0
    if not quotas:
        print("(no quotas)")
        return 0
    from mnemostack.quotas import TenantQuota

    print(f"{'TENANT':<24} {'MAX_POINTS':<12} {'MAX_RPS':<10} BURST")
    for q in quotas:
        mp = "unlimited" if q.get("max_points") is None else str(q["max_points"])
        rps = "unlimited" if q.get("max_rps") is None else f"{q['max_rps']:g}"
        # Show the burst that actually applies (explicit, else derived from the
        # rate), matching `quota set`'s output — '-' when there's no rate.
        eff = TenantQuota(max_rps=q.get("max_rps"), burst=q.get("burst")).effective_burst()
        burst = "-" if eff is None else str(eff)
        print(f"{q.get('tenant', ''):<24} {mp:<12} {rps:<10} {burst}")
    return 0


def cmd_quota_rm(args: argparse.Namespace) -> int:
    from mnemostack.quotas import QuotaStoreError

    try:
        removed = _quota_store(args).remove(args.tenant)
    except QuotaStoreError as e:
        _audit("quota.remove", tenant=args.tenant, outcome="error", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 1
    if removed:
        _audit("quota.remove", tenant=args.tenant)
        print(f"removed quota for tenant '{args.tenant}'")
        return 0
    _audit("quota.remove", tenant=args.tenant, outcome="error", reason="not_found")
    print(f"error: no quota set for tenant '{args.tenant}'", file=sys.stderr)
    return 1


def cmd_text_index(args: argparse.Namespace) -> int:
    """Create the full-text payload index the lexical text-search mode needs.

    Idempotent and non-destructive (safe on a pre-existing/mounted
    collection): a real Qdrant server REQUIRES this index before it will
    accept MatchText filters, which `recall.text_search: lexical` relies on.
    """
    gate_fields = _lexical_gate_fields()
    configured_fields = dict(_text_search_fields())
    if configured_fields:
        # Fields configured: refuse the same pairing the servers refuse to
        # boot on BEFORE mutating Qdrant — indexing the field keys and then
        # printing "lexical is ready" under text_search=sparse/off would be
        # a partial configuration change with a false success message.
        from mnemostack.config import ensure_text_fields_mode, resolve_text_search_mode

        try:
            rc = Config.load().recall
            ensure_text_fields_mode(
                resolve_text_search_mode(rc.text_search, rc.bm25_paths),
                configured_fields,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    store = VectorStore(collection=args.collection, dimension=1, host=args.qdrant)
    try:
        if not store.collection_exists():
            print(f"error: collection '{args.collection}' does not exist", file=sys.stderr)
            return 1
        for gate_field in gate_fields:
            store.ensure_text_index(gate_field)
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot create text index: {e}", file=sys.stderr)
        return 1
    listed = ", ".join(f"'{f}'" for f in gate_fields)
    print(
        f"full-text index ensured on payload field(s) {listed} of "
        f"'{args.collection}' — `recall.text_search: lexical` is ready"
    )
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    """Verify a citation against its CURRENT source document.

    Exit codes: 0 = the citation is still supported by the source (intact /
    source_changed / moved), 1 = it is not (changed / missing), 2 = it cannot
    be verified (unresolvable / no such point).
    """
    from mnemostack.provenance import resolve_citation

    store = VectorStore(collection=args.collection, dimension=1, host=args.qdrant)
    try:
        if not store.collection_exists():
            print(f"error: collection '{args.collection}' does not exist", file=sys.stderr)
            return 2
        # The CLI is the operator surface: bare source paths (no corpus root
        # in the payload) may resolve against the local filesystem here —
        # service surfaces (HTTP/MCP) never do, and they additionally confine
        # roots to the MNEMOSTACK_RESOLVE_ROOTS allowlist.
        res = resolve_citation(
            store,
            args.chunk_id,
            root=args.root,
            tenant=args.tenant,
            allow_unrooted=True,
            text_key=_payload_schema()[0],
        )
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot resolve: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"{res.verdict}: {res.detail}")
        print(f"  source:   {res.source}" + (f" -> {res.resolved_path}" if res.resolved_path else ""))
        print(f"  snapshot: {res.snapshot}" + (f" (captured {res.captured_at})" if res.captured_at else ""))
        if res.found_offset is not None and res.found_offset != res.stored_offset:
            print(f"  offset:   {res.stored_offset} -> {res.found_offset}")
    if res.supported:
        return 0
    return 2 if res.verdict == "unresolvable" else 1


def _lexical_gate_fields() -> list[str]:
    """The payload fields the lexical arm(s) gate on: the configured
    text_search_fields keys, or just text_key when unset. These are the
    fields that need a full-text index on a real server (text-index creates
    them, doctor live-checks them)."""
    fields = [k for k, _ in _text_search_fields()]
    return fields or [_payload_schema()[0]]


def cmd_sparse_backfill(args: argparse.Namespace) -> int:
    """Write sparse text encodings onto every existing point (idempotent).

    The migration path for enabling `recall.text_search: sparse` on an
    already-populated collection: nothing is re-embedded — only the named
    sparse vector is updated, batch by batch.
    """
    text_key = _payload_schema()[0]
    store = VectorStore(
        collection=args.collection,
        dimension=1,
        host=args.qdrant,
        sparse_text=True,
        text_key=text_key,
    )
    try:
        if not store.collection_exists():
            print(f"error: collection '{args.collection}' does not exist", file=sys.stderr)
            return 1
        # Sparse-only ensure: this command runs with a placeholder dense
        # dimension, so it must never trip the dense-dimension validation.
        # The backfill-needed refusal is exactly what we're here to resolve.
        store.ensure_sparse_space(require_backfilled=False)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        updated = store.backfill_sparse_text()
    except Exception as e:  # noqa: BLE001
        print(f"error: sparse backfill failed: {e}", file=sys.stderr)
        return 1
    print(
        f"sparse text encodings written onto {updated} point(s) of "
        f"'{args.collection}' — `recall.text_search: sparse` is ready"
    )
    return 0


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
    # timestamp_format is the other stable config enum: a typo here passes
    # config load but makes TemporalRetriever construction raise at serve
    # time — doctor must flag it, not report a false green.
    from mnemostack.config import TEXT_SEARCH_MODES
    from mnemostack.recall.retrievers import TemporalRetriever as _TR

    if cfg.recall.text_search in TEXT_SEARCH_MODES:
        add("config.text_search", "ok", cfg.recall.text_search)
    else:
        add(
            "config.text_search",
            "misconfig",
            f"invalid text_search '{cfg.recall.text_search}'",
            "set recall.text_search to one of: " + ", ".join(TEXT_SEARCH_MODES),
        )
    # text_search_fields only drives the `lexical` mode — configured with any
    # other mode it would be silently ignored (the servers refuse to boot on
    # this contradiction; doctor must show the same verdict, not a green).
    if cfg.recall.text_search_fields:
        from mnemostack.config import ensure_text_fields_mode, resolve_text_search_mode

        try:
            ensure_text_fields_mode(
                resolve_text_search_mode(cfg.recall.text_search, cfg.recall.bm25_paths),
                cfg.recall.text_search_fields,
            )
        except ValueError as e:
            add("config.text_search_fields", "misconfig", str(e))
        else:
            listed = ", ".join(
                f"{k}={v:g}" for k, v in cfg.recall.text_search_fields.items()
            )
            add("config.text_search_fields", "ok", listed)
    if cfg.recall.timestamp_format in _TR.TIMESTAMP_FORMATS:
        add("config.timestamp_format", "ok", cfg.recall.timestamp_format)
    else:
        add(
            "config.timestamp_format",
            "misconfig",
            f"invalid timestamp_format '{cfg.recall.timestamp_format}'",
            "set recall.timestamp_format to one of: "
            + ", ".join(_TR.TIMESTAMP_FORMATS),
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
    _doctor_qdrant(
        add,
        args.qdrant,
        args.collection,
        cfg.vector.health_timeout,
        expected_dim,
        text_search=cfg.recall.text_search,
    )

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
    _text_key, _ts_key, _ts_fmt = _payload_schema()
    pipeline = build_full_pipeline(
        state_store=FileStateStore(default_state_path()),
        graph_uri=getattr(args, "memgraph_uri", None) or None,
        graph_timeout=getattr(args, "graph_timeout", 5.0),
        text_key=_text_key,
        timestamp_key=_ts_key,
        timestamp_format=_ts_fmt,
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
    from mnemostack.config import resolve_text_search_mode

    provider = None
    store = None
    _lexical_mode = resolve_text_search_mode(
        _text_search_mode(), list(getattr(args, "bm25_path", []) or [])
    )
    _lexical_selected = _source_enabled_for_cli("bm25", source_filter)
    # `lexical` embeds the query, so it needs the provider; `qdrant_bm25` and
    # `sparse` only need a reachable store — constructing an embedding
    # provider for them would demand API keys a payload-only operation never
    # uses. ALL Qdrant-backed modes get the same loud collection preflight
    # (a typo'd collection must exit 2, not degrade to an empty report).
    _needs_provider = (
        _source_enabled_for_cli("vector", source_filter)
        or _source_enabled_for_cli("temporal", source_filter)
        or (_lexical_selected and _lexical_mode == "lexical")
    )
    _needs_store_only = _lexical_selected and _lexical_mode in ("qdrant_bm25", "sparse")
    if _needs_provider:
        provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    if _needs_provider or _needs_store_only:
        store = VectorStore(
            collection=args.collection,
            dimension=provider.dimension if provider is not None else 1,
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
        "timestamp_key": _payload_schema()[1],
        "timestamp_format": _payload_schema()[2],
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
    if source_filter is not None and "bm25" in source_filter:
        # "bm25" selects the configured LEXICAL ARM, whichever mode it is.
        source_filter = source_filter | {"qdrant_text", "sparse"}
    return source_filter is None or name in source_filter


def _graph_auth(args: argparse.Namespace) -> dict[str, Any]:
    """Graph auth (user/password/database) seeded onto the namespace by build_parser."""
    return {
        "user": getattr(args, "graph_user", "") or "",
        "password": getattr(args, "graph_password", "") or "",
        "database": getattr(args, "graph_database", None),
    }


@functools.lru_cache(maxsize=1)
def _payload_schema() -> tuple[str, str, str]:
    """(text_key, timestamp_key, timestamp_format) from config/env.

    The payload schema is a property of the DEPLOYMENT (one collection = one
    schema), not of an individual command, so the CLI reads it from the same
    config/env the servers use instead of per-command flags. Cached: one
    Config.load() per process (a CLI process runs one command).

    A config that fails to load falls back to the standard schema WITH a loud
    stderr warning — for several commands this is the only place the config
    file is parsed at all, so swallowing the error silently would make a
    configured `text_key` vanish with no clue why text came back empty.
    """
    try:
        rc = Config.load().recall
        return rc.text_key, rc.timestamp_key, rc.timestamp_format
    except Exception as e:  # noqa: BLE001 — fall back, but never silently
        print(
            f"warning: config failed to load ({e}); using the standard payload "
            "schema (text/timestamp/iso) — a configured text_key/timestamp_key "
            "is NOT in effect",
            file=sys.stderr,
        )
        return "text", "timestamp", "iso"


@functools.lru_cache(maxsize=1)
def _text_search_mode() -> str:
    """recall.text_search from config/env (same deployment-level resolution
    as _payload_schema; falls back to the historical default on a broken
    config, which _payload_schema already warned about loudly)."""
    try:
        return Config.load().recall.text_search
    except Exception:  # noqa: BLE001 — _payload_schema warned already
        return "auto"


@functools.lru_cache(maxsize=1)
def _text_search_fields() -> tuple[tuple[str, float], ...]:
    """recall.text_search_fields from config/env, as a tuple so the cached
    value cannot be mutated by a caller.

    Unlike _payload_schema this does NOT fall back on a broken config: a
    malformed fields value degrading `serve`/`text-index` to the body-only
    arm would be exactly the silent no-op the fields contract forbids —
    the error propagates (matching the servers, which refuse to boot)."""
    return tuple(Config.load().recall.text_search_fields.items())


def _indexing_store(args: argparse.Namespace, provider) -> VectorStore:
    """The write-side VectorStore for index commands. Under
    ``recall.text_search: sparse`` it maintains the sparse text space (the
    collection gains the space on ensure, every upsert carries the sparse
    encoding) — one config switch drives ingest AND recall."""
    text_key = _payload_schema()[0]
    sparse = _text_search_mode() == "sparse"
    return VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
        sparse_text=sparse,
        text_key=text_key,
    )


def _build_recaller(
    args: argparse.Namespace,
    provider,
    store,
    source_filter: set[str] | None = None,
) -> Recaller:
    """Build the same retriever-mode Recaller used by the service surfaces."""
    from mnemostack.config import ensure_text_fields_mode, resolve_text_search_mode
    from mnemostack.recall import QdrantSparseRetriever, build_qdrant_text_arms

    text_key, timestamp_key, timestamp_format = _payload_schema()
    schema_kw = {
        "text_key": text_key,
        "timestamp_key": timestamp_key,
        "timestamp_format": timestamp_format,
    }
    bm25_paths = list(getattr(args, "bm25_path", []) or [])
    mode = resolve_text_search_mode(_text_search_mode(), bm25_paths)
    ensure_text_fields_mode(mode, dict(_text_search_fields()))
    lexical_weights: dict[str, float] = {}
    retrievers: list[Retriever] = []
    if (
        provider is not None
        and store is not None
        and _source_enabled_for_cli("vector", source_filter)
    ):
        retrievers.append(
            VectorRetriever(
                embedding=provider,
                vector_store=store,
                text_key=text_key,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            )
        )
    if _source_enabled_for_cli("bm25", source_filter):
        if mode == "bm25":
            bm25_docs = build_bm25_docs(bm25_paths)
            if bm25_docs:
                retrievers.append(
                    BM25Retriever(
                        docs=bm25_docs,
                        timestamp_key=timestamp_key,
                        timestamp_format=timestamp_format,
                    )
                )
        elif mode == "qdrant_bm25" and store is not None:
            retrievers.append(
                BM25Retriever.from_qdrant(store.client, args.collection, text_key=schema_kw["text_key"], timestamp_key=schema_kw["timestamp_key"], timestamp_format=schema_kw["timestamp_format"])
            )
        elif mode == "lexical" and provider is not None and store is not None:
            arms, lexical_weights = build_qdrant_text_arms(
                embedding=provider,
                vector_store=store,
                text_key=schema_kw["text_key"],
                timestamp_key=schema_kw["timestamp_key"],
                timestamp_format=schema_kw["timestamp_format"],
                fields=dict(_text_search_fields()),
            )
            retrievers.extend(arms)
        elif mode == "sparse":
            sparse_store = VectorStore(
                collection=args.collection,
                dimension=1,  # sparse queries never touch the dense space
                host=args.qdrant,
                sparse_text=True,
                text_key=text_key,
            )
            retrievers.append(
                QdrantSparseRetriever(vector_store=sparse_store, text_key=schema_kw["text_key"], timestamp_key=schema_kw["timestamp_key"], timestamp_format=schema_kw["timestamp_format"])
            )
    memgraph_uri = getattr(args, "memgraph_uri", None)
    if _source_enabled_for_cli("memgraph", source_filter) and memgraph_uri:
        retrievers.append(MemgraphRetriever(uri=memgraph_uri, **_graph_auth(args)))
    if (
        provider is not None
        and store is not None
        and _source_enabled_for_cli("temporal", source_filter)
    ):
        retrievers.append(
            TemporalRetriever(
                embedding=provider,
                vector_store=store,
                text_key=text_key,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            )
        )
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
        retriever_weights=lexical_weights or None,
        text_key=text_key,
        timestamp_key=timestamp_key,
        timestamp_format=timestamp_format,
    )


def cmd_index(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if not target.exists():
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 2

    provider = get_provider(args.provider, **model_kwargs(_embedding_model(args)))
    store = _indexing_store(args, provider)
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
    from .quotas import QuotaExceededError

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
    store = _indexing_store(args, provider)

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
    max_points = _resolve_max_points(args, tenant)
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
    from .markdown.sync import ChunkSyncResult, build_link_map, upsert_markdown_chunks

    existing_ids = set(existing_payloads)
    skipped = sum(1 for c in chunks if c[0] in existing_ids)
    print(
        f"Indexing {len(chunks)} markdown chunks from {col.files} file(s)"
        f" — {len(chunks) - skipped} new, {skipped} already indexed (skipped)."
    )

    # Embed new chunks and refresh changed payloads (shared with the watcher's
    # per-file path so a full walk and a single-file update behave identically).
    from .markdown.sync import markdown_quota_check

    # Enforce the tenant's storage quota on the exact NET change (inserts minus
    # what the prune step will remove — per-source replacements plus full-root
    # deletions), checked after embedding but before any write.
    _quota_check = markdown_quota_check(
        store, tenant, max_points, existing_payloads, chunks, set(col.sources),
        prune=bool(args.prune and not args.recreate), full_root=full_root_walk,
        md_owned_only=False,  # bulk prune_stale_chunks removes every stale point of the source
    )
    initial_skipped = False
    try:
        cs = upsert_markdown_chunks(
            store, provider, chunks, existing_payloads, tenant=tenant,
            before_upsert=_quota_check,
        )
    except QuotaExceededError as e:
        # In --watch mode, a startup corpus over quota must NOT kill the daemon:
        # nothing was written (the check runs before any upsert), so warn and fall
        # through to the watch loop, which skips/retries files per the live quota.
        if not watching:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"warning: initial index skipped — {e}", file=sys.stderr)
        cs = ChunkSyncResult()
        initial_skipped = True
    inserted, refreshed, failed = cs.inserted, cs.refreshed, cs.failed
    failed_sources = cs.failed_sources

    # When the initial index was skipped over quota, NOTHING was upserted, so the
    # prune and graph-link steps below must not run: treating the collected
    # sources as "freshly refreshed" would delete the tenant's existing chunks
    # (and full-root edge reconciliation would drop their links) with no
    # replacement written. The watch loop re-indexes each file under the live
    # quota instead.
    pruned = 0
    if args.prune and not args.recreate and not initial_skipped:
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
    if graph_uri and not initial_skipped:
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
            args, store, provider, index_root, str(target), graph_uri, watch_baseline,
            retry_all=initial_skipped,
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
    *,
    retry_all: bool = False,
) -> int:
    """Run the incremental file-watch loop after the initial index (blocking).

    ``watch_root`` is the folder to observe (what was indexed); ``index_root`` is
    the corpus root the syncer scopes ids/sources/graph nodes to — identical in
    the common case, but distinct when ``--index-root`` pins a nested subtree.

    ``retry_all`` seeds every current file for retry — set when the initial index
    was skipped wholesale (over quota), so the watcher re-indexes the existing
    corpus under the live quota instead of leaving it unindexed until touched.
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

    _watch_tenant = getattr(args, "tenant", None) or None
    syncer = MarkdownSyncer(
        store,
        provider,
        index_root=index_root,
        chunk_size=args.chunk_size,
        graph=graph,
        subtree=watch_root,  # keep referrer re-resolution inside the watched tree
        tenant=_watch_tenant,
        # A resolver (not a fixed value) so `quota set/rm` takes effect mid-watch.
        max_points_resolver=lambda: _resolve_max_points(args, _watch_tenant),
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
    if retry_all:
        # The initial index was skipped over quota; queue the existing corpus so
        # a later `quota set` re-indexes it without waiting for each file's mtime
        # to change (poll_once only re-emits changed or already-failed paths).
        watcher.queue_all_for_retry()
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
            tenant=getattr(args, "tenant", None) or None,
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

    p_text_index = sub.add_parser(
        "text-index",
        parents=[common],
        help="Create the full-text payload index `recall.text_search: lexical` "
        "requires on a real Qdrant server (idempotent, non-destructive)",
    )
    p_text_index.set_defaults(func=cmd_text_index)

    p_resolve = sub.add_parser(
        "resolve",
        parents=[common],
        help="Verify a citation: resolve a chunk id back to its source "
        "document and report an honest verdict (intact / source_changed / "
        "moved / changed / missing / unresolvable)",
    )
    p_resolve.add_argument("chunk_id", help="the [id:...] value from recall/answer output")
    p_resolve.add_argument(
        "--root",
        default=None,
        help="Where relative sources are looked up (default: the payload's own index_root)",
    )
    p_resolve.add_argument("--tenant", default=None, help="Tenant scope for the lookup")
    p_resolve.add_argument(
        "--json", action="store_true", help="Emit the full resolution as JSON"
    )
    p_resolve.set_defaults(func=cmd_resolve)

    p_sparse_backfill = sub.add_parser(
        "sparse-backfill",
        parents=[common],
        help="Write sparse text encodings onto every existing point — the "
        "migration path for `recall.text_search: sparse` on a populated "
        "collection (idempotent, nothing re-embedded)",
    )
    p_sparse_backfill.set_defaults(func=cmd_sparse_backfill)

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

    p_tenant_export = sub.add_parser(
        "tenant-export",
        parents=[common],
        help="Export a tenant's vector points (vectors + payloads) as JSONL",
    )
    p_tenant_export.add_argument("--tenant", required=True, help="tenant_id to export")
    p_tenant_export.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file ('-' = stdout, the default; progress then goes to stderr). "
        "Written atomically: an existing file is replaced only after the export "
        "completes",
    )
    p_tenant_export.add_argument(
        "--no-vectors",
        action="store_true",
        help="Payloads only (smaller dump; restoring requires re-embedding)",
    )
    p_tenant_export.set_defaults(func=cmd_tenant_export)

    p_tenant_rm = sub.add_parser(
        "tenant-rm",
        parents=[common],
        help="Offboard a tenant: delete its points, graph records, keys, quota, and "
        "learning state (counts first; --dry-run / --yes)",
    )
    p_tenant_rm.add_argument("--tenant", required=True, help="tenant_id to remove")
    p_tenant_rm.add_argument(
        "--dry-run", action="store_true", help="Report per-store counts without deleting"
    )
    p_tenant_rm.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Confirm the irreversible cross-store deletion",
    )
    p_tenant_rm.add_argument(
        "--external-keys-revoked",
        action="store_true",
        help="With an external key store (MNEMOSTACK_KEYSTORE=openbao), confirm you "
        "have ALREADY revoked the tenant's key(s) there AND that running servers "
        "no longer accept them (their positive-verify cache has expired past "
        "MNEMOSTACK_OPENBAO_CACHE_TTL, or they were restarted) — tenant-rm can't "
        "verify this and otherwise refuses to sweep while a key could still write",
    )
    p_tenant_rm.add_argument(
        "--memgraph-uri",
        # A graph configured in config/env is swept by default; an EMPTY/unset
        # config value normalizes to None so it reads as "no graph configured"
        # (matching doctor's `if not cfg.graph.uri`) — not an explicit-empty-flag
        # error. Only an empty value passed on the CLI hits that error below.
        default=cfg.graph.uri or None,
        help="Also delete the tenant's graph nodes/edges at this bolt URI. "
        "Defaults to the configured graph (config/env), so a graph deployment is "
        "swept automatically; passes through as unset only when no graph is "
        "configured. Pass an explicit URI to override.",
    )
    p_tenant_rm.add_argument(
        "--no-graph",
        action="store_true",
        help="Explicitly skip graph deletion (a vector-only deployment). Required "
        "instead of --memgraph-uri when the stack config can't be loaded, so a "
        "config-file-only graph can't be silently missed.",
    )
    p_tenant_rm.add_argument(
        "--graph-timeout",
        type=float,
        default=cfg.graph.timeout,
        help="Memgraph connection timeout in seconds (default 5.0)",
    )
    p_tenant_rm.add_argument(
        "--keys-file",
        default=None,
        help="Key store path (default: $MNEMOSTACK_KEYS_FILE or ~/.config/mnemostack/keys.json)",
    )
    p_tenant_rm.add_argument(
        "--quotas-file",
        default=None,
        help="Quota store path (default: $MNEMOSTACK_QUOTAS_FILE or "
        "~/.config/mnemostack/quotas.json)",
    )
    p_tenant_rm.add_argument(
        "--state-path",
        default=None,
        help="Learning-state file (default: the server's default state path)",
    )
    p_tenant_rm.set_defaults(func=cmd_tenant_rm)

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

    # ---- quota ----
    p_quota = sub.add_parser(
        "quota", help="Manage per-tenant resource quotas (set / list / rm)"
    )
    qsub = p_quota.add_subparsers(dest="quota_action")

    def _quota_help(_args: argparse.Namespace) -> int:
        p_quota.print_help()
        return 2

    p_quota.set_defaults(func=_quota_help)
    from mnemostack.quotas import _UNSET
    _quotas_file_help = (
        "Quota store path (default: $MNEMOSTACK_QUOTAS_FILE or "
        "~/.config/mnemostack/quotas.json)"
    )
    pq_set = qsub.add_parser(
        "set",
        help="Set a tenant's quota (changes only the fields you pass)",
    )
    pq_set.add_argument("--tenant", required=True, help="tenant_id to limit")
    pq_set.add_argument(
        "--max-points",
        type=_int_or_none,
        default=_UNSET,
        metavar="N|none",
        help="Max vector points the tenant may store ('none' to clear the limit)",
    )
    pq_set.add_argument(
        "--max-rps",
        type=_float_or_none,
        default=_UNSET,
        metavar="R|none",
        help="Max sustained requests/sec at the authenticated HTTP surface "
        "('none' to clear)",
    )
    pq_set.add_argument(
        "--burst",
        type=_int_or_none,
        default=_UNSET,
        metavar="N|none",
        help="Rate-limit burst capacity (defaults to max-rps rounded up; 'none' to clear)",
    )
    pq_set.add_argument("--quotas-file", default=None, help=_quotas_file_help)
    pq_set.set_defaults(func=cmd_quota_set)
    pq_list = qsub.add_parser("list", help="List tenant quotas")
    pq_list.add_argument("--json", action="store_true", help="JSON output")
    pq_list.add_argument("--quotas-file", default=None, help=_quotas_file_help)
    pq_list.set_defaults(func=cmd_quota_list)
    pq_rm = qsub.add_parser("rm", help="Remove a tenant's quota")
    pq_rm.add_argument("--tenant", required=True, help="tenant_id whose quota to drop")
    pq_rm.add_argument("--quotas-file", default=None, help=_quotas_file_help)
    pq_rm.set_defaults(func=cmd_quota_rm)

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
        help="Retriever source to use ('bm25' selects the CONFIGURED lexical arm — see recall.text_search) (can be given multiple times; default: all available)",
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
        "--quotas-file",
        default=None,
        help="Quota store to enforce the tenant's max-points limit (default: "
        "$MNEMOSTACK_QUOTAS_FILE or ~/.config/mnemostack/quotas.json)",
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
    p_feedback.add_argument(
        "--tenant",
        default=None,
        metavar="ID",
        help="Record the feedback into this tenant's learning-state partition "
        "(default: unscoped / single-tenant)",
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
        "--quotas-file",
        default=None,
        help=(
            "Quota store for per-tenant request rate limits (--auth only; a tenant's "
            "max_rps is enforced). Default: $MNEMOSTACK_QUOTAS_FILE or "
            "~/.config/mnemostack/quotas.json"
        ),
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
    p_inspect.add_argument(
        "--auth",
        action="store_true",
        help=(
            "Admin console: require an admin service key on every /api call and "
            "unlock key/quota management (issue/revoke keys, set/remove quotas). "
            "Without it the inspector is a read-only data browser."
        ),
    )
    p_inspect.add_argument(
        "--keys-file",
        default=None,
        help="Service-key store path (default: $MNEMOSTACK_KEYS_FILE or ~/.config/mnemostack/keys.json)",
    )
    p_inspect.add_argument(
        "--quotas-file",
        default=None,
        help="Quota store path (default: $MNEMOSTACK_QUOTAS_FILE or ~/.config/mnemostack/quotas.json)",
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

    _schema_text, _schema_ts, _schema_fmt = _payload_schema()
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
        quotas_file=getattr(args, "quotas_file", None),
        text_key=_schema_text,
        timestamp_key=_schema_ts,
        timestamp_format=_schema_fmt,
        text_search=_text_search_mode(),
        text_search_fields=dict(_text_search_fields()),
        resolve_roots=[
            p for p in os.environ.get("MNEMOSTACK_RESOLVE_ROOTS", "").split(os.pathsep) if p
        ],
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
        from mnemostack.server import ServerConfig, _env_bool
    except ImportError as exc:
        print(
            f"error: server extra not installed ({exc}). Install with: "
            "pip install 'mnemostack[server]'",
            file=sys.stderr,
        )
        return 2

    _schema_text, _schema_ts, _schema_fmt = _payload_schema()
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
        # --auth turns the inspector into an admin console (key/quota management),
        # requiring an admin key on every /api call. Honor the env toggle too, for
        # parity with serve / mcp-serve.
        auth_enabled=getattr(args, "auth", False) or _env_bool("MNEMOSTACK_AUTH_ENABLED"),
        keys_file=getattr(args, "keys_file", None),
        quotas_file=getattr(args, "quotas_file", None),
        text_key=_schema_text,
        timestamp_key=_schema_ts,
        timestamp_format=_schema_fmt,
        text_search=_text_search_mode(),
        text_search_fields=dict(_text_search_fields()),
    )
    app = build_inspector_app(cfg)
    admin = cfg.auth_enabled
    if args.host == "0.0.0.0":
        print(
            "warning: the inspector exposes memory contents"
            + (" and tenant administration (key/quota management)" if admin else "")
            + "; binding to 0.0.0.0 makes it reachable on all interfaces — keep it "
            "behind auth/TLS or bound to localhost",
            file=sys.stderr,
        )
    mode = "admin console — admin key required" if admin else "read-only"
    print(f"mnemostack inspect: http://{args.host}:{args.port}  ({mode})")
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

    _schema_text, _schema_ts, _schema_fmt = _payload_schema()
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
        text_key=_schema_text,
        timestamp_key=_schema_ts,
        timestamp_format=_schema_fmt,
        text_search=_text_search_mode(),
        text_search_fields=dict(_text_search_fields()) or None,
        resolve_roots=[
            p for p in os.environ.get("MNEMOSTACK_RESOLVE_ROOTS", "").split(os.pathsep) if p
        ],
    )
    mcp.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    # `keys` and `quota` manage their own stores and need no stack config, so a
    # malformed unrelated config/env must not block managing a key or a quota.
    subcmd = next((a for a in raw if not a.startswith("-")), None)
    if subcmd in {"tenant-rm", "tenant-export"}:
        # Lifecycle commands only need vector/graph *defaults* from the config
        # (they never touch embedding/LLM/recall settings), but unlike `keys`
        # they DO benefit from the config's graph auth — so try the full load
        # first: a malformed unrelated value (e.g. MNEMOSTACK_TOKEN_BUDGET=notint)
        # must not block an emergency backup or offboarding.
        try:
            parser = build_parser()
        except Exception as e:  # noqa: BLE001
            # Fall back — but NEVER to destructive defaults: with the config
            # unreadable, the built-in http://localhost:6333/"mnemostack" seeds
            # could point tenant-rm --yes (or a trusted backup) at the wrong
            # store. Require the target to be named explicitly on the CLI.
            def _passed(flag: str) -> bool:
                return any(a == flag or a.startswith(flag + "=") for a in raw)

            missing = [f for f in ("--qdrant", "--collection") if not _passed(f)]
            # tenant-rm claims full removal, but with the config unreadable we
            # CAN'T see a graph configured in the config FILE — so require an
            # explicit graph decision (a URI, or --no-graph) rather than silently
            # skipping a graph that might exist. (--memgraph-uri's cfg-default is
            # None here since config_light built a bare Config.)
            if (
                subcmd == "tenant-rm"
                and not _passed("--memgraph-uri")
                and not _passed("--no-graph")
            ):
                missing.append("--memgraph-uri or --no-graph (config-file graph is invisible now)")
            if missing:
                print(
                    f"error: stack config failed to load ({e}); refusing to fall "
                    f"back to built-in defaults for {subcmd} — pass "
                    f"{' and '.join(missing)} explicitly (and graph credentials "
                    "via env if the graph is authenticated)",
                    file=sys.stderr,
                )
                return 2
            print(
                f"warning: stack config failed to load ({e}); using built-in "
                "defaults for everything not passed explicitly. Graph settings "
                "from the CONFIG FILE are invisible here — if it configures a "
                "graph, pass --memgraph-uri explicitly",
                file=sys.stderr,
            )
            parser = build_parser(config_light=True)
            # config_light built a bare Config() that skipped the graph-auth ENV
            # overrides, so re-apply them: an authenticated Memgraph must still
            # receive its credentials (else graph deletion fails and the sweep
            # proceeds, reporting removal while graph records survive).
            parser.set_defaults(
                graph_user=os.environ.get("MNEMOSTACK_GRAPH_USER") or None,
                graph_password=os.environ.get("MNEMOSTACK_GRAPH_PASSWORD") or None,
                graph_database=os.environ.get("MNEMOSTACK_GRAPH_DATABASE") or None,
            )
    else:
        parser = build_parser(config_light=subcmd in {"keys", "quota"})
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
