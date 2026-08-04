"""MCP server implementation — exposes mnemostack tools via MCP protocol.

Design notes:
- Components are lazy: graph_query tools register only if Memgraph is configured
- Tool names prefixed with `mnemostack_` to avoid collision with other MCP tools
- All tools accept JSON-serializable args and return JSON-serializable output
- Errors are returned as structured error objects, not exceptions
"""

from __future__ import annotations

import os
import threading
from typing import Annotated, Any

try:
    from fastmcp import FastMCP
    from pydantic import Field

    _FASTMCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment, misc]
    Field = None  # type: ignore[assignment]
    _FASTMCP_AVAILABLE = False

from ..config import Config, model_kwargs, provider_kwargs
from ..embeddings import get_provider
from ..feedback import apply_feedback
from ..llm import get_llm
from ..recall import (
    RERANK_MODES,
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    Recaller,
    RecallTrace,
    Reranker,
    TemporalRetriever,
    VectorRetriever,
    build_bm25_docs,
    build_full_pipeline,
    recall_flow,
    sum_tokens,
)
from ..recall.pipeline import FileStateStore, default_state_path
from ..vector import VectorStore


def _public_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {key: value for key, value in payload.items() if key != "_vector_floor_candidates"}


def build_server(
    collection: str = "mnemostack",
    embedding_provider: str = "gemini",
    embedding_model: str | None = None,
    llm_provider: str = "gemini",
    llm_model: str | None = None,
    qdrant_host: str = "http://localhost:6333",
    memgraph_uri: str | None = None,
    graph_timeout: float = 5.0,
    bm25_paths: list[str] | None = None,
    state_path: str | None = None,
    vector_floor: int = 0,
    rerank_mode: str = "relevant_only",
    token_budget: int | None = None,
    # graph auth appended at the tail to preserve positional back-compat.
    graph_user: str = "",
    graph_password: str = "",
    graph_database: str | None = None,
    # service-key auth (multi-tenant): the MCP process runs AS one principal.
    auth_enabled: bool = False,
    api_key: str | None = None,
    keys_file: str | None = None,
    # Payload schema of the collection recall reads (keyword-position tail;
    # a pre-existing collection keeps its own field names / numeric epochs).
    text_key: str = "text",
    timestamp_key: str = "timestamp",
    timestamp_format: str = "iso",
    text_search: str = "auto",
    text_search_fields: dict[str, float] | None = None,
    resolve_roots: list[str] | None = None,
    # Provider knobs — appended at the TAIL to preserve positional
    # back-compat for existing library callers.
    ollama_host: str | None = None,
    embedding_timeout: int | None = None,
) -> Any:
    """Build and return a configured FastMCP server.

    Args:
        collection: Qdrant collection name
        embedding_provider: embedding provider name (registered in mnemostack.embeddings)
        embedding_model: embedding model override (None uses provider default)
        ollama_host: Ollama endpoint for the embedding provider (None = the
            provider's own resolution: native OLLAMA_HOST env, then localhost)
        embedding_timeout: embedding request timeout in seconds (None =
            provider default)
        llm_provider: LLM provider name for answer generation
        llm_model: LLM model override (None uses provider default)
        qdrant_host: Qdrant URL
        memgraph_uri: if provided, register graph tools (e.g. bolt://localhost:7687)
        state_path: JSON state file for feedback / stateful recall stages
        vector_floor: protect top-N raw-vector candidates from later ranking stages
        rerank_mode: LLM reranker mode for service parity
        token_budget: default token budget applied to search/answer recall when
            the tool call does not pass one (None = no budget)

    Returns:
        FastMCP instance ready to .run()

    Raises:
        ImportError: if fastmcp not installed (install with mnemostack[mcp])
    """
    if not _FASTMCP_AVAILABLE:
        raise ImportError("fastmcp not installed. Install with: pip install 'mnemostack[mcp]'")
    if rerank_mode not in RERANK_MODES:
        allowed = ", ".join(sorted(RERANK_MODES))
        raise ValueError(f"rerank_mode must be one of: {allowed}")
    # Validate the fields config at BUILD time, not lazily inside the
    # recaller factory: a misconfigured deployment must fail before the MCP
    # server starts advertising tools whose every recall would then error
    # (possibly behind an unrelated embedding/vector failure). Both halves —
    # the weights themselves (parse) and the fields/mode pairing (ensure).
    from mnemostack.config import (
        ensure_text_fields_mode,
        parse_text_search_fields,
        resolve_text_search_mode,
    )

    text_search_fields = parse_text_search_fields(text_search_fields)
    ensure_text_fields_mode(
        resolve_text_search_mode(text_search, bm25_paths), text_search_fields
    )

    mcp = FastMCP("mnemostack")

    resolved_state_path = state_path or default_state_path()
    # The tool functions take a per-call `token_budget` parameter that
    # shadows the factory argument; keep the server-wide default reachable.
    # 0 or negative means "no budget" (apply_token_budget requires >= 1).
    default_token_budget = token_budget if token_budget is not None and token_budget > 0 else None

    # ----- Auth (multi-tenant). MCP's stdio transport is one client per process,
    # so (unlike HTTP) there is no per-request credential — the server is launched
    # WITH a service key that resolves the tenant + scopes for the whole process.
    # The key is re-verified on every tool call so a mid-session revocation takes
    # effect immediately (fail closed). Off by default: byte-identical, unscoped.
    key_store = None
    if auth_enabled:
        from ..auth import make_key_store

        # Backend selected by MNEMOSTACK_KEYSTORE (file default, openbao optional);
        # a selected-but-misconfigured backend raises here (boot fails loud).
        key_store = make_key_store(keys_file)
        if not api_key:
            raise ValueError(
                "auth enabled but no service key provided (set --api-key or MNEMOSTACK_API_KEY)"
            )
        if key_store.verify(api_key) is None:
            raise ValueError("auth enabled but the provided service key is invalid or revoked")

    class _AuthError(Exception):
        """Raised inside a tool on an auth failure; the tool's except maps it to a
        structured {"ok": False, "error": ...} (tools return errors, not raise)."""

    def _authorize(scope: str):
        """Principal for an authenticated call, or None when auth is off. Re-verifies
        the bound key each call (revocation takes effect at once); raises _AuthError
        on a revoked key or insufficient scope."""
        if not auth_enabled:
            return None
        principal = key_store.verify(api_key) if (key_store and api_key) else None
        if principal is None:
            raise _AuthError("service key invalid or revoked")
        if not principal.can(scope):
            raise _AuthError(f"key lacks '{scope}' scope")
        return principal

    def _tenant_of(principal: Any) -> str | None:
        return principal.tenant if principal is not None else None

    # Lazy-initialize components so server boots even if e.g. GEMINI_API_KEY missing.
    # Tool calls can run concurrently; the lock makes each component initialize
    # exactly once instead of racing into duplicate providers/clients.
    _components: dict[str, Any] = {}
    _components_lock = threading.RLock()

    def _component(name: str, factory):
        with _components_lock:
            if name not in _components:
                _components[name] = factory()
            return _components[name]

    def _get_embedding():
        return _component(
            "embedding",
            lambda: get_provider(
                embedding_provider,
                **provider_kwargs(
                    embedding_provider,
                    model=embedding_model,
                    ollama_host=ollama_host,
                    timeout=embedding_timeout,
                ),
            ),
        )

    def _get_vector():
        return _component(
            "vector",
            lambda: VectorStore(
                collection=collection,
                dimension=_get_embedding().dimension,
                host=qdrant_host,
            ),
        )

    def _get_vector_payload_only():
        # Invalidation is a payload write (retrieve + set_payload) that never
        # touches vectors, so it must not require the embedding provider —
        # otherwise a missing/unconfigured provider would break a pure payload
        # op. The dimension is unused by those calls.
        return _component(
            "vector_payload_only",
            lambda: VectorStore(collection=collection, dimension=1, host=qdrant_host),
        )

    def _build_recaller():
        from mnemostack.config import ensure_text_fields_mode, resolve_text_search_mode
        from mnemostack.recall import QdrantSparseRetriever, build_qdrant_text_arms

        emb = _get_embedding()
        vec = _get_vector()
        schema_kw = {
            "text_key": text_key,
            "timestamp_key": timestamp_key,
            "timestamp_format": timestamp_format,
        }
        mode = resolve_text_search_mode(text_search, bm25_paths)
        ensure_text_fields_mode(mode, text_search_fields or {})
        lexical_arms: list[Any] = []
        lexical_weights: dict[str, float] = {}
        if mode == "bm25":
            bm25_docs = build_bm25_docs(bm25_paths)
            if bm25_docs:
                lexical_arms.append(
                    BM25Retriever(
                        docs=bm25_docs,
                        timestamp_key=timestamp_key,
                        timestamp_format=timestamp_format,
                    )
                )
        elif mode == "qdrant_bm25":
            lexical_arms.append(
                BM25Retriever.from_qdrant(vec.client, collection, text_key=schema_kw["text_key"], timestamp_key=schema_kw["timestamp_key"], timestamp_format=schema_kw["timestamp_format"])
            )
        elif mode == "lexical":
            arms, lexical_weights = build_qdrant_text_arms(
                embedding=emb, vector_store=vec, fields=text_search_fields, **schema_kw
            )
            lexical_arms.extend(arms)
        elif mode == "sparse":
            sparse_store = VectorStore(
                collection=collection,
                dimension=1,  # sparse queries never touch the dense space
                host=qdrant_host,
                sparse_text=True,
                text_key=text_key,
            )
            lexical_arms.append(
                QdrantSparseRetriever(vector_store=sparse_store, text_key=schema_kw["text_key"], timestamp_key=schema_kw["timestamp_key"], timestamp_format=schema_kw["timestamp_format"])
            )
        retrievers = [
            VectorRetriever(
                embedding=emb,
                vector_store=vec,
                text_key=text_key,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            ),
            *lexical_arms,
            MemgraphRetriever(
                uri=memgraph_uri,
                user=graph_user,
                password=graph_password,
                database=graph_database,
                timeout=graph_timeout,
            )
            if memgraph_uri
            else None,
            TemporalRetriever(
                embedding=emb,
                vector_store=vec,
                text_key=text_key,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            ),
        ]
        return Recaller(
            retrievers=[r for r in retrievers if r is not None],
            vector_floor=max(0, int(vector_floor)),
            retriever_weights=lexical_weights or None,
            text_key=text_key,
            timestamp_key=timestamp_key,
            timestamp_format=timestamp_format,
        )

    def _get_recaller():
        return _component("recaller", _build_recaller)

    def _get_answer_gen():
        return _component(
            "answer",
            lambda: AnswerGenerator(
                llm=get_llm(llm_provider, **model_kwargs(llm_model)),
                recaller=_get_recaller(),
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            ),
        )

    def _get_pipeline():
        # The graph is tenant-scoped now: the GraphResurrection stage confines its
        # seed walk to the caller's tenant (threaded in via the recall's pipeline
        # context) and stamps `tenant_id` on what it injects, so it's safe to keep
        # in the pipeline under auth — a tenant only ever resurrects its own nodes.
        return _component(
            "pipeline",
            lambda: build_full_pipeline(
                state_store=FileStateStore(resolved_state_path),
                graph_uri=memgraph_uri,
                graph_user=graph_user,
                graph_password=graph_password,
                graph_database=graph_database,
                graph_timeout=graph_timeout,
                text_key=text_key,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            ),
        )

    def _get_reranker():
        return _component(
            "reranker",
            lambda: Reranker(
                llm=get_llm(llm_provider, **model_kwargs(llm_model)),
                max_items=20,
                rerank_mode=rerank_mode,
            ),
        )

    def _run_recall(
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
        token_budget_override: int | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
        tenant: str | None = None,
    ) -> tuple[list[Any], RecallTrace]:
        trace = RecallTrace()
        recaller = _get_recaller()
        try:
            reranker = _get_reranker()
        except Exception:  # noqa: BLE001 — LLM not configured; recall still works
            reranker = None
            trace.mark("reranker:unavailable")
        results = recall_flow(
            recaller,
            query,
            limit,
            pipeline=_get_pipeline(),
            reranker=reranker,
            filters=filters,
            trace=trace,
            # Per-call budget wins; unset falls back to the server-wide default.
            token_budget=(
                token_budget_override
                if token_budget_override is not None
                else default_token_budget
            ),
            include_invalidated=include_invalidated,
            as_of=as_of,
            tenant=tenant,
        )
        return results, trace

    def _get_feedback_pipeline():
        if "feedback_pipeline" not in _components:
            _components["feedback_pipeline"] = build_full_pipeline(
                state_store=FileStateStore(resolved_state_path),
                graph_uri=None,
                text_key=text_key,
                timestamp_key=timestamp_key,
                timestamp_format=timestamp_format,
            )
        return _components["feedback_pipeline"]

    # ---------- tools ----------

    @mcp.tool()
    def mnemostack_health() -> dict:
        """Check health of all mnemostack components.

        Read-only, no side effects, no authentication required. Returns a JSON
        object with ok (bool) and per-component status for the embedding
        provider, Qdrant vector store, and optional Memgraph graph database.
        Use this to verify the memory backend is reachable before issuing recall
        queries.
        """
        result: dict[str, Any] = {"ok": True, "components": {}}
        try:
            emb = _get_embedding()
            ok, msg = emb.health_check()
            result["components"]["embedding"] = {
                "ok": ok,
                "provider": emb.name,
                "dimension": emb.dimension,
                "message": msg,
            }
            if not ok:
                result["ok"] = False
        except Exception as e:  # noqa: BLE001
            result["components"]["embedding"] = {"ok": False, "error": str(e)}
            result["ok"] = False

        try:
            vec = _get_vector()
            exists = vec.collection_exists()
            count = vec.count() if exists else 0
            result["components"]["vector"] = {
                "ok": True,
                "collection": vec.collection,
                "exists": exists,
                "points": count,
            }
        except Exception as e:  # noqa: BLE001
            result["components"]["vector"] = {"ok": False, "error": str(e)}
            result["ok"] = False

        if memgraph_uri:
            try:
                from ..graph.factory import make_graph_store

                gs = make_graph_store(
                    memgraph_uri,
                    timeout=graph_timeout,
                    user=graph_user,
                    password=graph_password,
                    database=graph_database,
                )
                ok, msg = gs.health_check()
                result["components"]["graph"] = {
                    "ok": ok,
                    "nodes": gs.count_nodes() if ok else 0,
                    "edges": gs.count_edges() if ok else 0,
                    "message": msg,
                }
                gs.close()
                if not ok:
                    result["ok"] = False
            except Exception as e:  # noqa: BLE001
                result["components"]["graph"] = {"ok": False, "error": str(e)}
                result["ok"] = False

        return result

    @mcp.tool()
    def mnemostack_search(
        query: Annotated[
            str,
            Field(description="Natural language question or keyword to search memories for"),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum number of results to return (default 10)"),
        ] = 10,
        include_trace: Annotated[
            bool,
            Field(description="Include the per-retriever recall trace (debug; verbose)"),
        ] = False,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Payload filters applied inside every retriever: exact match "
                    '({"tenant": "a"}) or gte/lte range ({"timestamp": '
                    '{"gte": "2026-01-01"}}). Results never include points '
                    "outside the filtered scope."
                )
            ),
        ] = None,
        token_budget: Annotated[
            int | None,
            Field(
                description=(
                    "Hard cap on the total (estimated) text tokens of the "
                    "returned results; the final ranking is cut to the prefix "
                    "that fits. Unset uses the server-wide default, if any."
                ),
                ge=1,
            ),
        ] = None,
        include_invalidated: Annotated[
            bool,
            Field(
                description=(
                    "Include facts marked stale. Default false: memories with "
                    "an invalidated_at marker are hidden from recall."
                )
            ),
        ] = False,
        as_of: Annotated[
            str | None,
            Field(
                description=(
                    "Point-in-time recall (ISO-8601): return facts valid at "
                    "this world-time instant, ignoring later invalidation."
                )
            ),
        ] = None,
    ) -> dict:
        """Search indexed memories with hybrid recall.

        Read-only, no side effects, no authentication required. Use this when
        you need raw memory matches rather than a synthesized answer. Returns a
        JSON object with ok, query, count, results, tokens_estimate (estimated
        total text tokens of the results), and degraded (which
        components fell back while serving the call; empty when healthy).
        Results are ranked by reciprocal rank fusion of BM25, semantic, graph,
        and temporal retrievers when configured; each result includes id, text,
        score, sources, and payload. Stale facts (invalidated_at set) are
        hidden by default; use include_invalidated or as_of to see them.
        """
        try:
            principal = _authorize("read")
            results, trace = _run_recall(
                query, limit, filters, token_budget, include_invalidated, as_of,
                tenant=_tenant_of(principal),
            )
            response = {
                "ok": True,
                "query": query,
                "count": len(results),
                "degraded": trace.degraded,
                "tokens_estimate": sum_tokens(results),
                "results": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "score": round(r.score, 4),
                        "sources": r.sources,
                        "payload": _public_payload(r.payload),
                    }
                    for r in results
                ],
            }
            if include_trace:
                response["trace"] = trace.to_dict()
            return response
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "query": query}

    @mcp.tool()
    def mnemostack_answer(
        query: Annotated[
            str,
            Field(description="Natural language question or keyword to search memories for"),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum number of results to return (default 10)"),
        ] = 10,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Payload filters applied inside every retriever (exact match "
                    "or gte/lte ranges); the answer is generated only from "
                    "memories inside the filtered scope."
                )
            ),
        ] = None,
        token_budget: Annotated[
            int | None,
            Field(
                description=(
                    "Hard cap on the total (estimated) text tokens of the "
                    "memories fed to the answer LLM (same contract as "
                    "mnemostack_search). Unset uses the server-wide default."
                ),
                ge=1,
            ),
        ] = None,
        include_invalidated: Annotated[
            bool,
            Field(
                description="Include facts marked stale (default false; same as mnemostack_search)."
            ),
        ] = False,
        as_of: Annotated[
            str | None,
            Field(
                description="Point-in-time recall (ISO-8601); same contract as mnemostack_search."
            ),
        ] = None,
    ) -> dict:
        """Answer a question using retrieved memories.

        Read-only, no side effects, no authentication required. Use this when
        you want a concise factual answer synthesized from memory search results
        instead of the raw matches returned by mnemostack_search. Returns a JSON
        object with ok, query, answer text, confidence (0.0-1.0), sources,
        degraded (components that fell back while serving the call),
        fallback_recommended, tokens_estimate (estimated text tokens of the
        context memories), tokens_used (LLM-provider-reported usage for the
        answer call; null when unreported), and error. Stale facts are hidden
        by default; use include_invalidated or as_of to see them.
        """
        effective_budget = token_budget if token_budget is not None else default_token_budget
        try:
            principal = _authorize("read")
            tenant = _tenant_of(principal)
            memories, trace = _run_recall(
                query, limit, filters, effective_budget, include_invalidated, as_of, tenant=tenant
            )
            gen = _get_answer_gen()
            # recall_filters keeps the generator's retry sub-recalls inside
            # the same filtered scope; the budget must reach the generator
            # too, or those sub-recalls would prompt unbudgeted. tenant scopes
            # those retry sub-recalls to the caller's tenant as well.
            answer = gen.generate(
                query,
                memories,
                recall_filters=filters,
                token_budget=effective_budget,
                include_invalidated=include_invalidated,
                as_of=as_of,
                tenant=tenant,
            )
            # Prefer the generator's own estimate: its retry paths can swap
            # in a freshly recalled context pool. getattr: custom/duck-typed
            # answer generators may return objects predating these fields.
            tokens_estimate = getattr(answer, "context_tokens_estimate", None)
            if tokens_estimate is None:
                tokens_estimate = sum_tokens(memories)
            return {
                "ok": answer.ok,
                "query": query,
                "answer": answer.text,
                "confidence": round(answer.confidence, 3),
                "sources": answer.sources,
                "degraded": trace.degraded,
                "fallback_recommended": gen.should_fallback(answer),
                "tokens_estimate": tokens_estimate,
                "tokens_used": getattr(answer, "tokens_used", None),
                "error": answer.error,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "query": query}

    @mcp.tool()
    def mnemostack_resolve(
        chunk_id: Annotated[
            str,
            Field(description="The [id:...] value from a recall result or answer citation"),
        ],
    ) -> dict:
        """Verify a citation: resolve a chunk id back to its source document.

        Re-reads the CURRENT source and returns an honest verdict: intact /
        source_changed / moved (citation still supported), changed / missing
        (not supported by the current source), or unresolvable (cannot be
        verified from this process). Includes the snapshot-hash comparison and
        the fragment when locatable. Read-only; never mutates stored memory;
        runs outside the recall path. Resolution is confined to the corpus
        root recorded at ingest — there is deliberately no way for a caller
        to point it at another directory.
        """
        try:
            from ..provenance import resolve_citation

            principal = _authorize("read")
            # Payload-only store: resolving never embeds, so a missing or
            # unconfigured embedding provider must not break verification.
            # allowed_roots: the stored index_root is payload data, not a
            # security boundary — empty (unset) keeps this tool fail-closed.
            res = resolve_citation(
                _get_vector_payload_only(),
                chunk_id,
                tenant=_tenant_of(principal),
                text_key=text_key,
                allowed_roots=list(resolve_roots or []),
            )
            return {"ok": True, **res.to_dict()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "chunk_id": chunk_id}

    @mcp.tool()
    def mnemostack_invalidate(
        ids: Annotated[
            list[str | int],
            Field(description="Point id(s) to mark stale (string or integer)"),
        ],
        valid_until: Annotated[
            str | None,
            Field(
                description=(
                    "World-time the fact stopped being true (ISO-8601); "
                    "optional, separate from the system-time invalidation stamp."
                )
            ),
        ] = None,
        invalidated_at: Annotated[
            str | None,
            Field(description="System-time stamp (ISO-8601); default: now (UTC)"),
        ] = None,
        index_root: Annotated[
            str | None,
            Field(
                description=(
                    "Owner guard: when set, points owned by a different "
                    "index_root are skipped, so one root cannot invalidate "
                    "another's chunks in a shared collection."
                )
            ),
        ] = None,
    ) -> dict:
        """Mark memories stale by id, non-destructively.

        A write tool (parallel to mnemostack_graph_add_triple). Sets
        invalidated_at (and optionally valid_until) on each point's payload
        without deleting or re-embedding it; invalidated facts drop out of
        default recall but stay reachable via include_invalidated / as_of.
        Points that do not exist are skipped. Pass index_root in multi-root
        collections to avoid marking another root's chunks stale. Returns ok,
        requested, and invalidated (the number of points actually updated).
        """
        try:
            tenant = _tenant_of(_authorize("write"))
            # Coerce digit-only ids to int so numeric-id Qdrant collections can
            # be invalidated (UUID strings contain hyphens, so stay strings).
            coerced = [int(x) if isinstance(x, str) and x.isdigit() else x for x in ids]
            # tenant owner-guard: only pass it when set so a custom store without
            # the parameter (and the single-tenant path) is unaffected.
            tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
            updated = _get_vector_payload_only().invalidate(
                coerced,
                invalidated_at=invalidated_at,
                valid_until=valid_until,
                index_root=index_root,
                **tkw,
            )
            return {
                "ok": True,
                "requested": len(ids),
                "invalidated": updated,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "requested": len(ids)}

    @mcp.tool()
    def mnemostack_feedback(
        hit_id: str,
        signal: str,
        query: Annotated[
            str | None,
            Field(description="Natural language question or keyword associated with the feedback"),
        ] = None,
        query_type: str | None = None,
        source: str | None = None,
        sources: list[str] | None = None,
        reward: float | None = None,
    ) -> dict:
        """Record explicit feedback for stateful recall learning.

        Use signal='clicked' to also record inhibition-of-return exposure.
        Pass retriever labels from mnemostack_search results as sources so
        Q-learning can update source weights.
        """
        if signal not in {"useful", "irrelevant", "clicked"}:
            return {
                "ok": False,
                "error": "signal must be one of: useful, irrelevant, clicked",
            }
        if reward is not None and not 0.0 <= reward <= 1.0:
            return {"ok": False, "error": "reward must be in [0, 1]"}
        try:
            # write scope gates access; the learning state (Q-table + IoR) is
            # partitioned by the key's tenant, so feedback only ever moves this
            # tenant's own ranking state.
            tenant = _tenant_of(_authorize("write"))
            outcome = apply_feedback(
                _get_feedback_pipeline(),
                hit_id=hit_id,
                signal=signal,
                query=query,
                query_type=query_type,
                source=source,
                sources=sources or [],
                reward=reward,
                tenant=tenant,
            )
            return outcome.to_dict()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "hit_id": hit_id}

    # ---------- graph tools (optional) ----------

    if memgraph_uri:

        @mcp.tool()
        def mnemostack_graph_query(
            subject: str | None = None,
            predicate: str | None = None,
            obj: str | None = None,
            as_of: str | None = None,
            limit: Annotated[
                int,
                Field(description="Maximum number of graph triples to return (default 50)"),
            ] = 50,
        ) -> dict:
            """Query knowledge graph with optional SPO filters and point-in-time.

            All filter args are optional. If `as_of` is provided (ISO date),
            returns only facts valid at that date.
            """
            try:
                # Structured SPO query — the graph is tenant-scoped now, so under
                # auth we confine the query to the caller's tenant (both endpoints
                # pinned to `tenant`) rather than fail closed. Unscoped when auth
                # is off (tenant=None).
                tenant = _tenant_of(_authorize("read"))
                from ..graph.factory import make_graph_store

                gs = make_graph_store(
                    memgraph_uri,
                    timeout=graph_timeout,
                    user=graph_user,
                    password=graph_password,
                    database=graph_database,
                )
                triples = gs.query_triples(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    as_of=as_of,
                    limit=limit,
                    tenant=tenant,
                )
                gs.close()
                return {
                    "ok": True,
                    "count": len(triples),
                    "triples": [
                        {
                            "subject": t.subject,
                            "predicate": t.predicate,
                            "obj": t.obj,
                            "valid_from": t.valid_from,
                            "valid_until": t.valid_until,
                        }
                        for t in triples
                    ],
                }
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": str(e)}

        @mcp.tool()
        def mnemostack_graph_add_triple(
            subject: str,
            predicate: str,
            obj: str,
            valid_from: str | None = None,
            valid_until: str | None = None,
        ) -> dict:
            """Add a temporal fact (subject, predicate, object) to the graph.

            Nodes are created on demand. valid_from/valid_until are optional
            ISO date strings for point-in-time validity.
            """
            try:
                # Structured write — stamp the caller's tenant so the triple lands
                # in that tenant's isolated subgraph (its nodes/edges carry
                # `tenant`), never a shared namespace. Unscoped when auth is off.
                tenant = _tenant_of(_authorize("write"))
                from ..graph.factory import make_graph_store

                gs = make_graph_store(
                    memgraph_uri,
                    timeout=graph_timeout,
                    user=graph_user,
                    password=graph_password,
                    database=graph_database,
                )
                gs.add_triple(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    tenant=tenant,
                )
                gs.close()
                return {"ok": True, "subject": subject, "predicate": predicate, "obj": obj}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": str(e)}

    return mcp


def main() -> None:
    """Entry point for `python -m mnemostack.mcp.server` or `mnemostack-mcp`.

    Reads config from env vars:
        MNEMOSTACK_COLLECTION       (default: mnemostack)
        MNEMOSTACK_EMBEDDING        (default: gemini)
        MNEMOSTACK_EMBEDDING_MODEL  (default: provider default)
        MNEMOSTACK_LLM              (default: gemini)
        MNEMOSTACK_LLM_MODEL        (default: provider default)
        MNEMOSTACK_QDRANT_HOST      (default: http://localhost:6333)
        MNEMOSTACK_MEMGRAPH_URI     (default: none — graph tools disabled)
        MNEMOSTACK_GRAPH_TIMEOUT    (default: 5.0)
        MNEMOSTACK_BM25_PATHS       (default: none, os.pathsep-separated paths)
        MNEMOSTACK_STATE_PATH       (default: $XDG_STATE_HOME/mnemostack/server-state.json,
                                     falling back to ~/.local/state/mnemostack/server-state.json)
        MNEMOSTACK_RERANK_MODE      (default: relevant_only)
        MNEMOSTACK_TOKEN_BUDGET     (default: none — no recall token budget)
    """
    cfg = Config.load()
    auth_enabled = os.environ.get("MNEMOSTACK_AUTH_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mcp = build_server(
        collection=cfg.vector.collection,
        embedding_provider=cfg.embedding.provider,
        embedding_model=cfg.embedding.model,
        ollama_host=cfg.embedding.ollama_host,
        embedding_timeout=cfg.embedding.timeout,
        llm_provider=cfg.llm.provider,
        llm_model=cfg.llm.model,
        qdrant_host=cfg.vector.host,
        memgraph_uri=cfg.graph.uri,
        graph_user=cfg.graph.user,
        graph_password=cfg.graph.password,
        graph_database=cfg.graph.database,
        graph_timeout=cfg.graph.timeout,
        bm25_paths=list(cfg.recall.bm25_paths) or None,
        state_path=os.environ.get("MNEMOSTACK_STATE_PATH"),
        vector_floor=max(0, int(cfg.recall.vector_floor)),
        rerank_mode=cfg.recall.rerank_mode,
        text_key=cfg.recall.text_key,
        timestamp_key=cfg.recall.timestamp_key,
        timestamp_format=cfg.recall.timestamp_format,
        text_search=cfg.recall.text_search,
        text_search_fields=dict(cfg.recall.text_search_fields) or None,
        resolve_roots=[
            p for p in os.environ.get("MNEMOSTACK_RESOLVE_ROOTS", "").split(os.pathsep) if p
        ],
        token_budget=cfg.recall.token_budget,
        auth_enabled=auth_enabled,
        api_key=os.environ.get("MNEMOSTACK_API_KEY") or None,
        keys_file=os.environ.get("MNEMOSTACK_KEYS_FILE") or None,
    )
    mcp.run()


if __name__ == "__main__":
    main()
