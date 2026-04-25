"""MCP server implementation — exposes mnemostack tools via MCP protocol.

Design notes:
- Components are lazy: graph_query tools register only if Memgraph is configured
- Tool names prefixed with `mnemostack_` to avoid collision with other MCP tools
- All tools accept JSON-serializable args and return JSON-serializable output
- Errors are returned as structured error objects, not exceptions
"""
from __future__ import annotations

from typing import Any

try:
    from fastmcp import FastMCP

    _FASTMCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment, misc]
    _FASTMCP_AVAILABLE = False

from ..config import Config, model_kwargs
from ..embeddings import get_provider
from ..llm import get_llm
from ..recall import (
    AnswerGenerator,
    BM25Retriever,
    MemgraphRetriever,
    Recaller,
    TemporalRetriever,
    VectorRetriever,
    build_bm25_docs,
)
from ..vector import VectorStore


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
) -> Any:
    """Build and return a configured FastMCP server.

    Args:
        collection: Qdrant collection name
        embedding_provider: embedding provider name (registered in mnemostack.embeddings)
        embedding_model: embedding model override (None uses provider default)
        llm_provider: LLM provider name for answer generation
        llm_model: LLM model override (None uses provider default)
        qdrant_host: Qdrant URL
        memgraph_uri: if provided, register graph tools (e.g. bolt://localhost:7687)

    Returns:
        FastMCP instance ready to .run()

    Raises:
        ImportError: if fastmcp not installed (install with mnemostack[mcp])
    """
    if not _FASTMCP_AVAILABLE:
        raise ImportError(
            "fastmcp not installed. Install with: pip install 'mnemostack[mcp]'"
        )

    mcp = FastMCP("mnemostack")

    # Lazy-initialize components so server boots even if e.g. GEMINI_API_KEY missing
    _components: dict[str, Any] = {}

    def _get_embedding():
        if "embedding" not in _components:
            _components["embedding"] = get_provider(
                embedding_provider,
                **model_kwargs(embedding_model),
            )
        return _components["embedding"]

    def _get_vector():
        if "vector" not in _components:
            emb = _get_embedding()
            _components["vector"] = VectorStore(
                collection=collection, dimension=emb.dimension, host=qdrant_host
            )
        return _components["vector"]

    def _get_recaller():
        if "recaller" not in _components:
            emb = _get_embedding()
            vec = _get_vector()
            bm25_docs = build_bm25_docs(bm25_paths)
            retrievers = [
                VectorRetriever(embedding=emb, vector_store=vec),
                BM25Retriever(docs=bm25_docs) if bm25_docs else None,
                MemgraphRetriever(uri=memgraph_uri, timeout=graph_timeout)
                if memgraph_uri
                else None,
                TemporalRetriever(embedding=emb, vector_store=vec),
            ]
            _components["recaller"] = Recaller(
                retrievers=[r for r in retrievers if r is not None],
            )
        return _components["recaller"]

    def _get_answer_gen():
        if "answer" not in _components:
            _components["answer"] = AnswerGenerator(
                llm=get_llm(llm_provider, **model_kwargs(llm_model))
            )
        return _components["answer"]

    # ---------- tools ----------

    @mcp.tool()
    def mnemostack_health() -> dict:
        """Check health of mnemostack components (embedding, vector store, optional graph)."""
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
                from ..graph import GraphStore

                gs = GraphStore(uri=memgraph_uri, timeout=graph_timeout)
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
    def mnemostack_search(query: str, limit: int = 10) -> dict:
        """Hybrid recall over indexed memories.

        Returns top-K results ranked by reciprocal rank fusion of BM25 and
        semantic search. Each result has id, text, score, sources, payload.
        """
        try:
            recaller = _get_recaller()
            results = recaller.recall(query, limit=limit)
            return {
                "ok": True,
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "score": round(r.score, 4),
                        "sources": r.sources,
                        "payload": r.payload,
                    }
                    for r in results
                ],
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "query": query}

    @mcp.tool()
    def mnemostack_answer(query: str, limit: int = 10) -> dict:
        """Generate concise factual answer from retrieved memories.

        Uses hybrid recall to find relevant memories, then an LLM inference
        layer to synthesize a short answer with confidence score and citations.

        Returns: answer text, confidence (0.0-1.0), sources, fallback_recommended.
        """
        try:
            recaller = _get_recaller()
            memories = recaller.recall(query, limit=limit)
            gen = _get_answer_gen()
            answer = gen.generate(query, memories)
            return {
                "ok": answer.ok,
                "query": query,
                "answer": answer.text,
                "confidence": round(answer.confidence, 3),
                "sources": answer.sources,
                "fallback_recommended": gen.should_fallback(answer),
                "error": answer.error,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "query": query}

    # ---------- graph tools (optional) ----------

    if memgraph_uri:
        @mcp.tool()
        def mnemostack_graph_query(
            subject: str | None = None,
            predicate: str | None = None,
            obj: str | None = None,
            as_of: str | None = None,
            limit: int = 50,
        ) -> dict:
            """Query knowledge graph with optional SPO filters and point-in-time.

            All filter args are optional. If `as_of` is provided (ISO date),
            returns only facts valid at that date.
            """
            try:
                from ..graph import GraphStore

                gs = GraphStore(uri=memgraph_uri, timeout=graph_timeout)
                triples = gs.query_triples(
                    subject=subject, predicate=predicate, obj=obj,
                    as_of=as_of, limit=limit,
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
                from ..graph import GraphStore

                gs = GraphStore(uri=memgraph_uri, timeout=graph_timeout)
                gs.add_triple(
                    subject=subject, predicate=predicate, obj=obj,
                    valid_from=valid_from, valid_until=valid_until,
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
    """
    cfg = Config.load()
    mcp = build_server(
        collection=cfg.vector.collection,
        embedding_provider=cfg.embedding.provider,
        embedding_model=cfg.embedding.model,
        llm_provider=cfg.llm.provider,
        llm_model=cfg.llm.model,
        qdrant_host=cfg.vector.host,
        memgraph_uri=cfg.graph.uri,
        graph_timeout=cfg.graph.timeout,
        bm25_paths=list(cfg.recall.bm25_paths) or None,
    )
    mcp.run()


if __name__ == "__main__":
    main()
