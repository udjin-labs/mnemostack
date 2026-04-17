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
from .embeddings import get_provider, list_providers
from .llm import get_llm, list_llms
from .recall import AnswerGenerator, BM25Doc, Recaller
from .vector import VectorStore


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

    recaller = Recaller(embedding_provider=provider, vector_store=store)
    results = recaller.recall(args.query, limit=args.limit)

    if args.json:
        payload = [
            {
                "id": r.id,
                "score": round(r.score, 4),
                "text": r.text,
                "sources": r.sources,
                "payload": r.payload,
            }
            for r in results
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if not results:
            print("(no results)")
            return 0
        for i, r in enumerate(results, 1):
            preview = (r.text[:200] + "...") if len(r.text) > 200 else r.text
            sources = ",".join(r.sources) or "?"
            print(f"[{i}] score={r.score:.4f} ({sources})")
            print(f"    {preview}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
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

    recaller = Recaller(embedding_provider=provider, vector_store=store)
    results = recaller.recall(args.query, limit=args.limit)

    llm = get_llm(args.llm)
    gen = AnswerGenerator(llm=llm, confidence_threshold=args.min_confidence)
    answer = gen.generate(args.query, results)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "answer": answer.text,
                    "confidence": round(answer.confidence, 3),
                    "sources": answer.sources,
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

    print(f"ANSWER: {answer.text}")
    print(f"CONFIDENCE: {answer.confidence:.2f}")
    if answer.sources:
        print("SOURCES:")
        for s in answer.sources:
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


def cmd_index(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    store = VectorStore(
        collection=args.collection,
        dimension=provider.dimension,
        host=args.qdrant,
    )
    store.ensure_collection(recreate=args.recreate)

    target = Path(args.path)
    if not target.exists():
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 2

    files = (
        [target] if target.is_file()
        else sorted(target.rglob("*.md")) + sorted(target.rglob("*.txt"))
    )
    if not files:
        print(f"error: no .md/.txt files found under {target}", file=sys.stderr)
        return 2

    chunks: list[tuple[str, dict]] = []  # (text, payload) pairs
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i in range(0, len(text), args.chunk_size):
            chunk = text[i : i + args.chunk_size]
            if chunk.strip():
                chunks.append(
                    (
                        chunk,
                        {
                            "text": chunk,
                            "source": str(f.relative_to(target if target.is_dir() else target.parent)),
                            "offset": i,
                        },
                    )
                )

    print(f"Indexing {len(chunks)} chunks from {len(files)} file(s)...")
    point_id = store.count() + 1  # append after existing
    inserted = 0
    for text, payload in chunks:
        vec = provider.embed(text)
        if not vec:
            continue
        store.upsert(point_id, vec, payload)
        point_id += 1
        inserted += 1

    print(f"Done: inserted {inserted}/{len(chunks)} points into '{args.collection}'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mnemostack", description="Memory stack for AI agents")
    p.add_argument("--version", action="version", version=f"mnemostack {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider",
        default="ollama",
        choices=list_providers(),
        help="Embedding provider (default: ollama)",
    )
    common.add_argument(
        "--collection", default="mnemostack", help="Qdrant collection name"
    )
    common.add_argument(
        "--qdrant", default="http://localhost:6333", help="Qdrant URL"
    )

    p_health = sub.add_parser("health", parents=[common], help="Check stack health")
    p_health.set_defaults(func=cmd_health)

    p_search = sub.add_parser("search", parents=[common], help="Hybrid recall")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.set_defaults(func=cmd_search)

    p_answer = sub.add_parser(
        "answer", parents=[common], help="Synthesize concise answer from memories"
    )
    p_answer.add_argument("query", help="Question to answer")
    p_answer.add_argument("--limit", type=int, default=10, help="Max memories to consider")
    p_answer.add_argument(
        "--llm", default="gemini", choices=list_llms(), help="LLM provider for answer generation"
    )
    p_answer.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Fallback suggestion threshold",
    )
    p_answer.add_argument("--json", action="store_true", help="JSON output")
    p_answer.set_defaults(func=cmd_answer)

    p_index = sub.add_parser("index", parents=[common], help="Index files into vector store")
    p_index.add_argument("path", help="File or directory to index")
    p_index.add_argument("--chunk-size", type=int, default=800, help="Chunk size in chars")
    p_index.add_argument("--recreate", action="store_true", help="Drop existing collection")
    p_index.set_defaults(func=cmd_index)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
