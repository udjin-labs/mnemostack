"""Small corpus helpers shared by HTTP, CLI, and MCP recall paths."""
from __future__ import annotations

from pathlib import Path

from .bm25 import BM25Doc


def build_bm25_docs(paths: list[str] | None, chunk_size: int = 800) -> list[BM25Doc]:
    """Build a simple BM25 corpus from markdown/text files."""
    if not paths:
        return []

    docs: list[BM25Doc] = []
    for root in paths:
        p = Path(root)
        if not p.exists():
            continue
        targets = [p] if p.is_file() else sorted(p.rglob("*.md")) + sorted(p.rglob("*.txt"))
        for f in targets:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    docs.append(
                        BM25Doc(
                            id=f"{f}:{i}",
                            text=chunk,
                            payload={"source": str(f), "offset": i},
                        )
                    )
    return docs
