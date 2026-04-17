"""
Recall pipeline — hybrid retrieval + fusion + reranking + answer generation.

Components:
- BM25 (exact token matching)
- VectorStore (semantic search via Qdrant)
- RRF fusion (merges ranked lists)
- Reranker (optional)
- Answer generator (inference layer)

Usage:
    from mnemostack.recall import BM25, reciprocal_rank_fusion
    bm25 = BM25(documents)
    hits = bm25.search("query", limit=10)
"""

from .answer import Answer, AnswerGenerator
from .bm25 import BM25, BM25Doc, tokenize
from .fusion import reciprocal_rank_fusion
from .recaller import Recaller, RecallResult
from .reranker import Reranker

__all__ = [
    "BM25",
    "BM25Doc",
    "tokenize",
    "reciprocal_rank_fusion",
    "Recaller",
    "RecallResult",
    "Answer",
    "AnswerGenerator",
    "Reranker",
]
