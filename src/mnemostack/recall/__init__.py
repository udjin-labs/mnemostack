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

from .answer import Answer, AnswerGenerator, classify_question
from .bm25 import BM25, BM25Doc, tokenize
from .corpus import build_bm25_docs
from .expansion import QueryExpander
from .fusion import reciprocal_rank_fusion
from .pipeline import (
    ClassifyQuery,
    CuriosityBoost,
    ExactTokenRescue,
    FileStateStore,
    FreshnessBlend,
    GravityDampen,
    HubDampen,
    InhibitionOfReturn,
    InMemoryStateStore,
    Pipeline,
    PipelineContext,
    QLearningReranker,
    Stage,
    StateStore,
    build_full_pipeline,
    build_stateless_pipeline,
)
from .recaller import Recaller, RecallResult
from .render import compact_format, full_format
from .reranker import Reranker
from .retrievers import (
    BM25Retriever,
    HyDERetriever,
    MemgraphRetriever,
    Retriever,
    TemporalRetriever,
    VectorRetriever,
    bm25_docs_from_qdrant,
    extract_temporal,
)
from .specificity import detect_placeholders, resolve_specificity

__all__ = [
    "BM25",
    "BM25Doc",
    "build_bm25_docs",
    "tokenize",
    "reciprocal_rank_fusion",
    "Recaller",
    "RecallResult",
    "compact_format",
    "full_format",
    "QueryExpander",
    "Retriever",
    "VectorRetriever",
    "BM25Retriever",
    "bm25_docs_from_qdrant",
    "HyDERetriever",
    "MemgraphRetriever",
    "TemporalRetriever",

    "extract_temporal",
    "Answer",
    "AnswerGenerator",
    "classify_question",
    "detect_placeholders",
    "resolve_specificity",
    "Reranker",
    "Pipeline",
    "PipelineContext",
    "Stage",
    "StateStore",
    "InMemoryStateStore",
    "FileStateStore",
    "ClassifyQuery",
    "ExactTokenRescue",
    "GravityDampen",
    "HubDampen",
    "FreshnessBlend",
    "InhibitionOfReturn",
    "CuriosityBoost",
    "QLearningReranker",
    "build_full_pipeline",
    "build_stateless_pipeline",
]
