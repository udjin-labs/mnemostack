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
from .bm25 import BM25, BM25Doc, Tokenizer, tokenize
from .corpus import build_bm25_docs
from .expansion import QueryExpander
from .filters import payload_matches
from .flow import recall_flow, recall_flow_async
from .followup import rewrite_followup
from .fusion import reciprocal_rank_fusion
from .mca_prefilter import extract_exact_tokens, mca_prefilter
from .pipeline import (
    ClassifyQuery,
    CuriosityBoost,
    ExactTokenProtection,
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
from .query_expansion import expand_query
from .recaller import Recaller, RecallResult
from .render import compact_format, full_format
from .reranker import RERANK_MODES, Reranker
from .retrievers import (
    BM25Retriever,
    HyDERetriever,
    MemgraphRetriever,
    QdrantSparseRetriever,
    QdrantTextRetriever,
    Retriever,
    TemporalRetriever,
    VectorRetriever,
    bm25_docs_from_qdrant,
    extract_temporal,
)
from .scoring_reranker import RelevanceScorer, ScoringReranker
from .specificity import detect_placeholders, resolve_specificity
from .tokens import TokenCounter, apply_token_budget, estimate_tokens, sum_tokens
from .trace import DEGRADED_COUNTER, RecallTrace, RetrieverTrace, apply_rerank_safe
from .validity import filter_by_tenant, filter_by_validity, is_current, valid_at

__all__ = [
    "BM25",
    "BM25Doc",
    "Tokenizer",
    "build_bm25_docs",
    "tokenize",
    "reciprocal_rank_fusion",
    "RecallTrace",
    "RetrieverTrace",
    "apply_rerank_safe",
    "DEGRADED_COUNTER",
    "payload_matches",
    "recall_flow",
    "recall_flow_async",
    "filter_by_validity",
    "filter_by_tenant",
    "is_current",
    "valid_at",
    "TokenCounter",
    "estimate_tokens",
    "apply_token_budget",
    "sum_tokens",
    "rewrite_followup",
    "extract_exact_tokens",
    "mca_prefilter",
    "Recaller",
    "RecallResult",
    "compact_format",
    "full_format",
    "QueryExpander",
    "expand_query",
    "Retriever",
    "VectorRetriever",
    "BM25Retriever",
    "QdrantSparseRetriever",
    "QdrantTextRetriever",
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
    "RelevanceScorer",
    "ScoringReranker",
    "RERANK_MODES",
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
    "ExactTokenProtection",
    "build_full_pipeline",
    "build_stateless_pipeline",
]
