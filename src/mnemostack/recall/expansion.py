"""Query expansion — LLM-based query rewriting + multi-query retrieval.

For list-style and ambiguous queries, a single retrieval often misses parts of
the answer that live under different wordings. Example:
    Q: "What books has Tim read?"
    Ground truth: "Harry Potter, Game of Thrones, Name of the Wind, The Alchemist"
A single retrieval may surface only one of these if the others live in chunks
phrased as "I read X yesterday" or "X is my favorite". Query expansion generates
2-3 paraphrases, retrieves for each, and RRF-fuses the results.

Design:
- Optional stage on top of Recaller; does not replace it.
- Uses the same LLM you use for AnswerGenerator; small prompt, low token cost.
- Result is still a `list[RecallResult]` — drop-in for downstream pipelines.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.base import LLMProvider
from ..observability import counter, histogram
from .fusion import reciprocal_rank_fusion
from .recaller import Recaller, RecallResult
from .trace import RecallTrace

_EXPANSION_PROMPT = """Given a user question, produce {n} short paraphrases that
are likely to match different wordings in a memory store. Keep them diverse.
Output ONE paraphrase per line, no numbering, no quotes, no commentary.

Question: {query}

Paraphrases:"""


def _fresh_trace(caller_trace: RecallTrace | None) -> RecallTrace | None:
    """A trace of the caller's own type, or None when it asked for none."""
    if caller_trace is None:
        return None
    return type(caller_trace)()


def _absorb_expansion(
    caller_trace: RecallTrace | None, pass_trace: RecallTrace | None, variant: str
) -> None:
    """Fold one paraphrase's work into the caller's trace.

    Its retriever entries are the extra retrieval this expansion paid for,
    which is exactly what an operator reading a trace wants to see, so they
    carry over labelled `:expansion` and stamped with the paraphrase that
    produced them. Degradations and notes carry over verbatim and are NOT
    re-`mark()`ed: the paraphrase's own trace already emitted the
    process-wide counter for them, and marking again would count one
    failure twice in `/status.degraded_events`.
    """
    if caller_trace is None or pass_trace is None:
        return
    for entry in list(pass_trace.retrievers):
        entry.name = f"{entry.name}:expansion"
        if getattr(entry, "query", None) is None:
            entry.query = variant
        caller_trace.retrievers.append(entry)
    for tag in pass_trace.degraded:
        if tag not in caller_trace.degraded:
            caller_trace.degraded.append(tag)
    for tag in pass_trace.notes:
        if tag not in caller_trace.notes:
            caller_trace.notes.append(tag)


class QueryExpander:
    """LLM-based query paraphraser with multi-query fused retrieval.

    Args:
        recaller: the base Recaller used for each variant
        llm: LLM used to generate paraphrases
        n_variants: how many paraphrases to generate (2-3 is sensible)
        include_original: whether to always include the original query
        rrf_k: RRF dampening constant for cross-query fusion
        max_tokens: cap on paraphrase LLM output
        apply_to: predicate over query; expand only when predicate returns True.
                  Default: expand for list-like questions (what kinds, who are,
                  what are, which ones, where, what items, what people).
    """

    LIST_PATTERN = re.compile(
        r"^\s*(what (are|is|kind|kinds|type|types|items|things|people|books|places|ways)|who are|which|where has|where does|where did)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        recaller: Recaller,
        llm: LLMProvider,
        n_variants: int = 3,
        include_original: bool = True,
        rrf_k: int = 60,
        max_tokens: int = 120,
        apply_to=None,
    ):
        self.recaller = recaller
        self.llm = llm
        self.n_variants = n_variants
        self.include_original = include_original
        self.rrf_k = rrf_k
        self.max_tokens = max_tokens
        self.apply_to = apply_to or (lambda q: bool(self.LIST_PATTERN.match(q)))

    def generate_variants(self, query: str) -> list[str]:
        """Return paraphrases (excluding the original)."""
        prompt = _EXPANSION_PROMPT.format(n=self.n_variants, query=query)
        with histogram("mnemostack.query_expansion.llm_latency_ms"):
            resp = self.llm.generate(prompt, max_tokens=self.max_tokens)
        if not resp.ok:
            counter("mnemostack.query_expansion.errors", 1)
            return []
        lines = [ln.strip("-• \t\"'") for ln in resp.text.splitlines() if ln.strip()]
        # dedupe vs original, cap at n_variants
        seen = {query.lower().strip()}
        variants = []
        for ln in lines:
            k = ln.lower().strip()
            if k and k not in seen:
                seen.add(k)
                variants.append(ln)
            if len(variants) >= self.n_variants:
                break
        counter("mnemostack.query_expansion.variants_generated", len(variants))
        return variants

    def recall(
        self,
        query: str,
        limit: int = 10,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        filters: dict[str, Any] | None = None,
        *,
        trace: RecallTrace | None = None,
        include_invalidated: bool = False,
        as_of: str | None = None,
        tenant: str | None = None,
    ) -> list[RecallResult]:
        """Fused recall across original query + paraphrases.

        If `apply_to(query)` is False, falls back to plain Recaller.recall.

        The scoping keywords are forwarded to EVERY recall this makes, the
        original and each paraphrase alike. They were absent once, and the
        failure mode was silent: this class is a public export, so a
        multi-tenant caller reaching for it got an unscoped read — no
        tenant on the retrievers, no tenant backstop after fusion,
        retracted facts visible — with no error and nothing on a trace to
        show it (udjin-labs/mnemostack#166). A paraphrase is the same
        question asked differently; it must not be a wider one.

        `trace` records the whole expansion: the original query's recall
        writes into the caller's trace directly, and each paraphrase gets
        its own, folded back with its retriever entries labelled
        `:expansion` and carrying the paraphrase in `query`. Sharing one
        trace across passes would leave `fused` describing whichever pass
        ran last while the caller holds a merge of all of them.
        """
        counter("mnemostack.query_expansion.calls", 1)
        if not self.apply_to(query):
            counter("mnemostack.query_expansion.skipped", 1)
            return self.recaller.recall(
                query,
                limit=limit,
                vector_limit=vector_limit,
                bm25_limit=bm25_limit,
                filters=filters,
                trace=trace,
                include_invalidated=include_invalidated,
                as_of=as_of,
                tenant=tenant,
            )

        queries = [query] if self.include_original else []
        queries += self.generate_variants(query)
        if len(queries) <= 1:
            return self.recaller.recall(
                query,
                limit=limit,
                vector_limit=vector_limit,
                bm25_limit=bm25_limit,
                filters=filters,
                trace=trace,
                include_invalidated=include_invalidated,
                as_of=as_of,
                tenant=tenant,
            )

        # Recall for each query, merge via RRF across queries
        ranked_lists: list[list[tuple[RecallResult, float]]] = []
        for q in queries:
            pass_trace = trace if q == query else _fresh_trace(trace)
            res = self.recaller.recall(
                q,
                limit=limit * 2,
                vector_limit=vector_limit,
                bm25_limit=bm25_limit,
                filters=filters,
                trace=pass_trace,
                include_invalidated=include_invalidated,
                as_of=as_of,
                tenant=tenant,
            )
            if pass_trace is not None and pass_trace is not trace:
                _absorb_expansion(trace, pass_trace, q)
            ranked_lists.append([(r, r.score) for r in res])

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, limit=limit)
        # Rebuild RecallResult with fused score
        out: list[RecallResult] = []
        for item, rrf_score in fused:
            item.score = rrf_score
            out.append(item)
        if trace is not None:
            # `fused` is documented as the order recall RETURNED, and after
            # a cross-query merge no single pass produced it — the caller's
            # trace holds whatever the original query's pass wrote, or
            # nothing at all when `include_original=False`. `post_rerank`
            # is deliberately left alone: it means "the reranker's order
            # when a reranker ran", a claim about one component on the
            # candidates it was given, and no reranker was shown this
            # merge.
            trace.fused = [(str(r.id), float(r.score)) for r in out]
        counter("mnemostack.query_expansion.used_expansion", 1)
        return out
