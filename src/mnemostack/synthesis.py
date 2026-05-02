"""Entity-centric knowledge synthesis.

The public ``synthesize`` entry point gathers evidence about one entity from
whatever recall backends are available, deduplicates near-identical chunks, and
returns a structured result that can be rendered as Markdown or JSON.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from .ingest import stable_chunk_id
from .llm.base import LLMProvider
from .recall.recaller import Recaller, RecallResult
from .recall.retrievers import (
    BM25Retriever,
    MemgraphRetriever,
    Retriever,
    TemporalRetriever,
    VectorRetriever,
)

_TIMESTAMP_KEYS = ("timestamp", "created_at", "date", "time")
_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "but", "for",
    "from", "has", "have", "into", "its", "that", "the", "their", "them",
    "then", "there", "this", "was", "were", "with", "you", "your", "это",
    "как", "для", "или", "что", "она", "они", "его", "про", "на",
}


@dataclass
class SynthesisFact:
    """A single synthesized fact/evidence item."""

    text: str
    source: str
    timestamp: str | None = None
    relevance_score: float = 0.0
    subtopic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp,
            "relevance_score": self.relevance_score,
        }
        if self.subtopic:
            payload["subtopic"] = self.subtopic
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class SynthesisResult:
    """Structured result returned by :func:`synthesize`."""

    entity: str
    facts: list[SynthesisFact]
    related_entities: list[str] = field(default_factory=list)
    timeline: list[SynthesisFact] = field(default_factory=list)
    summary: str | None = None

    def markdown(self) -> str:
        lines = [f"# {self.entity}", ""]
        if self.summary:
            lines.extend(["## Summary", self.summary.strip(), ""])

        lines.append("## Facts")
        if not self.facts:
            lines.append("(no facts found)")
        else:
            grouped: dict[str, list[SynthesisFact]] = defaultdict(list)
            has_subtopics = any(f.subtopic for f in self.facts)
            for fact in self.facts:
                grouped[fact.subtopic or "General"].append(fact)
            for topic, facts in grouped.items():
                if has_subtopics:
                    lines.extend(["", f"### {topic}"])
                for fact in facts:
                    meta = f"source={fact.source}, score={fact.relevance_score:.3f}"
                    if fact.timestamp:
                        meta += f", time={fact.timestamp}"
                    lines.append(f"- {fact.text} ({meta})")
        lines.append("")

        if self.related_entities:
            lines.append("## Related entities")
            for ent in self.related_entities:
                lines.append(f"- {ent}")
            lines.append("")

        if self.timeline:
            lines.append("## Timeline")
            for fact in self.timeline:
                lines.append(f"- {fact.timestamp}: {fact.text} ({fact.source})")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "facts": [f.to_json() for f in self.facts],
            "related_entities": list(self.related_entities),
            "timeline": [f.to_json() for f in self.timeline],
            "summary": self.summary,
        }


def synthesize(
    entity: str,
    sources: list[str] | None = None,
    format: str = "markdown",
    max_results: int = 50,
    llm_summarize: bool = False,
    **kwargs: Any,
) -> SynthesisResult:
    """Collect known information about ``entity`` into a structured summary.

    Backends are intentionally optional. Callers may pass ``recaller=...`` or
    ``retrievers=[...]`` directly, or pass components used to construct default
    retrievers (``embedding_provider``/``vector_store``, ``bm25_docs``,
    ``memgraph_uri``). Missing or failing backends are skipped.
    """
    entity = entity.strip()
    if not entity:
        raise ValueError("entity must not be empty")
    if format not in {"markdown", "json"}:
        raise ValueError("format must be 'markdown' or 'json'")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    source_filter = _normalize_sources(sources)
    recaller = kwargs.get("recaller") or _build_recaller_from_kwargs(source_filter, kwargs)
    raw_results = _query_recaller(recaller, entity, max_results, kwargs.get("filters"))
    raw_results.extend(_query_retrievers(kwargs.get("retrievers"), entity, max_results, source_filter, kwargs.get("filters")))
    raw_results = [r for r in raw_results if _result_source_enabled(r, source_filter)]

    facts = _dedupe_facts(_facts_from_results(raw_results), max_results=max_results)
    if len(facts) >= int(kwargs.get("cluster_min_results", 8)):
        _assign_subtopics(facts, entity)
    timeline = _timeline(facts)
    related_entities = _related_entities(entity, raw_results, kwargs)
    summary = _summarize(entity, facts, related_entities, kwargs) if llm_summarize else None

    return SynthesisResult(
        entity=entity,
        facts=facts,
        related_entities=related_entities,
        timeline=timeline,
        summary=summary,
    )


def _normalize_sources(sources: list[str] | None) -> set[str] | None:
    if sources is None:
        return None
    aliases = {"graph": "memgraph"}
    return {aliases.get(s.lower(), s.lower()) for s in sources}


def _build_recaller_from_kwargs(source_filter: set[str] | None, kwargs: dict[str, Any]) -> Recaller | None:
    retrievers: list[Retriever] = []
    embedding = kwargs.get("embedding_provider") or kwargs.get("embedding")
    vector_store = kwargs.get("vector_store")
    if _source_enabled("vector", source_filter) and embedding is not None and vector_store is not None:
        retrievers.append(VectorRetriever(embedding=embedding, vector_store=vector_store))
    if _source_enabled("bm25", source_filter) and kwargs.get("bm25_docs"):
        retrievers.append(BM25Retriever(docs=list(kwargs["bm25_docs"])))
    memgraph_uri = kwargs.get("memgraph_uri")
    if _source_enabled("memgraph", source_filter) and (memgraph_uri or kwargs.get("graph_driver")):
        retrievers.append(
            MemgraphRetriever(
                uri=memgraph_uri or "bolt://localhost:7687",
                user=kwargs.get("memgraph_user", ""),
                password=kwargs.get("memgraph_password", ""),
                driver=kwargs.get("graph_driver"),
                timeout=float(kwargs.get("graph_timeout", 5.0)),
            )
        )
    if _source_enabled("temporal", source_filter) and embedding is not None and vector_store is not None:
        retrievers.append(TemporalRetriever(embedding=embedding, vector_store=vector_store))
    if not retrievers:
        return None
    return Recaller(retrievers=retrievers)


def _source_enabled(name: str, source_filter: set[str] | None) -> bool:
    return source_filter is None or name in source_filter


def _result_source_enabled(result: RecallResult, source_filter: set[str] | None) -> bool:
    if source_filter is None:
        return True
    sources = {str(s).lower() for s in (getattr(result, "sources", []) or [])}
    if "graph" in source_filter:
        source_filter = {*source_filter, "memgraph"}
    return bool(sources & source_filter)


def _query_recaller(
    recaller: Any,
    entity: str,
    max_results: int,
    filters: dict[str, Any] | None,
) -> list[RecallResult]:
    if recaller is None:
        return []
    try:
        return list(
            recaller.recall(
                entity,
                limit=max_results,
                vector_limit=max_results,
                bm25_limit=max_results,
                filters=filters,
            )
        )
    except TypeError:
        try:
            return list(recaller.recall(entity, limit=max_results, filters=filters))
        except TypeError:
            try:
                return list(recaller.recall(entity, limit=max_results))
            except Exception:
                return []
        except Exception:
            return []
    except Exception:
        return []


def _query_retrievers(
    retrievers: list[Any] | None,
    entity: str,
    max_results: int,
    source_filter: set[str] | None,
    filters: dict[str, Any] | None,
) -> list[RecallResult]:
    results: list[RecallResult] = []
    for retr in retrievers or []:
        name = str(getattr(retr, "name", "")).lower()
        if source_filter is not None and name not in source_filter:
            continue
        try:
            results.extend(retr.search(entity, limit=max_results, filters=filters))
        except TypeError:
            try:
                results.extend(retr.search(entity, limit=max_results))
            except Exception:
                continue
        except Exception:
            continue
    return results


def _facts_from_results(results: list[RecallResult]) -> list[SynthesisFact]:
    facts: list[SynthesisFact] = []
    for result in results:
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            continue
        sources = list(getattr(result, "sources", []) or [])
        payload = dict(getattr(result, "payload", {}) or {})
        source = ",".join(sources) or str(payload.get("source") or "unknown")
        facts.append(
            SynthesisFact(
                text=text,
                source=source,
                timestamp=_extract_timestamp(payload),
                relevance_score=float(getattr(result, "score", 0.0) or 0.0),
                metadata={"id": getattr(result, "id", None), **_safe_metadata(payload)},
            )
        )
    facts.sort(key=lambda f: f.relevance_score, reverse=True)
    return facts


def _extract_timestamp(payload: dict[str, Any]) -> str | None:
    for key in _TIMESTAMP_KEYS:
        value = payload.get(key)
        if value:
            return str(value)
    return None


def _safe_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in payload.items()
        if k not in {"text"} and isinstance(v, (str, int, float, bool, type(None)))
    }


def _dedupe_facts(facts: list[SynthesisFact], max_results: int) -> list[SynthesisFact]:
    kept: list[SynthesisFact] = []
    seen_hashes: set[str] = set()
    for fact in facts:
        normalized = _normalize_text(fact.text)
        if not normalized:
            continue
        content_id = stable_chunk_id("synthesis", 0, normalized)
        if content_id in seen_hashes:
            continue
        if any(_similar(normalized, _normalize_text(existing.text)) for existing in kept):
            continue
        seen_hashes.add(content_id)
        kept.append(fact)
        if len(kept) >= max_results:
            break
    return kept


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _similar(a: str, b: str, threshold: float = 0.92) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b)) > 0.75
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _assign_subtopics(facts: list[SynthesisFact], entity: str) -> None:
    entity_tokens = set(_tokens(entity))
    for fact in facts:
        tokens = [t for t in _tokens(fact.text) if t not in entity_tokens and t not in _STOPWORDS]
        if not tokens:
            fact.subtopic = "General"
            continue
        counts = Counter(tokens)
        fact.subtopic = counts.most_common(1)[0][0].capitalize()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w@-]+", text.lower(), flags=re.UNICODE)


def _timeline(facts: list[SynthesisFact]) -> list[SynthesisFact]:
    stamped = [f for f in facts if f.timestamp]
    return sorted(stamped, key=lambda f: _timestamp_sort_key(f.timestamp))


def _timestamp_sort_key(timestamp: str | None) -> tuple[int, str]:
    if not timestamp:
        return (1, "")
    raw = timestamp.strip()
    try:
        return (0, datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (0, raw)


def _related_entities(entity: str, results: list[RecallResult], kwargs: dict[str, Any]) -> list[str]:
    provider = kwargs.get("related_entity_provider")
    if provider is not None:
        try:
            return _unique_strings(provider(entity), exclude={entity})
        except Exception:
            return []
    related: list[str] = []
    for result in results:
        sources = {s.lower() for s in (getattr(result, "sources", []) or [])}
        payload = dict(getattr(result, "payload", {}) or {})
        if "memgraph" not in sources and payload.get("source") != "memgraph":
            continue
        text = getattr(result, "text", "") or ""
        related.extend(re.findall(r"->\s*([^;\n]+)", text))
        name = payload.get("name")
        if name:
            related.append(str(name))
    return _unique_strings(related, exclude={entity})


def _unique_strings(values: Any, exclude: set[str]) -> list[str]:
    excluded = {v.lower() for v in exclude}
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text.lower() in excluded or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    return out


def _summarize(
    entity: str,
    facts: list[SynthesisFact],
    related_entities: list[str],
    kwargs: dict[str, Any],
) -> str | None:
    llm: LLMProvider | None = kwargs.get("llm")
    if llm is None:
        return None
    evidence = "\n".join(f"- {f.text}" for f in facts[:20])
    related = ", ".join(related_entities[:20]) or "none"
    prompt = (
        "Summarize known information about the entity below using only the provided evidence. "
        "Be concise and factual.\n\n"
        f"Entity: {entity}\nRelated entities: {related}\nEvidence:\n{evidence}\n\nSummary:"
    )
    try:
        response = llm.generate(
            prompt,
            max_tokens=int(kwargs.get("summary_max_tokens", 300)),
            temperature=float(kwargs.get("summary_temperature", 0.0)),
        )
    except Exception:
        return None
    if getattr(response, "error", None):
        return None
    text = (getattr(response, "text", "") or "").strip()
    return text or None


def dumps(result: SynthesisResult, format: str = "markdown") -> str:
    """Render a synthesis result for callers that want a string."""
    if format == "markdown":
        return result.markdown()
    if format == "json":
        return json.dumps(result.to_json(), ensure_ascii=False, indent=2)
    raise ValueError("format must be 'markdown' or 'json'")


__all__ = ["SynthesisFact", "SynthesisResult", "synthesize", "dumps"]
