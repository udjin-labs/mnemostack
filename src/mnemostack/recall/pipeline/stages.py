"""Built-in pipeline stages for hybrid recall.

Stages are ported from the reference implementation in enhanced-recall.py.
Each stage is a small, independently-testable transformation.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from ..recaller import RecallResult
from .base import PipelineContext, Stage
from .state import StateStore


# ---------- defaults / stopwords ----------

STOPWORDS_RU = {
    "что", "как", "где", "когда", "кто", "зачем", "почему", "это",
    "мой", "мне", "на", "в", "с", "и", "или", "для", "по", "за",
    "от", "до", "не", "но", "а", "то", "ещё", "уже", "было", "был",
    "есть", "будет", "все", "так", "его", "её", "их", "мы", "они",
}
STOPWORDS_EN = {
    "what", "is", "my", "the", "a", "an", "do", "i", "know", "about",
    "have", "did", "on", "in", "for", "to", "of", "and", "or", "how",
    "why", "where", "when", "who", "should", "today", "done", "been",
}
STOPWORDS = STOPWORDS_RU | STOPWORDS_EN


# ---------- query classification ----------


class ClassifyQuery(Stage):
    """Categorize the query (person/project/decision/event/technical/general).

    Populates `context.query_type` for downstream stages (Q-learning).
    Does not modify results. The categories are intentionally coarse — they are
    used as reward buckets in Q-learning, not strict routing.
    """

    DEFAULT_MARKERS = {
        "person": {"кто", "who", "человек", "person"},
        "project": {"проект", "project"},
        "decision": {"решили", "decided", "почему", "why", "выбрали", "chose", "решение"},
        "event": {"когда", "when", "произошло", "happened"},
        "technical": {"как", "how", "настроить", "configure", "ошибка", "error", "fix", "баг"},
    }

    def __init__(self, markers: dict[str, set[str]] | None = None):
        self.markers = markers or self.DEFAULT_MARKERS

    def apply(self, context, results):
        q_lower = context.query.lower()
        for qtype, ms in self.markers.items():
            if any(m in q_lower for m in ms):
                context.query_type = qtype
                break
        else:
            context.query_type = "general"
        # Cache tokens too
        context.query_tokens = [
            t for t in re.findall(r"\w+", q_lower) if len(t) > 2 and t not in STOPWORDS
        ]
        return results


# ---------- exact-token detection / source-of-truth rescue ----------

_EXACT_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),          # IP address
    re.compile(r"\b\d{4,5}\b"),                            # port / numeric code
    re.compile(r"\b\d{4}\.\d+\.\d+\b"),                    # version like 2026.4.12
    re.compile(r"\b[a-z]+[-_]?\d+[a-z0-9-]*\b"),            # IDs/codes
]
_EXACT_MARKERS = {"ip", "порт", "port", "версия", "version", "id", "uuid", "токен", "api"}


def is_exact_token_query(query: str) -> bool:
    q = query.lower()
    if any(p.search(q) for p in _EXACT_PATTERNS):
        return True
    return any(m in q for m in _EXACT_MARKERS)


class ExactTokenRescue(Stage):
    """Boost results that contain exact-token answers (IP, port, version, etc.).

    Useful for infra queries where semantic similarity alone can miss the
    right answer buried in a terse MEMORY.md line.

    Args:
        target_sources: payload `source` prefixes that should be rescued
                        (e.g. ['MEMORY.md', 'TOOLS.md'])
        boost: score boost applied to matching items
    """

    def __init__(
        self,
        target_sources: list[str] | None = None,
        boost: float = 0.5,
    ):
        self.target_sources = target_sources or ["MEMORY.md", "TOOLS.md"]
        self.boost = boost

    def apply(self, context, results):
        if not results or not is_exact_token_query(context.query):
            return results
        answer_re = re.compile(r"\b\d{4,5}\b|\b\d{1,3}(?:\.\d{1,3}){3}\b|\b\d{4}\.\d+")
        for r in results:
            source = str(r.payload.get("source", ""))
            if not any(name in source for name in self.target_sources):
                continue
            if answer_re.search(r.text):
                r.score += self.boost
                r.payload.setdefault("flags", []).append("exact_rescue")
        results.sort(key=lambda x: -x.score)
        return results


# ---------- dampening ----------


class GravityDampen(Stage):
    """Penalize results that don't actually contain any query key term.

    Counters the "hub memory" problem: a popular memory dominates embedding
    similarity without really containing the information we asked for.
    """

    def __init__(self, penalty: float = 0.5, min_score: float = 0.1):
        self.penalty = penalty
        self.min_score = min_score

    def apply(self, context, results):
        key_terms = set(context.query_tokens) or (
            set(context.query.lower().split()) - STOPWORDS
        )
        if not key_terms:
            return results
        for r in results:
            if r.score <= self.min_score:
                continue
            text_lower = r.text.lower()
            if not any(t in text_lower for t in key_terms):
                r.score *= self.penalty
                r.payload["dampened"] = "gravity"
        results.sort(key=lambda x: -x.score)
        return results


class HubDampen(Stage):
    """Penalize results whose graph node is in the top 10% by degree.

    Requires `hub_degrees` (dict of id → node degree). If empty, stage is a no-op.
    """

    def __init__(self, hub_degrees: dict[Any, int] | None = None, p90_floor: float = 0.4):
        self.hub_degrees = hub_degrees or {}
        self.p90_floor = p90_floor

    def apply(self, context, results):
        if not self.hub_degrees:
            return results
        degrees = sorted(self.hub_degrees.values())
        if not degrees:
            return results
        p90 = degrees[int(len(degrees) * 0.9)] if len(degrees) > 1 else degrees[-1]
        max_deg = degrees[-1]
        if max_deg <= p90:
            return results
        for r in results:
            deg = self.hub_degrees.get(r.id, 0)
            if deg > p90:
                ratio = (deg - p90) / (max_deg - p90)
                penalty = 1.0 - (1.0 - self.p90_floor) * ratio
                r.score *= max(self.p90_floor, penalty)
                r.payload["dampened"] = "hub"
        results.sort(key=lambda x: -x.score)
        return results


# ---------- freshness ----------


class FreshnessBlend(Stage):
    """Blend similarity with recency decay.

    Score becomes `(1-weight)*old_score + weight*freshness`, where freshness
    is `exp(-ln(2) * age_days / halflife_days)`. Items from the last
    `echo_window_minutes` are penalized (likely meta-noise from current
    conversation).
    """

    def __init__(
        self,
        weight: float = 0.2,
        halflife_days: int = 14,
        echo_window_minutes: int = 10,
        echo_penalty: float = 0.5,
    ):
        self.weight = weight
        self.halflife_days = halflife_days
        self.echo_window_minutes = echo_window_minutes
        self.echo_penalty = echo_penalty

    def apply(self, context, results):
        now = datetime.now(timezone.utc)
        for r in results:
            ts_dt = self._parse_timestamp(r.payload)
            if ts_dt is None:
                ts_dt = self._date_from_source(r.payload.get("source", ""))
            freshness = 0.5
            age_minutes = None
            if ts_dt is not None:
                age_days = max(0.0, (now - ts_dt).total_seconds() / 86400)
                freshness = math.exp(-math.log(2) * age_days / self.halflife_days)
                age_minutes = (now - ts_dt).total_seconds() / 60
            if age_minutes is not None and age_minutes < self.echo_window_minutes:
                freshness *= self.echo_penalty
                r.payload["echo_penalty"] = True
            r.score = (1 - self.weight) * r.score + self.weight * freshness
            r.payload["freshness"] = round(freshness, 3)
        results.sort(key=lambda x: -x.score)
        return results

    @staticmethod
    def _parse_timestamp(payload: dict[str, Any]) -> datetime | None:
        ts = payload.get("timestamp")
        if not ts:
            return None
        try:
            if isinstance(ts, str):
                ts = ts.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts)
            else:
                dt = ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _date_from_source(source: str) -> datetime | None:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", source)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


# ---------- stateful stages: IOR, curiosity ----------


class InhibitionOfReturn(Stage):
    """Penalize memories recalled in the recent past (default 24h).

    State key: 'ior_log' = [{id, timestamp}]
    Usage:
        stage = InhibitionOfReturn(state_store=FileStateStore('state.json'))
        # After emitting results, call stage.record_recall(result.id)
    """

    STATE_KEY = "ior_log"

    def __init__(
        self,
        state_store: StateStore,
        penalty_per_recall: float = 0.03,
        max_penalty: float = 0.15,
        window_hours: int = 24,
        keep_last: int = 500,
    ):
        self.store = state_store
        self.penalty_per_recall = penalty_per_recall
        self.max_penalty = max_penalty
        self.window_hours = window_hours
        self.keep_last = keep_last

    def apply(self, context, results):
        log = self.store.get(self.STATE_KEY) or []
        now = datetime.now(timezone.utc)
        window = timedelta(hours=self.window_hours)
        counts: dict[str, int] = defaultdict(int)
        for entry in log:
            try:
                ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                if now - ts < window:
                    counts[str(entry.get("id", ""))] += 1
            except (KeyError, ValueError, TypeError):
                continue
        for r in results:
            n = counts.get(str(r.id), 0)
            if n > 0:
                penalty = min(self.max_penalty, self.penalty_per_recall * n)
                r.score *= (1 - penalty)
                r.payload["ior_penalty"] = round(penalty, 3)
        results.sort(key=lambda x: -x.score)
        return results

    def record_recall(self, memory_id: Any) -> None:
        """Append a recall event for memory_id (call after emitting results)."""
        ts = datetime.now(timezone.utc).isoformat()
        def _update(current):
            log = list(current or [])
            log.append({"id": memory_id, "timestamp": ts})
            return log[-self.keep_last :]
        self.store.update(self.STATE_KEY, _update)


class CuriosityBoost(Stage):
    """Boost old, rarely-recalled memories to encourage exploration.

    Uses IOR log as source of truth for recall counts (both stages typically
    share the same state_store).
    """

    IOR_KEY = "ior_log"

    def __init__(
        self,
        state_store: StateStore,
        bonus: float = 0.05,
        min_age_days: int = 7,
        max_recalls: int = 2,
    ):
        self.store = state_store
        self.bonus = bonus
        self.min_age_days = min_age_days
        self.max_recalls = max_recalls

    def apply(self, context, results):
        log = self.store.get(self.IOR_KEY) or []
        counts: dict[str, int] = defaultdict(int)
        for entry in log:
            counts[str(entry.get("id", ""))] += 1
        now = datetime.now(timezone.utc)
        for r in results:
            if counts.get(str(r.id), 0) > self.max_recalls:
                continue
            created = r.payload.get("timestamp") or r.payload.get("created")
            if created:
                try:
                    ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_days = (now - ts).days
                    if age_days >= self.min_age_days:
                        r.score += self.bonus
                        r.payload["curiosity_boosted"] = True
                except (ValueError, TypeError):
                    pass
            else:
                # No date info — half bonus
                r.score += self.bonus * 0.5
        results.sort(key=lambda x: -x.score)
        return results


# ---------- Q-learning reranker ----------


class QLearningReranker(Stage):
    """Rerank results by source Q-value given query type.

    Each result must have `payload['retrieval_source']` (e.g. 'vector', 'bm25',
    'graph'). Stage boosts scores from sources that historically performed
    well for the query type. Updates happen via `record_feedback(memory_id,
    query_type, source, reward)`.
    """

    STATE_KEY = "q_table"

    def __init__(
        self,
        state_store: StateStore,
        alpha: float = 0.15,
        default_q: float = 0.5,
        decay_rate: float = 0.995,
        q_min: float = 0.05,
        q_max: float = 0.95,
        boost_weight: float = 0.3,
    ):
        self.store = state_store
        self.alpha = alpha
        self.default_q = default_q
        self.decay_rate = decay_rate
        self.q_min = q_min
        self.q_max = q_max
        self.boost_weight = boost_weight

    def apply(self, context, results):
        q_table = self.store.get(self.STATE_KEY) or {}
        for r in results:
            source = r.payload.get("retrieval_source") or (
                r.sources[0] if r.sources else "unknown"
            )
            entry = q_table.get(source, {}).get(context.query_type, {})
            q_val = entry.get("q", self.default_q)
            # Additive boost: results from high-Q sources climb up
            r.score += self.boost_weight * (q_val - self.default_q)
            r.payload["q_value"] = round(q_val, 3)
        results.sort(key=lambda x: -x.score)
        return results

    def record_feedback(
        self,
        source: str,
        query_type: str,
        reward: float,
    ) -> None:
        """Update Q-table based on observed reward (1.0 = user used this result)."""
        def _update(current):
            table = dict(current or {})
            src_table = dict(table.get(source, {}))
            entry = dict(src_table.get(query_type, {"q": self.default_q, "n": 0}))
            # Decay toward default first (prevents drift)
            entry["q"] = entry["q"] * self.decay_rate + self.default_q * (1 - self.decay_rate)
            # Standard Q-learning update
            entry["q"] += self.alpha * (reward - entry["q"])
            entry["q"] = max(self.q_min, min(self.q_max, entry["q"]))
            entry["n"] = entry.get("n", 0) + 1
            src_table[query_type] = entry
            table[source] = src_table
            return table
        self.store.update(self.STATE_KEY, _update)
