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

from ..mca_prefilter import extract_exact_tokens
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
        q_tokens = set(re.findall(r"\w+", q_lower))
        for qtype, ms in self.markers.items():
            if q_tokens & ms:
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
    tokens = set(re.findall(r"\w+", q))
    return bool(tokens & _EXACT_MARKERS)


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


DEFAULT_TECHNICAL_QUERY_DAMPENING_SCALE = 0.6
TECHNICAL_QUERY_SCORE_FLOOR_RAW_THRESHOLD = 0.75
# Disabled: keep the stage/helper wired for compatibility, but make it no-op.
TECHNICAL_QUERY_SCORE_FLOOR_RATIO = 0.0


def _exact_tokens_for_context(context: PipelineContext) -> list[str]:
    tokens = context.extras.get("exact_tokens")
    if tokens is None:
        tokens = extract_exact_tokens(context.query)
        context.extras["exact_tokens"] = tokens
    return list(tokens)


def _is_technical_exact_query(context: PipelineContext) -> bool:
    return bool(_exact_tokens_for_context(context))


def _scale_dampening_factor(factor: float, dampening_scale: float) -> float:
    """Move a multiplicative penalty factor toward 1.0 by dampening_scale.

    A normal factor of 0.5 with scale=0.6 becomes 0.7, i.e. only 60% of the
    original score reduction is applied.
    """
    scale = max(0.0, min(1.0, dampening_scale))
    return 1.0 - (1.0 - factor) * scale


def _raw_vector_score(result) -> float | None:
    try:
        raw = result.payload.get("raw_vector_score")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _apply_technical_score_floor(context: PipelineContext, result) -> None:
    if not _is_technical_exact_query(context):
        return
    raw = _raw_vector_score(result)
    if raw is None or raw < TECHNICAL_QUERY_SCORE_FLOOR_RAW_THRESHOLD:
        return
    floor = raw * TECHNICAL_QUERY_SCORE_FLOOR_RATIO
    if result.score < floor:
        result.score = floor
        result.payload["score_floor"] = "technical_raw_vector"


class TechnicalScoreFloor(Stage):
    """Protect strong vector hits for exact-token queries after all reranking."""

    def apply(self, context, results):
        if not results or not _is_technical_exact_query(context):
            return results
        for r in results:
            _apply_technical_score_floor(context, r)
        results.sort(key=lambda x: -x.score)
        return results


class GravityDampen(Stage):
    """Penalize results that don't actually contain any query key term.

    Counters the "hub memory" problem: a popular memory dominates embedding
    similarity without really containing the information we asked for.
    """

    def __init__(
        self,
        penalty: float = 0.5,
        min_score: float = 0.1,
        technical_query_dampening_scale: float = DEFAULT_TECHNICAL_QUERY_DAMPENING_SCALE,
    ):
        self.penalty = penalty
        self.min_score = min_score
        self.technical_query_dampening_scale = technical_query_dampening_scale

    def apply(self, context, results):
        key_terms = set(context.query_tokens) or (
            set(context.query.lower().split()) - STOPWORDS
        )
        if not key_terms:
            return results
        dampening_factor = self.penalty
        if _is_technical_exact_query(context):
            dampening_factor = _scale_dampening_factor(
                self.penalty, self.technical_query_dampening_scale
            )
        for r in results:
            if r.score <= self.min_score:
                continue
            text_lower = r.text.lower()
            if not any(t in text_lower for t in key_terms):
                r.score *= dampening_factor
                _apply_technical_score_floor(context, r)
                r.payload["dampened"] = "gravity"
        results.sort(key=lambda x: -x.score)
        return results


class HubDampen(Stage):
    """Penalize results whose graph node is in the top 10% by degree.

    Requires `hub_degrees` (dict of id → node degree). If empty, stage is a no-op.
    """

    def __init__(
        self,
        hub_degrees: dict[Any, int] | None = None,
        p90_floor: float = 0.4,
        technical_query_dampening_scale: float = DEFAULT_TECHNICAL_QUERY_DAMPENING_SCALE,
    ):
        self.hub_degrees = hub_degrees or {}
        self.p90_floor = p90_floor
        self.technical_query_dampening_scale = technical_query_dampening_scale

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
        technical_query = _is_technical_exact_query(context)
        for r in results:
            deg = self.hub_degrees.get(r.id, 0)
            if deg > p90:
                ratio = (deg - p90) / (max_deg - p90)
                penalty = 1.0 - (1.0 - self.p90_floor) * ratio
                dampening_factor = max(self.p90_floor, penalty)
                if technical_query:
                    dampening_factor = _scale_dampening_factor(
                        dampening_factor, self.technical_query_dampening_scale
                    )
                r.score *= dampening_factor
                _apply_technical_score_floor(context, r)
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
        always_current_files: tuple[str, ...] = (
            "MEMORY.md",
            "TOOLS.md",
            "AGENTS.md",
            "USER.md",
            "IDENTITY.md",
            "SOUL.md",
            "RULES.md",
            "HEALTHCHECK.md",
        ),
        always_current_freshness: float = 0.8,
    ):
        self.weight = weight
        self.halflife_days = halflife_days
        self.echo_window_minutes = echo_window_minutes
        self.echo_penalty = echo_penalty
        self.always_current_files = always_current_files
        self.always_current_freshness = always_current_freshness

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
            # Always-current files (MEMORY.md, AGENTS.md, etc.) get a high
            # static freshness so they don't lose to today's transcripts.
            src = str(
                (r.payload.get("source") if r.payload else "")
                or (r.id if isinstance(r.id, str) else "")
            )
            if any(name in src for name in self.always_current_files):
                if freshness < self.always_current_freshness:
                    freshness = self.always_current_freshness
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

    Each result must have ``RecallResult.sources`` populated (e.g.
    ``['vector']``, ``['bm25']``, ``['graph']``). One result may carry
    multiple contributing sources when RRF fused them; Q-value is
    averaged across the listed sources. Falls back to ``['vector']``
    when ``sources`` is absent/empty so historical payloads that
    predate the ``sources`` field still score.

    Stage boosts scores from sources that historically performed well
    for the query type. Updates happen via ``record_feedback(memory_id,
    query_type, source, reward)``.
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
        min_samples: int = 10,
        exploration_bonus: float = 0.1,
        blend_weight: float = 0.2,
        use_blend: bool = True,
    ):
        self.store = state_store
        self.alpha = alpha
        self.default_q = default_q
        self.decay_rate = decay_rate
        self.q_min = q_min
        self.q_max = q_max
        self.boost_weight = boost_weight
        self.min_samples = min_samples
        self.exploration_bonus = exploration_bonus
        self.blend_weight = blend_weight
        self.use_blend = use_blend

    def _q_value_with_ucb(
        self, q_table: dict, source: str, query_type: str
    ) -> float:
        """Get Q-value with UCB1 exploration bonus for under-sampled sources.

        Mirrors legacy enhanced-recall.py:get_q_value so cold-start behaviour
        matches: sources with < min_samples get an exploration kick so the
        reranker can move them up instead of leaving them at default_q.
        """
        entry = q_table.get(source, {}).get(query_type, {})
        q = entry.get("q", self.default_q)
        n = entry.get("n", 0)
        if n < self.min_samples:
            total_n = 0
            for src_entry in q_table.values():
                qt_entry = src_entry.get(query_type, {})
                total_n += qt_entry.get("n", 0)
            if total_n > 0 and n > 0:
                q += self.exploration_bonus * math.sqrt(
                    math.log(total_n) / n
                )
            else:
                q += self.exploration_bonus
        return q

    def apply(self, context, results):
        q_table = self.store.get(self.STATE_KEY) or {}
        for r in results:
            # Q can mix multiple sources (Vector + BM25 etc.) for one item.
            source_qs = []
            for source in (r.sources or ["vector"]):
                source_qs.append(
                    self._q_value_with_ucb(q_table, source, context.query_type)
                )
            avg_q = sum(source_qs) / len(source_qs) if source_qs else self.default_q
            if self.use_blend:
                # Multiplicative blend: mirrors legacy behaviour — raw scores
                # still matter but a high-Q source gets a material lift even
                # when score values are tiny (RRF scale ~0.016).
                r.score = (1 - self.blend_weight) * r.score + self.blend_weight * avg_q
            else:
                r.score += self.boost_weight * (avg_q - self.default_q)
            r.payload["q_value"] = round(avg_q, 3)
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
