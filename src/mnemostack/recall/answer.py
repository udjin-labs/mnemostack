"""Answer generation — inference layer on top of hybrid recall.

Synthesizes concise factual answers from retrieved memories, with explicit
confidence scoring and source citations. Designed to be honest about
uncertainty: low confidence → caller should fall back to raw evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLMProvider
from ..observability import counter, histogram
from .recaller import RecallResult

_DEFAULT_PROMPT = """Answer a question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. Answer with the SHORTEST factual answer possible. No filler, no sentence form.
2. For list questions (what kinds of, who are, which ones): list ALL mentioned items, comma-separated. Be exhaustive.
3. For date questions (when): absolute date (e.g. "7 May 2023", "2022"). Convert relative time ("yesterday", "last week") using timestamps in memories if present.
4. For identity/label questions: exact label, no explanation.
5. If multiple memories contradict, prefer the most recent one.
6. If the memories genuinely don't contain an answer, reply: "Not in memory."
7. If the answer requires inference from multiple memories, do it — but only with clear evidence.
8. NO meta commentary, NO explanation, JUST the answer.

After your answer, on a NEW line, output ONLY:
CONFIDENCE: <float 0.0-1.0>

Where:
- 1.0 = directly stated in one memory
- 0.7-0.9 = clear inference from multiple memories
- 0.4-0.6 = partial/uncertain
- 0.0-0.3 = weak or contradictory evidence

QUESTION: {query}

ANSWER:"""


@dataclass
class Answer:
    """Synthesized answer with provenance."""

    text: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    raw: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class AnswerGenerator:
    """Generate concise answers from retrieved memories.

    Args:
        llm: LLM provider (usually Gemini Flash with thinkingBudget=0)
        max_memories: how many top memories to include in the prompt
        max_tokens: LLM output budget
        confidence_threshold: callers can use .should_fallback(answer) to decide
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_memories: int = 15,
        max_tokens: int = 200,
        confidence_threshold: float = 0.5,
        prompt_template: str | None = None,
    ):
        self.llm = llm
        self.max_memories = max_memories
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        self.prompt_template = prompt_template or _DEFAULT_PROMPT

    def generate(self, query: str, memories: list[RecallResult]) -> Answer:
        """Synthesize answer from retrieved memories."""
        counter("mnemostack.answer.calls", 1)
        if not memories:
            counter("mnemostack.answer.empty_memory", 1)
            return Answer(
                text="Not in memory.",
                confidence=0.0,
                sources=[],
                raw="",
            )

        context = self._format_context(memories[: self.max_memories])
        prompt = self.prompt_template.format(context=context, query=query)

        with histogram("mnemostack.answer.llm_latency_ms"):
            resp = self.llm.generate(prompt, max_tokens=self.max_tokens)
        if not resp.ok:
            counter("mnemostack.answer.errors", 1)
            return Answer(
                text="",
                confidence=0.0,
                sources=[],
                raw="",
                error=resp.error,
            )

        text, confidence = self._parse_response(resp.text)
        bucket = "high" if confidence >= 0.7 else ("medium" if confidence >= 0.4 else "low")
        counter("mnemostack.answer.by_confidence", 1, labels={"bucket": bucket})
        return Answer(
            text=text,
            confidence=confidence,
            sources=self._extract_sources(memories[: self.max_memories]),
            raw=resp.text,
        )

    def should_fallback(self, answer: Answer) -> bool:
        """Whether caller should show raw memories instead of this answer."""
        if answer.error:
            return True
        return answer.confidence < self.confidence_threshold

    @staticmethod
    def _format_context(memories: list[RecallResult]) -> str:
        lines = []
        for i, m in enumerate(memories, 1):
            text = m.text.strip().replace("\n", " ")[:400]
            source = m.payload.get("source", "")
            ts = m.payload.get("timestamp", "")
            prefix = f"[{i}]"
            if ts:
                prefix = f"{prefix} [{ts[:10]}]"
            if source:
                prefix = f"{prefix} ({source})"
            lines.append(f"{prefix} {text}")
        return "\n".join(lines)

    @staticmethod
    def _extract_sources(memories: list[RecallResult]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for m in memories:
            src = m.payload.get("source", "")
            if src and src not in seen:
                seen.add(src)
                out.append(src)
        return out[:5]

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, float]:
        """Extract answer and confidence from LLM output."""
        lines = raw.strip().split("\n")
        answer_lines: list[str] = []
        confidence = 0.5
        for line in lines:
            match = re.match(r"CONFIDENCE:\s*(-?[0-9.]+)", line.strip(), re.IGNORECASE)
            if match:
                try:
                    confidence = max(0.0, min(1.0, float(match.group(1))))
                except ValueError:
                    pass
            else:
                answer_lines.append(line)
        return "\n".join(answer_lines).strip(), confidence
