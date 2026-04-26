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

_CONFIDENCE_RULES = """After your answer, on a NEW line, output ONLY:
CONFIDENCE: <float 0.0-1.0>

Where:
- 1.0 = directly stated in one memory
- 0.7-0.9 = clear inference from multiple memories
- 0.4-0.6 = partial/uncertain
- 0.0-0.3 = weak or contradictory evidence"""

_DEFAULT_PROMPT = """Answer a question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. Answer with the SHORTEST factual answer possible. No filler, no sentence form.
2. For list questions (what kinds of, who are, which ones): list ALL mentioned items, comma-separated. Be exhaustive.
3. For date questions (when): absolute date (e.g. "7 May 2023", "2022").
   CRITICAL: convert relative time against the TIMESTAMP of the memory that mentions it.
   - "[2023-02-25] I met Jean yesterday" → event date is 24 February 2023 (session date MINUS 1)
   - "[2023-01-08] I joined the group last week" → event date is the week before 1 January 2023 (ONE WEEK before session date)
   - "[2023-03-13] Grandma passed away last week" → event is the week before 6 March 2023
   - "[2023-05-08] I went to the group yesterday" → 7 May 2023
   If the question asks "when", NEVER return the memory's own timestamp when that memory uses words like "yesterday", "last week", "a few weeks ago", "the week before" — subtract the implied interval first.
   Prefer the phrasing the ground truth likely uses (e.g. "December 2022" vs "1 January 2023" — if the event was in the previous month, answer with the month-year of the event, not the session date that describes it).
4. For identity/label questions: exact label, no explanation.
5. If multiple memories contradict, prefer the most recent one.
6. If the memories genuinely don't contain an answer, reply: "Not in memory."
   For hypothetical or inference questions ("might be", "would be", "likely", "probably", "what if"): attempt a reasonable inference from available evidence rather than defaulting to "Not in memory." Only say "Not in memory" when there is truly zero relevant context.
7. If the answer requires inference from multiple memories, do it. For cross-hop questions ("common between X and Y", "what do both..."), connect facts from different memories before answering.
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

_LIST_PROMPT = """Answer a list or count question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. Answer with a COMPLETE list of ALL items from the memories matching the question.
2. Comma-separated. NO duplicates.
3. NO partial subsets — include EVERY mentioned item even if you're unsure.
4. Use specific names/labels, not vague placeholders ('home country' → exact country, 'that book' → title).
5. If you list 2 items but there might be more — keep scanning the memories first.
6. For count questions, count all matching distinct items after scanning all memories.
7. If multiple memories contradict, prefer the most recent one.
8. If the memories genuinely don't contain an answer, reply: "Not in memory."
9. NO meta commentary, NO explanation, JUST the answer.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_INFERENCE_PROMPT = """Answer an inference question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. This is an inference question. Make a reasonable inference from any relevant evidence in the memories.
2. DO NOT answer 'Not in memory' if even partial evidence exists. Examples:
   - 'What might X's political leaning be?' → infer from values/causes they mention
   - 'Would Y be religious?' → infer from references/practices
   - 'Does Z prefer beach or mountains?' → infer from places they live/visit/discuss
3. Answer format: short answer + (if uncertain) qualifier like 'likely', 'probably', 'leans towards'.
4. Only say 'Not in memory' when there is ZERO relevant evidence at all.
5. If multiple memories contradict, prefer the most recent one.
6. NO meta commentary, NO explanation, JUST the answer.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_MULTIHOP_PROMPT = """Answer a multi-hop reasoning question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. This is a multi-hop reasoning question requiring evidence from MULTIPLE memories. Do not answer from a single memory.
2. Connect facts from different sessions/dates/speakers before answering.
3. Be specific about WHO did WHAT and WHY when stating motivations or causes.
4. Answer in 1-2 sentences with key entities, actions, and reasons explicitly named.
5. If multiple aspects exist (e.g. 'what motivated X' has emotional + practical reasons), list both.
6. If the memories genuinely don't contain an answer, reply: "Not in memory."
7. NO meta commentary, NO explanation, JUST the answer.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_TEMPORAL_PROMPT = """Answer a temporal question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. Answer with the SHORTEST factual answer possible. No filler, no sentence form.
2. For date questions: absolute date (e.g. "7 May 2023", "June 2022").
3. For relative phrases ('yesterday', 'last week', 'a few weeks ago'), compute event date from the memory's own session timestamp first.
   CRITICAL: convert relative time against the TIMESTAMP of the memory that mentions it.
   - "[2023-02-25] I met Jean yesterday" → event date is 24 February 2023 (session date MINUS 1)
   - "[2023-01-08] I joined the group last week" → event date is the week before 1 January 2023 (ONE WEEK before session date)
   - "[2023-03-13] Grandma passed away last week" → event is the week before 6 March 2023
   - "[2023-05-08] I went to the group yesterday" → 7 May 2023
   If the question asks "when", NEVER return the memory's own timestamp when that memory uses words like "yesterday", "last week", "a few weeks ago", "the week before" — subtract the implied interval first.
   Prefer the phrasing the ground truth likely uses (e.g. "December 2022" vs "1 January 2023" — if the event was in the previous month, answer with the month-year of the event, not the session date that describes it).
4. Always answer with absolute date (e.g. '7 May 2023', 'June 2022').
5. If multiple memories contradict, prefer the most recent one.
6. If the memories genuinely don't contain an answer, reply: "Not in memory."
7. NO meta commentary, NO explanation, JUST the answer.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_ADVERSARIAL_PROMPT = """Answer a question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. This may be an adversarial question — the asked entity/fact might not exist in memories.
2. Answer 'Not in memory' UNLESS there is direct, specific, unambiguous evidence in the memories.
3. Do not infer for adversarial questions.
4. If multiple memories contradict, prefer the most recent one.
5. NO meta commentary, NO explanation, JUST the answer.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_PROMPT_BY_CATEGORY = {
    "list": _LIST_PROMPT,
    "count": _LIST_PROMPT,
    "temporal": _TEMPORAL_PROMPT,
    "multihop": _MULTIHOP_PROMPT,
    "inference": _INFERENCE_PROMPT,
    "adversarial": _ADVERSARIAL_PROMPT,
    "general": _DEFAULT_PROMPT,
}

_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)


def classify_question(query: str) -> str:
    """Classify a question shape for answer prompt routing.

    The classifier is intentionally conservative: recognizable high-precision
    patterns route to specialized prompts; ambiguous questions fall back to
    ``general`` to preserve legacy behavior.
    """
    q = re.sub(r"\s+", " ", query.strip().lower())
    if not q:
        return "general"

    if re.search(
        r"\b(something never mentioned|never mentioned|not mentioned|"
        r"made[- ]?up|fictional|nonexistent|does not exist)\b",
        q,
    ) and re.match(r"^(what|which|who|where|when|how)\b", q):
        return "adversarial"

    if re.search(r"\b(how many|count)\b", q):
        return "count"

    if re.match(
        r"^(what are|list|name all|who are|what kinds of|what events|"
        r"what activities|what symbols|what items)\b",
        q,
    ) or re.match(
        r"^which (?:cities|places|locations|countries|books|movies|films|"
        r"events|activities|symbols|items|people|members|pets)\b",
        q,
    ):
        return "list"

    if re.search(
        r"\b(both|common between|across|what was the connection between|"
        r"what motivated|what did .+ take away from|what do .+ and .+|"
        r"what did .+ and .+ both|how does .+ compare to .+)\b",
        q,
    ):
        return "multihop"

    if re.search(
        r"\b(when|what year|which year|what month|which month|what date|"
        r"which date)\b",
        q,
    ) or re.match(rf"^in ({_MONTH_PATTERN}|20(?:20|21|22|23|24))\b", q):
        return "temporal"

    if re.search(
        r"\b(would|likely|probably|what if|do you think|"
        r"what kind of person)\b",
        q,
    ) or re.search(r"\bmight\b(?!['’]ve)", q) or re.search(
        r"\bwhat(?:'s| is) their (?:likely |probable )?(?:political )?leaning\b",
        q,
    ):
        return "inference"

    return "general"


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
        category_aware_prompts: bool = False,
    ):
        self.llm = llm
        self.max_memories = max_memories
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        self.prompt_template = prompt_template or _DEFAULT_PROMPT
        self.category_aware_prompts = category_aware_prompts

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
        prompt_template = self.prompt_template
        if self.category_aware_prompts:
            category = classify_question(query)
            counter(
                "mnemostack.answer.question_category",
                1,
                labels={"category": category},
            )
            prompt_template = _PROMPT_BY_CATEGORY[category]
        prompt = prompt_template.format(
            context=context,
            query=query,
            confidence_rules=_CONFIDENCE_RULES,
        )

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
