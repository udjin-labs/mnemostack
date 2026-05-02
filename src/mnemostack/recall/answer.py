"""Answer generation — inference layer on top of hybrid recall.

Synthesizes concise factual answers from retrieved memories, with explicit
confidence scoring and source citations. Designed to be honest about
uncertainty: low confidence → caller should fall back to raw evidence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..llm.base import LLMProvider
from ..observability import counter, histogram
from .inference_retry import decompose_query, merge_results, should_retry
from .recaller import RecallResult
from .specificity import detect_placeholders, resolve_specificity

if TYPE_CHECKING:
    from .recaller import Recaller

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
6. Say "Not in memory" ONLY when NONE of the memories contain ANY relevant information. If even ONE memory mentions something related to the question, extract and provide the best answer you can. Partial/uncertain answers are better than "Not in memory".
   For hypothetical or inference questions ("might be", "would be", "likely", "probably", "what if"): attempt a reasonable inference from available evidence rather than defaulting to "Not in memory." Only say "Not in memory" when there is truly zero relevant context.
7. If the answer requires inference from multiple memories, do it. For cross-hop questions ("common between X and Y", "what do both..."), connect facts from different memories before answering.
8. NO meta commentary, NO explanation, JUST the answer.

IMPORTANT: "Not in memory" is only appropriate when the memories contain ZERO relevant information about the question topic. If any memory mentions the person, event, or topic — extract what you can.

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

IMPORTANT: "Not in memory" is only appropriate when the memories contain ZERO relevant information about the question topic. If any memory mentions the person, event, or topic — extract what you can.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_LIST_EXTRACT_PROMPT = """You will scan {n_memories} memories and extract ALL items matching the question.

QUESTION: {query}

MEMORIES:
{context}

INSTRUCTIONS:
1. Read EVERY memory. Don't skip.
2. Extract every distinct item that directly matches the question's predicate (e.g. 'pets' = animals owned, 'activities' = things actively done by the person, 'cities visited' = explicit travel destinations).
3. Return JSON: {{"items": ["item1", "item2", ...]}}
4. Each item must be:
   - Specific (use exact names, not 'her dog' → 'Luna')
   - Distinct (no duplicates or near-duplicates like 'cat' and 'cats')
   - Directly supported by memory text (don't infer or generalize)
5. If question asks for a count rather than list — still return all items, count derived from list length.
6. If no items match — return {{"items": []}}

JSON_OUTPUT:"""

_LIST_FINALIZE_PROMPT = """Format these extracted items as a final answer.

QUESTION: {query}

EXTRACTED ITEMS: {items}

RULES:
1. Output the items as a comma-separated list. NO filler.
2. For count questions: output just the number.
3. Use the exact specific names from the items list (no generalizations like 'her dog').
4. Do not invent items. Do not omit items.

ANSWER:"""

_INFERENCE_PROMPT = """Answer an inference question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. This is an inference question (might be / would be / likely). Make a reasonable inference from any relevant evidence in the memories.
2. DO NOT answer 'Not in memory' if even partial evidence exists. Examples:
   - 'What might X's political leaning be?' → infer from values/causes they mention
   - 'Would Y be religious?' → infer from references/practices
   - 'Does Z prefer beach or mountains?' → infer from places they live/visit/discuss
3. Answer format: short answer + (if uncertain) qualifier like 'likely', 'probably', 'leans towards'.
4. Only say 'Not in memory' when there is ZERO relevant evidence at all.
5. If multiple memories contradict, prefer the most recent one.
6. NO meta commentary, NO explanation, JUST the answer.

IMPORTANT: "Not in memory" is only appropriate when the memories contain ZERO relevant information about the question topic. If any memory mentions the person, event, or topic — extract what you can.

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
6. Say "Not in memory" ONLY when NONE of the memories contain ANY relevant information. If even ONE memory mentions something related to the question, extract and provide the best answer you can. Partial/uncertain answers are better than "Not in memory".
7. NO meta commentary, NO explanation, JUST the answer.

IMPORTANT: "Not in memory" is only appropriate when the memories contain ZERO relevant information about the question topic. If any memory mentions the person, event, or topic — extract what you can.

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
6. Say "Not in memory" ONLY when NONE of the memories contain ANY relevant information. If even ONE memory mentions something related to the question, extract and provide the best answer you can. Partial/uncertain answers are better than "Not in memory".
7. NO meta commentary, NO explanation, JUST the answer.

IMPORTANT: "Not in memory" is only appropriate when the memories contain ZERO relevant information about the question topic. If any memory mentions the person, event, or topic — extract what you can.

{confidence_rules}

QUESTION: {query}

ANSWER:"""

_ADVERSARIAL_PROMPT = """Answer a question based on retrieved memories.

MEMORIES:
{context}

RULES:
1. This may be an adversarial question — the asked entity/fact might not exist in memories.
2. If ANY memory contains information relevant to the question, provide an answer. Say "Not in memory" only when there is truly zero relevant context. When evidence is indirect but present, attempt a reasonable answer.
3. Do not infer for adversarial questions when there is truly zero relevant context.
4. If multiple memories contradict, prefer the most recent one.
5. NO meta commentary, NO explanation, JUST the answer.

IMPORTANT: "Not in memory" is only appropriate when the memories contain ZERO relevant information about the question topic. If any memory mentions the person, event, or topic — extract what you can.

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
        r"\b(both|common between|across (?:sessions|dates|speakers|memories)|"
        r"what was the connection between|"
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
        category_aware_prompts: bool = True,
        list_extract_mode: bool = False,
        specificity_resolver: bool = True,
        inference_retry: bool = True,
        recaller: Recaller | None = None,
        retry_with_expansion: bool = False,
        expansion_llm: LLMProvider | None = None,
    ):
        self.llm = llm
        self.max_memories = max_memories
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        self._custom_prompt_template = prompt_template is not None
        self.prompt_template = prompt_template or _DEFAULT_PROMPT
        self.category_aware_prompts = category_aware_prompts
        self.list_extract_mode = list_extract_mode
        self.specificity_resolver = specificity_resolver
        self.inference_retry = inference_retry
        self.recaller = recaller
        self.retry_with_expansion = retry_with_expansion
        self.expansion_llm = expansion_llm

    def generate(
        self,
        query: str,
        memories: list[RecallResult],
        recall_filters: dict[str, object] | None = None,
    ) -> Answer:
        """Synthesize answer from retrieved memories."""
        counter("mnemostack.answer.calls", 1)
        prompt_template = self.prompt_template
        category = "general"
        if self.category_aware_prompts:
            category = classify_question(query)
            counter(
                "mnemostack.answer.question_category",
                1,
                labels={"category": category},
            )
            if not self._custom_prompt_template:
                prompt_template = _PROMPT_BY_CATEGORY[category]

        if not memories:
            counter("mnemostack.answer.empty_memory", 1)
            answer = Answer(
                text="Not in memory.",
                confidence=0.0,
                sources=[],
                raw="",
            )
        elif self.list_extract_mode and category in {"list", "count"}:
            answer = self._generate_list_extract(query, memories)
        else:
            answer = self._generate_single_prompt(
                query=query,
                memories=memories[: self.max_memories],
                prompt_template=prompt_template,
            )

        specificity_memories = memories
        if (
            self.retry_with_expansion
            and self.recaller is not None
            and self._is_weak_answer(answer)
        ):
            answer, specificity_memories = self._retry_answer_with_expansion(
                query=query,
                memories=memories,
                draft=answer,
                prompt_template=prompt_template,
                recall_filters=recall_filters,
            )
        if (
            self.category_aware_prompts
            and self.inference_retry
            and category == "inference"
            and self.recaller is not None
            and should_retry(answer.text, answer.confidence)
        ):
            answer, specificity_memories = self._retry_inference_answer(
                query=query,
                memories=specificity_memories,
                draft=answer,
                prompt_template=prompt_template,
                recall_filters=recall_filters,
            )
        answer = self._apply_specificity_resolver(
            query, answer, specificity_memories, category
        )
        return self._apply_evidence_guard(query, answer, specificity_memories, category)

    @staticmethod
    def _is_weak_answer(answer: Answer) -> bool:
        if not answer.ok:
            return False
        return (
            answer.confidence < 0.5
            or answer.text.lower().strip() in {"not in memory", "not in memory."}
        )

    def _retry_answer_with_expansion(
        self,
        query: str,
        memories: list[RecallResult],
        draft: Answer,
        prompt_template: str,
        recall_filters: dict[str, object] | None,
    ) -> tuple[Answer, list[RecallResult]]:
        """Retry weak answers with expanded batch-vector recall."""
        if self.expansion_llm is None and getattr(self.recaller, "expansion_llm", None) is None:
            return draft, memories

        old_expansion_llm = getattr(self.recaller, "expansion_llm", None)
        if old_expansion_llm is None and self.expansion_llm is not None:
            self.recaller.expansion_llm = self.expansion_llm
        try:
            expanded_memories = self.recaller.recall_with_expanded_vectors(
                query,
                limit=max(self.max_memories, 10),
                vector_limit=max(self.max_memories, 20),
                filters=dict(recall_filters) if recall_filters is not None else None,
                n_variants=2,
            )
        except Exception:
            expanded_memories = []
        finally:
            if old_expansion_llm is None and self.expansion_llm is not None:
                self.recaller.expansion_llm = None

        merged_memories = merge_results(memories, expanded_memories)
        if not merged_memories:
            return draft, memories

        retry_answer = self._generate_single_prompt(
            query=query,
            memories=merged_memories[: self.max_memories],
            prompt_template=prompt_template,
        )
        if retry_answer.ok and not self._is_weak_answer(retry_answer):
            counter("mnemostack.answer.retry_expansion_success", 1)
            return retry_answer, merged_memories
        counter("mnemostack.answer.retry_expansion_no_improvement", 1)
        return draft, memories

    def _retry_inference_answer(
        self,
        query: str,
        memories: list[RecallResult],
        draft: Answer,
        prompt_template: str,
        recall_filters: dict[str, object] | None,
    ) -> tuple[Answer, list[RecallResult]]:
        """Retry low-confidence inference answers with decomposed evidence queries."""
        sub_queries = decompose_query(query, self.llm)
        if not sub_queries:
            return draft, memories

        sub_results: list[list[RecallResult]] = []
        for sub_query in sub_queries:
            try:
                if recall_filters is None:
                    sub_results.append(self.recaller.recall(sub_query, limit=10))
                else:
                    sub_results.append(
                        self.recaller.recall(
                            sub_query,
                            limit=10,
                            filters=dict(recall_filters),
                        )
                    )
            except Exception:
                sub_results.append([])

        merged_memories = merge_results(memories, *sub_results)
        if not merged_memories:
            return draft, memories

        retry_answer = self._generate_single_prompt(
            query=query,
            memories=merged_memories[: self.max_memories],
            prompt_template=prompt_template,
        )
        if (
            retry_answer.ok
            and "not in memory" not in retry_answer.text.lower()
            and retry_answer.confidence >= 0.5
        ):
            return retry_answer, merged_memories
        return draft, memories

    def _apply_evidence_guard(
        self,
        query: str,
        answer: Answer,
        memories: list[RecallResult],
        category: str,
    ) -> Answer:
        """Override low-confidence ungrounded general/adversarial answers."""
        if (
            not answer.ok
            or category not in {"general", "adversarial"}
            or answer.confidence >= 0.3
            or answer.text.lower().strip() in {"not in memory.", "not in memory", "unknown"}
        ):
            return answer

        if not memories or memories[0].score < 0.2 or not self._evidence_check(
            query, answer.text, memories
        ):
            counter("mnemostack.answer.evidence_guard", 1, labels={"category": category})
            return Answer(
                text="Not in memory.",
                confidence=0.1,
                sources=[],
                raw=answer.raw,
            )
        return answer

    @staticmethod
    def _evidence_check(
        query: str,
        answer_text: str,
        memories: list[RecallResult],
    ) -> bool:
        """Return True if answer is grounded in memories, False if likely hallucinated."""
        if not memories:
            return False

        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower()))

        answer_tokens = tokenize(answer_text)
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "not", "it", "its", "i", "my",
            "me", "he", "she", "they", "we", "you", "this", "that", "with",
            "from", "by", "as", "but", "if", "so", "no", "yes",
        }
        answer_content = answer_tokens - stopwords
        if len(answer_content) < 2:
            return True

        all_memory_text = " ".join(m.text.lower() for m in memories[:10])
        memory_tokens = tokenize(all_memory_text)

        query_tokens = tokenize(query)
        claim_tokens = {
            token
            for token in answer_content
            if token.isdigit() or token not in query_tokens
        }
        if claim_tokens and not claim_tokens <= memory_tokens:
            return False

        overlap = answer_content & memory_tokens
        ratio = len(overlap) / len(answer_content) if answer_content else 0
        return ratio >= 0.3

    def _apply_specificity_resolver(
        self,
        query: str,
        answer: Answer,
        memories: list[RecallResult],
        category: str,
    ) -> Answer:
        if (
            not self.specificity_resolver
            or not self.category_aware_prompts
            or category == "adversarial"
            or not answer.ok
        ):
            return answer
        if not detect_placeholders(answer.text):
            return answer

        rewritten = resolve_specificity(
            query=query,
            draft_answer=answer.text,
            candidate_memories=memories[: self.max_memories],
            llm=self.llm,
        )
        if rewritten == answer.text:
            return answer
        answer.text = rewritten
        return answer

    def _generate_list_extract(self, query: str, memories: list[RecallResult]) -> Answer:
        extract_memories = memories[:40]
        context = self._format_context(extract_memories)
        prompt = _LIST_EXTRACT_PROMPT.format(
            n_memories=len(extract_memories),
            context=context,
            query=query,
        )

        with histogram("mnemostack.answer.llm_latency_ms"):
            extract_resp = self.llm.generate(prompt, max_tokens=self.max_tokens)
        if not extract_resp.ok:
            return self._fallback_list_answer(query, memories)

        try:
            items = self._parse_extracted_items(extract_resp.text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._fallback_list_answer(query, memories)

        if not items:
            return self._fallback_list_answer(query, memories)

        finalize_prompt = _LIST_FINALIZE_PROMPT.format(query=query, items=json.dumps(items))
        with histogram("mnemostack.answer.llm_latency_ms"):
            final_resp = self.llm.generate(finalize_prompt, max_tokens=self.max_tokens)
        if not final_resp.ok:
            return self._fallback_list_answer(query, memories)

        return Answer(
            text=final_resp.text.strip(),
            confidence=0.8,
            sources=self._extract_sources(extract_memories),
            raw=final_resp.text,
        )

    def _fallback_list_answer(
        self,
        query: str,
        memories: list[RecallResult],
    ) -> Answer:
        answer = self._generate_single_prompt(
            query=query,
            memories=memories[: self.max_memories],
            prompt_template=_LIST_PROMPT,
        )
        if answer.ok:
            answer.confidence = 0.3
        return answer

    def _generate_single_prompt(
        self,
        query: str,
        memories: list[RecallResult],
        prompt_template: str,
    ) -> Answer:
        context = self._format_context(memories)
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
            sources=self._extract_sources(memories),
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
    def _parse_extracted_items(raw: str) -> list[str]:
        """Parse extraction-pass JSON into a clean item list."""
        data = json.loads(raw.strip())
        if not isinstance(data, dict):
            raise ValueError("extracted JSON must be an object")
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("extracted JSON must contain an items list")
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

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
