"""LLM-based extractor: text → knowledge graph triples.

Given free-form text, extracts (subject, predicate, object) triples suitable
for GraphStore.add_triple(). Uses a small, deterministic prompt that asks
the LLM to respond in a strict JSON format.
"""
from __future__ import annotations

import json
import re

from ..llm.base import LLMProvider
from .store import GraphStore, Triple

_EXTRACT_PROMPT = """Extract structured facts from the text as knowledge graph triples.

A triple is (subject, predicate, object). Subject and object are entity names
(people, projects, places, things). Predicate is a short verb or relation
like 'works_on', 'decided', 'uses', 'located_in', 'caused', 'member_of'.

RULES:
1. Return ONLY a JSON array. No prose, no markdown, no explanations.
2. Each item: {{"subject": "...", "predicate": "...", "object": "...", "valid_from": "YYYY-MM-DD or null"}}
3. Use predicate in snake_case lowercase (works_on, not WORKS_ON).
4. Use specific entity names, not generic terms ("Alice" not "the developer").
5. If a date is mentioned, set valid_from. Otherwise null.
6. Extract AT MOST {max_triples} most significant triples. Skip trivial facts.
7. If no extractable facts, return: []

TEXT:
{text}

JSON:"""


class TripleExtractor:
    """Extract structured triples from free-form text using an LLM.

    Args:
        llm: LLM provider (Gemini Flash works well)
        max_triples: cap on triples extracted per call
        max_tokens: LLM response budget (JSON array can get long)
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_triples: int = 10,
        max_tokens: int = 1000,
    ):
        self.llm = llm
        self.max_triples = max_triples
        self.max_tokens = max_tokens

    def extract(self, text: str) -> list[Triple]:
        """Extract triples from text. Returns empty list on failure."""
        if not text.strip():
            return []

        prompt = _EXTRACT_PROMPT.format(text=text.strip(), max_triples=self.max_triples)
        resp = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=0.0)
        if not resp.ok or not resp.text:
            return []

        data = self._parse_json(resp.text)
        if data is None:
            return []

        triples: list[Triple] = []
        for item in data[: self.max_triples]:
            if not isinstance(item, dict):
                continue
            subj = (item.get("subject") or "").strip()
            pred = (item.get("predicate") or "").strip()
            obj = (item.get("object") or "").strip()
            if not (subj and pred and obj):
                continue
            triples.append(
                Triple(
                    subject=subj,
                    predicate=pred,
                    obj=obj,
                    valid_from=item.get("valid_from") or None,
                )
            )
        return triples

    def extract_and_store(
        self,
        text: str,
        graph: GraphStore,
        subject_label: str = "Entity",
        obj_label: str = "Entity",
    ) -> list[Triple]:
        """Extract triples and persist them in the graph. Returns added triples."""
        triples = self.extract(text)
        for t in triples:
            graph.add_triple(
                subject=t.subject,
                predicate=t.predicate,
                obj=t.obj,
                valid_from=t.valid_from,
                subject_label=subject_label,
                obj_label=obj_label,
            )
        return triples

    @staticmethod
    def _parse_json(raw: str) -> list | None:
        """Find the first JSON array in the response.

        LLMs sometimes prepend markdown or prose despite instructions.
        This handles ```json fences and leading/trailing text.
        """
        text = raw.strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Find first [ ... ]
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
