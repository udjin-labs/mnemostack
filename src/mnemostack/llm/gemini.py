"""Gemini LLM provider (Flash / Flash Lite / Pro)."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from urllib.error import HTTPError

from .base import LLMProvider, LLMResponse


class GeminiLLM(LLMProvider):
    """Gemini text generation via Google Generative Language API.

    Default model is `gemini-2.5-flash` with thinking disabled
    (thinkingBudget=0) — fast, cheap, good for factual answers.

    For higher quality use `gemini-2.5-pro` (note: Pro does NOT support
    thinkingBudget=0 as of 2026-04; thinking tokens will consume your
    max_tokens budget, set higher maxOutputTokens).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        thinking_budget: int | None = 0,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set (env var or explicit api_key param)")
        self.thinking_budget = thinking_budget
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> LLMResponse:
        url = f"{self.BASE_URL}/{self.model}:generateContent"
        gen_config: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if self.thinking_budget is not None and "flash" in self.model.lower():
            # thinkingBudget=0 only supported on Flash variants, not Pro
            gen_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                text = self._extract_text(data)
                if text is None:
                    return LLMResponse(
                        text="",
                        error=f"empty response (finish reason: {self._finish_reason(data)})",
                    )
                return LLMResponse(
                    text=text.strip(),
                    tokens_used=data.get("usageMetadata", {}).get("totalTokenCount"),
                )
            except HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return LLMResponse(text="", error=f"HTTP {e.code}: {e.reason}")
            except Exception as e:  # noqa: BLE001
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return LLMResponse(text="", error=str(e))
        return LLMResponse(text="", error="exhausted retries")

    @staticmethod
    def _extract_text(data: dict) -> str | None:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        return parts[0].get("text")

    @staticmethod
    def _finish_reason(data: dict) -> str:
        candidates = data.get("candidates") or []
        if candidates:
            return candidates[0].get("finishReason", "unknown")
        return "no_candidates"
