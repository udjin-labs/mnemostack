"""OpenAI-compatible LLM provider (LiteLLM, vLLM, llama.cpp server, gateways)."""

from __future__ import annotations

import json
import os
import urllib.request
from urllib.error import HTTPError

from .base import LLMProvider, LLMResponse

#: Env var holding the Bearer token for the gateway. Read here, in the
#: provider, rather than threaded through ``llm_kwargs``/ServerConfig/CLI —
#: a secret has no business in a YAML config file, and every construction
#: surface would otherwise grow another positional-tail knob just to carry
#: it. Same pattern as ``GEMINI_API_KEY`` in the gemini provider.
API_KEY_ENV = "MNEMOSTACK_LLM_API_KEY"


class OpenAICompatLLM(LLMProvider):
    """LLM behind any endpoint speaking ``POST {base}/v1/chat/completions``.

    This is the universal adapter: LiteLLM proxies, vLLM, llama.cpp server,
    LM Studio, OpenRouter and the cloud vendors' own OpenAI-compatible
    endpoints all accept this protocol, so one provider covers them all.

    Both ``host`` and ``model`` are required — a gateway's base URL and its
    model names are deployment-specific, and any default we picked would
    dial the wrong place silently. Misconfiguration fails loud at
    construction (like the gemini provider's missing-key check), not as a
    per-call error.

    Requests are single-shot — no transient-error retry, unlike the gemini
    provider. Deliberate: the gateways this provider targets (LiteLLM and
    friends) implement retries and fallbacks themselves, and every shipped
    caller of ``generate()`` is fail-open on error.

    Auth: ``api_key`` argument, else the ``MNEMOSTACK_LLM_API_KEY`` env var.
    Unset, empty, or the literal string ``none`` (any case) sends no
    ``Authorization`` header — keyless vLLM and llama.cpp deployments must
    not receive a bogus Bearer token, which some servers reject.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: int = 60,
        api_key: str | None = None,
    ):
        if not host:
            raise ValueError(
                "openai LLM provider requires a base URL: set llm.host in the "
                "config or MNEMOSTACK_LLM_HOST (e.g. http://gateway:4000)"
            )
        if not model:
            raise ValueError(
                "openai LLM provider requires a model name: set llm.model in "
                "the config or pass --llm-model (gateways have no default)"
            )
        self.model = model
        # Gateways commonly live on bare "host:port" inside a network — same
        # scheme normalization as the ollama providers.
        if "://" not in host:
            host = f"http://{host}"
        host = host.rstrip("/")
        # The OpenAI SDK convention is a base_url that already ends in /v1;
        # LiteLLM docs hand out bare hosts. Accept both, never double the
        # path segment.
        if not host.endswith("/v1"):
            host = f"{host}/v1"
        self.host = host
        self.timeout = timeout
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if key and key.lower() != "none":
            self.api_key: str | None = key
        else:
            self.api_key = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # max_tokens (not the newer max_completion_tokens): the field
            # every compatible server actually implements.
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.host}/chat/completions"
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except HTTPError as exc:
            # The response body usually names the actual problem (unknown
            # model, quota, auth) — surface a bounded snippet of it.
            try:
                detail = exc.read().decode(errors="replace")[:300]
            except Exception:
                detail = ""
            return LLMResponse(
                text="", error=f"{self.name} HTTP {exc.code}: {detail or exc.reason}"
            )
        except Exception as exc:
            return LLMResponse(text="", error=f"{self.name} request failed: {exc}")

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return LLMResponse(
                text="",
                error=f"{self.name} unexpected response shape: {str(data)[:300]}",
            )
        if not isinstance(content, str):
            # Some multimodal upstreams hand back content as a list of parts —
            # name the real problem, not a token budget the operator will chase.
            return LLMResponse(
                text="",
                error=f"{self.name} unexpected content type: {type(content).__name__}",
            )
        text = content.strip()
        if not text:
            # Whitespace-only counts: a truthy "\n" reporting ok would degrade
            # reranking and expansion as silently as a genuinely empty string.
            return LLMResponse(text="", error=f"{self.name} returned empty content")
        # isinstance gates, not `or {}`: a malformed truthy usage ("n/a", a
        # list) must degrade to tokens_used=None, never raise past the
        # base-class never-raise contract.
        usage = data.get("usage")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        return LLMResponse(
            text=text,
            tokens_used=tokens if isinstance(tokens, int) else None,
        )
