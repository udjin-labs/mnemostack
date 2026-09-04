"""OpenAI-compatible LLM provider (LiteLLM, vLLM, llama.cpp server, gateways)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any
from urllib.error import HTTPError

from .base import LLMProvider, LLMResponse


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect. urllib's default handler copies the request
    headers — ``Authorization`` included — into the redirected request, so a
    gateway 3xx to another origin would hand the bearer token to that
    destination. An authenticated API POST has no legitimate redirect to
    follow; a 3xx surfaces as a normal HTTP error response instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)

#: Env var holding the Bearer token for the gateway. Read here, in the
#: provider, rather than threaded through ``llm_kwargs``/ServerConfig/CLI —
#: a secret has no business in a YAML config file, and every construction
#: surface would otherwise grow another positional-tail knob just to carry
#: it. Same pattern as ``GEMINI_API_KEY`` in the gemini provider.
API_KEY_ENV = "MNEMOSTACK_LLM_API_KEY"

# Bound successful response bodies too, not only HTTP error details. A
# misconfigured or hostile compatible endpoint must not be able to exhaust the
# serving process's memory with an unbounded response.
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


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

    Reasoning models against the cloud OpenAI endpoint (o1 family and
    successors) reject ``max_tokens`` in favor of ``max_completion_tokens``
    and refuse a non-default ``temperature``. Pass
    ``token_param="max_completion_tokens"`` to rename the budget field and
    ``options={"temperature": None}`` to drop a field; ``options`` values
    otherwise override the defaults (same idea as ``OllamaLLM.options``).
    Gateways normally translate these quirks themselves (LiteLLM's
    ``drop_params``), so the defaults stay the fields every compatible
    server implements.

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
        token_param: str = "max_tokens",
        options: dict[str, Any] | None = None,
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
        self.token_param = token_param
        self.options = dict(options or {})
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
            # max_tokens by default (not the newer max_completion_tokens):
            # the field every compatible server actually implements; see the
            # class docstring for the reasoning-model knobs.
            self.token_param: max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        for opt_key, opt_value in self.options.items():
            if opt_value is None:
                payload.pop(opt_key, None)
            else:
                payload[opt_key] = opt_value
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.host}/chat/completions"
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
            with _OPENER.open(req, timeout=self.timeout) as resp:
                body = resp.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    return LLMResponse(
                        text="", error=f"{self.name} response exceeds {MAX_RESPONSE_BYTES} bytes"
                    )
                data = json.loads(body)
        except HTTPError as exc:
            # The response body usually names the actual problem (unknown
            # model, quota, auth) — surface a bounded snippet of it.
            try:
                detail = exc.read(300).decode(errors="replace")
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
        # bool is an int subclass and a negative count is as malformed as a
        # string — either would corrupt callers that sum usage.
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            tokens = None
        return LLMResponse(text=text, tokens_used=tokens)
