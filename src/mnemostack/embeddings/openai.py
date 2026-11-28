"""OpenAI embedding provider (``POST /v1/embeddings``)."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import urllib.request
from datetime import timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

#: Env var holding the API key. Read here, in the provider: a secret has no
#: business in a YAML config file. Same pattern as ``GEMINI_API_KEY``.
API_KEY_ENV = "OPENAI_API_KEY"

# A batch of 256 vectors of 3072 floats is about 20 MB of JSON. The bound
# stops a misbehaving endpoint from exhausting the process memory.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024

_REQUEST_WIDE_400_CODES = frozenset({
    "invalid_api_key",
    "model_not_found",
    "insufficient_quota",
    "billing_hard_limit_reached",
})
_KNOWN_ERROR_CODES = _REQUEST_WIDE_400_CODES | frozenset({
    "rate_limit_exceeded",
    "context_length_exceeded",
    "server_error",
    "invalid_request_error",
})
_MAX_RETRY_DELAY = 60.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: urllib would copy ``Authorization`` into the
    redirected request and hand the key to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _backoff(attempt: int, base: float = 1.0) -> float:
    """Exponential backoff with jitter so concurrent retries don't synchronize."""
    return base * (2**attempt) * (0.5 + random.random())


def _retry_delay(headers: Message | None) -> float | None:
    """Read a server retry delay in seconds, capped at one minute."""
    if headers is None:
        return None
    for header, scale in (("retry-after-ms", 0.001), ("Retry-After", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except ValueError:
            if header != "Retry-After":
                continue
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = max(0.0, date.timestamp() - time.time())
            except (ValueError, OverflowError, OSError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return min(seconds, _MAX_RETRY_DELAY)
    return None


def _content_length(headers: Message) -> int:
    try:
        return int(headers.get("Content-Length", "0"))
    except ValueError:
        return 0


class _Failure(Exception):
    """One failed request. ``reason`` is safe to log and to show: it holds a
    status code, an error code of the API or an exception class name, never
    the request text, the key or a response message."""

    def __init__(self, reason: str, *, provider_down: bool):
        super().__init__(reason)
        self.reason = reason
        self.provider_down = provider_down


def _error_code(error: HTTPError) -> str:
    """Read a recognized machine-readable error code."""
    try:
        with error:
            body = json.loads(error.read(65536))
        detail = body.get("error") if isinstance(body, dict) else None
        if not isinstance(detail, dict):
            return ""
        code = detail.get("code")
        if code is None:
            code = detail.get("type")
    except Exception:  # noqa: BLE001 — an unreadable body just has no code
        return ""
    if not isinstance(code, str) or code not in _KNOWN_ERROR_CODES:
        return ""
    return code


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI embeddings API.

    Default model is ``text-embedding-3-small`` with 1536-dim output.
    Set the ``OPENAI_API_KEY`` env var or pass ``api_key`` directly.

    Every returned vector is checked against the model's dimension. A
    response of another size is a failure, never stored data.
    """

    #: embed_batch handles content failures and provider outages internally.
    _batch_includes_per_item_fallback = True

    URL = "https://api.openai.com/v1/embeddings"
    DEFAULT_MODEL = "text-embedding-3-small"
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    #: Inputs per request. The API accepts more; smaller requests keep one
    #: failure from losing a large group.
    BATCH_SIZE = 256

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        if model not in self.MODEL_DIMS:
            # The dimension is declared, not guessed: a wrong guess would
            # create a collection of the wrong size.
            raise ValueError(f"Unknown OpenAI embedding model. Supported: {sorted(self.MODEL_DIMS)}")
        self.model = model
        key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        if not key:
            raise ValueError(f"{API_KEY_ENV} not set (env var or explicit api_key param)")
        if any(not " " <= char <= "~" for char in key):
            raise ValueError(f"{API_KEY_ENV} must contain only printable ASCII characters")
        self.api_key = key
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self._dim = self.MODEL_DIMS[model]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, text: str) -> list[float]:
        try:
            return self._request([text])[0]
        except _Failure as failure:
            reason = failure.reason
            if not failure.provider_down:
                logger.warning("openai embedding content failure (%s)", reason)
        raise RuntimeError(f"OpenAI embedding request failed: {reason}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, in input order. Empty list for a failed item."""
        out: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.BATCH_SIZE):
                self._embed_group(texts[start : start + self.BATCH_SIZE], out)
        except _Failure as failure:
            missing = len(texts) - len(out)
            logger.error(
                "openai embedding failed (%s): %d item(s) not embedded", failure.reason, missing
            )
            out.extend([] for _ in range(missing))
        return out

    def _embed_group(self, texts: list[str], out: list[list[float]]) -> None:
        """Append vectors in order, splitting content rejections down to single items."""
        try:
            out.extend(self._request(texts))
            return
        except _Failure as failure:
            if failure.provider_down:
                raise
            if len(texts) == 1:
                logger.warning("openai embedding content failure (%s)", failure.reason)
                out.append([])
                return
        middle = len(texts) // 2
        self._embed_group(texts[:middle], out)
        self._embed_group(texts[middle:], out)

    def _request(self, texts: list[str]) -> list[list[float]]:
        """One call for ``texts``; vectors in input order, or ``_Failure``."""
        reason = ""
        provider_down = True
        for attempt in range(self.max_retries):
            delay = None
            try:
                body = json.dumps(
                    {"model": self.model, "input": texts, "encoding_format": "float"}
                ).encode()
                request = urllib.request.Request(
                    self.URL,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                with _OPENER.open(request, timeout=self.timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) <= MAX_RESPONSE_BYTES:
                        expected = _content_length(response.headers)
                        if len(raw) < expected:
                            raise IncompleteRead(raw, expected - len(raw))
            except HTTPError as error:
                code = _error_code(error)
                quota_exhausted = code == "insufficient_quota"
                retryable = not quota_exhausted and (
                    error.code in (408, 429) or 500 <= error.code < 600
                )
                if retryable:
                    delay = _retry_delay(error.headers)
                reason = f"HTTP {error.code}"
                if code and code not in self.api_key and self.api_key not in code:
                    reason += f" {code}"
                content_error = error.code == 413 or (
                    error.code == 400 and code not in _REQUEST_WIDE_400_CODES
                )
                provider_down = not content_error
            except Exception as error:  # noqa: BLE001 — transport failure
                reason = f"transport error {type(error).__name__}"
                retryable = not isinstance(error, (ValueError, TypeError))
            else:
                return self._vectors(raw, len(texts))
            if not retryable or attempt + 1 == self.max_retries:
                break
            logger.warning(
                "openai embed retry (%s), attempt %d/%d", reason, attempt + 1, self.max_retries
            )
            time.sleep(delay if delay is not None else _backoff(attempt))
        raise _Failure(reason, provider_down=provider_down)

    def _vectors(self, raw: bytes, expected: int) -> list[list[float]]:
        """Parse and validate one response: one vector per input, placed by
        the API's own ``index``, each of the model's dimension."""
        if len(raw) > MAX_RESPONSE_BYTES:
            raise _Failure("response too large", provider_down=True)
        placed: dict[int, list[float]] = {}
        malformed = False
        try:
            data: Any = json.loads(raw)
            rows = data["data"]
            for row in rows:
                index, vector = row["index"], row["embedding"]
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index in placed
                    or not isinstance(vector, list)
                ):
                    raise ValueError
                placed[index] = vector
        except Exception:  # noqa: BLE001 — any malformed shape is one failure
            malformed = True
        if malformed:
            raise _Failure("malformed response", provider_down=True)
        if sorted(placed) != list(range(expected)):
            raise _Failure("response indices do not match the inputs", provider_down=True)
        for vector in placed.values():
            if len(vector) != self._dim:
                raise _Failure("dimension mismatch", provider_down=True)
            try:
                valid = all(
                    type(value) in (int, float) and math.isfinite(value) for value in vector
                )
            except OverflowError:
                valid = False
            if not valid:
                raise _Failure("malformed response", provider_down=True)
        return [placed[i] for i in range(expected)]
