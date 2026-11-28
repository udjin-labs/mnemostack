"""Built-in OpenAI embedding provider (`POST /v1/embeddings`).

Covers the request contract, batch ordering, dimension validation, API
failures, what reaches the logs, the embedding-space fingerprint of an
existing collection, and selection of the provider on every surface: CLI,
HTTP server and MCP.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import socket
import urllib.request
from email.message import Message
from http.client import HTTPConnection
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.response import addinfourl

import pytest
from qdrant_client import QdrantClient

import mnemostack.embeddings.openai as openai_module
from mnemostack.config import provider_kwargs
from mnemostack.embeddings import EmbeddingProvider, get_provider, list_providers, register_provider
from mnemostack.embeddings.openai import API_KEY_ENV, OpenAIProvider
from mnemostack.embeddings.roles import EMBEDDING_SPACE_KEY, SpaceGuard, embed_documents_resilient
from mnemostack.ingest import IngestItem, Ingestor, ingest_remote_items
from mnemostack.vector import VectorStore

KEY = "your-key-here"
SECRET_TEXT = "the launch code is PINEAPPLE-42"


def _vector(text: str, dim: int = 1536) -> list[float]:
    vector = [0.0] * dim
    if not dim:
        return vector
    for word in text.lower().split():
        vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dim] += 1.0
    return vector


class _Response(io.BytesIO):
    def __init__(self, body, *, headers=None):
        super().__init__(body)
        self.headers = _headers(headers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Opener:
    """Stands in for the module's no-redirect opener."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[dict] = []

    def open(self, request, timeout=0):
        body = json.loads(request.data)
        self.requests.append(
            {
                "url": request.full_url, "method": request.get_method(),
                "headers": dict(request.header_items()), "body": body, "timeout": timeout,
            }
        )
        return self.handler(body, len(self.requests))


def _ok(body, *, dim=1536, order=None):
    rows = [
        {"object": "embedding", "index": i, "embedding": _vector(text, dim)}
        for i, text in enumerate(body["input"])
    ]
    if order == "reversed":
        rows.reverse()
    return _Response(json.dumps({"object": "list", "data": rows, "model": body["model"]}).encode())


def _headers(values):
    headers = Message()
    for name, value in (values or {}).items():
        headers[name] = value
    return headers


def _http_error(status, code="some_code", message="irrelevant", *, headers=None):
    return _http_error_detail(
        status, {"message": message, "type": "invalid_request_error", "code": code}, headers=headers,
    )


def _http_error_detail(status, detail, *, headers=None):
    payload = io.BytesIO(json.dumps({"error": detail}).encode())
    return HTTPError(OpenAIProvider.URL, status, "reason", _headers(headers), payload)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    attempts = []

    def block_network(*args, **kwargs):
        attempts.append(True)
        pytest.fail("OpenAI embedding tests must run offline")

    monkeypatch.setattr(socket, "getaddrinfo", block_network)
    monkeypatch.setattr(socket.socket, "connect", block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", block_network)
    yield
    assert not attempts, "A component attempted a network connection"


@pytest.fixture
def api(monkeypatch):
    """Install a fake API; returns a function that sets the handler."""
    monkeypatch.setattr(openai_module, "time", SimpleNamespace(sleep=lambda _s: None))
    monkeypatch.setenv(API_KEY_ENV, KEY)

    def install(handler=lambda body, _n: _ok(body)):
        opener = _Opener(handler)
        monkeypatch.setattr(openai_module, "_OPENER", opener)
        return opener

    return install


# ------------------------------------------------------------ construction


def test_registered_as_a_builtin_with_the_documented_default():
    assert "openai" in list_providers()
    provider = get_provider("openai", api_key=KEY)
    assert isinstance(provider, OpenAIProvider)
    assert (provider.name, provider.dimension) == ("openai:text-embedding-3-small", 1536)
    assert get_provider("openai", api_key=KEY, model="text-embedding-3-large").dimension == 3072


def test_key_comes_from_the_environment_and_is_required(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=API_KEY_ENV):
        OpenAIProvider()
    monkeypatch.setenv(API_KEY_ENV, KEY)
    assert OpenAIProvider().api_key == KEY


def test_key_whitespace_is_stripped_for_explicit_and_environment_keys(api, monkeypatch):
    opener = api()
    padded_key = f" \t{KEY}\r\n"
    monkeypatch.setenv(API_KEY_ENV, padded_key)
    for provider in (OpenAIProvider(), OpenAIProvider(api_key=padded_key)):
        assert provider.embed(SECRET_TEXT) == _vector(SECRET_TEXT)
    assert [request["headers"]["Authorization"] for request in opener.requests] == [
        f"Bearer {KEY}", f"Bearer {KEY}",
    ]
    with pytest.raises(ValueError, match=f"{API_KEY_ENV} not set"):
        OpenAIProvider(api_key=" \t\n")


@pytest.mark.parametrize("character", ["\n", "\r", "\t", "\x00", "\x7f", "\x80", "\ufeff", "\u200b", "é"])
def test_keys_with_non_printable_ascii_characters_are_rejected(character):
    key = f"{character}{KEY}suffix" if character in {"\ufeff", "\u200b"} else f"{KEY}{character}suffix"
    with pytest.raises(ValueError, match=f"{API_KEY_ENV} must contain only printable ASCII") as failure:
        OpenAIProvider(api_key=key)
    assert key not in str(failure.value)
    assert KEY not in str(failure.value)


def test_unknown_model_is_refused_instead_of_guessing_a_dimension():
    with pytest.raises(ValueError, match="Unknown OpenAI embedding model") as failure:
        OpenAIProvider(model=KEY, api_key=KEY)
    assert KEY not in str(failure.value)


def test_timeout_reaches_the_builtin_but_not_a_custom_provider_of_that_name(monkeypatch):
    assert provider_kwargs("openai", model="text-embedding-3-small", timeout=7) == {
        "model": "text-embedding-3-small", "timeout": 7,
    }
    import mnemostack.embeddings.registry as registry

    class Custom(EmbeddingProvider):
        dimension = 1536
        name = "openai:text-embedding-3-small"

        def embed(self, text):
            return _vector(text)

        def embed_batch(self, texts):
            return [_vector(t) for t in texts]

    monkeypatch.setitem(registry._REGISTRY, "openai", OpenAIProvider)
    register_provider("openai", Custom)
    # A late lookup of another built-in must not replace the custom class.
    registry._lazy_register_builtins()
    assert type(get_provider("openai", **provider_kwargs("openai", timeout=7))) is Custom


# ---------------------------------------------------------------- requests


def test_single_embedding_request_contract(api):
    opener = api()
    vector = OpenAIProvider(timeout=11).embed("hello world")
    assert vector == _vector("hello world") and len(vector) == 1536
    (request,) = opener.requests
    assert request["url"] == "https://api.openai.com/v1/embeddings"
    assert request["method"] == "POST"
    assert request["body"] == {
        "model": "text-embedding-3-small", "input": ["hello world"], "encoding_format": "float",
    }
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert request["timeout"] == 11


def test_batch_is_one_request_and_keeps_input_order(api):
    opener = api(lambda body, _n: _ok(body, order="reversed"))
    texts = [f"text number {i}" for i in range(5)]
    vectors = OpenAIProvider().embed_batch(texts)
    assert vectors == [_vector(t) for t in texts]
    assert len(opener.requests) == 1 and opener.requests[0]["body"]["input"] == texts


def test_large_batch_is_split_and_stays_in_order(api, monkeypatch):
    monkeypatch.setattr(OpenAIProvider, "BATCH_SIZE", 2)
    opener = api(lambda body, _n: _ok(body, order="reversed"))
    texts = [f"t{i}" for i in range(5)]
    assert OpenAIProvider().embed_batch(texts) == [_vector(t) for t in texts]
    assert [len(r["body"]["input"]) for r in opener.requests] == [2, 2, 1]
    assert OpenAIProvider().embed_batch([]) == []


def test_role_methods_use_the_neutral_text(api):
    opener = api()
    provider = OpenAIProvider()
    provider.embed_query("q")
    provider.embed_documents(["d1", "d2"])
    assert [r["body"]["input"] for r in opener.requests] == [["q"], ["d1", "d2"]]


# ------------------------------------------------------- response validation


@pytest.mark.parametrize("dim", [1535, 3072, 0])
def test_dimension_mismatch_fails_single_and_batch(api, dim, caplog):
    opener = api(lambda body, _n: _ok(body, dim=dim))
    provider = OpenAIProvider()
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        provider.embed("x")
    with caplog.at_level(logging.WARNING):
        assert provider.embed_batch(["a", "b", "c"]) == [[], [], []]
    # A wrong dimension is a configuration fault: no per-item replay.
    assert len(opener.requests) == 2
    assert "dimension mismatch" in caplog.text


def test_dimension_mismatch_in_a_later_vector_rejects_the_batch(api):
    rows = [
        {"index": 0, "embedding": _vector("a")},
        {"index": 1, "embedding": _vector("b", 1535)},
    ]
    api(lambda _body, _n: _Response(json.dumps({"data": rows}).encode()))
    assert OpenAIProvider().embed_batch(["a", "b"]) == [[], []]


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "embedding": _vector("a")}],  # one vector for two inputs
        [{"index": 0, "embedding": _vector("a")}, {"index": 0, "embedding": _vector("b")}],
        [{"index": 0, "embedding": _vector("a")}, {"index": 2, "embedding": _vector("b")}],
        [{"index": 0, "embedding": _vector("a")}, {"index": 1, "embedding": "not a list"}],
        [{"index": 0, "embedding": _vector("a")}, {"index": 1, "embedding": ["x"] * 1536}],
        [{"index": 0, "embedding": _vector("a")}, {"index": 1, "embedding": [float("nan")] * 1536}],
        [{"index": 0, "embedding": _vector("a")}, {"index": 1, "embedding": [float("inf")] * 1536}],
        [{"index": 0, "embedding": _vector("a")}, {"index": 1, "embedding": [10**400] * 1536}],
        "not a list",
    ],
)
def test_malformed_or_misaligned_response_is_a_failure(api, rows):
    api(lambda _body, _n: _Response(json.dumps({"data": rows}).encode()))
    assert OpenAIProvider().embed_batch(["a", "b"]) == [[], []]


def test_non_json_response_is_a_failure(api):
    api(lambda _body, _n: _Response(b"<html>gateway</html>"))
    with pytest.raises(RuntimeError, match="malformed response"):
        OpenAIProvider().embed("x")


def test_oversized_response_fails_without_retries(api, monkeypatch):
    body = b"{} "
    monkeypatch.setattr(openai_module, "MAX_RESPONSE_BYTES", len(body) - 1)
    opener = api(lambda _body, _n: _Response(body, headers={"Content-Length": "100"}))
    with pytest.raises(RuntimeError, match="response too large"):
        OpenAIProvider().embed("x")
    assert len(opener.requests) == 1


def test_cut_response_is_retried_then_succeeds(api, monkeypatch):
    delays = []
    monkeypatch.setattr(openai_module.time, "sleep", delays.append)
    monkeypatch.setattr(openai_module.random, "random", lambda: 0.5)

    def handler(body, n):
        if n == 1:
            return _Response(b'{"data": [', headers={"Content-Length": "200"})
        return _ok(body)

    opener = api(handler)
    assert OpenAIProvider().embed("x") == _vector("x")
    assert len(opener.requests) == 2
    assert delays == [1]


# ------------------------------------------------------------- API failures


@pytest.mark.parametrize("detail", [{"code": "rate_limit_exceeded"}, {}])
def test_rate_limit_is_retried_then_succeeds(api, detail):
    def handler(body, n):
        if n < 3:
            raise _http_error_detail(429, detail)
        return _ok(body)

    opener = api(handler)
    assert OpenAIProvider(max_retries=3).embed("x") == _vector("x")
    assert len(opener.requests) == 3


@pytest.mark.parametrize("api_key", [KEY, "quota"])
@pytest.mark.parametrize("status", [400, 429])
def test_quota_exhaustion_skips_retries_and_stops_the_batch(api, monkeypatch, caplog, api_key, status):
    monkeypatch.setattr(OpenAIProvider, "BATCH_SIZE", 2)
    delays = []
    monkeypatch.setattr(openai_module.time, "sleep", delays.append)

    def handler(_body, _n):
        raise _http_error(status, "insufficient_quota", headers={"Retry-After": "5"})

    opener = api(handler)
    assert OpenAIProvider(api_key=api_key).embed_batch(["a", "b", "c"]) == [[], [], []]
    assert [request["body"]["input"] for request in opener.requests] == [["a", "b"]]
    assert delays == []
    reason = f"HTTP {status}"
    if api_key == KEY:
        reason += " insufficient_quota"
    assert caplog.messages == [f"openai embedding failed ({reason}): 3 item(s) not embedded"]


@pytest.mark.parametrize(
    ("status", "headers", "delay"),
    [
        (408, {"Retry-After": "4"}, 4),
        (429, {"Retry-After-Ms": "2250", "Retry-After": "9"}, 2.25),
        (503, {"Retry-After": "5"}, 5),
        (429, {"Retry-After": "Thu, 01 Jan 1970 00:00:15 GMT"}, 5),
        (429, {"Retry-After": "Thu, 01 Jan 1970 00:00:01 GMT"}, 0),
        (503, {"Retry-After": SECRET_TEXT}, 1),
        (429, {"retry-after-ms": "-1", "Retry-After": "3"}, 3),
        (429, {"Retry-After": "120"}, 60),
    ],
)
def test_server_retry_delay(api, monkeypatch, caplog, status, headers, delay):
    delays = []
    monkeypatch.setattr(openai_module.time, "sleep", delays.append)
    monkeypatch.setattr(openai_module.time, "time", lambda: 10, raising=False)
    monkeypatch.setattr(openai_module.random, "random", lambda: 0.5)

    def handler(body, n):
        if n == 1:
            raise _http_error(status, headers=headers)
        return _ok(body)

    api(handler)
    assert OpenAIProvider().embed("x") == _vector("x")
    assert delays == [delay]
    assert SECRET_TEXT not in caplog.text


@pytest.mark.parametrize("header", ["retry-after-ms", "Retry-After"])
@pytest.mark.parametrize("value", ["nan", "inf", "1e400"])
def test_nonfinite_retry_delays_fall_back_to_backoff(api, monkeypatch, header, value):
    delays = []
    monkeypatch.setattr(openai_module.time, "sleep", delays.append)
    monkeypatch.setattr(openai_module.random, "random", lambda: 0.5)

    def handler(body, n):
        if n == 1:
            raise _http_error(429, headers={header: value})
        return _ok(body)

    api(handler)
    assert OpenAIProvider().embed("x") == _vector("x")
    assert delays == [1]


def test_server_errors_exhaust_retries_without_per_item_replay(api):
    def handler(_body, _n):
        raise _http_error(503, "server_error")

    opener = api(handler)
    provider = OpenAIProvider(max_retries=2)
    texts = ["a", "b", "c"]
    assert embed_documents_resilient(provider, texts) == [[], [], []]
    assert [request["body"]["input"] for request in opener.requests] == [texts, texts]
    with pytest.raises(RuntimeError, match="HTTP 503 server_error"):
        provider.embed("a")


@pytest.mark.parametrize("status", [307, 401, 403, 404])
def test_nonretryable_provider_errors_stop_the_batch(api, status):
    def handler(_body, _n):
        raise _http_error(status, "invalid_api_key")

    opener = api(handler)
    assert OpenAIProvider().embed_batch(["a", "b"]) == [[], []]
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    "code", ["invalid_api_key", "model_not_found", "insufficient_quota", "billing_hard_limit_reached"],
)
def test_provider_wide_400_does_not_split_the_batch(api, code):
    def handler(_body, _n):
        raise _http_error(400, code)

    opener = api(handler)
    texts = [f"text number {i}" for i in range(64)]
    assert OpenAIProvider().embed_batch(texts) == [[] for _ in texts]
    assert [request["body"]["input"] for request in opener.requests] == [texts]


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (400, {"code": "context_length_exceeded"}),
        (400, {}),
        (400, {"code": "unknown_input_error"}),
        (413, {"code": "invalid_request_error"}),
        (413, {"code": "model_not_found"}),
    ],
)
def test_content_error_splits_in_halves_until_the_groups_fit(api, status, detail):
    def handler(body, _n):
        if len(body["input"]) > 8:
            raise _http_error_detail(status, detail)
        return _ok(body)

    opener = api(handler)
    texts = [f"text number {i}" for i in range(32)]
    assert OpenAIProvider().embed_batch(texts) == [_vector(text) for text in texts]
    assert [len(r["body"]["input"]) for r in opener.requests] == [32, 16, 8, 8, 16, 8, 8]


def test_content_error_is_logged_for_each_rejected_item_and_later_groups_run(api, caplog, monkeypatch):
    monkeypatch.setattr(OpenAIProvider, "BATCH_SIZE", 2)

    def handler(_body, _n):
        raise _http_error(400, "context_length_exceeded", message=SECRET_TEXT)

    opener = api(handler)
    texts = [f"{SECRET_TEXT} {index}" for index in range(3)]
    with caplog.at_level(logging.WARNING):
        assert OpenAIProvider().embed_batch(texts) == [[], [], []]
    assert [r["body"]["input"] for r in opener.requests] == [texts[:2], [texts[0]], [texts[1]], [texts[2]]]
    assert caplog.messages == ["openai embedding content failure (HTTP 400 context_length_exceeded)"] * 3
    assert SECRET_TEXT not in caplog.text


def test_content_error_only_discards_the_rejected_text(api):
    texts = ["first good text", "rejected text", "second good text"]

    def handler(body, _n):
        if texts[1] in body["input"]:
            raise _http_error(400, "context_length_exceeded")
        return _ok(body)

    opener = api(handler)
    assert OpenAIProvider().embed_batch(texts) == [_vector(texts[0]), [], _vector(texts[2])]
    assert [r["body"]["input"] for r in opener.requests] == [
        texts, texts[:1], texts[1:], texts[1:2], texts[2:],
    ]


def test_invalid_request_error_with_null_code_only_discards_the_rejected_text(api, caplog):
    texts = [f"text number {index}" for index in range(16)]
    rejected = texts[5]

    def handler(body, _n):
        if rejected in body["input"]:
            raise _http_error_detail(400, {
                "message": "Input validation failed",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            })
        return _ok(body)

    opener = api(handler)
    assert OpenAIProvider().embed_batch(texts) == [
        [] if text == rejected else _vector(text) for text in texts
    ]
    assert len(opener.requests) == 9
    assert caplog.messages == ["openai embedding content failure (HTTP 400 invalid_request_error)"]


def test_provider_failure_inside_the_fallback_stops_the_rest(api, monkeypatch):
    monkeypatch.setattr(OpenAIProvider, "BATCH_SIZE", 4)

    def handler(body, n):
        if n == 1:
            raise _http_error(400, "context_length_exceeded")
        if n == 3:
            raise URLError("connection reset")
        return _ok(body)

    opener = api(handler)
    texts = ["a", "b", "c", "d", "e"]
    assert OpenAIProvider(max_retries=1).embed_batch(texts) == [_vector("a"), _vector("b"), [], [], []]
    assert [r["body"]["input"] for r in opener.requests] == [texts[:4], texts[:2], texts[2:4]]


def test_transport_error_is_retried_with_backoff_and_reported_by_class_name(api, monkeypatch):
    def handler(_body, _n):
        raise URLError(f"cannot reach host with {KEY}")

    opener = api(handler)
    delays = []
    monkeypatch.setattr(openai_module.time, "sleep", delays.append)
    monkeypatch.setattr(openai_module.random, "random", lambda: 0.5)
    with pytest.raises(RuntimeError, match="transport error URLError") as failure:
        OpenAIProvider(max_retries=3).embed("x")
    assert len(opener.requests) == 3 and KEY not in str(failure.value)
    assert delays == [1, 2]


@pytest.mark.parametrize("max_retries", [0, -3])
def test_nonpositive_retry_count_still_attempts_once(api, max_retries):
    def handler(_body, _n):
        raise _http_error(503)

    opener = api(handler)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        OpenAIProvider(max_retries=max_retries).embed(SECRET_TEXT)
    assert len(opener.requests) == 1


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_local_errors_do_not_retry(api, monkeypatch, error_type):
    def handler(_body, _n):
        raise error_type(f"{KEY} {SECRET_TEXT}")

    opener = api(handler)
    delays = []
    monkeypatch.setattr(openai_module.time, "sleep", delays.append)
    with pytest.raises(RuntimeError, match=error_type.__name__):
        OpenAIProvider().embed(SECRET_TEXT)
    assert len(opener.requests) == 1
    assert delays == []


def test_health_check_reports_failure_without_details(api):
    def handler(_body, _n):
        raise _http_error(401, "invalid_api_key", message=f"Incorrect API key provided: {KEY}")

    api(handler)
    healthy, message = OpenAIProvider().health_check()
    assert healthy is False and "HTTP 401 invalid_api_key" in message and KEY not in message


# ------------------------------------------------------------------- logging


def test_key_and_input_text_never_reach_logs_or_errors(api, caplog):
    """The API can quote the input and the key in its error message, and a
    transport error can carry anything: neither may be logged or raised."""
    failures = [
        lambda: _http_error(400, "bad\nrequest " + SECRET_TEXT, message=f"input was: {SECRET_TEXT} key {KEY}"),
        lambda: _http_error(401, KEY),
        lambda: _http_error(401, "PINEAPPLE-42"),
        lambda: _http_error(401, {"unexpected": SECRET_TEXT}),
        lambda: _http_error(500, None, message=f"{SECRET_TEXT} {KEY}"),
        lambda: URLError(f"{SECRET_TEXT} {KEY}"),
        lambda: ValueError(f"{SECRET_TEXT} {KEY}"),
    ]
    raised = []
    with caplog.at_level(logging.DEBUG):
        for make in failures:
            def handler(_body, _n, make=make):
                raise make()

            api(handler)
            provider = OpenAIProvider(max_retries=2)
            assert provider.embed_batch([SECRET_TEXT, "another private input"]) == [[], []]
            with pytest.raises(RuntimeError) as failure:
                provider.embed(SECRET_TEXT)
            raised.append(str(failure.value) + repr(failure.value.__cause__))
    seen = caplog.text + " ".join(raised)
    assert caplog.text  # the failures were logged
    for forbidden in (KEY, "PINEAPPLE", "launch code"):
        assert forbidden not in seen


def _assert_clean_exception_chain(error, *secrets):
    pending = [error]
    seen = set()
    while pending:
        link = pending.pop()
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        for secret in secrets:
            assert secret not in str(link)
            assert secret not in repr(link)
        pending.extend((link.__cause__, link.__context__))
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("kind", ["transport", "http", "malformed"])
def test_request_exception_chain_is_clean(api, kind):
    def handler(_body, _n):
        message = f"received {KEY} and {SECRET_TEXT}"
        if kind == "transport":
            raise URLError(message)
        if kind == "http":
            raise _http_error(401, "invalid_api_key", message=message)
        return _Response(message.encode())

    api(handler)
    reason = {
        "transport": "transport error URLError",
        "http": "HTTP 401 invalid_api_key",
        "malformed": "malformed response",
    }[kind]
    with pytest.raises(openai_module._Failure, match=reason) as failure:
        OpenAIProvider(max_retries=1)._request([SECRET_TEXT])
    _assert_clean_exception_chain(failure.value, KEY, SECRET_TEXT)


def test_trailing_newline_rejected_by_real_header_validation_without_exposing_key(api, monkeypatch):
    key = f"{KEY}\n"

    def check_header(request, timeout):
        connection = HTTPConnection("example.invalid", timeout=timeout)
        connection.putrequest("POST", "/embeddings")
        authorization = request.get_header("Authorization")
        assert authorization == f"Bearer {key}"
        connection.putheader("Authorization", authorization)

    opener = api()
    monkeypatch.setattr(opener, "open", check_header)
    provider = OpenAIProvider()
    provider.api_key = key
    with pytest.raises(openai_module._Failure, match="transport error ValueError") as failure:
        provider._request([SECRET_TEXT])
    _assert_clean_exception_chain(failure.value, KEY, key, SECRET_TEXT)


@pytest.mark.parametrize(
    "code",
    [
        "rate_limit_exceeded",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "context_length_exceeded",
        "invalid_api_key",
        "model_not_found",
        "server_error",
        "invalid_request_error",
    ],
)
def test_known_error_code_is_reported(api, code):
    def handler(_body, _n):
        raise _http_error(400, code)

    api(handler)
    with pytest.raises(RuntimeError) as failure:
        OpenAIProvider().embed("rate")
    assert str(failure.value) == f"OpenAI embedding request failed: HTTP 400 {code}"


@pytest.mark.parametrize(
    ("detail", "suffix"),
    [
        ({"code": "custom_code", "type": "invalid_request_error"}, ""),
        ({"code": "echo_private_input"}, ""),
        ({"type": "invalid_request_error"}, " invalid_request_error"),
        ({"code": None, "type": "invalid_request_error"}, " invalid_request_error"),
        ({"code": None, "type": "custom_type"}, ""),
        ({"type": "custom_type"}, ""),
        ({}, ""),
    ],
)
def test_error_type_is_used_for_missing_or_null_codes_and_must_be_listed(api, detail, suffix):
    def handler(_body, _n):
        raise _http_error_detail(401, detail)

    api(handler)
    with pytest.raises(RuntimeError) as failure:
        OpenAIProvider().embed("private_input")
    assert str(failure.value) == f"OpenAI embedding request failed: HTTP 401{suffix}"


def test_http_error_body_is_closed_before_retry(api):
    error = _http_error(503, "server_error")

    def handler(body, n):
        if n == 1:
            raise error
        assert error.closed
        return _ok(body)

    api(handler)
    assert OpenAIProvider().embed("hello") == _vector("hello")


def test_redirects_are_refused(monkeypatch):
    opener = openai_module._OPENER
    (handler,) = (h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler))
    requests = []

    def redirect(request):
        requests.append(request)
        assert len(requests) == 1
        response = addinfourl(
            io.BytesIO(b""), {"location": "https://redirect.example.invalid"}, request.full_url, 302
        )
        response.msg = "Found"
        return response

    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda _host: True)
    monkeypatch.setattr(handler, "https_open", redirect)
    with pytest.raises(RuntimeError, match="HTTP 302"):
        OpenAIProvider(api_key=KEY, max_retries=1).embed(SECRET_TEXT)
    assert len(requests) == 1


# ------------------------------------------------- existing collection, space


class _LegacyOpenAI(EmbeddingProvider):
    """The shape of a custom provider that deployments used before the
    built-in existed: a class registered as ``openai``."""

    dimension = 1536
    name = "openai:text-embedding-3-small"

    def embed(self, text):
        return _vector(text)

    def embed_batch(self, texts):
        return [_vector(t) for t in texts]


@pytest.fixture
def qdrant(monkeypatch):
    import mnemostack.vector.qdrant as qdrant_module

    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda *a, **k: client)
    yield client
    client.close()


def test_fingerprints_equal_those_of_the_custom_provider():
    builtin, legacy = OpenAIProvider(api_key=KEY), _LegacyOpenAI()
    assert builtin.document_space_fingerprint() == legacy.document_space_fingerprint()
    assert builtin.query_profile_fingerprint() == legacy.query_profile_fingerprint()
    # Pinned: a change of this value would orphan every stamped collection.
    assert builtin.document_space_fingerprint() == "es1:1c04cd76fa69690face539c3800f00e5"
    assert builtin._legacy_space_compatible()


def test_existing_collection_stays_readable_and_writable_without_reindexing(api, qdrant):
    store = VectorStore(collection="existing", dimension=1536)
    store.ensure_collection()
    old = Ingestor(_LegacyOpenAI(), store).ingest(
        [IngestItem(text="the staging database runs postgres sixteen", source="notes", offset=0)]
    )
    stamped = store.retrieve_payload(old.ids[0])[EMBEDDING_SPACE_KEY]
    assert stamped == "es1:1c04cd76fa69690face539c3800f00e5"

    opener = api()
    builtin = get_provider("openai")
    assert builtin.dimension == qdrant.get_collection("existing").config.params.vectors.size == 1536
    # The write-side guard compares the stored stamp with the provider's.
    assert SpaceGuard(store, builtin, recheck_seconds=0.0, fail_closed=True).ensure() == stamped
    # Writable: the remote write path runs the same guard and stamps alike.
    results = ingest_remote_items(
        builtin, store, [IngestItem(text="the release checklist lives in the wiki", source="notes", offset=1)]
    )
    assert [r.status for r in results] == ["stored"]
    assert store.retrieve_payload(results[0].id)[EMBEDDING_SPACE_KEY] == stamped
    # Readable: a query embedded by the built-in finds the OLD point first.
    hits = store.search(builtin.embed_query("staging database postgres"), limit=1)
    assert hits[0].id == old.ids[0]
    assert store.count() == 2 and opener.requests


def test_collection_of_another_space_is_refused(api, qdrant):
    """The guard still works: same dimension, another model."""
    from mnemostack.embeddings.roles import EmbeddingSpaceError

    store = VectorStore(collection="existing", dimension=1536)
    store.ensure_collection()
    api()
    Ingestor(get_provider("openai", model="text-embedding-ada-002"), store).ingest(
        [IngestItem(text="older model", source="notes")]
    )
    with pytest.raises(EmbeddingSpaceError):
        ingest_remote_items(get_provider("openai"), store, [IngestItem(text="new", source="notes", offset=1)])


# ---------------------------------------------------------------- surfaces


@pytest.fixture
def shared_qdrant(monkeypatch, tmp_path, api, qdrant):
    """One in-memory Qdrant behind every VectorStore, the fake OpenAI API,
    and no config file: the surfaces are selected by their standard options."""
    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("MNEMOSTACK_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.delenv("MNEMOSTACK_GRAPH_URI", raising=False)
    monkeypatch.delenv("MNEMOSTACK_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MNEMOSTACK_LLM", raising=False)
    for name in ("MNEMOSTACK_PROVIDER", "MNEMOSTACK_EMBEDDING", "MNEMOSTACK_EMBEDDING_PROVIDER", "MNEMOSTACK_TENANT"):
        monkeypatch.delenv(name, raising=False)
    return api()


def test_cli_index_and_search_with_the_openai_provider(shared_qdrant, tmp_path, capsys):
    import mnemostack.cli as cli

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "infra.md").write_text("The staging database runs PostgreSQL on port 5433.\n")
    (notes / "food.md").write_text("The canteen serves soup on Thursdays.\n")
    options = ["--provider", "openai", "--collection", "cli-openai"]
    assert cli.main(["index", str(notes), *options]) == 0
    capsys.readouterr()
    assert cli.main([
        "search", "staging database port", *options, "--limit", "1", "--json", "--raw", "--memgraph-uri", "",
    ]) == 0
    (hit,) = json.loads(capsys.readouterr().out)
    assert "PostgreSQL" in hit["text"]
    assert "vector" in hit["sources"]
    assert shared_qdrant.requests[-1]["body"]["input"] == ["staging database port"]
    assert all(r["body"]["model"] == "text-embedding-3-small" for r in shared_qdrant.requests)


@pytest.mark.parametrize("config_source", ["flags", "env", "yaml"])
async def test_http_and_mcp_share_one_collection_and_one_embedding_space(
    shared_qdrant, qdrant, monkeypatch, tmp_path, config_source,
):
    """The setup-guide scenario: `serve` and `mcp-serve` with the same
    provider and collection, selected by the standard configuration."""
    pytest.importorskip("fastapi")
    fastmcp = pytest.importorskip("fastmcp")
    import httpx
    import uvicorn

    import mnemostack.cli as cli
    import mnemostack.server as http_server

    # Both servers use this LLM setting for reranking as well.
    options = ["--llm", "none", "--memgraph-uri", ""]
    monkeypatch.delenv("MNEMOSTACK_RETRY_ON_WEAK", raising=False)
    if config_source == "flags":
        options += ["--provider", "openai", "--collection", "shared", "--embedding-timeout", "7"]
    elif config_source == "env":
        monkeypatch.setenv("MNEMOSTACK_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("MNEMOSTACK_VECTOR_COLLECTION", "shared")
        monkeypatch.setenv("MNEMOSTACK_EMBEDDING_TIMEOUT", "7")
    else:
        config = tmp_path / "config.yaml"
        config.write_text("embedding:\n  provider: openai\n  timeout: 7\nvector:\n  collection: shared\n")
        monkeypatch.setenv("MNEMOSTACK_CONFIG", str(config))

    apps, servers = [], []
    monkeypatch.setattr(http_server, "_make_probe_client", lambda *args: qdrant)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))
    monkeypatch.setattr(fastmcp.FastMCP, "run", lambda server: servers.append(server))
    assert cli.main(["serve", *options]) == 0
    assert cli.main(["mcp-serve", *options]) == 0
    (app,) = apps
    (server,) = servers

    async def call(name, **arguments):
        result = await server.call_tool(name, arguments)
        return getattr(result, "structured_content", None) or json.loads(result.content[0].text)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://memory.example.com") as http:
        stored = await http.post(
            "/memories", json={"items": [{"text": "The on-call rota changes every Monday", "source": "chat/http"}]}
        )
        assert stored.status_code == 200 and stored.json()["stored"] == 1, stored.text

        found = await call("mnemostack_search", query="on-call rota Monday", limit=1)
        assert "on-call rota" in found["results"][0]["text"]
        remembered = await call("mnemostack_remember", text="Backups run at two in the night", source="chat/mcp")
        assert remembered["ok"] and remembered["stored"] == 1
        recalled = await http.post("/recall", json={"query": "when do backups run", "limit": 1})
        assert recalled.status_code == 200 and "Backups run" in recalled.json()["results"][0]["text"]
    assert {r["body"]["model"] for r in shared_qdrant.requests} == {"text-embedding-3-small"}
    assert {r["timeout"] for r in shared_qdrant.requests} == {7}
    spaces = {
        hit.payload[EMBEDDING_SPACE_KEY]
        for hit in VectorStore(collection="shared", dimension=1536).scroll()
    }
    assert spaces == {"es1:1c04cd76fa69690face539c3800f00e5"}


@pytest.mark.parametrize(
    ("configured", "selected", "key_env"),
    [("gemini", "openai", API_KEY_ENV), ("openai", "gemini", "GEMINI_API_KEY")],
)
def test_doctor_hint_names_the_selected_providers_key(monkeypatch, tmp_path, capsys, configured, selected, key_env):
    import mnemostack.cli as cli

    config = tmp_path / "config.yaml"
    config.write_text(f"embedding:\n  provider: {configured}\n")
    monkeypatch.setenv("MNEMOSTACK_CONFIG", str(config))
    for name in ("MNEMOSTACK_PROVIDER", "MNEMOSTACK_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for probe in ("_doctor_qdrant", "_doctor_llm", "_doctor_graph"):
        monkeypatch.setattr(cli, probe, lambda *args, **kwargs: None)
    assert cli.main(["doctor", "--provider", selected, "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    (embedding,) = (check for check in report["checks"] if check["section"] == "embedding")
    assert embedding["hint"] == f"set {key_env} (required for the {selected} provider)"


def test_doctor_hint_skips_present_key_and_custom_openai_provider(monkeypatch):
    import mnemostack.cli as cli
    from mnemostack.config import Config
    from mnemostack.embeddings import registry

    cfg = Config()
    monkeypatch.setenv(API_KEY_ENV, KEY)
    assert cli._api_key_hint(cfg, "openai") is None
    monkeypatch.delenv(API_KEY_ENV)
    monkeypatch.setitem(registry._REGISTRY, "openai", _LegacyOpenAI)
    assert cli._api_key_hint(cfg, "openai") is None
