"""Unit tests for the smoke suite's client/server pair guard.

``tests/test_qdrant_server_smoke.py`` skips entirely without a live server,
so the verdict logic of ``_assert_supported_pair`` is exercised here against
mocked versions: the guard must fail on officially-unsupported pairs (major
mismatch or minor diff > 1) and on unparseable version strings, and stay
silent on supported ones.
"""

from __future__ import annotations

import pytest
from test_qdrant_server_smoke import _assert_supported_pair


def _with_client(monkeypatch, version: str) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _name: version)


@pytest.mark.parametrize(
    ("client", "server"),
    [
        ("1.15.1", "1.15.4"),  # aligned
        ("1.14.3", "1.15.4"),  # minor diff 1, client older
        ("1.18.0", "1.17.2"),  # minor diff 1, client newer
        ("1.18.0", "1.18.3-dev"),  # pre-release suffix on the patch part
    ],
)
def test_supported_pairs_pass(monkeypatch, client, server):
    _with_client(monkeypatch, client)
    _assert_supported_pair(server)


@pytest.mark.parametrize(
    ("client", "server"),
    [
        ("1.18.0", "1.15.4"),  # the drift PR #129 exists to prevent
        ("1.14.3", "1.18.3"),  # declared minimum vs current server
        ("2.18.0", "1.18.3"),  # major mismatch alone (minor diff is 0)
    ],
)
def test_unsupported_pairs_fail(monkeypatch, client, server):
    _with_client(monkeypatch, client)
    with pytest.raises(pytest.fail.Exception, match="officially-unsupported pair"):
        _assert_supported_pair(server)


@pytest.mark.parametrize("server", ["", "dev", "unknown.version"])
def test_unparseable_server_version_fails_closed(monkeypatch, server):
    _with_client(monkeypatch, "1.18.0")
    with pytest.raises(pytest.fail.Exception, match="cannot parse versions"):
        _assert_supported_pair(server)


def test_unparseable_client_version_fails_closed(monkeypatch):
    _with_client(monkeypatch, "not-a-version")
    with pytest.raises(pytest.fail.Exception, match="cannot parse versions"):
        _assert_supported_pair("1.18.3")
