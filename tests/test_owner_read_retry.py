"""Owner-read transport faults; no server, models or external research fixtures."""
from __future__ import annotations

import asyncio
import copy
import json

import grpc
import httpcore
import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from mnemostack.vector.qdrant import TenantConflictError, VectorStore


class RpeSubclass(httpx.RemoteProtocolError):
    pass


class Wire:
    """Synthetic peer state; request serialization and error wrapping stay real."""

    def __init__(self, reads=(), writes=(), owners=None):
        self.read_faults = list(reads)
        self.write_faults = list(writes)
        self.points = {
            i: {"id": i, "vector": [1.0, 0.0], "payload": {"tenant_id": owner}}
            for i, owner in (owners or {101: "alpha"}).items()
        }
        self.requests = []
        self.puts = 0
        self.commits = 0

    @property
    def reads(self):
        return [r for r in self.requests if r["method"] == "POST"]

    def __call__(self, request):
        assert request.url.host == "owner.invalid"
        assert request.url.path == "/collections/owner_test/points"
        body = json.loads(request.content)
        self.requests.append({
            "method": request.method, "path": request.url.path,
            "body": body, "bytes": request.content,
            "timeout": request.extensions["timeout"],
        })
        if request.method == "POST":
            event = self.read_faults.pop(0) if self.read_faults else None
            if callable(event):
                event(self)
            elif isinstance(event, BaseException):
                raise event
            elif event == "core-rpe":
                from httpx._transports.default import map_httpcore_exceptions

                with map_httpcore_exceptions():
                    raise httpcore.RemoteProtocolError("synthetic closure")
            elif event in (401, 403, 422, 500, 429):
                return httpx.Response(event, json={"status": {"error": "synthetic"}})
            elif event == "malformed":
                return httpx.Response(200, json={"result": "not-records", "status": "ok"})
            rows = []
            for pid in body["ids"]:
                if pid not in self.points:
                    continue
                payload = self.points[pid]["payload"]
                fields = body.get("with_payload")
                if isinstance(fields, list):
                    payload = {k: payload[k] for k in fields if k in payload}
                rows.append({"id": pid, "payload": payload, "vector": None})
            return httpx.Response(200, json={"result": rows, "status": "ok", "time": 0})
        assert request.method == "PUT"
        self.puts += 1
        event = self.write_faults.pop(0) if self.write_faults else None
        if isinstance(event, BaseException):
            raise event
        for point in body["points"]:
            self.points[point["id"]] = copy.deepcopy(point)
        self.commits += 1
        if event == "committed-rpe":
            raise httpx.RemoteProtocolError("response lost after synthetic commit")
        return httpx.Response(200, json={
            "result": {"operation_id": self.commits, "status": "completed"},
            "status": "ok", "time": 0,
        })


@pytest.fixture
def make_store():
    clients = []

    def make(wire, grpc_mode=False):
        client = QdrantClient(
            url="http://owner.invalid:6333", timeout=30,
            prefer_grpc=grpc_mode, check_compatibility=False,
            trust_env=False, transport=httpx.MockTransport(wire),
        )
        clients.append(client)
        store = VectorStore.__new__(VectorStore)
        store.collection = "owner_test"
        store.client = client
        return store

    yield make
    for client in clients:
        client.close()


def payload():
    return {
        "text": "Synthetic record: значение.", "source": "synthetic-source",
        "offset": 42, "source_content_hash": "synthetic-version",
        "valid_from": "2026-01-01T00:00:00Z", "valid_until": "2027-01-01T00:00:00Z",
        "invalidated_at": None, "tenant_id": "untrusted-override",
    }


def upsert(store, ids=(101,), tenant="alpha", batch_size=100):
    points = [(pid, [1.0, 0.0], payload()) for pid in ids]
    before = copy.deepcopy(points)
    try:
        return store.upsert_batch(points, batch_size=batch_size, tenant=tenant)
    finally:
        assert points == before


def assert_owner_reads(wire, ids, count):
    assert len(wire.reads) == count
    for request in wire.reads:
        assert request["body"] == {
            "ids": list(ids), "with_payload": ["tenant_id"], "with_vector": False,
        }
        assert set(request["timeout"].values()) == {30}
    assert all(r == wire.reads[0] for r in wire.reads)


def test_normal_path_exact_payload_and_owner_projection(make_store):
    wire = Wire()
    assert upsert(make_store(wire)) == 1
    assert_owner_reads(wire, [101], 1)
    assert wire.puts == wire.commits == 1
    expected = payload()
    expected["tenant_id"] = "alpha"
    assert wire.points[101] == {"id": 101, "vector": [1.0, 0.0], "payload": expected}


@pytest.mark.parametrize("tenant", ["alpha", None])
@pytest.mark.parametrize("event", ["rpe", "core-rpe"])
def test_transient_owner_read_persists_once(make_store, tenant, event):
    first = httpx.RemoteProtocolError("synthetic closure") if event == "rpe" else event
    wire = Wire(reads=[first])
    assert upsert(make_store(wire), tenant=tenant) == 1
    assert_owner_reads(wire, [101], 2)
    assert wire.puts == wire.commits == 1
    assert wire.points[101]["payload"]["tenant_id"] == "alpha"


def test_persistent_owner_failure_stops_after_two(make_store):
    wire = Wire(reads=[httpx.RemoteProtocolError("one"), httpx.RemoteProtocolError("two")])
    with pytest.raises(ResponseHandlingException) as caught:
        upsert(make_store(wire))
    assert type(caught.value.source) is httpx.RemoteProtocolError
    assert_owner_reads(wire, [101], 2)
    assert wire.puts == wire.commits == 0


@pytest.mark.parametrize("status", [401, 403, 422, 500, 429])
def test_http_errors_are_not_retried(make_store, status):
    wire = Wire(reads=[status])
    with pytest.raises(UnexpectedResponse) as caught:
        upsert(make_store(wire))
    assert caught.value.status_code == status
    assert_owner_reads(wire, [101], 1)
    assert wire.puts == 0


@pytest.mark.parametrize("fault", [
    httpx.ReadTimeout("timeout"), httpx.WriteTimeout("timeout"),
    httpx.ConnectTimeout("timeout"), httpx.PoolTimeout("timeout"),
    httpx.ConnectError("closure"), httpx.LocalProtocolError("closure"),
    RpeSubclass("subtype"), ValueError("RemoteProtocolError words only"),
    ResponseHandlingException(httpx.RemoteProtocolError("nested wrapper")),
])
def test_other_wrapped_errors_are_not_retried(make_store, fault):
    wire = Wire(reads=[fault])
    with pytest.raises(ResponseHandlingException) as caught:
        upsert(make_store(wire))
    assert caught.value.source is fault
    assert_owner_reads(wire, [101], 1)
    assert wire.puts == 0


def test_malformed_success_is_validation_error_not_retry(make_store):
    wire = Wire(reads=["malformed"])
    with pytest.raises(ResponseHandlingException) as caught:
        upsert(make_store(wire))
    assert type(caught.value.source).__name__ == "ValidationError"
    assert_owner_reads(wire, [101], 1)
    assert wire.puts == 0


def test_cancellation_is_not_retried(make_store):
    wire = Wire(reads=[asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        upsert(make_store(wire))
    assert_owner_reads(wire, [101], 1)
    assert wire.puts == 0


def test_second_error_is_preserved_without_third_attempt(make_store):
    timeout = httpx.ReadTimeout("second failure")
    wire = Wire(reads=[httpx.RemoteProtocolError("first"), timeout])
    with pytest.raises(ResponseHandlingException) as caught:
        upsert(make_store(wire))
    assert caught.value.source is timeout
    assert_owner_reads(wire, [101], 2)
    assert wire.puts == 0


@pytest.mark.parametrize("owners", [{101: "beta"}, {101: "alpha", 102: "beta"}])
def test_conflict_blocks_entire_chunk(make_store, owners):
    wire = Wire(owners=owners)
    with pytest.raises(TenantConflictError):
        upsert(make_store(wire), ids=tuple(owners))
    assert_owner_reads(wire, tuple(owners), 1)
    assert wire.puts == 0


def test_owner_change_on_first_failure_is_seen(make_store):
    def change(wire):
        wire.points[102]["payload"]["tenant_id"] = "beta"
        raise httpx.RemoteProtocolError("owner changed before fresh read")

    wire = Wire(reads=[change], owners={101: "alpha", 102: "alpha"})
    with pytest.raises(TenantConflictError):
        upsert(make_store(wire), ids=(101, 102))
    assert_owner_reads(wire, [101, 102], 2)
    assert wire.puts == 0


@pytest.mark.parametrize("initial_owner", [None, "alpha"])
def test_owner_disappears_before_retry_preserves_legacy_semantics(make_store, initial_owner):
    def remove(wire):
        wire.points.pop(101)
        raise httpx.RemoteProtocolError("point removed")

    wire = Wire(reads=[remove], owners={101: initial_owner})
    assert upsert(make_store(wire)) == 1
    assert_owner_reads(wire, [101], 2)
    assert wire.points[101]["payload"]["tenant_id"] == "alpha"


def test_unscoped_retry_preserves_mixed_owners_and_new_point(make_store):
    wire = Wire(reads=[httpx.RemoteProtocolError("first")], owners={101: "alpha", 102: "beta"})
    assert upsert(make_store(wire), ids=(101, 102, 103), tenant=None) == 3
    assert_owner_reads(wire, [101, 102, 103], 2)
    assert wire.points[101]["payload"]["tenant_id"] == "alpha"
    assert wire.points[102]["payload"]["tenant_id"] == "beta"
    assert "tenant_id" not in wire.points[103]["payload"]
    assert wire.puts == 1


@pytest.mark.parametrize("persistent", [False, True])
def test_later_batch_failure_keeps_prior_commit(make_store, persistent):
    faults = [None, httpx.RemoteProtocolError("later")]
    if persistent:
        faults.append(httpx.RemoteProtocolError("still closed"))
    wire = Wire(reads=faults)
    store = make_store(wire)
    if persistent:
        with pytest.raises(ResponseHandlingException):
            upsert(store, ids=(101, 102, 103), batch_size=2)
    else:
        assert upsert(store, ids=(101, 102, 103), batch_size=2) == 3
    assert [r["body"]["ids"] for r in wire.reads] == [[101, 102], [103], [103]]
    assert wire.puts == wire.commits == (1 if persistent else 2)
    assert wire.points[101]["payload"]["source"] == "synthetic-source"
    assert wire.points[102]["payload"]["source"] == "synthetic-source"
    assert (103 in wire.points) == (not persistent)


@pytest.mark.parametrize("event", ["committed-rpe", httpx.ReadTimeout("unknown write")])
def test_upsert_unknown_result_never_retried(make_store, event):
    wire = Wire(writes=[event])
    with pytest.raises(ResponseHandlingException):
        upsert(make_store(wire))
    assert_owner_reads(wire, [101], 1)
    assert wire.puts == 1
    assert wire.commits == (1 if event == "committed-rpe" else 0)


def test_generic_retrieve_has_no_retry(make_store):
    wire = Wire(reads=[httpx.RemoteProtocolError("generic")])
    with pytest.raises(ResponseHandlingException):
        make_store(wire).client.retrieve("owner_test", [101])
    assert len(wire.reads) == 1
    assert wire.puts == 0


def test_grpc_error_is_not_retried_or_rerouted_to_rest(make_store):
    class FailedGrpc:
        calls = 0

        def Get(self, request, timeout=None):
            self.calls += 1
            assert len(request.ids) == 1
            assert timeout == 30
            raise grpc.RpcError("synthetic unavailable")

    wire = Wire()
    store = make_store(wire, grpc_mode=True)
    stub = FailedGrpc()
    store.client._client._grpc_points_client_pool = [stub]
    with pytest.raises(grpc.RpcError):
        upsert(store)
    assert stub.calls == 1
    assert wire.requests == []


def test_custom_retry_middleware_is_outside_transport_attempt_bound(make_store):
    wire = Wire(reads=[httpx.RemoteProtocolError("x") for _ in range(4)])
    store = make_store(wire)

    def caller_retry(request, send):
        try:
            return send(request)
        except ResponseHandlingException:
            return send(request)

    store.client.http.client.add_middleware(caller_retry)
    with pytest.raises(ResponseHandlingException):
        upsert(store)
    # Two SDK calls can become four HTTP attempts with caller-added retries.
    # This is a compatibility limitation, not the supported production default.
    assert_owner_reads(wire, [101], 4)
    assert wire.puts == 0


def test_empty_batch_does_not_read_or_write(make_store):
    wire = Wire()
    assert upsert(make_store(wire), ids=()) == 0
    assert wire.requests == []


@pytest.mark.parametrize("tenant", ["alpha", None])
def test_public_single_point_upsert_transient_owner_read(make_store, tenant):
    wire = Wire(reads=[httpx.RemoteProtocolError("transient protocol fault")])
    original = payload()
    before = copy.deepcopy(original)
    make_store(wire).upsert(101, [1.0, 0.0], original, tenant=tenant)
    assert original == before
    assert_owner_reads(wire, [101], 2)
    assert wire.puts == wire.commits == 1
    assert wire.points[101]["payload"]["tenant_id"] == "alpha"
