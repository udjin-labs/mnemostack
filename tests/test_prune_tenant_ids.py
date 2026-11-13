"""Legacy prune preserves the owner scope and native point ID types."""
import pytest

from mnemostack.ingest import prune_stale_chunks


@pytest.mark.parametrize("tenant", [None, "alpha"])
@pytest.mark.parametrize("stale", [17, "017", "f4c117e3-bf52-4a2b-a821-891c583f8cf8"])
def test_prune_preserves_ids_and_scope(tenant, stale):
    calls = []

    class Store:
        def iter_ids(self, **kwargs):
            calls.append(("read", kwargs))
            return iter(["fresh", stale])

        def delete_points(self, ids, **kwargs):
            calls.append(("delete", ids, kwargs))
            return len(ids)

    assert prune_stale_chunks(Store(), {"note.md": {"fresh"}}, index_root="/synthetic", tenant=tenant) == 1
    scope = {"tenant": tenant} if tenant is not None else {}
    assert calls == [("read", {"filters": {"source": "note.md", "index_root": "/synthetic"}, **scope}), ("delete", [stale], scope)]
    assert type(calls[1][1][0]) is type(stale)


def test_prune_compares_integer_ids_to_documented_string_fresh_ids():
    class Store:
        def iter_ids(self, filters):
            return iter([17])

        def delete_points(self, ids):
            pytest.fail("A fresh integer ID must not be deleted")

    assert prune_stale_chunks(Store(), {"note.md": {"17"}}) == 0


def test_unscoped_custom_store_needs_no_tenant_keyword():
    deleted = []

    class Store:
        def iter_ids(self, filters):
            assert filters == {"source": "note.md"}
            return iter([23])

        def delete_points(self, ids):
            deleted.extend(ids)
            return len(ids)

    assert prune_stale_chunks(Store(), {"note.md": set()}) == 1
    assert deleted == [23] and type(deleted[0]) is int


def test_prune_retains_delete_owner_backstop_after_read():
    class Store:
        owner = "alpha"

        def iter_ids(self, filters, tenant):
            assert tenant == self.owner
            self.owner = "beta"
            return iter([31])

        def delete_points(self, ids, tenant=None):
            if tenant is not None and tenant != self.owner:
                raise ValueError("tenant mismatch")
            self.owner = None
            return len(ids)

    store = Store()
    with pytest.raises(ValueError, match="tenant mismatch"):
        prune_stale_chunks(store, {"note.md": set()}, tenant="alpha")
    assert store.owner == "beta"


def test_prune_does_not_retry_failed_delete_or_continue_to_next_source():
    calls = []

    class Store:
        def iter_ids(self, **kwargs):
            calls.append("read")
            return iter([41])

        def delete_points(self, ids, **kwargs):
            calls.append("write")
            raise RuntimeError("uncertain write")

    with pytest.raises(RuntimeError, match="uncertain write"):
        prune_stale_chunks(Store(), {"first.md": set(), "second.md": set()}, tenant="alpha")
    assert calls == ["read", "write"]


def test_prune_empty_map_never_reads_or_deletes():
    assert prune_stale_chunks(object(), {}) == 0
