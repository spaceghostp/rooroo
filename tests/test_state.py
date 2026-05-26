"""State accessor behavior + change-log honesty."""

from __future__ import annotations

import pytest

from harness import FileSystemState, InMemoryState


def test_inmemory_put_get_roundtrip():
    s = InMemoryState()
    s.put("doc", {"title": "hi"})
    assert s.get("doc") == {"title": "hi"}


def test_inmemory_update_merges():
    s = InMemoryState({"doc": {"title": "hi"}})
    s.update("doc", {"author": "x"})
    assert s.get("doc") == {"title": "hi", "author": "x"}


def test_inmemory_update_on_nondict_raises():
    s = InMemoryState({"doc": "string"})
    with pytest.raises(TypeError):
        s.update("doc", {"k": "v"})


def test_change_log_records_each_mutation():
    s = InMemoryState()
    s.put("a", 1)
    s.put("b", 2)
    s.update("c", {"x": 3})
    s.delete("a")
    changes = s.changes_since(0)
    assert [c.op for c in changes] == ["put", "put", "update", "delete"]
    assert changes[0].before is None
    assert changes[0].after == 1
    assert changes[3].before == 1 and changes[3].after is None


def test_get_returns_deep_copy_not_reference():
    """Important for §5: state must be observable, not a shared mutable."""
    s = InMemoryState({"doc": {"nested": [1, 2]}})
    fetched = s.get("doc")
    fetched["nested"].append(3)
    assert s.get("doc")["nested"] == [1, 2]


def test_filesystem_state_roundtrip(tmp_path):
    s = FileSystemState(tmp_path)
    s.put("plan/v1", {"steps": [1, 2, 3]})
    assert s.get("plan/v1") == {"steps": [1, 2, 3]}
    assert "plan/v1" in list(s.keys())


def test_filesystem_state_delete(tmp_path):
    s = FileSystemState(tmp_path)
    s.put("k", "v")
    s.delete("k")
    assert s.get("k") is None
