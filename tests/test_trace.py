"""Trace is append-only, immutable, and persists to JSONL."""

from __future__ import annotations

from harness import Cost, Outcome, Trace
from harness.types import ActionType


def test_append_returns_entry_with_step_id():
    t = Trace()
    entry = t.append(
        action_type=ActionType.TOOL,
        action_input={"tool": "x"},
        action_output={"ok": True},
        cost=Cost(),
        outcome=Outcome.OK,
    )
    assert entry.step_id
    assert entry.action_type == ActionType.TOOL


def test_entries_are_copied_on_read():
    t = Trace()
    t.append(
        action_type=ActionType.TOOL,
        action_input={},
        action_output={},
        cost=Cost(),
        outcome=Outcome.OK,
    )
    listing = t.entries()
    listing.clear()
    assert len(t) == 1  # internal list unaffected


def test_cost_rollup_sums_across_entries():
    t = Trace()
    for _ in range(3):
        t.append(
            action_type=ActionType.TOOL,
            action_input={},
            action_output={},
            cost=Cost(tokens_in=1, tokens_out=2, wall_ms=10, dollars=0.1),
            outcome=Outcome.OK,
        )
    total = t.cost_rollup()
    assert total.tokens_in == 3
    assert total.tokens_out == 6
    assert total.wall_ms == 30
    assert abs(total.dollars - 0.3) < 1e-9


def test_jsonl_persistence_and_reload(tmp_path):
    path = tmp_path / "trace.jsonl"
    t = Trace(path=path)
    t.append(
        action_type=ActionType.TOOL,
        action_input={"a": 1},
        action_output={"b": 2},
        cost=Cost(tokens_in=5),
        outcome=Outcome.OK,
        notes="hi",
    )
    reloaded = Trace.load(path)
    assert len(reloaded) == 1
    e = reloaded.entries()[0]
    assert e.action_input == {"a": 1}
    assert e.notes == "hi"


def test_by_type_filter():
    t = Trace()
    for at in (ActionType.PLAN, ActionType.TOOL, ActionType.TOOL, ActionType.FINISH):
        t.append(
            action_type=at,
            action_input={},
            action_output={},
            cost=Cost(),
            outcome=Outcome.OK,
        )
    assert len(t.by_type(ActionType.TOOL)) == 2


def test_trace_appends_across_sessions_on_same_path(tmp_path):
    """§8: trace is append-only across runs sharing a session_dir.

    A second ``Trace(path)`` against an existing file must load prior
    entries (so cost rollups and ``by_type`` queries see history) and
    keep appending — not truncate.
    """
    path = tmp_path / "trace.jsonl"

    # First "run": write two entries.
    t1 = Trace(path=path)
    t1.append(
        action_type=ActionType.TOOL,
        action_input={"run": 1, "i": 0},
        action_output={},
        cost=Cost(tokens_in=1),
        outcome=Outcome.OK,
    )
    t1.append(
        action_type=ActionType.TOOL,
        action_input={"run": 1, "i": 1},
        action_output={},
        cost=Cost(tokens_in=2),
        outcome=Outcome.OK,
    )

    # Second "run": open the same path. Prior entries must be visible.
    t2 = Trace(path=path)
    assert len(t2) == 2
    assert t2.entries()[0].action_input == {"run": 1, "i": 0}
    t2.append(
        action_type=ActionType.TOOL,
        action_input={"run": 2, "i": 0},
        action_output={},
        cost=Cost(tokens_in=4),
        outcome=Outcome.OK,
    )

    # On-disk: three entries total. Cost rollup spans both runs.
    reloaded = Trace.load(path)
    assert len(reloaded) == 3
    assert t2.cost_rollup().tokens_in == 1 + 2 + 4
