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
