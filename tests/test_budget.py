"""Budget semantics, including carving sub-budgets."""

from __future__ import annotations

import pytest

from harness import Budget
from harness.errors import BudgetExhausted


def test_iterations_dimension_exhausts():
    b = Budget(max_iterations=2, max_tokens=1000, max_wall_seconds=60)
    b.consume_iteration()
    b.check()  # 1 used, still ok
    b.consume_iteration()
    with pytest.raises(BudgetExhausted) as ei:
        b.check()
    assert ei.value.args[0] == "iterations"


def test_tokens_dimension_exhausts():
    b = Budget(max_iterations=10, max_tokens=10, max_wall_seconds=60)
    b.consume_tokens(10)
    with pytest.raises(BudgetExhausted) as ei:
        b.check()
    assert ei.value.args[0] == "tokens"


def test_carved_child_propagates_token_consumption_to_parent():
    parent = Budget(max_iterations=10, max_tokens=100, max_wall_seconds=60)
    child = parent.carve(max_tokens=50, max_iterations=5, max_wall_seconds=30)
    child.consume_tokens(40)
    # Child still has 10 tokens; parent has 60 left.
    assert child.remaining_tokens() == 10
    assert parent.remaining_tokens() == 60


def test_carved_child_clamps_to_parent_remaining():
    parent = Budget(max_iterations=10, max_tokens=100, max_wall_seconds=60)
    parent.consume_tokens(80)
    child = parent.carve(max_tokens=50)  # asked for 50, parent only has 20 left
    assert child.max_tokens == 20


def test_child_iterations_do_not_propagate_to_parent():
    parent = Budget(max_iterations=10, max_tokens=1000, max_wall_seconds=60)
    child = parent.carve(max_iterations=3)
    child.consume_iteration()
    child.consume_iteration()
    assert child.iterations_used == 2
    assert parent.iterations_used == 0


def test_snapshot_contains_all_dimensions():
    b = Budget(max_tokens=100, max_iterations=10, max_wall_seconds=60)
    b.consume_tokens(5)
    snap = b.snapshot()
    assert snap["tokens_used"] == 5
    assert snap["max_tokens"] == 100
    assert "wall_elapsed" in snap
