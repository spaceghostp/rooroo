"""Sub-agent contract: typed task/result, depth caps, structured failures."""

from __future__ import annotations

import pytest

from harness import Budget, Registry, SubAgent, SubAgentResult, SubAgentTask
from harness.errors import (
    AllowlistViolation,
    DepthLimitExceeded,
    SchemaContractError,
)


def test_happy_path(uppercase_subagent):
    res = uppercase_subagent.run({"text": "hi"})
    assert res.ok is True
    assert res.output == {"text": "HI"}


def test_task_validation_returns_structured_error(uppercase_subagent):
    res = uppercase_subagent.run({"not_text": "x"})
    assert res.ok is False
    assert res.error.kind == "task_validation"


def test_depth_2_without_justification_refused(uppercase_subagent):
    """P3: depth 2 requires explicit justification."""
    with pytest.raises(DepthLimitExceeded):
        uppercase_subagent.run({"text": "x"}, depth=2)


def test_depth_2_with_justification_allowed():
    class _Task(SubAgentTask): text: str
    class _Result(SubAgentResult): text: str

    class _Recursive(SubAgent):
        name = "rec"
        description = "Permitted at depth 2."
        task_schema = _Task
        result_schema = _Result
        depth2_justification = "search space is recursive (codebase walk)"

        def _run(self, task, scratch):
            return _Result(text=task.text)

    res = _Recursive().run({"text": "x"}, depth=2)
    assert res.ok is True


def test_depth_3_refused_even_with_justification():
    class _Task(SubAgentTask): pass
    class _Result(SubAgentResult): pass

    class _D2(SubAgent):
        name = "d2"
        description = "OK at depth 2."
        task_schema = _Task
        result_schema = _Result
        depth2_justification = "justified"

        def _run(self, task, scratch):
            return _Result()

    with pytest.raises(DepthLimitExceeded):
        _D2().run({}, depth=3)


def test_missing_schemas_rejected_at_class_creation():
    """Acceptance criterion 7: refuse to register schemaless sub-agents."""
    with pytest.raises(SchemaContractError):
        class _Bad(SubAgent):
            name = "bad"
            # no task_schema or result_schema

            def _run(self, task, scratch):  # pragma: no cover
                return {}


def test_internal_exception_is_caught_not_raised():
    class _Task(SubAgentTask): pass
    class _Result(SubAgentResult): pass

    class _Crashy(SubAgent):
        name = "crashy"
        description = "Raises."
        task_schema = _Task
        result_schema = _Result

        def _run(self, task, scratch):
            raise RuntimeError("nope")

    res = _Crashy().run({})
    assert res.ok is False
    assert res.error.kind == "subagent_internal_error"


# ----- build_inner_registry / carve_budget helpers (§4.2, P6) ----------


def test_build_inner_registry_filters_to_allowlist(echo_tool, boom_tool):
    """The inner Registry exposes only the allowlisted tools, nothing else."""

    class _T(SubAgentTask): pass
    class _R(SubAgentResult): pass

    class _Limited(SubAgent):
        name = "limited"
        description = "Only sees the echo tool."
        task_schema = _T
        result_schema = _R
        tool_allowlist = ("echo",)

        def _run(self, task, scratch):
            return _R()

    parent = Registry()
    parent.register_tool(echo_tool)
    parent.register_tool(boom_tool)

    inner = _Limited().build_inner_registry(parent)
    assert set(inner.tools) == {"echo"}
    # Sub-agents / verifiers do NOT cross the boundary.
    assert inner.subagents == {}
    assert inner.verifiers == {}


def test_build_inner_registry_rejects_unknown_allowlist_entry(echo_tool):
    """An allowlist that references a tool the parent doesn't own is a wiring
    bug; catch it at construction time, not when the inner coordinator tries
    to dispatch and fails opaquely."""

    class _T(SubAgentTask): pass
    class _R(SubAgentResult): pass

    class _Typo(SubAgent):
        name = "typo"
        description = "Allowlist references a non-existent tool."
        task_schema = _T
        result_schema = _R
        tool_allowlist = ("ecoh",)  # typo

        def _run(self, task, scratch):
            return _R()

    parent = Registry()
    parent.register_tool(echo_tool)

    with pytest.raises(AllowlistViolation):
        _Typo().build_inner_registry(parent)


def test_carve_budget_consumption_propagates_to_parent():
    """P6 / §7: sub-budget consumption rolls back to the parent's totals."""

    class _T(SubAgentTask): pass
    class _R(SubAgentResult): pass

    class _Sub(SubAgent):
        name = "carver"
        description = "Carves a sub-budget."
        task_schema = _T
        result_schema = _R

        def _run(self, task, scratch):
            return _R()

    parent_budget = Budget(max_tokens=1000, max_iterations=10, max_wall_seconds=60)
    child = _Sub().carve_budget(parent_budget, max_tokens=200)

    child.consume_tokens(50)
    # Parent now sees the consumption — that's the spec contract.
    assert parent_budget.tokens_used == 50
    # Child's own cap is what we asked for.
    assert child.max_tokens == 200


def test_carve_budget_without_parent_uses_class_defaults():
    """Standalone tests / examples can carve without a parent budget; the
    helper falls back to the sub-agent's default_max_* values."""

    class _T(SubAgentTask): pass
    class _R(SubAgentResult): pass

    class _Defaulty(SubAgent):
        name = "defaulty"
        description = "Defaults-only budget."
        task_schema = _T
        result_schema = _R
        default_max_tokens = 9_999
        default_max_iterations = 7

        def _run(self, task, scratch):
            return _R()

    budget = _Defaulty().carve_budget(None)
    assert budget.max_tokens == 9_999
    assert budget.max_iterations == 7
    assert budget.parent is None  # unparented


def test_run_passes_parent_context_into_scratch(echo_tool):
    """Sub-agent _run sees parent_registry and parent_budget in scratch."""

    seen: dict = {}

    class _T(SubAgentTask): pass
    class _R(SubAgentResult): pass

    class _Inspector(SubAgent):
        name = "inspector"
        description = "Records scratch contents."
        task_schema = _T
        result_schema = _R

        def _run(self, task, scratch):
            seen.update(scratch)
            return _R()

    parent = Registry()
    parent.register_tool(echo_tool)
    parent_budget = Budget(max_tokens=100)

    _Inspector().run({}, parent_registry=parent, parent_budget=parent_budget)
    assert seen["parent_registry"] is parent
    assert seen["parent_budget"] is parent_budget
    assert seen["depth"] == 1
