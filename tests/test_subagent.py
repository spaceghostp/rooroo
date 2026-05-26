"""Sub-agent contract: typed task/result, depth caps, structured failures."""

from __future__ import annotations

import pytest

from harness import SubAgent, SubAgentResult, SubAgentTask
from harness.errors import DepthLimitExceeded, SchemaContractError


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
