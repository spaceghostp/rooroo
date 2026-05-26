"""Acceptance criteria §10 of the spec, one test per criterion.

These tests serve as the conformance checklist. If you change the
harness, these must still pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Registry,
    ScriptedPlanner,
    SubAgent,
    SubAgentAction,
    SubAgentResult,
    SubAgentTask,
    ToolAction,
    Trace,
    Verdict,
    Verifier,
    VerifyAction,
)
from harness.errors import DepthLimitExceeded, SchemaContractError
from harness.types import ActionType, Outcome


# ----- AC1: one coordinator, structured result with full trace ----------


def test_ac1_session_returns_structured_result_with_trace(
    tmp_path, echo_tool, uppercase_subagent, nonempty_verifier
):
    registry = Registry()
    registry.register_tool(echo_tool)
    registry.register_subagent(uppercase_subagent)
    registry.register_verifier(nonempty_verifier)

    coord = Coordinator(
        planner=ScriptedPlanner(
            actions=[
                ToolAction(tool="echo", arguments={"text": "hi"}),
                FinishAction(result={"done": True}),
            ],
            finish_result={"done": True},
        ),
        registry=registry,
        session_dir=tmp_path,
    )
    res = coord.run("AC1")

    assert res.session_id
    assert res.result == {"done": True}
    assert res.trace_path is not None
    assert Path(res.trace_path).exists()
    assert len(coord.trace) > 0


# ----- AC2: invoke a tool, sub-agent, AND verifier in one session -------


def test_ac2_all_three_primitives_in_one_session(
    echo_tool, uppercase_subagent, nonempty_verifier
):
    registry = Registry()
    registry.register_tool(echo_tool)
    registry.register_subagent(uppercase_subagent)
    registry.register_verifier(nonempty_verifier)

    actions = [
        ToolAction(tool="echo", arguments={"text": "x"}),
        SubAgentAction(subagent="uppercase", task={"text": "x"}),
        VerifyAction(verifier="nonempty", target={"text": "X"}),
        FinishAction(result={}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
    )
    coord.run("AC2")

    seen = {e.action_type for e in coord.trace.entries()}
    assert {ActionType.TOOL, ActionType.SUBAGENT, ActionType.VERIFY}.issubset(seen)


# ----- AC3: terminate cleanly on budget exhaustion with a partial result


def test_ac3_budget_exhaustion_returns_partial_result(echo_tool):
    registry = Registry()
    registry.register_tool(echo_tool)
    coord = Coordinator(
        planner=ScriptedPlanner(
            actions=[
                ToolAction(tool="echo", arguments={"text": "a"}),
                ToolAction(tool="echo", arguments={"text": "b"}),
            ],
        ),
        registry=registry,
        budget=Budget(max_iterations=1),
    )
    res = coord.run("AC3")
    assert res.incomplete is True
    assert "iterations" in (res.incomplete_reason or "")


# ----- AC4: refuse depth-violating sub-agent invocations ----------------


def test_ac4_depth_limit_refusal():
    """No sub-agent may be invoked at depth 3+. Depth 2 needs justification."""

    class _T(SubAgentTask): pass
    class _R(SubAgentResult): pass

    class _Plain(SubAgent):
        name = "plain"
        description = "depth-1 only"
        task_schema = _T
        result_schema = _R

        def _run(self, task, scratch):
            return _R()

    with pytest.raises(DepthLimitExceeded):
        _Plain().run({}, depth=2)
    with pytest.raises(DepthLimitExceeded):
        _Plain().run({}, depth=3)


# ----- AC5: trace is deterministically replayable given the same outputs


def test_ac5_trace_replay_from_disk_is_deterministic(
    tmp_path, echo_tool, uppercase_subagent
):
    registry = Registry()
    registry.register_tool(echo_tool)
    registry.register_subagent(uppercase_subagent)

    actions = [
        ToolAction(tool="echo", arguments={"text": "a"}),
        SubAgentAction(subagent="uppercase", task={"text": "a"}),
        FinishAction(result={"x": 1}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions, finish_result={"x": 1}),
        registry=registry,
        session_dir=tmp_path,
    )
    res = coord.run("AC5")

    # Load the trace back; it should round-trip with identical structured
    # inputs/outputs (timestamps and step_ids excepted).
    reloaded = Trace.load(res.trace_path)
    assert len(reloaded) == len(coord.trace)
    for original, restored in zip(coord.trace.entries(), reloaded.entries()):
        assert original.action_type == restored.action_type
        assert original.action_input == restored.action_input
        assert original.action_output == restored.action_output
        assert original.outcome == restored.outcome


# ----- AC6: verifier-grounding audit ------------------------------------


def test_ac6_verifier_grounding_audit_all_retry_eligible_touch_evidence(
    nonempty_verifier, commentary_verifier
):
    """Every retry-eligible verifier must demonstrably touch evidence.

    'retry-eligible' = ``passed=False`` triggers retries. We audit by running
    the verifier and checking that the verdict in the failing case carries
    at least one evidence item. The commentary verifier (which would be
    rejected at registration time if we required ``evidence`` at class
    declaration; here we catch it at verdict time) must NEVER appear as
    ``passed=True`` without evidence.
    """
    # Real verifier: failure case still has evidence.
    res = nonempty_verifier.run({"text": ""})
    assert res.verdict.passed is False
    assert res.verdict.evidence, "retry-eligible verdict must carry evidence"

    # Commentary verifier: claims pass with no evidence; harness demotes it.
    res2 = commentary_verifier.run({"text": "hi"})
    assert res2.verdict.passed is False
    assert "GROUNDING_FAILED" in (res2.verdict.notes or "")


# ----- AC7: reject schemaless sub-agent specs at registration time -------


def test_ac7_schemaless_subagent_rejected_at_class_creation():
    with pytest.raises(SchemaContractError):
        class _Bad(SubAgent):
            name = "bad"
            description = "missing schemas"
            # NO task_schema, NO result_schema

            def _run(self, task, scratch):  # pragma: no cover
                return {}
