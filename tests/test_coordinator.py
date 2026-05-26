"""Coordinator loop semantics: planning, dispatch, trace, budget."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Registry,
    ScriptedPlanner,
    SubAgentAction,
    ToolAction,
    Verdict,
    Verifier,
    VerifyAction,
)
from harness.errors import UnknownPrimitive
from harness.types import ActionType, Outcome


def _make_registry(echo_tool, uppercase_subagent, nonempty_verifier):
    r = Registry()
    r.register_tool(echo_tool)
    r.register_subagent(uppercase_subagent)
    r.register_verifier(nonempty_verifier)
    return r


def test_runs_full_session_with_all_three_primitives(
    echo_tool, uppercase_subagent, nonempty_verifier
):
    """Acceptance criterion 2: invoke a tool, a sub-agent, AND a verifier."""
    registry = _make_registry(echo_tool, uppercase_subagent, nonempty_verifier)

    actions = [
        ToolAction(tool="echo", arguments={"text": "hi"}),
        SubAgentAction(subagent="uppercase", task={"text": "hi"}),
        VerifyAction(verifier="nonempty", target={"text": "HI"}),
        FinishAction(result={"final": "HI"}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions, finish_result={"final": "HI"}),
        registry=registry,
        budget=Budget(max_iterations=10),
    )
    res = coord.run("test")

    # Acceptance criterion 1: structured result + trace
    assert res.result == {"final": "HI"}
    assert res.incomplete is False
    assert len(coord.trace) >= 4  # plan steps + executed actions

    types_seen = {e.action_type for e in coord.trace.entries()}
    assert ActionType.TOOL in types_seen
    assert ActionType.SUBAGENT in types_seen
    assert ActionType.VERIFY in types_seen


def test_budget_exhaustion_yields_partial_result_not_exception(echo_tool):
    """Acceptance criterion 3: clean termination on budget exhaustion."""
    registry = Registry()
    registry.register_tool(echo_tool)

    # max_iterations=1 means: plan+act once and the next plan check fails.
    actions = [
        ToolAction(tool="echo", arguments={"text": "a"}),
        ToolAction(tool="echo", arguments={"text": "b"}),
        ToolAction(tool="echo", arguments={"text": "c"}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
        budget=Budget(max_iterations=2, max_tokens=10_000, max_wall_seconds=60),
    )
    res = coord.run("test")
    assert res.incomplete is True
    assert res.incomplete_reason is not None
    assert "iterations" in res.incomplete_reason
    # Final trace entry records the budget outcome.
    last = coord.trace.entries()[-1]
    assert last.outcome == Outcome.BUDGET


def test_unknown_tool_raises_unknown_primitive(uppercase_subagent):
    registry = Registry()
    registry.register_subagent(uppercase_subagent)
    coord = Coordinator(
        planner=ScriptedPlanner(actions=[ToolAction(tool="missing", arguments={})]),
        registry=registry,
    )
    with pytest.raises(UnknownPrimitive):
        coord.run("test")


def test_trace_is_appended_in_action_order(
    echo_tool, uppercase_subagent, nonempty_verifier
):
    registry = _make_registry(echo_tool, uppercase_subagent, nonempty_verifier)
    actions = [
        ToolAction(tool="echo", arguments={"text": "x"}),
        SubAgentAction(subagent="uppercase", task={"text": "x"}),
        FinishAction(result={}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
    )
    coord.run("test")
    types = [e.action_type for e in coord.trace.entries()]
    # plan, tool, plan, subagent, plan(final w/ finish)
    assert types.count(ActionType.PLAN) >= 3
    assert types.index(ActionType.TOOL) < types.index(ActionType.SUBAGENT)


def test_filesystem_session_persists_state_and_trace(
    tmp_path, echo_tool, uppercase_subagent, nonempty_verifier
):
    registry = _make_registry(echo_tool, uppercase_subagent, nonempty_verifier)
    actions = [ToolAction(tool="echo", arguments={"text": "x"}), FinishAction(result={})]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
        session_dir=tmp_path,
    )
    res = coord.run("test")
    assert res.trace_path is not None
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "state").is_dir()


def test_planner_does_not_see_full_trace():
    """§5: 'Trace is append-only and never read by the coordinator's plan step
    except as explicitly summarized state.'

    We verify by checking that ``PlanRequest`` has no trace field at all.
    """
    from harness.planner import PlanRequest

    fields = set(PlanRequest.__dataclass_fields__.keys())
    assert "trace" not in fields
    assert "trace_entries" not in fields
    assert "transcript" not in fields


# ----- §4.3: verifier max_retries is enforced by the coordinator --------


class _AlwaysFailsTarget(BaseModel):
    note: str = ""


class _AlwaysFailsVerifier(Verifier):
    """Returns a grounded failed verdict every time. Used to prove the
    coordinator caps the loop at ``max_retries`` failures (§4.3)."""

    name = "always_fails"
    description = "Always returns passed=False, with evidence."
    target_schema = _AlwaysFailsTarget
    evidence_sources = ("schema_validate",)
    max_retries = 2  # so the third failure trips the cap

    def _verify(self, target):
        return Verdict(
            passed=False,
            evidence=["schema_validate: synthetic failure"],
            suggested_revisions=["impossible to satisfy"],
        )


def test_coordinator_caps_verifier_retries():
    """After ``max_retries`` failed verdicts on the same verifier, the
    coordinator terminates with ``incomplete=True`` regardless of what
    the planner emits next.

    With ``max_retries=2``, the third verify call exhausts the budget
    and the loop ends.
    """
    registry = Registry()
    registry.register_verifier(_AlwaysFailsVerifier())

    # Planner emits four verify actions; the coordinator should cut us
    # off after the third fail (i.e. the third dispatched verify).
    actions = [
        VerifyAction(verifier="always_fails", target={"note": "try 1"}),
        VerifyAction(verifier="always_fails", target={"note": "try 2"}),
        VerifyAction(verifier="always_fails", target={"note": "try 3"}),
        VerifyAction(verifier="always_fails", target={"note": "try 4"}),
        FinishAction(result={"unreachable": True}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions, finish_result={"unreachable": True}),
        registry=registry,
        budget=Budget(max_iterations=20, max_tokens=10_000, max_wall_seconds=10),
    )
    res = coord.run("retry-cap test")

    assert res.incomplete is True
    assert (res.incomplete_reason or "").startswith("retry_budget_exhausted:")
    assert "always_fails" in (res.incomplete_reason or "")
    # Exactly three verify dispatches landed on the trace.
    from harness.types import ActionType
    assert len(coord.trace.by_type(ActionType.VERIFY)) == 3


def test_coordinator_retry_counter_only_counts_failures():
    """A passing verdict does not consume the retry budget."""

    class _AlternateTarget(BaseModel):
        attempt: int

    class _AlternatesVerifier(Verifier):
        """Fails on even-numbered attempts, passes on odd ones."""

        name = "alternates"
        description = "Pass/fail based on attempt index."
        target_schema = _AlternateTarget
        evidence_sources = ("schema_validate",)
        max_retries = 1  # only one failure permitted

        def _verify(self, target):
            if target.attempt % 2 == 0:
                return Verdict(
                    passed=False,
                    evidence=[f"schema_validate: attempt {target.attempt} fails"],
                )
            return Verdict(
                passed=True,
                evidence=[f"schema_validate: attempt {target.attempt} passes"],
            )

    registry = Registry()
    registry.register_verifier(_AlternatesVerifier())

    # fail(0) -> pass(1) -> fail(2). Only two failures total, but they
    # are interleaved with a success. The counter does not reset on
    # success — both failures count toward max_retries=1, so the second
    # failure ends the session.
    actions = [
        VerifyAction(verifier="alternates", target={"attempt": 0}),
        VerifyAction(verifier="alternates", target={"attempt": 1}),
        VerifyAction(verifier="alternates", target={"attempt": 2}),
        VerifyAction(verifier="alternates", target={"attempt": 3}),
        FinishAction(result={}),
    ]
    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
    )
    res = coord.run("interleaved-retry test")

    assert res.incomplete is True
    assert "retry_budget_exhausted:alternates" in (res.incomplete_reason or "")
