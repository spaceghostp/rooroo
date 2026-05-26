"""End-to-end demo: tool + sub-agent + verifier in one session.

Run with:

    python -m examples.echo_session

The scripted planner walks through:

  1. Call ``add`` tool to compute 2 + 3
  2. Delegate to ``classify`` sub-agent on the sum
  3. Verify the sub-agent's output with ``schema_check`` verifier
  4. Finish with a structured result

This is the same shape a real LLM-backed coordinator would produce — just
with the actions hand-rolled so the example has no API dependency.

Note on the sub-agent: ``ClassifySubAgent`` is a deliberately trivial
``_run`` (a static if/elif over an int) so this example can stay the
smallest possible primitives-coverage demo. The SPEC §4.2 "When NOT to
use a sub-agent" test would correctly say "this should be a tool" if
the rest of the system were real. For the canonical depth-1 sub-agent
shape — inner coordinator, allowlist-filtered registry, carved
budget — see ``examples/bugfix_session.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    Tool,
    ToolAction,
    Verdict,
    Verifier,
    VerifyAction,
)


# -------- Tool: add ----------------------------------------------------------


class AddIn(BaseModel):
    a: int
    b: int


class AddOut(BaseModel):
    sum: int


class AddTool(Tool):
    name = "add"
    description = "Add two integers."
    input_schema = AddIn
    output_schema = AddOut
    side_effects = ()  # pure

    def _execute(self, payload: AddIn) -> AddOut:
        return AddOut(sum=payload.a + payload.b)


# -------- SubAgent: classify -------------------------------------------------


class ClassifyTask(SubAgentTask):
    value: int


class ClassifyResult(SubAgentResult):
    label: str
    value: int


class ClassifySubAgent(SubAgent):
    name = "classify"
    description = "Label an integer as 'small', 'medium', or 'large'."
    task_schema = ClassifyTask
    result_schema = ClassifyResult

    def _run(self, task: ClassifyTask, scratch: dict) -> ClassifyResult:
        if task.value < 10:
            label = "small"
        elif task.value < 100:
            label = "medium"
        else:
            label = "large"
        return ClassifyResult(label=label, value=task.value)


# -------- Verifier: schema_check (grounded on schema_validate) ---------------


class SchemaCheckTarget(BaseModel):
    label: str
    value: int


class SchemaCheckVerifier(Verifier):
    name = "schema_check"
    description = "Verify the classifier output adheres to the expected shape."
    target_schema = SchemaCheckTarget
    evidence_sources = ("schema_validate",)

    def _verify(self, target: SchemaCheckTarget) -> Verdict:
        # Touch external evidence: the (separately maintained) canonical
        # vocabulary of allowed labels. In a real verifier this would be
        # a file fetch or DB query; here it's an inline constant whose
        # MEMBERSHIP CHECK constitutes the grounded signal.
        allowed = {"small", "medium", "large"}
        if target.label in allowed:
            return Verdict(
                passed=True,
                evidence=[
                    f"schema_validate: target conforms to SchemaCheckTarget",
                    f"vocabulary_check: '{target.label}' ∈ {sorted(allowed)}",
                ],
            )
        return Verdict(
            passed=False,
            evidence=[f"vocabulary_check: '{target.label}' ∉ {sorted(allowed)}"],
            suggested_revisions=[f"Use one of {sorted(allowed)}"],
        )


def main() -> None:
    registry = Registry()
    registry.register_tool(AddTool())
    registry.register_subagent(ClassifySubAgent())
    registry.register_verifier(SchemaCheckVerifier())

    # The scripted planner emits the action sequence and observes results
    # along the way. ``finish_result`` is what gets bundled into the
    # ``SessionResult`` if the script runs out.
    actions = [
        ToolAction(tool="add", arguments={"a": 2, "b": 3}, rationale="initial compute"),
        SubAgentAction(
            subagent="classify",
            task={"value": 5},
            rationale="delegate labelling",
        ),
        VerifyAction(
            verifier="schema_check",
            target={"label": "small", "value": 5},
            rationale="gate the deliverable",
        ),
        FinishAction(result={"label": "small", "value": 5}),
    ]

    planner = ScriptedPlanner(actions=actions, finish_result={"label": "small", "value": 5})
    coordinator = Coordinator(
        planner=planner,
        registry=registry,
        budget=Budget(max_iterations=10, max_tokens=10_000, max_wall_seconds=30),
        session_dir=Path(".sessions/echo"),
    )

    result = coordinator.run("Compute 2+3 and classify the result.")
    print(json.dumps(result.model_dump(), indent=2, default=str))
    print(f"\nTrace at: {result.trace_path}")
    print(f"State at: {result.state_path}")
    print(f"Steps recorded: {len(coordinator.trace)}")


if __name__ == "__main__":
    main()
