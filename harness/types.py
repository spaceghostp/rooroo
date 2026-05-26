"""Shared types for the agent harness.

The harness keeps three things typed:

- The primitive boundaries (tool/sub-agent input + output) — Pydantic models
- The coordinator's plan output — the ``Action`` discriminated union
- The trace entry shape — see ``trace.py``

Everything that crosses a primitive boundary or gets persisted is a Pydantic
model so we can serialize it to JSON and validate it on the way back in.
"""

from __future__ import annotations

import enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class ActionType(str, enum.Enum):
    """The kinds of work the coordinator can dispatch on a given iteration."""

    PLAN = "plan"
    TOOL = "tool"
    SUBAGENT = "subagent"
    VERIFY = "verify"
    FINISH = "finish"


class Outcome(str, enum.Enum):
    """How a step ended. ``BUDGET`` is a first-class outcome per §7."""

    OK = "ok"
    ERROR = "error"
    BUDGET = "budget"


class Cost(BaseModel):
    """Cost rollup for a single step. Filled in by the executor."""

    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: int = 0
    dollars: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            wall_ms=self.wall_ms + other.wall_ms,
            dollars=self.dollars + other.dollars,
        )


class StructuredError(BaseModel):
    """Tools and sub-agents return this on failure rather than raising.

    Coordinator code can pattern-match on ``kind`` to decide retry strategy.
    """

    kind: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    """Verifier output. See §4.3.

    ``evidence`` is what makes the verdict honest: it is the externally-sourced
    signal the verifier consulted. If the list is empty, the harness flags
    the verifier as commentary, not verification (see ``Verifier.run``).
    """

    passed: bool
    evidence: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] | None = None
    notes: str | None = None


# --- Plan / Action discriminated union -----------------------------------
#
# The planner emits exactly one of these per iteration. The coordinator
# does not interpret freeform planner text — only structured actions.


class ToolAction(BaseModel):
    type: Literal[ActionType.TOOL] = ActionType.TOOL
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class SubAgentAction(BaseModel):
    type: Literal[ActionType.SUBAGENT] = ActionType.SUBAGENT
    subagent: str
    task: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class VerifyAction(BaseModel):
    type: Literal[ActionType.VERIFY] = ActionType.VERIFY
    verifier: str
    target: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class FinishAction(BaseModel):
    type: Literal[ActionType.FINISH] = ActionType.FINISH
    result: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


Action = Union[ToolAction, SubAgentAction, VerifyAction, FinishAction]


class Plan(BaseModel):
    """The planner returns this. ``terminate`` short-circuits the loop."""

    terminate: bool = False
    next_action: Action | None = None
    notes: str | None = None


class SessionResult(BaseModel):
    """What ``Coordinator.run`` returns to the caller.

    ``incomplete`` is set when the loop ended on budget exhaustion or a
    non-recoverable primitive failure — never raised as an exception, per §7.
    """

    intent: str
    result: dict[str, Any] = Field(default_factory=dict)
    incomplete: bool = False
    incomplete_reason: str | None = None
    cost: Cost = Field(default_factory=Cost)
    trace_path: str | None = None
    state_path: str | None = None
    session_id: str
