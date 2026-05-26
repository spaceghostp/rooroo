"""The Coordinator. See §5.

One per session (P1). Runs ``plan → act → observe → verify`` over the
three primitive types. Checks budget before every action, not just at
iteration boundaries. Treats budget exhaustion as a first-class outcome
(P6) and never raises it past ``run()``.

The coordinator owns:

- the planner (which produces a ``Plan`` per iteration),
- the registry (tools/sub-agents/verifiers),
- the state (typed, external) and trace (append-only),
- the budget (with sub-budgets for sub-agents).

The coordinator does NOT own:

- the planner's prompt logic (lives in concrete ``Planner`` subclasses),
- the primitives' internals (they have their own files),
- any cross-session memory (deferred to v2, §6).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import Budget
from .errors import (
    AllowlistViolation,
    BudgetExhausted,
    DepthLimitExceeded,
    RetryBudgetExhausted,
    SchemaContractError,
    UnknownPrimitive,
)
from .planner import PlanRequest, Planner
from .primitives import SubAgent, Tool, Verifier
from .primitives.subagent import SubAgentInvocation
from .primitives.tool import ToolResult
from .primitives.verifier import VerifierResult
from .state import FileSystemState, InMemoryState, State
from .trace import Trace
from .types import (
    Action,
    ActionType,
    Cost,
    FinishAction,
    Outcome,
    SessionResult,
    SubAgentAction,
    ToolAction,
    VerifyAction,
)


@dataclass
class Registry:
    """Holds the primitives available to a coordinator.

    A sub-agent's coordinator gets its own ``Registry`` filtered by the
    sub-agent's ``tool_allowlist`` so the tool surface really is a
    subset (§4.2). Use ``Registry.subset(names)`` (or
    ``SubAgent.build_inner_registry(parent)``) to construct the filtered
    view — that's the canonical way to honor the allowlist.
    """

    tools: dict[str, Tool] = field(default_factory=dict)
    subagents: dict[str, SubAgent] = field(default_factory=dict)
    verifiers: dict[str, Verifier] = field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        if not tool.name:
            raise SchemaContractError("Tool missing name")
        self.tools[tool.name] = tool

    def register_subagent(self, sub: SubAgent) -> None:
        if not sub.name:
            raise SchemaContractError("SubAgent missing name")
        self.subagents[sub.name] = sub

    def register_verifier(self, v: Verifier) -> None:
        if not v.name:
            raise SchemaContractError("Verifier missing name")
        self.verifiers[v.name] = v

    def manifest(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "tools": [t.manifest() for t in self.tools.values()],
            "subagents": [s.manifest() for s in self.subagents.values()],
            "verifiers": [v.manifest() for v in self.verifiers.values()],
        }

    def subset(self, tool_names: "tuple[str, ...] | list[str]") -> "Registry":
        """Return a new Registry holding only the named tools.

        Sub-agents and verifiers are NOT carried across: each sub-agent
        builds its own inner sub-agent/verifier set as part of declaring
        its loop. The tool surface is the only thing P-2 lets a sub-agent
        inherit from its parent.

        Raises ``AllowlistViolation`` if any name is not present in this
        registry — that's a typo or a stale allowlist, and we'd rather
        catch it at coordinator construction than silently drop a tool.
        """
        missing = [n for n in tool_names if n not in self.tools]
        if missing:
            raise AllowlistViolation(
                f"tool_allowlist references tools not in parent registry: {missing}"
            )
        return Registry(tools={n: self.tools[n] for n in tool_names})


class Coordinator:
    """The single coordinator per session.

    Construct with a planner, a registry, and (optionally) a session
    directory for filesystem-backed state + trace. Call ``run(intent)``
    to drive the loop.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        registry: Registry,
        budget: Budget | None = None,
        session_dir: str | Path | None = None,
        depth: int = 0,
        session_id: str | None = None,
    ) -> None:
        self.planner = planner
        self.registry = registry
        self.budget = budget or Budget()
        self.depth = depth
        self.session_id = session_id or uuid.uuid4().hex

        if session_dir is not None:
            session_path = Path(session_dir)
            session_path.mkdir(parents=True, exist_ok=True)
            self.state: State = FileSystemState(session_path / "state")
            self.trace = Trace(path=session_path / "trace.jsonl")
            self._session_path: Path | None = session_path
        else:
            self.state = InMemoryState()
            self.trace = Trace()
            self._session_path = None

        # §4.3: cap each verifier at its ``max_retries`` failed verdicts in
        # a single session. After that, the loop terminates with
        # ``incomplete=True`` instead of looping forever on a stuck verdict.
        self._verifier_failures: dict[str, int] = {}

    # ---- main loop ------------------------------------------------------

    def run(self, intent: str) -> SessionResult:
        last_observation: dict[str, Any] | None = None
        incomplete = False
        incomplete_reason: str | None = None
        finish_result: dict[str, Any] = {}

        try:
            while True:
                # Budget check #1: before planning.
                self._require_budget()

                self.budget.consume_iteration()

                request = PlanRequest(
                    intent=intent,
                    iteration=self.budget.iterations_used,
                    state_summary=self.state.summary(),
                    tools=[t.manifest() for t in self.registry.tools.values()],
                    subagents=[s.manifest() for s in self.registry.subagents.values()],
                    verifiers=[v.manifest() for v in self.registry.verifiers.values()],
                    last_observation=last_observation,
                    budget_snapshot=self.budget.snapshot(),
                )

                plan = self.planner.plan(request)

                # Trace the plan itself — it is a coordinator decision per §8.
                self.trace.append(
                    action_type=ActionType.PLAN,
                    action_input={
                        "iteration": request.iteration,
                        "intent": intent,
                    },
                    action_output=plan.model_dump(),
                    cost=Cost(),  # planner cost is its own concern
                    outcome=Outcome.OK,
                    notes=plan.notes,
                )

                if plan.terminate or plan.next_action is None or isinstance(
                    plan.next_action, FinishAction
                ):
                    if isinstance(plan.next_action, FinishAction):
                        finish_result = plan.next_action.result
                    break

                # Budget check #2: between planning and acting.
                self._require_budget()

                last_observation = self._execute(plan.next_action)

        except BudgetExhausted as e:
            incomplete = True
            incomplete_reason = f"budget_exhausted:{e.args[0] if e.args else 'unknown'}"
            self.trace.append(
                action_type=ActionType.FINISH,
                action_input={},
                action_output={"reason": incomplete_reason},
                cost=Cost(),
                outcome=Outcome.BUDGET,
            )
        except RetryBudgetExhausted as e:
            incomplete = True
            incomplete_reason = f"retry_budget_exhausted:{e.args[0] if e.args else 'unknown'}"
            self.trace.append(
                action_type=ActionType.FINISH,
                action_input={},
                action_output={"reason": incomplete_reason},
                cost=Cost(),
                outcome=Outcome.ERROR,
            )
        except DepthLimitExceeded as e:
            # P3 is a hard refusal. We do NOT swallow it — it should be
            # impossible at runtime if the harness is wired correctly.
            # But we DO ensure the trace records it before re-raising.
            self.trace.append(
                action_type=ActionType.FINISH,
                action_input={},
                action_output={"reason": str(e)},
                cost=Cost(),
                outcome=Outcome.ERROR,
            )
            raise

        return SessionResult(
            intent=intent,
            result=finish_result,
            incomplete=incomplete,
            incomplete_reason=incomplete_reason,
            cost=self.trace.cost_rollup(),
            trace_path=self.trace.path,
            state_path=str(self._session_path / "state") if self._session_path else None,
            session_id=self.session_id,
        )

    # ---- execution per action type --------------------------------------

    def _execute(self, action: Action) -> dict[str, Any]:
        """Dispatch one action, record the trace entry, and return a
        structured observation the planner will see on the next turn."""

        if isinstance(action, ToolAction):
            return self._execute_tool(action)
        if isinstance(action, SubAgentAction):
            return self._execute_subagent(action)
        if isinstance(action, VerifyAction):
            return self._execute_verify(action)
        # FinishAction handled in the loop; never reaches here.
        raise AssertionError(f"Unhandled action type: {type(action).__name__}")

    def _execute_tool(self, action: ToolAction) -> dict[str, Any]:
        tool = self.registry.tools.get(action.tool)
        if tool is None:
            raise UnknownPrimitive(f"tool '{action.tool}' not registered")
        result: ToolResult = tool.call(action.arguments)
        # Tool wall_ms is the only fillable cost component for tools.
        self.budget.consume_tokens(0)
        self.trace.append(
            action_type=ActionType.TOOL,
            action_input={"tool": action.tool, "arguments": action.arguments,
                          "rationale": action.rationale},
            action_output=result.model_dump(),
            cost=result.cost,
            outcome=result.outcome,
        )
        return result.model_dump()

    def _execute_subagent(self, action: SubAgentAction) -> dict[str, Any]:
        sub = self.registry.subagents.get(action.subagent)
        if sub is None:
            raise UnknownPrimitive(f"sub-agent '{action.subagent}' not registered")
        # depth==1 for the parent's first-level sub-agents (the parent is
        # the depth-0 coordinator). A sub-agent's *own* coordinator would
        # invoke at depth=2.
        invocation_depth = self.depth + 1
        # Pass the parent's registry + budget so the sub-agent can build a
        # filtered inner registry (§4.2) and carve a sub-budget (§7).
        result: SubAgentInvocation = sub.run(
            action.task,
            depth=invocation_depth,
            parent_registry=self.registry,
            parent_budget=self.budget,
        )
        self.trace.append(
            action_type=ActionType.SUBAGENT,
            action_input={
                "subagent": action.subagent,
                "task": action.task,
                "rationale": action.rationale,
                "depth": invocation_depth,
            },
            action_output=result.model_dump(),
            cost=result.cost,
            outcome=result.outcome,
        )
        return result.model_dump()

    def _execute_verify(self, action: VerifyAction) -> dict[str, Any]:
        verifier = self.registry.verifiers.get(action.verifier)
        if verifier is None:
            raise UnknownPrimitive(f"verifier '{action.verifier}' not registered")
        result: VerifierResult = verifier.run(action.target)
        self.trace.append(
            action_type=ActionType.VERIFY,
            action_input={
                "verifier": action.verifier,
                "target": action.target,
                "rationale": action.rationale,
            },
            action_output=result.model_dump(),
            cost=result.cost,
            outcome=result.outcome,
        )
        # §4.3: count failed verdicts (including grounding-demoted ones)
        # per verifier and halt the loop once we cross ``max_retries``.
        if result.ok and result.verdict is not None and not result.verdict.passed:
            count = self._verifier_failures.get(action.verifier, 0) + 1
            self._verifier_failures[action.verifier] = count
            if count > verifier.max_retries:
                raise RetryBudgetExhausted(action.verifier)
        return result.model_dump()

    # ---- helpers --------------------------------------------------------

    def _require_budget(self) -> None:
        self.budget.check()
