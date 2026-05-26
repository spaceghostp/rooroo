"""Planner abstraction — the LLM-shaped piece of the coordinator.

A ``Planner`` is the thing that, given (intent, state-summary, registry
manifest), returns a structured ``Plan``. The coordinator never gives the
planner the raw trace; only the typed state summary and the most recent
observations (per §5: 'Trace is append-only and never read by the
coordinator's plan step except as explicitly summarized state').

Two planners ship:

- ``ScriptedPlanner``: emits a pre-recorded sequence of actions. Tests
  and examples use this so we have no LLM dependency. It's also exactly
  the right shape for replay (acceptance criterion 5).
- ``LLMPlannerBase``: a tiny abstract class for real LLM-backed planners.
  Concrete subclasses (per-provider) live outside this package.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Action, FinishAction, Plan


@dataclass
class PlanRequest:
    """The structured snapshot a planner gets each iteration.

    The fields here are exactly what the spec permits the planner to see:
    intent, current state summary, registry manifest of available
    primitives, last observation, iteration index. Notably absent: the
    full trace. That's deliberate (§5).
    """

    intent: str
    iteration: int
    state_summary: dict[str, Any]
    tools: list[dict[str, Any]]
    subagents: list[dict[str, Any]]
    verifiers: list[dict[str, Any]]
    last_observation: dict[str, Any] | None = None
    budget_snapshot: dict[str, Any] = field(default_factory=dict)


class Planner(abc.ABC):
    """Returns a ``Plan`` for the next iteration of the coordinator loop."""

    @abc.abstractmethod
    def plan(self, request: PlanRequest) -> Plan: ...


class ScriptedPlanner(Planner):
    """Emits a pre-recorded sequence of Actions, then terminates.

    Useful for tests, examples, and replay. The script is a list of
    ``Action`` instances; once the list is exhausted, the planner returns
    a terminating ``FinishAction`` with whatever ``finish_result`` was set.
    """

    def __init__(
        self,
        actions: list[Action],
        finish_result: dict[str, Any] | None = None,
        on_observation: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self._actions = list(actions)
        self._finish_result = finish_result or {}
        self._on_observation = on_observation
        self._index = 0

    def plan(self, request: PlanRequest) -> Plan:
        if self._on_observation is not None:
            self._on_observation(request.last_observation)
        if self._index >= len(self._actions):
            return Plan(
                terminate=True,
                next_action=FinishAction(result=self._finish_result),
            )
        action = self._actions[self._index]
        self._index += 1
        terminate = isinstance(action, FinishAction)
        return Plan(terminate=terminate, next_action=action)


class LLMPlannerBase(Planner):
    """Skeleton for LLM-backed planners.

    Concrete implementations (e.g. for the Anthropic SDK) live outside the
    harness package — they belong with the per-provider code so the core
    stays dependency-free. A subclass implements ``_call_model`` returning
    the raw structured action dict; this base handles validation into the
    ``Plan`` shape.
    """

    @abc.abstractmethod
    def _call_model(self, request: PlanRequest) -> dict[str, Any]: ...

    def plan(self, request: PlanRequest) -> Plan:
        raw = self._call_model(request)
        # Defer to Pydantic to discriminate the union by ``type``:
        return Plan.model_validate(raw)
