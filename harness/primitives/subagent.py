"""Sub-agents — bounded reasoning units invoked as typed functions. See §4.2.

A sub-agent runs *its own* coordinator loop with an independent context,
its own system prompt, and a tool allowlist. From the parent's POV it's
a typed function: structured task in, structured result out.

Depth enforcement (P3): a sub-agent at depth 1 may not invoke another
sub-agent unless ``depth2_justification`` is non-empty. No depth 3.
"""

from __future__ import annotations

import abc
import time
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import DepthLimitExceeded, SchemaContractError
from ..types import Cost, Outcome, StructuredError

TaskModel = TypeVar("TaskModel", bound=BaseModel)
ResultModel = TypeVar("ResultModel", bound=BaseModel)


class SubAgentTask(BaseModel):
    """Marker base — sub-agent task schemas inherit from this."""


class SubAgentResult(BaseModel):
    """Marker base — sub-agent result schemas inherit from this."""


class SubAgentInvocation(BaseModel):
    """What ``SubAgent.run`` returns to its caller (the parent coordinator).

    Mirrors ``ToolResult`` so the coordinator's executor can treat both
    primitives uniformly when it comes time to write the trace.
    """

    ok: bool
    output: dict[str, Any] | None = None
    error: StructuredError | None = None
    cost: Cost = Cost()
    outcome: Outcome = Outcome.OK
    incomplete: bool = False
    incomplete_reason: str | None = None


class SubAgent(abc.ABC, Generic[TaskModel, ResultModel]):
    """Subclasses define ``task_schema`` and ``result_schema`` and implement
    ``_run(task, scratch)``, which is where the sub-agent's own coordinator
    loop lives.

    For depth-2 invocations (a sub-agent invoking another sub-agent), the
    subclass MUST set a non-empty ``depth2_justification`` — a one-line
    statement that goes in the spec record for design review.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    task_schema: ClassVar[type[BaseModel]]
    result_schema: ClassVar[type[BaseModel]]
    tool_allowlist: ClassVar[tuple[str, ...]] = ()
    system_prompt: ClassVar[str] = ""
    # Carving defaults if the parent doesn't override at invocation time.
    default_max_tokens: ClassVar[int] = 50_000
    default_max_iterations: ClassVar[int] = 10
    default_max_wall_seconds: ClassVar[float] = 120.0
    # Required if this sub-agent is intended to be invoked from another sub-agent.
    depth2_justification: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        if not cls.name:
            raise SchemaContractError(f"SubAgent {cls.__name__} missing 'name'")
        if not hasattr(cls, "task_schema") or not hasattr(cls, "result_schema"):
            raise SchemaContractError(
                f"SubAgent {cls.__name__} must declare task_schema and result_schema "
                "(acceptance criterion 7)"
            )

    # ---- subclass override ----------------------------------------------

    @abc.abstractmethod
    def _run(self, task: TaskModel, scratch: dict[str, Any]) -> ResultModel:
        """Run the sub-agent's loop. May invoke its allowlisted tools.

        ``scratch`` is a per-invocation in-memory dict; treat it as the
        sub-agent's working state. It is independent from the parent's
        state per §6.
        """

    # ---- harness-facing entrypoint --------------------------------------

    def run(self, task_arguments: dict[str, Any], depth: int = 1) -> SubAgentInvocation:
        started = time.monotonic()

        if depth > 2:
            raise DepthLimitExceeded(
                f"SubAgent '{self.name}' invoked at depth {depth}; max is 2 per P3"
            )
        if depth == 2 and not self.depth2_justification:
            raise DepthLimitExceeded(
                f"SubAgent '{self.name}' cannot be invoked at depth 2 without "
                "'depth2_justification' (P3). Add a one-line justification to the class."
            )

        try:
            task = self.task_schema(**task_arguments)
        except ValidationError as e:
            return SubAgentInvocation(
                ok=False,
                error=StructuredError(
                    kind="task_validation",
                    message=str(e),
                    details={"errors": e.errors()},
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        scratch: dict[str, Any] = {}
        try:
            raw_result = self._run(task, scratch)
        except Exception as e:  # never bubble to parent coordinator
            return SubAgentInvocation(
                ok=False,
                error=StructuredError(
                    kind="subagent_internal_error",
                    message=f"{type(e).__name__}: {e}",
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        try:
            validated = self.result_schema.model_validate(
                raw_result.model_dump() if isinstance(raw_result, BaseModel) else raw_result
            )
        except ValidationError as e:
            return SubAgentInvocation(
                ok=False,
                error=StructuredError(
                    kind="result_validation",
                    message=str(e),
                    details={"errors": e.errors()},
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        return SubAgentInvocation(
            ok=True,
            output=validated.model_dump(),
            cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
            outcome=Outcome.OK,
        )

    @classmethod
    def manifest(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "task_schema": cls.task_schema.model_json_schema(),
            "result_schema": cls.result_schema.model_json_schema(),
            "tool_allowlist": list(cls.tool_allowlist),
            "depth2_justification": cls.depth2_justification or None,
            "default_budget": {
                "tokens": cls.default_max_tokens,
                "iterations": cls.default_max_iterations,
                "wall_seconds": cls.default_max_wall_seconds,
            },
        }
