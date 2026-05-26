"""Tools — deterministic, typed, no LLM inside. See §4.1."""

from __future__ import annotations

import abc
import time
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import SchemaContractError
from ..types import Cost, Outcome, StructuredError

InModel = TypeVar("InModel", bound=BaseModel)
OutModel = TypeVar("OutModel", bound=BaseModel)


class ToolResult(BaseModel):
    """What ``Tool.call`` returns regardless of success/failure.

    Tools never raise to the coordinator. ``error`` is ``None`` on success,
    populated on failure. ``output`` is the validated Pydantic dump on success.
    """

    ok: bool
    output: dict[str, Any] | None = None
    error: StructuredError | None = None
    cost: Cost = Cost()
    outcome: Outcome = Outcome.OK


class Tool(abc.ABC, Generic[InModel, OutModel]):
    """A typed, side-effect-declared function.

    Subclasses MUST:

    - set ``name``, ``description``, ``input_schema``, ``output_schema``
    - declare ``side_effects`` (free-form tags like ``"network"``, ``"fs:write"``)
    - declare ``cost_budget_*`` so registration-time review can sanity-check
    - implement ``_execute(payload)`` returning an ``OutModel`` instance

    They MUST NOT raise from ``_execute``; if they need to signal failure,
    return ``None`` or raise ``ToolFailure`` (which we catch). All other
    exceptions are caught and turned into a structured error so the
    coordinator can't be destabilized by a buggy tool.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    side_effects: ClassVar[tuple[str, ...]] = ()
    cost_budget_wall_ms: ClassVar[int] = 5_000
    cost_budget_dollars: ClassVar[float] = 0.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return  # still abstract; don't enforce yet
        if not cls.name:
            raise SchemaContractError(f"Tool {cls.__name__} missing 'name'")
        if not hasattr(cls, "input_schema") or not hasattr(cls, "output_schema"):
            raise SchemaContractError(
                f"Tool {cls.__name__} must declare input_schema and output_schema"
            )

    # ---- subclass override ----------------------------------------------

    @abc.abstractmethod
    def _execute(self, payload: InModel) -> OutModel: ...

    # ---- harness-facing entrypoint --------------------------------------

    def call(self, arguments: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            payload = self.input_schema(**arguments)
        except ValidationError as e:
            return ToolResult(
                ok=False,
                error=StructuredError(
                    kind="input_validation",
                    message=str(e),
                    details={"errors": e.errors()},
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        try:
            raw = self._execute(payload)
        except ToolFailure as e:
            return ToolResult(
                ok=False,
                error=StructuredError(kind=e.kind, message=str(e), details=e.details),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )
        except Exception as e:  # last-resort: never bubble to coordinator
            return ToolResult(
                ok=False,
                error=StructuredError(
                    kind="tool_internal_error",
                    message=f"{type(e).__name__}: {e}",
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        try:
            validated = self.output_schema.model_validate(
                raw.model_dump() if isinstance(raw, BaseModel) else raw
            )
        except ValidationError as e:
            return ToolResult(
                ok=False,
                error=StructuredError(
                    kind="output_validation",
                    message=str(e),
                    details={"errors": e.errors()},
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        return ToolResult(
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
            "input_schema": cls.input_schema.model_json_schema(),
            "output_schema": cls.output_schema.model_json_schema(),
            "side_effects": list(cls.side_effects),
            "cost_budget": {
                "wall_ms": cls.cost_budget_wall_ms,
                "dollars": cls.cost_budget_dollars,
            },
        }


class ToolFailure(Exception):
    """Raise from inside ``_execute`` when you have a structured failure
    that should reach the coordinator as a ``StructuredError``."""

    def __init__(self, kind: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = details
