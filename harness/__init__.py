"""Agent harness — one coordinator, three composable primitives.

See ``docs/SPEC.md`` for the full design contract. The locked principles
(P1–P7) and the acceptance criteria in §10 are the binding pieces; this
package implements them.
"""

from .budget import Budget
from .coordinator import Coordinator, Registry
from .errors import (
    BudgetExhausted,
    DepthLimitExceeded,
    GroundingError,
    HarnessError,
    SchemaContractError,
    UnknownPrimitive,
)
from .planner import LLMPlannerBase, PlanRequest, Planner, ScriptedPlanner
from .primitives import SubAgent, SubAgentResult, SubAgentTask, Tool, Verifier
from .primitives.tool import ToolFailure, ToolResult
from .primitives.subagent import SubAgentInvocation
from .primitives.verifier import VerifierResult
from .state import FileSystemState, InMemoryState, State, StateChange
from .trace import Trace, TraceEntry
from .types import (
    Action,
    ActionType,
    Cost,
    FinishAction,
    Outcome,
    Plan,
    SessionResult,
    StructuredError,
    SubAgentAction,
    ToolAction,
    Verdict,
    VerifyAction,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Coordinator",
    "Registry",
    "Budget",
    # State / trace
    "State",
    "StateChange",
    "InMemoryState",
    "FileSystemState",
    "Trace",
    "TraceEntry",
    # Planner
    "Planner",
    "ScriptedPlanner",
    "LLMPlannerBase",
    "PlanRequest",
    # Primitives
    "Tool",
    "ToolFailure",
    "ToolResult",
    "SubAgent",
    "SubAgentTask",
    "SubAgentResult",
    "SubAgentInvocation",
    "Verifier",
    "VerifierResult",
    # Types
    "Action",
    "ActionType",
    "ToolAction",
    "SubAgentAction",
    "VerifyAction",
    "FinishAction",
    "Plan",
    "Cost",
    "Outcome",
    "Verdict",
    "StructuredError",
    "SessionResult",
    # Errors
    "HarnessError",
    "BudgetExhausted",
    "DepthLimitExceeded",
    "GroundingError",
    "SchemaContractError",
    "UnknownPrimitive",
]
