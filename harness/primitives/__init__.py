"""Three primitives, as specified in §4: Tools, Sub-agents, Verifiers.

A sub-agent may not invoke another sub-agent unless explicitly justified
(see ``SubAgent.depth2_justification``). Verifiers may not be used as
critics — they must touch external evidence.

Each primitive declares its JSON schemas via Pydantic models. The schemas
are exported as part of registration; coordinator planners may inspect
them at registration time and only valid action arguments will be accepted
into ``run()``.
"""

from .tool import Tool, ToolResult
from .subagent import SubAgent, SubAgentResult, SubAgentTask
from .verifier import Verifier

__all__ = [
    "Tool",
    "ToolResult",
    "SubAgent",
    "SubAgentResult",
    "SubAgentTask",
    "Verifier",
]
