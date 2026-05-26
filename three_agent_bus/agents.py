from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from .bus import Bus
from .llm import LLM
from .plan import Plan


@dataclass
class AgentState:
    role: str
    plan: Plan
    bus: Bus
    task_brief: Optional[str]
    tick: int


@dataclass
class Action:
    type: str
    payload: dict


class Agent(ABC):
    @abstractmethod
    def decide(self, state: AgentState) -> list[Action]:
        ...


ScriptedDecider = Callable[[AgentState], list[Action]]


class ScriptedAgent(Agent):
    def __init__(self, decider: ScriptedDecider) -> None:
        self._decider = decider

    def decide(self, state: AgentState) -> list[Action]:
        return self._decider(state)


EXECUTOR_SYSTEM_PROMPT = """\
You are the Executor in a three-agent system. Your job is to complete a task by writing artifacts (code, files). In this prototype, you announce artifact changes via bus messages rather than producing real files.

You share an async message bus with an Adversary who raises objections grounded in plan constraints. A Supervisor enforces protocol rules mechanically.

You may emit a JSON array of actions per turn. Each action is one of:
- {"action": "observation", "topic": "<artifact path>", "content": "...", "artifact_change": true/false}
- {"action": "override", "topic": "...", "objection_id": "<id from bus>", "reason": "..."}
- {"action": "amendment", "topic": "...", "operation": "add", "constraint": {"id": "C-N", "kind": "...", "scope": "...", "description": "..."}, "rationale": "..."}
- {"action": "amendment", "topic": "...", "operation": "remove", "target_constraint_id": "C-N", "rationale": "..."}
- {"action": "vote", "topic": "...", "amendment_id": "<id from bus>", "vote": "accept" | "reject"}
- {"action": "done"}
- {"action": "silent"}

Respond with ONLY a JSON array. No prose. No code fences.\
"""


ADVERSARY_SYSTEM_PROMPT = """\
You are the Adversary in a three-agent system. Your job is to find substantive flaws in the Executor's work before they ship.

ON EACH TURN, ask: "What is the strongest objection I could raise that I haven't already raised?" If the answer doesn't clear a meaningful threshold, stay silent. Forced ranking is the difference between signal and nitpicking — silence is a valid output.

You may emit a JSON array of actions per turn. Each action is one of:
- {"action": "objection", "topic": "<artifact>", "cites_constraint": "C-N", "rationale": "..."}  (MUST cite a real constraint ID from the plan)
- {"action": "question", "topic": "...", "content": "..."}
- {"action": "amendment", "topic": "...", "operation": "add", "constraint": {"id": "C-N", "kind": "...", "scope": "...", "description": "..."}, "rationale": "..."}
- {"action": "amendment", "topic": "...", "operation": "remove", "target_constraint_id": "C-N", "rationale": "..."}
- {"action": "vote", "topic": "...", "amendment_id": "<id from bus>", "vote": "accept" | "reject"}
- {"action": "silent"}

Respond with ONLY a JSON array. No prose. No code fences.\
"""


class LLMAgent(Agent):
    def __init__(
        self,
        llm: LLM,
        role: str,
        max_recent_messages: int = 50,
    ) -> None:
        if role not in ("executor", "adversary"):
            raise ValueError(f"role must be 'executor' or 'adversary', got {role!r}")
        self._llm = llm
        self._role = role
        self._max_recent = max_recent_messages
        self._system = (
            EXECUTOR_SYSTEM_PROMPT if role == "executor" else ADVERSARY_SYSTEM_PROMPT
        )

    def decide(self, state: AgentState) -> list[Action]:
        prompt = self._build_user_prompt(state)
        response = self._llm.call(system=self._system, user=prompt)
        return self._parse_actions(response.text)

    def _build_user_prompt(self, state: AgentState) -> str:
        lines: list[str] = []
        lines.append(f"## Plan (version {state.plan.version})")
        actives = state.plan.active_constraints()
        if not actives:
            lines.append("Active constraints: (none)")
        else:
            lines.append("Active constraints:")
            for c in actives:
                lines.append(
                    f"  #{c.id} [{c.kind}] (scope: {c.scope}): {c.description}"
                )
        lines.append("")
        lines.append("## Recent bus messages")
        recent = state.bus.all()[-self._max_recent :]
        if not recent:
            lines.append("  (none)")
        for m in recent:
            lines.append(
                f"[id={m.id} seq={m.seq} sender={m.sender} type={m.type} topic={m.topic}]"
            )
            for k, v in m.payload.items():
                lines.append(f"  {k}: {v}")
        lines.append("")
        if state.task_brief and state.role == "executor":
            lines.append("## Task brief")
            lines.append(state.task_brief)
            lines.append("")
        lines.append(f"## Your turn (tick {state.tick}, role: {state.role})")
        lines.append("Respond with a JSON array of actions.")
        return "\n".join(lines)

    def _parse_actions(self, text: str) -> list[Action]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match is None:
            return []
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        actions: list[Action] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            atype = item.get("action")
            if not isinstance(atype, str):
                continue
            payload = {k: v for k, v in item.items() if k != "action"}
            actions.append(Action(type=atype, payload=payload))
        return actions
