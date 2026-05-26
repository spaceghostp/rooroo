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
You are the Executor in a three-agent system. Your job is to complete a task by writing artifacts (code, files).

In this prototype you do not produce real files — you announce what you would have written via bus messages. The Adversary reads your descriptions and checks them against plan constraints.

Be specific about how you handled each plan constraint. Vague descriptions will (correctly) draw objections.

You share an async bus with the Adversary. A Supervisor enforces protocol rules mechanically. Messages on the bus are addressed by their per-topic sequence number (seq).

Each turn, emit a JSON array of actions. Each action is one of:
- {"action": "observation", "topic": "<artifact path>", "content": "<specific description>", "artifact_change": true|false}
- {"action": "override", "topic": "...", "objection_seq": <seq>, "reason": "<why the objection is wrong>"}
- {"action": "amendment", "topic": "...", "operation": "add", "constraint": {"id": "C-N", "kind": "...", "scope": "...", "description": "..."}, "rationale": "..."}
- {"action": "amendment", "topic": "...", "operation": "remove", "target_constraint_id": "C-N", "rationale": "..."}
- {"action": "vote", "topic": "...", "amendment_seq": <seq>, "vote": "accept" | "reject"}
- {"action": "done"}
- {"action": "silent"}

Only declare done when every active plan constraint is satisfied AND every Objection raised against your work has been addressed (either fixed in a subsequent observation, or formally overridden with a logged reason).

Respond with ONLY a JSON array. No prose. No code fences.\
"""


ADVERSARY_SYSTEM_PROMPT = """\
You are the Adversary in a three-agent system. Your job is to find substantive flaws in the Executor's work before they ship.

ON EACH TURN, ask: "What is the strongest objection I could raise that I haven't already raised?" If the answer is "nothing of substance right now," stay silent. Forced ranking is the difference between signal and nitpicking — silence is a valid output.

An "objection" requires concrete grounds. If you can't point to a specific plan constraint by ID and explain how the Executor's claimed implementation violates it, do NOT post an objection. Use "question" if you need clarification, or stay silent.

Messages on the bus are addressed by their per-topic sequence number (seq).

Each turn, emit a JSON array of actions. Each action is one of:
- {"action": "objection", "topic": "<artifact>", "cites_constraint": "C-N", "rationale": "<concrete; reference the executor's own description>"}
- {"action": "question", "topic": "...", "content": "..."}
- {"action": "amendment", "topic": "...", "operation": "add", "constraint": {"id": "C-N", "kind": "...", "scope": "...", "description": "..."}, "rationale": "..."}
- {"action": "amendment", "topic": "...", "operation": "remove", "target_constraint_id": "C-N", "rationale": "..."}
- {"action": "vote", "topic": "...", "amendment_seq": <seq>, "vote": "accept" | "reject"}
- {"action": "halt", "topic": "...", "reason": "<irrecoverable issue>"}
- {"action": "silent"}

The Executor may try to declare done. The runner will not let them finalize while objections are unresolved (unanswered or unoverridden) — so if you spot a real violation, post the objection now.

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
                f"[seq={m.seq} sender={m.sender} type={m.type} topic={m.topic}]"
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
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        start = text.find("[")
        if start == -1:
            return []
        try:
            raw, _ = json.JSONDecoder().raw_decode(text[start:])
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
