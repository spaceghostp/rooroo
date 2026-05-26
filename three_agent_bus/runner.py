from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agents import Action, Agent, AgentState
from .bus import Bus
from .plan import Plan
from .supervisor import Finding, Supervisor


@dataclass
class RunResult:
    ticks: int
    terminated: str
    findings: list[Finding]


class Runner:
    def __init__(
        self,
        plan: Plan,
        bus: Bus,
        supervisor: Supervisor,
        executor: Agent,
        adversary: Agent,
        task_brief: str = "",
        max_ticks: int = 20,
    ) -> None:
        self.plan = plan
        self.bus = bus
        self.supervisor = supervisor
        self.executor = executor
        self.adversary = adversary
        self.task_brief = task_brief
        self.max_ticks = max_ticks
        self._executor_done = False

    def step(self, tick: int) -> Optional[str]:
        if not self._executor_done:
            exec_actions = self.executor.decide(
                AgentState(
                    role="executor",
                    plan=self.plan,
                    bus=self.bus,
                    task_brief=self.task_brief,
                    tick=tick,
                )
            )
            for a in exec_actions:
                if a.type == "done":
                    self._executor_done = True
                else:
                    self._post(sender="executor", action=a)

        adv_actions = self.adversary.decide(
            AgentState(
                role="adversary",
                plan=self.plan,
                bus=self.bus,
                task_brief=None,
                tick=tick,
            )
        )
        for a in adv_actions:
            self._post(sender="adversary", action=a)

        self.supervisor.apply_ratified_amendments()
        findings = self.supervisor.scan()

        for f in findings:
            if f.kind == "budget_exceeded":
                return "budget"

        if self._executor_done and not self._unresolved_objection_ids():
            return "done"
        return None

    def run(self) -> RunResult:
        for tick in range(1, self.max_ticks + 1):
            reason = self.step(tick)
            if reason is not None:
                return RunResult(
                    ticks=tick,
                    terminated=reason,
                    findings=self.supervisor.scan(),
                )
        return RunResult(
            ticks=self.max_ticks,
            terminated="tick_cap",
            findings=self.supervisor.scan(),
        )

    def _post(self, sender: str, action: Action) -> None:
        a = action
        if a.type == "silent":
            return
        topic = a.payload.get("topic", "")
        if a.type == "observation":
            self.bus.post(
                topic=topic,
                sender=sender,
                type="observation",
                payload={
                    "content": a.payload.get("content", ""),
                    "artifact_change": a.payload.get("artifact_change", False),
                },
            )
        elif a.type == "override":
            self.bus.post(
                topic=topic,
                sender=sender,
                type="override",
                payload={
                    "objection_id": a.payload.get("objection_id", ""),
                    "reason": a.payload.get("reason", ""),
                },
            )
        elif a.type == "objection":
            self.bus.post(
                topic=topic,
                sender=sender,
                type="objection",
                payload={
                    "cites_constraint": a.payload.get("cites_constraint", ""),
                    "rationale": a.payload.get("rationale", ""),
                },
            )
        elif a.type == "question":
            self.bus.post(
                topic=topic,
                sender=sender,
                type="question",
                payload={"content": a.payload.get("content", "")},
            )
        elif a.type == "amendment":
            payload = {
                "operation": a.payload.get("operation", ""),
                "rationale": a.payload.get("rationale", ""),
            }
            if "constraint" in a.payload:
                payload["constraint"] = a.payload["constraint"]
            if "target_constraint_id" in a.payload:
                payload["target_constraint_id"] = a.payload["target_constraint_id"]
            self.bus.post(topic=topic, sender=sender, type="amendment", payload=payload)
        elif a.type == "vote":
            self.bus.post(
                topic=topic,
                sender=sender,
                type="vote",
                payload={
                    "amendment_id": a.payload.get("amendment_id", ""),
                    "vote": a.payload.get("vote", ""),
                },
            )

    def _unresolved_objection_ids(self) -> list[str]:
        out: list[str] = []
        for m in self.bus.by_type("objection"):
            cid = m.payload.get("cites_constraint")
            if cid and self.plan.get(cid) is not None and not self.plan.is_active(cid):
                continue
            answered = False
            for later in self.bus.for_topic(m.topic):
                if later.seq <= m.seq:
                    continue
                if (
                    later.type == "override"
                    and later.payload.get("objection_id") == m.id
                ):
                    answered = True
                    break
            if not answered:
                out.append(m.id)
        return out
