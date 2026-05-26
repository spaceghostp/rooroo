from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import amendments as am
from .bus import Bus
from .plan import Plan
from .types import Constraint, Message


@dataclass
class SupervisorConfig:
    max_amendments_per_topic: int = 5
    max_objection_repeats: int = 3
    max_messages_per_topic: int = 100
    max_unanswered_objection_lag: int = 3
    max_overrides_per_topic: int = 3


@dataclass
class Finding:
    kind: str
    topic: Optional[str]
    detail: str
    refs: list[str] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        plan: Plan,
        bus: Bus,
        config: Optional[SupervisorConfig] = None,
    ) -> None:
        self.plan = plan
        self.bus = bus
        self.config = config or SupervisorConfig()

    def scan(self) -> list[Finding]:
        checks = (
            self._stale_citations,
            self._ping_pong,
            self._amendment_storms,
            self._budget,
            self._unanswered_objections,
            self._override_excess,
            self._drift,
        )
        return [f for check in checks for f in check()]

    def applied_amendment_ids(self) -> set[str]:
        return {
            m.payload["amendment_id"]
            for m in self.bus.by_type("applied")
            if m.sender == "supervisor"
        }

    def apply_ratified_amendments(self) -> list[int]:
        applied = self.applied_amendment_ids()
        new_versions: list[int] = []
        for amendment, _vote in am.ratified(self.bus):
            if amendment.id in applied:
                continue
            op = amendment.payload.get("operation")
            if op == "add":
                c_data = amendment.payload.get("constraint") or {}
                cid = c_data.get("id", "")
                if cid and not self.plan.is_active(cid):
                    v = self.plan.add(
                        Constraint(
                            id=cid,
                            kind=c_data["kind"],
                            scope=c_data["scope"],
                            description=c_data["description"],
                        )
                    )
                    new_versions.append(v)
            elif op == "remove":
                cid = amendment.payload.get("target_constraint_id")
                if cid and self.plan.is_active(cid):
                    v = self.plan.remove(cid)
                    new_versions.append(v)
            self.bus.post(
                topic=amendment.topic,
                sender="supervisor",
                type="applied",
                payload={
                    "amendment_id": amendment.id,
                    "plan_version": self.plan.version,
                },
            )
        return new_versions

    def _stale_citations(self) -> list[Finding]:
        out: list[Finding] = []
        for m in self.bus.by_type("objection"):
            cid = m.payload.get("cites_constraint")
            if not cid:
                continue
            c = self.plan.get(cid)
            if c is None:
                out.append(
                    Finding(
                        "stale_citation",
                        m.topic,
                        f"objection {m.id} cites unknown constraint {cid}",
                        [m.id],
                    )
                )
            elif not c.is_active():
                out.append(
                    Finding(
                        "stale_citation",
                        m.topic,
                        f"objection {m.id} cites amended-out constraint {cid}",
                        [m.id, cid],
                    )
                )
        return out

    def _ping_pong(self) -> list[Finding]:
        groups: dict[tuple[str, str], list[Message]] = {}
        for m in self.bus.by_type("objection"):
            key = (m.topic, m.payload.get("cites_constraint", ""))
            groups.setdefault(key, []).append(m)
        out: list[Finding] = []
        for (topic, cid), msgs in groups.items():
            if len(msgs) >= self.config.max_objection_repeats:
                out.append(
                    Finding(
                        "stuck",
                        topic,
                        f"objection ping-pong: constraint {cid} cited {len(msgs)} times",
                        [m.id for m in msgs],
                    )
                )
        return out

    def _amendment_storms(self) -> list[Finding]:
        out: list[Finding] = []
        for topic, n in am.per_topic(self.bus).items():
            if n >= self.config.max_amendments_per_topic:
                out.append(
                    Finding(
                        "amendment_storm",
                        topic,
                        f"{n} amendments on {topic} (cap {self.config.max_amendments_per_topic})",
                    )
                )
        return out

    def _budget(self) -> list[Finding]:
        counts: dict[str, int] = {}
        for m in self.bus.all():
            counts[m.topic] = counts.get(m.topic, 0) + 1
        out: list[Finding] = []
        for topic, n in counts.items():
            if n >= self.config.max_messages_per_topic:
                out.append(
                    Finding(
                        "budget_exceeded",
                        topic,
                        f"{n} messages on {topic} (cap {self.config.max_messages_per_topic})",
                    )
                )
        return out

    def _unanswered_objections(self) -> list[Finding]:
        out: list[Finding] = []
        for m in self.bus.by_type("objection"):
            answered = False
            for later in self.bus.for_topic(m.topic):
                if later.seq <= m.seq:
                    continue
                if later.type == "override" and later.payload.get("objection_id") == m.id:
                    answered = True
                    break
                if later.type == "fix" and later.payload.get("resolves_objection") == m.id:
                    answered = True
                    break
            cid = m.payload.get("cites_constraint")
            if cid and self.plan.get(cid) is not None and not self.plan.is_active(cid):
                answered = True
            if answered:
                continue
            lag = sum(1 for x in self.bus.for_topic(m.topic) if x.seq > m.seq)
            if lag >= self.config.max_unanswered_objection_lag:
                out.append(
                    Finding(
                        "unanswered_objection",
                        m.topic,
                        f"objection {m.id} unanswered after {lag} subsequent messages",
                        [m.id],
                    )
                )
        return out

    def _override_excess(self) -> list[Finding]:
        counts: dict[str, int] = {}
        for m in self.bus.by_type("override"):
            counts[m.topic] = counts.get(m.topic, 0) + 1
        out: list[Finding] = []
        for topic, n in counts.items():
            if n >= self.config.max_overrides_per_topic:
                out.append(
                    Finding(
                        "override_excess",
                        topic,
                        f"{n} overrides on {topic} (cap {self.config.max_overrides_per_topic})",
                    )
                )
        return out

    def _drift(self) -> list[Finding]:
        scopes = self.plan.scopes()
        out: list[Finding] = []
        for m in self.bus.all():
            if m.sender != "executor":
                continue
            if not m.payload.get("artifact_change"):
                continue
            if m.topic in scopes or "global" in scopes:
                continue
            out.append(
                Finding(
                    "drift",
                    m.topic,
                    f"executor changed artifact on topic {m.topic} with no active constraint scoped to it",
                    [m.id],
                )
            )
        return out
