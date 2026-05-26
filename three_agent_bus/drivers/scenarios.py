from __future__ import annotations

import sys
from typing import Callable

from ..bus import Bus
from ..plan import Plan
from ..supervisor import Supervisor
from ..types import Constraint


def _new() -> tuple[Plan, Bus, Supervisor]:
    plan = Plan()
    bus = Bus()
    return plan, bus, Supervisor(plan, bus)


def _report(label: str, plan: Plan, sup: Supervisor) -> None:
    findings = sup.scan()
    print(f"\n=== {label} ===")
    print(f"plan version: {plan.version}")
    print(f"active constraints: {[c.id for c in plan.active_constraints()]}")
    print(f"findings ({len(findings)}):")
    if not findings:
        print("  (none)")
    for f in findings:
        print(f"  [{f.kind}] topic={f.topic} :: {f.detail}")


def happy_path() -> None:
    plan, bus, sup = _new()
    plan.add(
        Constraint(
            "C-1",
            "required",
            "src/db/schema.sql",
            "users table must have created_at column",
        )
    )
    bus.post(
        "src/db/schema.sql",
        "executor",
        "observation",
        {"content": "wrote schema with created_at", "artifact_change": True},
    )
    bus.post(
        "src/db/schema.sql",
        "adversary",
        "observation",
        {"content": "verified created_at present"},
    )
    _report("happy_path (no findings expected)", plan, sup)


def stale_citation() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/auth.py", "use bcrypt"))
    plan.add(Constraint("C-2", "forbidden", "src/auth.py", "no plaintext storage"))
    bus.post(
        "src/auth.py",
        "adversary",
        "objection",
        {"cites_constraint": "C-2", "rationale": "found plaintext path"},
    )
    plan.remove("C-2")
    bus.post(
        "src/auth.py",
        "adversary",
        "objection",
        {"cites_constraint": "C-2", "rationale": "raising again"},
    )
    bus.post(
        "src/auth.py",
        "adversary",
        "objection",
        {"cites_constraint": "C-99", "rationale": "wrong id"},
    )
    _report("stale_citation (expect 3 stale findings)", plan, sup)


def ping_pong() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/api.py", "rate limit on POST /login"))
    for i in range(4):
        bus.post(
            "src/api.py",
            "adversary",
            "objection",
            {"cites_constraint": "C-1", "rationale": f"still missing #{i}"},
        )
    _report("ping_pong (expect stuck)", plan, sup)


def amendment_flow() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/api.py", "rate limit 100/min"))
    amend = bus.post(
        "src/api.py",
        "executor",
        "amendment",
        {
            "operation": "add",
            "constraint": {
                "id": "C-2",
                "kind": "limit",
                "scope": "src/api.py",
                "description": "burst window 10s",
            },
            "rationale": "tighter window matches real traffic",
        },
    )
    bus.post(
        "src/api.py",
        "adversary",
        "vote",
        {"amendment_id": amend.id, "vote": "accept"},
    )
    sup.apply_ratified_amendments()
    _report("amendment_flow (expect C-2 added)", plan, sup)


def amendment_storm() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/api.py", "rate limit"))
    for i in range(6):
        bus.post(
            "src/api.py",
            "executor",
            "amendment",
            {
                "operation": "add",
                "constraint": {
                    "id": f"C-{100 + i}",
                    "kind": "limit",
                    "scope": "src/api.py",
                    "description": f"variant {i}",
                },
                "rationale": "tuning",
            },
        )
    _report("amendment_storm (expect storm finding)", plan, sup)


def drift() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/api.py", "rate limit"))
    bus.post(
        "src/billing/charge.py",
        "executor",
        "observation",
        {"content": "modified charge logic", "artifact_change": True},
    )
    _report("drift (expect drift finding)", plan, sup)


def override_excess() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/api.py", "rate limit"))
    for i in range(4):
        obj = bus.post(
            "src/api.py",
            "adversary",
            "objection",
            {"cites_constraint": "C-1", "rationale": f"weak #{i}"},
        )
        bus.post(
            "src/api.py",
            "executor",
            "override",
            {"objection_id": obj.id, "reason": f"acceptable trade-off {i}"},
        )
    _report("override_excess (expect override_excess finding)", plan, sup)


def unanswered_objection() -> None:
    plan, bus, sup = _new()
    plan.add(Constraint("C-1", "required", "src/api.py", "input validation"))
    bus.post(
        "src/api.py",
        "adversary",
        "objection",
        {"cites_constraint": "C-1", "rationale": "missing validation"},
    )
    for i in range(4):
        bus.post(
            "src/api.py",
            "executor",
            "observation",
            {"content": f"unrelated work step {i}", "artifact_change": False},
        )
    _report("unanswered_objection (expect unanswered finding)", plan, sup)


SCENARIOS: dict[str, Callable[[], None]] = {
    "happy_path": happy_path,
    "stale_citation": stale_citation,
    "ping_pong": ping_pong,
    "amendment_flow": amendment_flow,
    "amendment_storm": amendment_storm,
    "drift": drift,
    "override_excess": override_excess,
    "unanswered_objection": unanswered_objection,
}


def main() -> None:
    args = sys.argv[1:]
    targets = args if args else list(SCENARIOS.keys())
    for name in targets:
        if name not in SCENARIOS:
            print(f"unknown scenario: {name}", file=sys.stderr)
            print(f"available: {', '.join(SCENARIOS)}", file=sys.stderr)
            sys.exit(1)
        SCENARIOS[name]()


if __name__ == "__main__":
    main()
