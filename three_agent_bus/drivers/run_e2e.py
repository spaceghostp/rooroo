from __future__ import annotations

import sys
from typing import Callable

from ..agents import Action, AgentState, LLMAgent, ScriptedAgent
from ..bus import Bus
from ..llm import ScriptedLLM
from ..plan import Plan
from ..runner import RunResult, Runner
from ..supervisor import Supervisor
from ..types import Constraint


def _new_plan_with(constraints: list[Constraint]) -> Plan:
    plan = Plan()
    for c in constraints:
        plan.add(c)
    return plan


def _report(label: str, runner: Runner, result: RunResult) -> None:
    print(f"\n=== {label} ===")
    print(f"terminated: {result.terminated} after {result.ticks} tick(s)")
    print(f"plan version: {runner.plan.version}")
    print(
        f"active constraints: {[c.id for c in runner.plan.active_constraints()]}"
    )
    print(f"bus messages ({len(runner.bus.all())}):")
    for m in runner.bus.all():
        print(
            f"  seq={m.seq} sender={m.sender:>10} type={m.type:<12} topic={m.topic}"
        )
    print(f"findings ({len(result.findings)}):")
    if not result.findings:
        print("  (none)")
    for f in result.findings:
        print(f"  [{f.kind}] {f.detail}")


def happy_run() -> None:
    plan = _new_plan_with(
        [Constraint("C-1", "required", "src/api.py", "must validate POST input")]
    )
    bus = Bus()
    sup = Supervisor(plan, bus)

    exec_script = iter(
        [
            [
                Action(
                    "observation",
                    {
                        "topic": "src/api.py",
                        "content": "wrote input validation",
                        "artifact_change": True,
                    },
                )
            ],
            [
                Action(
                    "observation",
                    {
                        "topic": "src/api.py",
                        "content": "added type checks",
                        "artifact_change": True,
                    },
                )
            ],
            [Action("done", {})],
        ]
    )
    adv_script = iter(
        [
            [Action("silent", {})],
            [
                Action(
                    "observation",
                    {"topic": "src/api.py", "content": "verified validation present"},
                )
            ],
            [Action("silent", {})],
        ]
    )

    runner = Runner(
        plan,
        bus,
        sup,
        executor=ScriptedAgent(lambda s: next(exec_script)),
        adversary=ScriptedAgent(lambda s: next(adv_script)),
        task_brief="Add input validation to POST endpoint.",
        max_ticks=5,
    )
    _report("happy_run (clean termination)", runner, runner.run())


def objection_then_override() -> None:
    plan = _new_plan_with(
        [Constraint("C-1", "required", "src/api.py", "rate limit on /login")]
    )
    bus = Bus()
    sup = Supervisor(plan, bus)

    def executor_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/api.py",
                        "content": "wrote /login handler",
                        "artifact_change": True,
                    },
                )
            ]
        if s.tick == 2:
            objs = s.bus.by_type("objection")
            if objs:
                return [
                    Action(
                        "override",
                        {
                            "topic": "src/api.py",
                            "objection_id": objs[-1].id,
                            "reason": "rate limiting handled by upstream proxy, not application",
                        },
                    )
                ]
            return [Action("silent", {})]
        if s.tick == 3:
            return [Action("done", {})]
        return [Action("silent", {})]

    def adversary_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "objection",
                    {
                        "topic": "src/api.py",
                        "cites_constraint": "C-1",
                        "rationale": "no rate limit visible in the handler",
                    },
                )
            ]
        return [Action("silent", {})]

    runner = Runner(
        plan,
        bus,
        sup,
        executor=ScriptedAgent(executor_decide),
        adversary=ScriptedAgent(adversary_decide),
        task_brief="Add /login endpoint with rate limit (C-1).",
        max_ticks=5,
    )
    _report(
        "objection_then_override (override clears the path to done)",
        runner,
        runner.run(),
    )


def amendment_ratification() -> None:
    plan = _new_plan_with(
        [Constraint("C-1", "required", "src/api.py", "rate limit 100/min")]
    )
    bus = Bus()
    sup = Supervisor(plan, bus)

    def executor_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "amendment",
                    {
                        "topic": "src/api.py",
                        "operation": "add",
                        "constraint": {
                            "id": "C-2",
                            "kind": "limit",
                            "scope": "src/api.py",
                            "description": "burst window 10s",
                        },
                        "rationale": "10s burst window handles login retries",
                    },
                )
            ]
        if s.tick == 3:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/api.py",
                        "content": "wrote burst-aware rate limiter",
                        "artifact_change": True,
                    },
                )
            ]
        if s.tick == 4:
            return [Action("done", {})]
        return [Action("silent", {})]

    def adversary_decide(s: AgentState) -> list[Action]:
        if s.tick == 2:
            amends = s.bus.by_type("amendment")
            if amends:
                return [
                    Action(
                        "vote",
                        {
                            "topic": "src/api.py",
                            "amendment_id": amends[-1].id,
                            "vote": "accept",
                        },
                    )
                ]
        return [Action("silent", {})]

    runner = Runner(
        plan,
        bus,
        sup,
        executor=ScriptedAgent(executor_decide),
        adversary=ScriptedAgent(adversary_decide),
        task_brief="Tune /login rate limiting (constraint C-1).",
        max_ticks=8,
    )
    _report(
        "amendment_ratification (C-2 ratified and applied)",
        runner,
        runner.run(),
    )


def llm_smoke() -> None:
    """LLMAgent path exercised with a ScriptedLLM — no API calls, proves wiring."""
    plan = _new_plan_with(
        [Constraint("C-1", "required", "src/api.py", "must validate POST input")]
    )
    bus = Bus()
    sup = Supervisor(plan, bus)

    exec_llm = ScriptedLLM(
        [
            '[{"action": "observation", "topic": "src/api.py", "content": "wrote validator", "artifact_change": true}]',
            '[{"action": "done"}]',
        ]
    )
    adv_llm = ScriptedLLM(
        [
            '[{"action": "silent"}]',
            '[{"action": "silent"}]',
        ]
    )

    runner = Runner(
        plan,
        bus,
        sup,
        executor=LLMAgent(exec_llm, role="executor"),
        adversary=LLMAgent(adv_llm, role="adversary"),
        task_brief="Add input validation.",
        max_ticks=5,
    )
    _report("llm_smoke (LLMAgent path wired via ScriptedLLM)", runner, runner.run())


SCENARIOS: dict[str, Callable[[], None]] = {
    "happy_run": happy_run,
    "objection_then_override": objection_then_override,
    "amendment_ratification": amendment_ratification,
    "llm_smoke": llm_smoke,
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
