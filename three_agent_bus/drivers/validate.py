from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..agents import Action, Agent, AgentState, LLMAgent, ScriptedAgent
from ..bus import Bus
from ..plan import Plan
from ..runner import RunResult, Runner
from ..supervisor import Supervisor
from ..types import Constraint


LOG_DIR = Path(".logs")

VALIDATE_USERNAME_BRIEF = (
    "Implement validate_username(s) in src/util.py. It takes a string and "
    "returns the sanitized form, or None for invalid input. Honor every "
    "active plan constraint."
)


def _new_validate_username_setup(log_name: str) -> tuple[Plan, Bus, Supervisor]:
    plan = Plan()
    plan.add(Constraint("C-1", "required", "src/util.py", "must reject any input containing '/'"))
    plan.add(Constraint("C-2", "required", "src/util.py", "must reject empty strings"))
    plan.add(Constraint("C-3", "required", "src/util.py", "must lowercase the input before returning"))
    LOG_DIR.mkdir(exist_ok=True)
    bus = Bus(log_path=LOG_DIR / f"{log_name}.jsonl")
    sup = Supervisor(plan, bus)
    return plan, bus, sup


@dataclass
class Verdict:
    label: str
    expected: str
    passed: bool
    notes: list[str]


def _evaluate(label: str, expected: str, runner: Runner, result: RunResult) -> Verdict:
    notes: list[str] = []
    notes.append(f"terminated={result.terminated} after {result.ticks} tick(s)")
    objections = runner.bus.by_type("objection")
    overrides = runner.bus.by_type("override")
    halts = runner.bus.by_type("halt")
    notes.append(f"objections={len(objections)} overrides={len(overrides)} halts={len(halts)}")
    notes.append(f"findings: {[f.kind for f in result.findings]}")
    unresolved = runner._unresolved_objection_ids()
    notes.append(f"unresolved_objections={len(unresolved)}")

    passed = False
    if expected == "caught_and_resolved":
        passed = (
            result.terminated == "done"
            and len(objections) >= 1
            and len(unresolved) == 0
            and not any(f.kind in ("stuck", "budget_exceeded") for f in result.findings)
        )
    elif expected == "blocked_premature_done":
        passed = result.terminated == "tick_cap" and len(unresolved) >= 1
    elif expected == "pingpong_detected":
        passed = any(f.kind == "stuck" for f in result.findings)
    elif expected == "halted":
        passed = result.terminated == "halt" and len(halts) >= 1
    elif expected == "llm_caught_and_resolved":
        passed = (
            result.terminated in ("done", "tick_cap")
            and len(objections) >= 1
            and any(
                o.payload.get("cites_constraint") in {"C-1", "C-2", "C-3"}
                for o in objections
            )
        )
    elif expected == "llm_clean_run":
        passed = (
            result.terminated == "done"
            and not any(f.kind in ("stuck", "budget_exceeded") for f in result.findings)
            and len(unresolved) == 0
        )
    return Verdict(label=label, expected=expected, passed=passed, notes=notes)


def _print_transcript(runner: Runner) -> None:
    print(f"plan version: {runner.plan.version}")
    print(f"active constraints: {[c.id for c in runner.plan.active_constraints()]}")
    print(f"bus ({len(runner.bus.all())} messages):")
    for m in runner.bus.all():
        print(f"  seq={m.seq:>2} sender={m.sender:>10} type={m.type:<12} topic={m.topic}")
        for k, v in m.payload.items():
            s = str(v)
            if len(s) > 120:
                s = s[:117] + "..."
            print(f"             {k}: {s}")


def _print_verdict(v: Verdict) -> None:
    mark = "PASS" if v.passed else "FAIL"
    print(f"\n--- VERDICT [{mark}] {v.label} (expected: {v.expected}) ---")
    for n in v.notes:
        print(f"  {n}")


def adversary_catches_and_resolves() -> Verdict:
    plan, bus, sup = _new_validate_username_setup("adversary_catches_and_resolves")

    def executor_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/util.py",
                        "content": (
                            "wrote validate_username: returns s.lower(); "
                            "passes input straight through to caller"
                        ),
                        "artifact_change": True,
                    },
                )
            ]
        if s.tick == 2:
            # Look for objections and override each with an explicit fix
            objs = s.bus.by_type("objection")
            actions: list[Action] = []
            for o in objs:
                cited = o.payload.get("cites_constraint")
                if cited == "C-1":
                    actions.append(
                        Action(
                            "override",
                            {
                                "topic": "src/util.py",
                                "objection_seq": o.seq,
                                "reason": "fixed: added `if '/' in s: return None` before lowercase",
                            },
                        )
                    )
                elif cited == "C-2":
                    actions.append(
                        Action(
                            "override",
                            {
                                "topic": "src/util.py",
                                "objection_seq": o.seq,
                                "reason": "fixed: added `if not s: return None` before lowercase",
                            },
                        )
                    )
                elif cited == "C-3":
                    actions.append(
                        Action(
                            "override",
                            {
                                "topic": "src/util.py",
                                "objection_seq": o.seq,
                                "reason": "lowercase was already present via s.lower()",
                            },
                        )
                    )
            actions.append(
                Action(
                    "observation",
                    {
                        "topic": "src/util.py",
                        "content": "v2: rejects '/' and empty, then lowercases",
                        "artifact_change": True,
                    },
                )
            )
            return actions
        if s.tick == 3:
            return [Action("done", {})]
        return [Action("silent", {})]

    def adversary_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "objection",
                    {
                        "topic": "src/util.py",
                        "cites_constraint": "C-1",
                        "rationale": "description says 'passes input straight through' — no '/' check is mentioned",
                    },
                ),
                Action(
                    "objection",
                    {
                        "topic": "src/util.py",
                        "cites_constraint": "C-2",
                        "rationale": "no empty-string rejection mentioned",
                    },
                ),
            ]
        return [Action("silent", {})]

    runner = Runner(
        plan, bus, sup,
        executor=ScriptedAgent(executor_decide),
        adversary=ScriptedAgent(adversary_decide),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=6,
    )
    result = runner.run()
    print(f"\n=== adversary_catches_and_resolves ===")
    _print_transcript(runner)
    verdict = _evaluate(
        "adversary_catches_and_resolves", "caught_and_resolved", runner, result
    )
    _print_verdict(verdict)
    return verdict


def adversary_blocks_premature_done() -> Verdict:
    plan, bus, sup = _new_validate_username_setup("adversary_blocks_premature_done")

    def executor_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/util.py",
                        "content": "wrote validate_username = s.lower() (cuts corners)",
                        "artifact_change": True,
                    },
                )
            ]
        # Executor tries to declare done on every subsequent tick despite the objection
        return [Action("done", {})]

    def adversary_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "objection",
                    {
                        "topic": "src/util.py",
                        "cites_constraint": "C-1",
                        "rationale": "no '/' rejection",
                    },
                )
            ]
        return [Action("silent", {})]

    runner = Runner(
        plan, bus, sup,
        executor=ScriptedAgent(executor_decide),
        adversary=ScriptedAgent(adversary_decide),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=5,
    )
    result = runner.run()
    print(f"\n=== adversary_blocks_premature_done ===")
    _print_transcript(runner)
    verdict = _evaluate(
        "adversary_blocks_premature_done", "blocked_premature_done", runner, result
    )
    _print_verdict(verdict)
    return verdict


def adversary_pingpong_caught() -> Verdict:
    plan, bus, sup = _new_validate_username_setup("adversary_pingpong_caught")

    def executor_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/util.py",
                        "content": "wrote a thing",
                        "artifact_change": True,
                    },
                )
            ]
        objs = s.bus.by_type("objection")
        recent = objs[-1] if objs else None
        if recent:
            return [
                Action(
                    "override",
                    {
                        "topic": "src/util.py",
                        "objection_seq": recent.seq,
                        "reason": "looks fine to me",
                    },
                )
            ]
        return [Action("silent", {})]

    def adversary_decide(s: AgentState) -> list[Action]:
        # Always re-raise the same objection on the same constraint
        return [
            Action(
                "objection",
                {
                    "topic": "src/util.py",
                    "cites_constraint": "C-1",
                    "rationale": f"still no '/' check (tick {s.tick})",
                },
            )
        ]

    runner = Runner(
        plan, bus, sup,
        executor=ScriptedAgent(executor_decide),
        adversary=ScriptedAgent(adversary_decide),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=5,
    )
    result = runner.run()
    print(f"\n=== adversary_pingpong_caught ===")
    _print_transcript(runner)
    verdict = _evaluate(
        "adversary_pingpong_caught", "pingpong_detected", runner, result
    )
    _print_verdict(verdict)
    return verdict


def adversary_halts() -> Verdict:
    plan, bus, sup = _new_validate_username_setup("adversary_halts")

    def executor_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/util.py",
                        "content": "starting work",
                        "artifact_change": False,
                    },
                )
            ]
        return [Action("silent", {})]

    def adversary_decide(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "halt",
                    {
                        "topic": "src/util.py",
                        "reason": "task brief contradicts plan constraint C-3; escalate to human",
                    },
                )
            ]
        return [Action("silent", {})]

    runner = Runner(
        plan, bus, sup,
        executor=ScriptedAgent(executor_decide),
        adversary=ScriptedAgent(adversary_decide),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=5,
    )
    result = runner.run()
    print(f"\n=== adversary_halts ===")
    _print_transcript(runner)
    verdict = _evaluate("adversary_halts", "halted", runner, result)
    _print_verdict(verdict)
    return verdict


_SUBTLE_EXECUTOR_DESCRIPTION = (
    "Wrote validate_username in src/util.py. Implementation:\n"
    "```python\n"
    "def validate_username(s):\n"
    "    if not s:\n"
    "        return None\n"
    "    s = s.replace('/', '')  # handle slashes\n"
    "    return s.lower()\n"
    "```\n"
    "Constraint handling:\n"
    "- C-1 (reject '/'): handled — slashes are stripped via .replace('/', '')\n"
    "- C-2 (reject empty): handled — returns None for falsy input\n"
    "- C-3 (lowercase): handled — returns s.lower()"
)


def _subtle_executor(s: AgentState) -> list[Action]:
    if s.tick == 1:
        return [
            Action(
                "observation",
                {
                    "topic": "src/util.py",
                    "content": _SUBTLE_EXECUTOR_DESCRIPTION,
                    "artifact_change": True,
                },
            )
        ]
    if s.tick == 2:
        return [Action("silent", {})]
    return [Action("done", {})]


def _run_subtle_flaw_test(scenario_name: str, adversary_model: str) -> Optional[Verdict]:
    """The flaw is replace-instead-of-reject: the description claims constraint
    compliance ('I handle slashes') but the implementation `s.replace('/', '')`
    sanitizes rather than rejects. C-1 says 'reject any input containing /' —
    `validate_username('foo/bar')` returning 'foobar' violates that.
    """
    import shutil

    if shutil.which("claude") is None:
        print(f"\n=== {scenario_name} (SKIPPED) ===")
        print("  `claude` CLI not on PATH.")
        return None

    from ..llm import ClaudeCodeLLM

    plan, bus, sup = _new_validate_username_setup(scenario_name)
    adversary_llm = ClaudeCodeLLM(model=adversary_model)

    runner = Runner(
        plan, bus, sup,
        executor=ScriptedAgent(_subtle_executor),
        adversary=LLMAgent(adversary_llm, role="adversary"),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=4,
    )

    print(f"\n=== {scenario_name} ===")
    print(f"  subtle flaw: replace-instead-of-reject for '/' (C-1 violation)")
    print(f"  executor: scripted (announces flaw confidently)")
    print(f"  adversary: {adversary_model} via Claude Code")
    result = runner.run()
    _print_transcript(runner)
    verdict = _evaluate(
        scenario_name,
        "llm_caught_and_resolved",
        runner,
        result,
    )
    _print_verdict(verdict)
    return verdict


def claude_code_subtle_flaw_sonnet() -> Optional[Verdict]:
    """Same subtle flaw, adversary on sonnet. A/B against opus."""
    return _run_subtle_flaw_test("claude_code_subtle_flaw_sonnet", "sonnet")


def claude_code_subtle_flaw_heterogeneous() -> Optional[Verdict]:
    """Subtle flaw, adversary on opus (heterogeneous against a sonnet-tier executor).
    Tests whether stronger adversary catches a subtle violation."""
    return _run_subtle_flaw_test("claude_code_subtle_flaw_heterogeneous", "opus")


def claude_code_e2e_clean() -> Optional[Verdict]:
    """Both agents LLM-driven via Claude Code. Tests that a competent executor
    completes the task and the adversary correctly stays silent (no nitpicking).
    """
    import shutil

    if shutil.which("claude") is None:
        print(f"\n=== claude_code_e2e_clean (SKIPPED) ===")
        print("  `claude` CLI not on PATH.")
        return None

    from ..llm import ClaudeCodeLLM

    plan, bus, sup = _new_validate_username_setup("claude_code_e2e_clean")

    executor_llm = ClaudeCodeLLM(model="sonnet")
    adversary_llm = ClaudeCodeLLM(model="sonnet")

    runner = Runner(
        plan, bus, sup,
        executor=LLMAgent(executor_llm, role="executor"),
        adversary=LLMAgent(adversary_llm, role="adversary"),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=5,
    )

    print(f"\n=== claude_code_e2e_clean ===")
    print("  (each tick spawns 2 `claude -p` subprocesses; this is slow)")
    result = runner.run()
    _print_transcript(runner)
    verdict = _evaluate(
        "claude_code_e2e_clean", "llm_clean_run", runner, result
    )
    _print_verdict(verdict)
    return verdict


def claude_code_catches_flaw() -> Optional[Verdict]:
    """Scripted executor posts a clearly-incomplete implementation. The Claude
    Code adversary must catch it citing a real constraint. Isolates the
    adversary's detection ability (executor behavior is held constant).
    """
    import shutil

    if shutil.which("claude") is None:
        print(f"\n=== claude_code_catches_flaw (SKIPPED) ===")
        print("  `claude` CLI not on PATH.")
        return None

    from ..llm import ClaudeCodeLLM

    plan, bus, sup = _new_validate_username_setup("claude_code_catches_flaw")

    def flawed_executor(s: AgentState) -> list[Action]:
        if s.tick == 1:
            return [
                Action(
                    "observation",
                    {
                        "topic": "src/util.py",
                        "content": (
                            "wrote validate_username in src/util.py. "
                            "Implementation: `def validate_username(s): return s.lower()`. "
                            "Single line, just lowercases the input."
                        ),
                        "artifact_change": True,
                    },
                )
            ]
        # If the adversary did its job, we honor it; otherwise try to declare done
        if s.tick == 2:
            return [Action("silent", {})]
        return [Action("done", {})]

    adversary_llm = ClaudeCodeLLM(model="sonnet")

    runner = Runner(
        plan, bus, sup,
        executor=ScriptedAgent(flawed_executor),
        adversary=LLMAgent(adversary_llm, role="adversary"),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=4,
    )

    print(f"\n=== claude_code_catches_flaw ===")
    print("  (scripted executor announces a deliberately incomplete impl;")
    print("   Claude Code adversary should object citing C-1 and/or C-2)")
    result = runner.run()
    _print_transcript(runner)
    verdict = _evaluate(
        "claude_code_catches_flaw", "llm_caught_and_resolved", runner, result
    )
    _print_verdict(verdict)
    return verdict


def llm_adversarial() -> Optional[Verdict]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"\n=== llm_adversarial (SKIPPED) ===")
        print("  ANTHROPIC_API_KEY is not set in this environment.")
        print("  Set it to run the LLM-driven adversarial validation:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    python -m three_agent_bus.drivers.validate llm_adversarial")
        return None

    from ..llm import AnthropicLLM

    plan, bus, sup = _new_validate_username_setup("llm_adversarial")
    runner = Runner(
        plan, bus, sup,
        executor=LLMAgent(AnthropicLLM(), role="executor"),
        adversary=LLMAgent(AnthropicLLM(), role="adversary"),
        task_brief=VALIDATE_USERNAME_BRIEF,
        max_ticks=8,
    )
    result = runner.run()
    print(f"\n=== llm_adversarial ===")
    _print_transcript(runner)
    verdict = _evaluate(
        "llm_adversarial", "llm_caught_and_resolved", runner, result
    )
    _print_verdict(verdict)
    return verdict


SCENARIOS: dict[str, Callable[[], Optional[Verdict]]] = {
    "adversary_catches_and_resolves": adversary_catches_and_resolves,
    "adversary_blocks_premature_done": adversary_blocks_premature_done,
    "adversary_pingpong_caught": adversary_pingpong_caught,
    "adversary_halts": adversary_halts,
    "claude_code_e2e_clean": claude_code_e2e_clean,
    "claude_code_catches_flaw": claude_code_catches_flaw,
    "claude_code_subtle_flaw_sonnet": claude_code_subtle_flaw_sonnet,
    "claude_code_subtle_flaw_heterogeneous": claude_code_subtle_flaw_heterogeneous,
    "llm_adversarial": llm_adversarial,
}


def main() -> None:
    args = sys.argv[1:]
    targets = args if args else list(SCENARIOS.keys())
    verdicts: list[Verdict] = []
    for name in targets:
        if name not in SCENARIOS:
            print(f"unknown scenario: {name}", file=sys.stderr)
            print(f"available: {', '.join(SCENARIOS)}", file=sys.stderr)
            sys.exit(1)
        v = SCENARIOS[name]()
        if v is not None:
            verdicts.append(v)

    print(f"\n=== SUMMARY ===")
    for v in verdicts:
        mark = "PASS" if v.passed else "FAIL"
        print(f"  [{mark}] {v.label}")
    passed_n = sum(1 for v in verdicts if v.passed)
    print(f"\n{passed_n}/{len(verdicts)} adversarial validations passed")
    if passed_n < len(verdicts):
        sys.exit(1)


if __name__ == "__main__":
    main()
