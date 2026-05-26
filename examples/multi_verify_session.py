"""Playbook Pattern 5 — Multi-Verifier Gate (AND-composition).

Generate a small Python function, then gate the result through THREE
independent grounded verifiers. All three must pass before
``FinishAction``:

  1. ``syntax_check``   — ast.parse                  (evidence: execute_code)
  2. ``type_check``     — every param has an annotation (evidence: schema_validate)
  3. ``unit_test``      — run a canonical test against the generated fn
                                                     (evidence: execute_code)

The planner here is a small custom class — not ``ScriptedPlanner`` —
because the gate logic is the whole point: walk the verifier list,
stop on the first failure (escalating to the generator for a retry),
and only ``FinishAction`` when every verdict in a row is ``passed=True``.

Composition policy is explicitly AND. The spec leaves verifier
composition as an open question (§11); this example commits to one
policy and makes it visible in the planner.

Run:

    python -m examples.multi_verify_session
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Plan,
    PlanRequest,
    Planner,
    Registry,
    Tool,
    ToolAction,
    Verdict,
    Verifier,
    VerifyAction,
)


# -------- Tool: generate_function -------------------------------------------


class GenIn(BaseModel):
    name: str
    spec: str  # e.g. "double", "square"


class GenOut(BaseModel):
    code: str


class GenerateFunctionTool(Tool):
    """Deterministic 'generator' — a templated mapping from spec to code.

    In production this would call an LLM; here it's a switch statement
    so the example has no API dependency. The shape (typed in, typed
    out, retryable) is identical.
    """

    name = "generate_function"
    description = "Emit a Python function source string from a typed spec."
    input_schema = GenIn
    output_schema = GenOut
    side_effects = ()

    _TEMPLATES = {
        # version 1: missing type annotation — fails type_check
        "double_v1": "def double(x):\n    return x + x\n",
        # version 2: type-annotated but wrong behavior — fails unit_test
        "double_v2": "def double(x: int) -> int:\n    return x  # bug: identity\n",
        # version 3: correct
        "double_v3": "def double(x: int) -> int:\n    return x * 2\n",
    }

    def _execute(self, payload):
        code = self._TEMPLATES.get(payload.spec, "")
        return GenOut(code=code)


# -------- Verifiers ----------------------------------------------------------


class CodeTarget(BaseModel):
    code: str
    function_name: str


class SyntaxCheckVerifier(Verifier):
    name = "syntax_check"
    description = "Parse the source with ast.parse and surface syntax errors."
    target_schema = CodeTarget
    evidence_sources = ("execute_code",)

    def _verify(self, target):
        try:
            ast.parse(target.code)
        except SyntaxError as e:
            return Verdict(
                passed=False,
                evidence=[f"execute_code: ast.parse SyntaxError at line {e.lineno}: {e.msg}"],
                suggested_revisions=[f"Fix syntax at line {e.lineno}"],
            )
        return Verdict(passed=True, evidence=["execute_code: ast.parse succeeded"])


class TypeCheckVerifier(Verifier):
    name = "type_check"
    description = "Confirm every parameter and the return value carry annotations."
    target_schema = CodeTarget
    evidence_sources = ("schema_validate",)

    def _verify(self, target):
        try:
            tree = ast.parse(target.code)
        except SyntaxError:
            return Verdict(
                passed=False,
                evidence=["schema_validate: cannot parse source"],
                suggested_revisions=["fix syntax first"],
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == target.function_name:
                missing = [
                    a.arg for a in node.args.args if a.annotation is None
                ]
                if missing:
                    return Verdict(
                        passed=False,
                        evidence=[
                            f"schema_validate: function {node.name} parameters without annotations: {missing}",
                        ],
                        suggested_revisions=[
                            f"Add type annotations for parameters: {missing}"
                        ],
                    )
                if node.returns is None:
                    return Verdict(
                        passed=False,
                        evidence=[f"schema_validate: function {node.name} missing return annotation"],
                        suggested_revisions=["Add a return-type annotation"],
                    )
                return Verdict(
                    passed=True,
                    evidence=[f"schema_validate: function {node.name} fully annotated"],
                )
        return Verdict(
            passed=False,
            evidence=[f"schema_validate: no function named {target.function_name} found"],
        )


class UnitTestVerifier(Verifier):
    name = "unit_test"
    description = "Execute the function with canonical inputs and check outputs."
    target_schema = CodeTarget
    evidence_sources = ("execute_code",)

    # Canonical examples; in a real verifier these come from the spec.
    _CASES = [(3, 6), (0, 0), (-2, -4)]

    def _verify(self, target):
        ns: dict[str, Any] = {}
        try:
            exec(compile(target.code, "<gen>", "exec"), ns)
        except Exception as e:
            return Verdict(
                passed=False,
                evidence=[f"execute_code: import failed: {type(e).__name__}: {e}"],
                suggested_revisions=["fix the module-level error"],
            )
        fn = ns.get(target.function_name)
        if fn is None or not callable(fn):
            return Verdict(
                passed=False,
                evidence=[f"execute_code: no callable named {target.function_name}"],
            )
        evidence = []
        for x, expected in self._CASES:
            try:
                got = fn(x)
            except Exception as e:
                return Verdict(
                    passed=False,
                    evidence=[*evidence, f"execute_code: {target.function_name}({x}) raised {e}"],
                )
            if got != expected:
                return Verdict(
                    passed=False,
                    evidence=[*evidence, f"execute_code: {target.function_name}({x}) = {got}, expected {expected}"],
                    suggested_revisions=[
                        f"behavior wrong: {target.function_name}({x}) should return {expected}"
                    ],
                )
            evidence.append(f"execute_code: {target.function_name}({x}) = {got} ✓")
        return Verdict(passed=True, evidence=evidence)


# -------- Custom planner: drives the AND-gate -------------------------------


VERSIONS = ["double_v1", "double_v2", "double_v3"]
GATE = ["syntax_check", "type_check", "unit_test"]


class GatePlanner(Planner):
    """Walks the verifier list. On the first failure, retries the
    generator with the next version. Finishes when all verifiers pass
    in a row, or exhausts the generator versions.
    """

    def __init__(self) -> None:
        self._version_idx = 0
        self._gate_idx = 0
        self._current_code: str | None = None
        self._last_finished = False

    def plan(self, request: PlanRequest) -> Plan:
        if self._last_finished:
            return Plan(terminate=True, next_action=FinishAction(result={"code": self._current_code}))

        obs = request.last_observation or {}

        # Just generated code → start verifying from the top.
        if "output" in obs and "code" in (obs.get("output") or {}):
            self._current_code = obs["output"]["code"]
            self._gate_idx = 0
            return self._next_verify()

        # Just got a verdict → either advance, retry-with-next-version, or finish.
        verdict = (obs.get("verdict") or None) if obs else None
        if verdict is not None:
            if not verdict["passed"]:
                # Failure: bump generator version and re-generate.
                self._version_idx += 1
                if self._version_idx >= len(VERSIONS):
                    return Plan(
                        terminate=True,
                        next_action=FinishAction(
                            result={
                                "error": "all versions exhausted",
                                "last_failed_gate": GATE[self._gate_idx],
                                "last_evidence": verdict["evidence"],
                            }
                        ),
                    )
                return self._next_generate()
            # Passed: advance through the gate.
            self._gate_idx += 1
            if self._gate_idx >= len(GATE):
                self._last_finished = True
                return Plan(
                    terminate=True,
                    next_action=FinishAction(result={"code": self._current_code, "passed_all_gates": True}),
                )
            return self._next_verify()

        # Cold start: generate first version.
        return self._next_generate()

    def _next_generate(self) -> Plan:
        return Plan(
            next_action=ToolAction(
                tool="generate_function",
                arguments={"name": "double", "spec": VERSIONS[self._version_idx]},
                rationale=f"attempt {self._version_idx + 1}",
            )
        )

    def _next_verify(self) -> Plan:
        return Plan(
            next_action=VerifyAction(
                verifier=GATE[self._gate_idx],
                target={"code": self._current_code, "function_name": "double"},
                rationale=f"gate {self._gate_idx + 1}/{len(GATE)}",
            )
        )


# -------- Drive --------------------------------------------------------------


def main() -> None:
    registry = Registry()
    registry.register_tool(GenerateFunctionTool())
    registry.register_verifier(SyntaxCheckVerifier())
    registry.register_verifier(TypeCheckVerifier())
    registry.register_verifier(UnitTestVerifier())

    coord = Coordinator(
        planner=GatePlanner(),
        registry=registry,
        budget=Budget(max_iterations=30, max_tokens=50_000, max_wall_seconds=15),
        session_dir=Path(".sessions/multi_verify"),
    )
    result = coord.run("Generate `double(x)` that satisfies syntax + types + tests.")
    print(json.dumps(result.model_dump(), indent=2, default=str))

    # Show the full gate trail so the AND-composition is visible.
    from harness.types import ActionType

    print("\nGate trail (one line per verdict):")
    for entry in coord.trace.by_type(ActionType.VERIFY):
        v = entry.action_output.get("verdict", {})
        verifier = entry.action_input.get("verifier")
        status = "PASS" if v.get("passed") else "FAIL"
        first_evidence = (v.get("evidence") or ["—"])[0]
        print(f"  {verifier:14s} {status}  | {first_evidence}")


if __name__ == "__main__":
    main()
