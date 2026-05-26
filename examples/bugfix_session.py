"""Playbook Pattern 3 — Tool + Sub-agent + Verifier (canonical depth-1).

A miniature "bug-fix-PR" agent against a fake in-memory filesystem.

Shape:
  - Tools: ``read_file``, ``write_file`` (deterministic IO over a dict)
  - Sub-agent: ``locate_bug`` — has its own loop, its own tool subset
    (read-only), produces a typed ``(file_path, line, hypothesis)`` result
  - Verifier: ``run_tests`` — actually executes the in-memory test
    function and reports pass/fail with evidence (P4 grounded)

The "bug" is a hardcoded greeting that should say "Hello, World!" but
currently says "Hello, Bug!". The test function checks string equality.

Run:

    python -m examples.bugfix_session
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Registry,
    ScriptedPlanner,
    SubAgent,
    SubAgentAction,
    SubAgentResult,
    SubAgentTask,
    Tool,
    ToolAction,
    ToolFailure,
    Verdict,
    Verifier,
    VerifyAction,
)


# -------- Fake filesystem ----------------------------------------------------
# Stand-in for a real codebase; the tools all close over this dict so they
# remain deterministic and don't need an actual disk.

FAKE_FS: dict[str, str] = {
    "greet.py": (
        "def greet():\n"
        "    return 'Hello, Bug!'\n"
    ),
    "test_greet.py": (
        "from greet import greet\n"
        "def test_greet():\n"
        "    assert greet() == 'Hello, World!'\n"
    ),
}


# -------- Tools --------------------------------------------------------------


class ReadIn(BaseModel):
    path: str


class ReadOut(BaseModel):
    content: str


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file from the working tree."
    input_schema = ReadIn
    output_schema = ReadOut
    side_effects = ("fs:read",)

    def _execute(self, payload):
        if payload.path not in FAKE_FS:
            raise ToolFailure("not_found", f"no file at {payload.path}")
        return ReadOut(content=FAKE_FS[payload.path])


class WriteIn(BaseModel):
    path: str
    content: str


class WriteOut(BaseModel):
    bytes_written: int


class WriteFileTool(Tool):
    name = "write_file"
    description = "Overwrite a file in the working tree."
    input_schema = WriteIn
    output_schema = WriteOut
    side_effects = ("fs:write",)

    def _execute(self, payload):
        FAKE_FS[payload.path] = payload.content
        return WriteOut(bytes_written=len(payload.content.encode()))


# -------- Sub-agent: locate_bug ---------------------------------------------


class LocateTask(SubAgentTask):
    symptom: str = Field(..., description="What the user reports going wrong.")


class LocateResult(SubAgentResult):
    file_path: str
    snippet: str
    hypothesis: str


class LocateBugAgent(SubAgent):
    """In a real implementation, ``_run`` would have its own Coordinator
    looping over (read_file, grep) until it found a suspect. To keep this
    example dependency-free we short-circuit with a focused search — but
    the *boundary* (typed task in, typed result out, independent context)
    is exactly what the spec calls for.
    """

    name = "locate_bug"
    description = "Given a symptom, return the file + snippet most likely at fault."
    task_schema = LocateTask
    result_schema = LocateResult
    tool_allowlist = ("read_file",)  # strict subset; cannot write
    system_prompt = "You are a careful bug-locator. Cite file:line for every claim."

    def _run(self, task: LocateTask, scratch: dict[str, Any]) -> LocateResult:
        for path, content in FAKE_FS.items():
            if path.startswith("test_"):
                continue
            if "Bug" in content:
                return LocateResult(
                    file_path=path,
                    snippet=content,
                    hypothesis=(
                        f"{path} returns a literal containing 'Bug' but the "
                        f"symptom ({task.symptom!r}) suggests the literal "
                        "should be 'World'."
                    ),
                )
        return LocateResult(
            file_path="",
            snippet="",
            hypothesis="No suspect file found in the working tree.",
        )


# -------- Verifier: run_tests (grounded — actually executes) ----------------


class TestTarget(BaseModel):
    # An empty target — the verifier reads the fake FS and runs whatever
    # test_*.py files it finds. The target is a marker that we're verifying
    # the current tree state, not a specific artifact.
    note: str = ""


class RunTestsVerifier(Verifier):
    name = "run_tests"
    description = "Execute the test files in the fake FS and report pass/fail."
    target_schema = TestTarget
    evidence_sources = ("execute_code",)  # P4: we actually run things
    max_retries = 2

    def _verify(self, target):
        import sys
        import types as _types

        evidence: list[str] = []
        failures: list[str] = []

        # Register each non-test source as a real module in sys.modules so
        # `from greet import greet` inside the test files actually resolves.
        # This is the verifier's "external reality": it gets a verdict by
        # actually executing code, not by reading the diff and offering
        # opinions on it.
        injected: list[str] = []
        try:
            for dep_path, dep_source in FAKE_FS.items():
                if dep_path.startswith("test_") or not dep_path.endswith(".py"):
                    continue
                mod_name = dep_path[:-3]
                module = _types.ModuleType(mod_name)
                exec(compile(dep_source, dep_path, "exec"), module.__dict__)
                sys.modules[mod_name] = module
                injected.append(mod_name)

            for path, source in FAKE_FS.items():
                if not path.startswith("test_"):
                    continue
                ns: dict[str, Any] = {"__name__": path[:-3]}
                try:
                    exec(compile(source, path, "exec"), ns)
                except Exception as e:
                    failures.append(f"{path}: import error {type(e).__name__}: {e}")
                    continue
                for fname, fval in ns.items():
                    if not fname.startswith("test_") or not callable(fval):
                        continue
                    try:
                        fval()
                        evidence.append(f"execute_code: {path}::{fname} PASSED")
                    except AssertionError as e:
                        failures.append(f"execute_code: {path}::{fname} FAILED ({e or 'assertion'})")
                    except Exception as e:
                        failures.append(f"execute_code: {path}::{fname} CRASHED: {e}")
        finally:
            for mod_name in injected:
                sys.modules.pop(mod_name, None)

        if failures:
            return Verdict(
                passed=False,
                evidence=evidence + failures,
                suggested_revisions=[
                    "Patch the source so the failing assertion holds."
                ],
            )
        return Verdict(passed=True, evidence=evidence or ["execute_code: no tests found"])


# -------- Drive --------------------------------------------------------------


def main() -> None:
    registry = Registry()
    registry.register_tool(ReadFileTool())
    registry.register_tool(WriteFileTool())
    registry.register_subagent(LocateBugAgent())
    registry.register_verifier(RunTestsVerifier())

    fixed_source = "def greet():\n    return 'Hello, World!'\n"

    # A real LLM-backed planner would observe each step's output and
    # adapt; we script the sequence the planner would converge on.
    actions = [
        # 1. Delegate locate to the read-only sub-agent.
        SubAgentAction(
            subagent="locate_bug",
            task={"symptom": "greet() returns the wrong string"},
            rationale="parent context shouldn't carry the codebase walk",
        ),
        # 2. Patch via the write tool (parent retains the write capability).
        ToolAction(
            tool="write_file",
            arguments={"path": "greet.py", "content": fixed_source},
            rationale="apply the fix the sub-agent located",
        ),
        # 3. Ground the patch by actually running the tests.
        VerifyAction(
            verifier="run_tests",
            target={"note": "post-patch verification"},
            rationale="gate the deliverable on grounded evidence (P4)",
        ),
        FinishAction(result={"patched": "greet.py", "tests": "passing"}),
    ]

    coord = Coordinator(
        planner=ScriptedPlanner(
            actions=actions,
            finish_result={"patched": "greet.py", "tests": "passing"},
        ),
        registry=registry,
        budget=Budget(max_iterations=12, max_tokens=50_000, max_wall_seconds=30),
        session_dir=Path(".sessions/bugfix"),
    )

    result = coord.run("Fix the greet() bug; gate the result on the tests.")
    print(json.dumps(result.model_dump(), indent=2, default=str))
    print(f"\nTrace: {result.trace_path}")
    print(f"Steps: {len(coord.trace)}")
    # Show the verifier's grounded verdict so the example *demonstrates* P4.
    from harness.types import ActionType

    verify_entries = coord.trace.by_type(ActionType.VERIFY)
    if verify_entries:
        verdict = verify_entries[-1].action_output.get("verdict")
        print("\nFinal verifier verdict:")
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
