"""Playbook Pattern 3 — Tool + Sub-agent + Verifier (canonical depth-1).

A miniature "bug-fix-PR" agent against a fake in-memory filesystem.

Shape:
  - Tools: ``list_files``, ``read_file``, ``write_file`` (deterministic IO
    over a dict).
  - Sub-agent: ``locate_bug`` — runs its OWN inner Coordinator with a
    read-only registry filtered through ``self.tool_allowlist``, looping
    over (list_files → read_file) until it finds the suspect. Returns a
    typed ``(file_path, line, hypothesis)`` result.
  - Verifier: ``run_tests`` — actually executes the in-memory test
    function and reports pass/fail with evidence (P4 grounded).

The parent planner is observation-driven: it dispatches ``locate_bug``,
reads the typed result, and emits the write/verify chain *from the
hypothesis*. Nothing about the fix is precomputed.

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
    Plan,
    PlanRequest,
    Planner,
    Registry,
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


class ListIn(BaseModel):
    pass


class ListOut(BaseModel):
    paths: list[str]


class ListFilesTool(Tool):
    name = "list_files"
    description = "List every path in the working tree."
    input_schema = ListIn
    output_schema = ListOut
    side_effects = ("fs:read",)

    def _execute(self, payload):
        return ListOut(paths=sorted(FAKE_FS.keys()))


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
    expected: str = Field(..., description="The literal the symptom implies should appear.")


class LocateResult(SubAgentResult):
    file_path: str
    fixed_content: str
    hypothesis: str


class _LocatePlanner(Planner):
    """Inner planner for ``LocateBugAgent``. Lists once, then walks files
    sequentially. On each ``read_file`` result it inspects the content
    for the symptom marker; once found it finishes with the patch.

    All decisions come from observations — this is an actual
    plan→observe→re-plan loop, which is what the SPEC §4.2 "When to use
    a sub-agent" rationale is about.
    """

    def __init__(self, task: LocateTask) -> None:
        self._task = task
        self._paths: list[str] | None = None
        self._cursor = 0
        self._finish_payload: dict[str, Any] | None = None

    def plan(self, request: PlanRequest) -> Plan:
        obs = request.last_observation or {}
        out = obs.get("output") if obs else None

        # --- step 1: pull the candidate list once -----------------------
        if self._paths is None:
            if out and "paths" in out:
                self._paths = [p for p in out["paths"] if not p.startswith("test_") and p.endswith(".py")]
                return self._next_read()
            return Plan(
                next_action=ToolAction(tool="list_files", arguments={}, rationale="enumerate candidates")
            )

        # --- step 2: inspect each file's content ------------------------
        if out and "content" in out:
            path = self._paths[self._cursor - 1]
            content = out["content"]
            if "Bug" in content:
                fixed = content.replace("Bug", "World")
                self._finish_payload = {
                    "file_path": path,
                    "fixed_content": fixed,
                    "hypothesis": (
                        f"{path} contains a literal with 'Bug' but the symptom "
                        f"({self._task.symptom!r}) implies it should read "
                        f"{self._task.expected!r}; rewrite literal."
                    ),
                }
                return Plan(terminate=True, next_action=FinishAction(result=self._finish_payload))
            # no bug here — advance
            if self._cursor < len(self._paths):
                return self._next_read()

        # exhausted without a finding
        self._finish_payload = {
            "file_path": "",
            "fixed_content": "",
            "hypothesis": "No suspect file found in the working tree.",
        }
        return Plan(terminate=True, next_action=FinishAction(result=self._finish_payload))

    def _next_read(self) -> Plan:
        assert self._paths is not None
        path = self._paths[self._cursor]
        self._cursor += 1
        return Plan(
            next_action=ToolAction(
                tool="read_file",
                arguments={"path": path},
                rationale=f"inspect candidate {self._cursor}/{len(self._paths)}",
            )
        )


class LocateBugAgent(SubAgent):
    name = "locate_bug"
    description = "Given a symptom, find the source file at fault and propose a fix."
    task_schema = LocateTask
    result_schema = LocateResult
    tool_allowlist = ("list_files", "read_file")  # strict subset; cannot write
    system_prompt = "You are a careful bug-locator. Cite file:line for every claim."
    default_max_iterations = 8

    def _run(self, task: LocateTask, scratch: dict[str, Any]) -> LocateResult:
        # Build the filtered inner registry from the parent's tools (§4.2):
        # only ``list_files`` and ``read_file`` cross the boundary, regardless
        # of what the parent has registered.
        inner_registry = self.build_inner_registry(scratch["parent_registry"])
        inner_budget = self.carve_budget(scratch["parent_budget"])

        inner = Coordinator(
            planner=_LocatePlanner(task),
            registry=inner_registry,
            budget=inner_budget,
            depth=scratch["depth"],  # =1 when called from the depth-0 parent
        )
        inner_result = inner.run(f"Locate the source of: {task.symptom!r}")
        # The planner's FinishAction.result is what we typed-return.
        payload = inner_result.result
        return LocateResult(
            file_path=payload.get("file_path", ""),
            fixed_content=payload.get("fixed_content", ""),
            hypothesis=payload.get("hypothesis", "no hypothesis"),
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


# -------- Parent planner: observes locate result, drives the fix -----------


class _ParentPlanner(Planner):
    """Dispatches locate, observes the typed result, then emits write +
    verify *from that result*. This is the wiring the playbook calls for:
    the sub-agent finds something the parent didn't know, and the parent
    plans the next step on what it learned.
    """

    def __init__(self, task_inputs: dict[str, str]) -> None:
        self._task_inputs = task_inputs
        self._phase = "locate"
        self._fix: dict[str, str] | None = None

    def plan(self, request: PlanRequest) -> Plan:
        obs = request.last_observation or {}

        if self._phase == "locate":
            self._phase = "await_locate"
            return Plan(
                next_action=SubAgentAction(
                    subagent="locate_bug",
                    task=self._task_inputs,
                    rationale="parent context shouldn't carry the codebase walk",
                )
            )

        if self._phase == "await_locate":
            out = obs.get("output") or {}
            if not out.get("file_path"):
                return Plan(
                    terminate=True,
                    next_action=FinishAction(
                        result={"error": "locator returned no candidate",
                                "hypothesis": out.get("hypothesis", "")}
                    ),
                )
            self._fix = {"path": out["file_path"], "content": out["fixed_content"]}
            self._phase = "write"
            return Plan(
                next_action=ToolAction(
                    tool="write_file",
                    arguments=self._fix,
                    rationale=f"apply fix from locator hypothesis: {out['hypothesis']!r}",
                )
            )

        if self._phase == "write":
            self._phase = "verify"
            return Plan(
                next_action=VerifyAction(
                    verifier="run_tests",
                    target={"note": f"post-patch verification of {self._fix['path']}"},
                    rationale="gate the deliverable on grounded evidence (P4)",
                )
            )

        # self._phase == "verify": read the verdict and finish.
        verdict = (obs.get("verdict") or {}) if obs else {}
        result = {
            "patched": self._fix["path"] if self._fix else None,
            "tests": "passing" if verdict.get("passed") else "failing",
            "evidence": verdict.get("evidence", []),
        }
        return Plan(terminate=True, next_action=FinishAction(result=result))


# -------- Drive --------------------------------------------------------------


def main() -> None:
    # Reset the bug each run so the example is reproducible if FAKE_FS
    # was modified by a prior in-process run.
    FAKE_FS["greet.py"] = "def greet():\n    return 'Hello, Bug!'\n"

    registry = Registry()
    registry.register_tool(ListFilesTool())
    registry.register_tool(ReadFileTool())
    registry.register_tool(WriteFileTool())
    registry.register_subagent(LocateBugAgent())
    registry.register_verifier(RunTestsVerifier())

    parent_planner = _ParentPlanner(
        task_inputs={
            "symptom": "greet() returns the wrong string",
            "expected": "Hello, World!",
        }
    )

    coord = Coordinator(
        planner=parent_planner,
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
