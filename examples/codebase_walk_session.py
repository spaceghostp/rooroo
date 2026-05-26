"""Playbook Pattern 4 — Depth-2 Sub-agent (recursive search space).

A codebase is the canonical recursive search space — the spec permits
depth-2 sub-agents exactly because tree-shaped exploration doesn't
collapse into a single reasoning loop. This example shows the full
chain:

  Top Coordinator (depth=0)
    └─ WalkDirAgent (depth=1)            -- runs its OWN inner Coordinator
         └─ ExtractSymbolsAgent (depth=2) -- has depth2_justification (P3)

If ``ExtractSymbolsAgent.depth2_justification`` were empty, the inner
coordinator's call into ``extract_symbols.run(depth=2)`` would raise
``DepthLimitExceeded``. We demonstrate that refusal at the bottom of
the run, against a sibling sub-agent that doesn't carry justification.

Run:

    python -m examples.codebase_walk_session
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from harness import (
    Budget,
    Coordinator,
    DepthLimitExceeded,
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
)


# -------- Fake codebase ------------------------------------------------------

FAKE_REPO: dict[str, Any] = {
    "src/": {
        "auth.py": "def login(user):\n    return True\n\ndef logout(user):\n    pass\n",
        "billing.py": "def charge(amount):\n    return amount * 1.05\n",
        "util/": {
            "log.py": "def info(msg):\n    print(msg)\n\ndef warn(msg):\n    print('!', msg)\n",
        },
    },
    "README.md": "# fake repo\n",
}


def _walk_path(repo: dict[str, Any], path: str) -> Any:
    if path in ("", "/"):
        return repo
    node = repo
    for part in [p for p in path.strip("/").split("/") if p]:
        key = part if part in node else part + "/"
        if key not in node:
            raise KeyError(path)
        node = node[key]
    return node


# -------- Tools (used by both sub-agents) ------------------------------------


class ListIn(BaseModel):
    path: str


class ListOut(BaseModel):
    entries: list[str]  # trailing '/' marks directories


class ListDirTool(Tool):
    name = "list_dir"
    description = "List entries at a path in the working tree."
    input_schema = ListIn
    output_schema = ListOut
    side_effects = ("fs:read",)

    def _execute(self, payload):
        try:
            node = _walk_path(FAKE_REPO, payload.path)
        except KeyError:
            raise ToolFailure("not_found", f"no path {payload.path!r}")
        if not isinstance(node, dict):
            raise ToolFailure("not_a_dir", f"{payload.path!r} is a file")
        return ListOut(entries=sorted(node.keys()))


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
        try:
            node = _walk_path(FAKE_REPO, payload.path)
        except KeyError:
            raise ToolFailure("not_found", f"no file {payload.path!r}")
        if isinstance(node, dict):
            raise ToolFailure("is_a_dir", f"{payload.path!r} is a directory")
        return ReadOut(content=node)


# -------- Inner sub-agent (depth=2): extract_symbols ------------------------


class ExtractTask(SubAgentTask):
    file_path: str


class ExtractResult(SubAgentResult):
    file_path: str
    symbols: list[str]


class ExtractSymbolsAgent(SubAgent):
    name = "extract_symbols"
    description = "Parse a Python source file and return top-level def names."
    task_schema = ExtractTask
    result_schema = ExtractResult
    tool_allowlist = ("read_file",)
    # P3 — explicit justification recorded in the class:
    depth2_justification = (
        "Called from walk_dir which is itself a sub-agent. The codebase is "
        "a recursive search space (dir → file → symbol); this is the "
        "terminal extraction step. Bounded by token + iteration budget."
    )

    _DEF = re.compile(r"^def\s+(\w+)\(", re.MULTILINE)

    def _run(self, task: ExtractTask, scratch: dict) -> ExtractResult:
        try:
            node = _walk_path(FAKE_REPO, task.file_path)
        except KeyError:
            return ExtractResult(file_path=task.file_path, symbols=[])
        if not isinstance(node, str):
            return ExtractResult(file_path=task.file_path, symbols=[])
        return ExtractResult(
            file_path=task.file_path,
            symbols=self._DEF.findall(node),
        )


# -------- Outer sub-agent (depth=1): walk_dir -------------------------------


class WalkTask(SubAgentTask):
    root: str


class WalkResult(SubAgentResult):
    by_file: dict[str, list[str]]


class WalkDirAgent(SubAgent):
    """Recursively walks ``root`` and, for each .py file found, dispatches
    to ``extract_symbols``. Runs its own inner Coordinator at ``depth=1``
    so the inner coordinator's calls into ``extract_symbols`` resolve at
    ``depth=2`` — which is exactly when ``depth2_justification`` is
    consulted by the harness.
    """

    name = "walk_dir"
    description = "Walk a directory tree and collect symbols from each .py file."
    task_schema = WalkTask
    result_schema = WalkResult
    tool_allowlist = ("list_dir", "read_file")

    def _run(self, task: WalkTask, scratch: dict) -> WalkResult:
        # Build the inner registry — tools subset + the one depth-2 sub-agent.
        inner_registry = Registry()
        inner_registry.register_tool(ListDirTool())
        inner_registry.register_tool(ReadFileTool())
        inner_registry.register_subagent(ExtractSymbolsAgent())

        # Walk the tree breadth-first, gathering .py files. We do this with
        # real tool calls inside the inner coordinator so the harness sees
        # the work and the depth chain stays honest.
        py_files: list[str] = []

        def _collect(path: str) -> list[str]:
            tool = inner_registry.tools["list_dir"]
            res = tool.call({"path": path})
            if not res.ok:
                return []
            return res.output["entries"]

        stack = [task.root]
        while stack:
            here = stack.pop()
            for name in _collect(here):
                child = f"{here.rstrip('/')}/{name}" if here else name
                if name.endswith("/"):
                    stack.append(child.rstrip("/"))
                elif name.endswith(".py"):
                    py_files.append(child)

        # Now dispatch extract_symbols once per file via the inner
        # coordinator. depth=1 here means the inner coord will invoke
        # sub-agents at depth=2.
        actions: list = []
        for fp in py_files:
            actions.append(
                SubAgentAction(
                    subagent="extract_symbols",
                    task={"file_path": fp},
                    rationale=f"extract from {fp}",
                )
            )
        actions.append(FinishAction(result={"files": py_files}))

        inner = Coordinator(
            planner=ScriptedPlanner(actions=actions, finish_result={"files": py_files}),
            registry=inner_registry,
            budget=Budget(max_iterations=2 * len(py_files) + 4, max_tokens=50_000),
            depth=1,  # critical: makes inner sub-agent invocations land at depth=2
        )
        inner.run(f"Symbol-walk under {task.root}")

        # Pull the typed sub-agent outputs back out of the inner trace.
        from harness.types import ActionType

        by_file: dict[str, list[str]] = {}
        for entry in inner.trace.by_type(ActionType.SUBAGENT):
            out = entry.action_output.get("output") or {}
            if "file_path" in out and "symbols" in out:
                by_file[out["file_path"]] = out["symbols"]

        return WalkResult(by_file=by_file)


# -------- A sibling sub-agent that LACKS justification — used for the refusal demo


class _UnjustifiedTask(SubAgentTask):
    pass


class _UnjustifiedResult(SubAgentResult):
    pass


class UnjustifiedDepth2Agent(SubAgent):
    name = "no_just"
    description = "Identical shape, no depth2_justification — invoking at depth=2 is refused."
    task_schema = _UnjustifiedTask
    result_schema = _UnjustifiedResult
    # depth2_justification deliberately left empty

    def _run(self, task, scratch):
        return _UnjustifiedResult()


# -------- Drive --------------------------------------------------------------


def main() -> None:
    registry = Registry()
    registry.register_tool(ListDirTool())
    registry.register_tool(ReadFileTool())
    registry.register_subagent(WalkDirAgent())

    coord = Coordinator(
        planner=ScriptedPlanner(
            actions=[
                SubAgentAction(
                    subagent="walk_dir",
                    task={"root": "src"},
                    rationale="delegate codebase exploration to depth-1 walker",
                ),
                FinishAction(result={}),
            ],
        ),
        registry=registry,
        budget=Budget(max_iterations=10, max_tokens=200_000),
        session_dir=Path(".sessions/codebase_walk"),
    )
    result = coord.run("Index every Python symbol under src/.")
    print(json.dumps(result.model_dump(), indent=2, default=str))

    from harness.types import ActionType

    walker_entry = coord.trace.by_type(ActionType.SUBAGENT)[0]
    print("\nDepth-1 walker reported:")
    print(json.dumps(walker_entry.action_output["output"], indent=2))

    # ---- Refusal demo: depth-2 without justification is a hard refusal ----
    print("\nRefusal demo (P3): invoking an unjustified sub-agent at depth=2 →")
    try:
        UnjustifiedDepth2Agent().run({}, depth=2)
    except DepthLimitExceeded as e:
        print(f"  DepthLimitExceeded raised as expected: {e}")


if __name__ == "__main__":
    main()
