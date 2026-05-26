"""Playbook Pattern 6 — Resumable Long-Run.

Process a queue of items across multiple budget windows. The first
``Coordinator.run`` exhausts its iteration budget before the queue is
empty and returns ``incomplete=True`` (a first-class outcome per §7).
A second ``Coordinator.run`` against the same ``session_dir`` reads
the persisted state and picks up where the first one stopped.

What this demonstrates:

  - P5: external state is the substrate for resumption (not the trace)
  - P6: ``incomplete=True`` is the handoff, not an exception
  - The planner derives "where am I" from ``state_summary``, never from
    the previous run's trace

The "work" is trivially small — uppercase 10 strings — so the example
runs in milliseconds. The shape is what matters: a planner that reads
state, picks the next item, dispatches a tool that writes state, and
finishes when the queue is empty.

Run:

    python -m examples.resumable_session
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import BaseModel

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Plan,
    PlanRequest,
    Planner,
    Registry,
    State,
    Tool,
    ToolAction,
)

SESSION_DIR = Path(".sessions/resumable")

QUEUE_KEY = "queue"
PROCESSED_KEY = "processed"

INPUT_ITEMS = ["alpha", "bravo", "charlie", "delta", "echo",
               "foxtrot", "golf", "hotel", "india", "juliet"]


# -------- Tool: process_item ------------------------------------------------
#
# A real tool would mutate external systems; here it just transforms a
# string. The important property is that its output is what the planner
# uses to *update state* on the next iteration.


class ProcessIn(BaseModel):
    item: str


class ProcessOut(BaseModel):
    original: str
    transformed: str


class ProcessItemTool(Tool):
    name = "process_item"
    description = "Uppercase a single queue item."
    input_schema = ProcessIn
    output_schema = ProcessOut
    side_effects = ()
    cost_budget_wall_ms = 5

    def _execute(self, payload):
        return ProcessOut(original=payload.item, transformed=payload.item.upper())


# -------- Planner: state-driven, resumption-safe ----------------------------


class ResumablePlanner(Planner):
    """All decisions are derived from state — never from the trace. That's
    what makes the same planner class work for both the initial run and
    the resumed run.

    Constructed unbound; call ``bind(state)`` once the coordinator has
    created its state handle. The planner writes results back to state
    before emitting the next action, which is how continuity survives a
    budget-exhausted run.
    """

    def __init__(self) -> None:
        self._state: State | None = None

    def bind(self, state: State) -> "ResumablePlanner":
        self._state = state
        return self

    def plan(self, request: PlanRequest) -> Plan:
        assert self._state is not None, "call bind(state) before run()"
        # Step 1: if the previous tool returned, record it to state.
        obs = request.last_observation or {}
        out = obs.get("output") if obs else None
        if out and "transformed" in out:
            processed = self._state.get(PROCESSED_KEY) or []
            processed.append(out)
            self._state.put(PROCESSED_KEY, processed)
            queue = self._state.get(QUEUE_KEY) or []
            queue = [i for i in queue if i != out["original"]]
            self._state.put(QUEUE_KEY, queue)

        # Step 2: decide what to do based purely on state.
        queue = self._state.get(QUEUE_KEY) or []
        if not queue:
            return Plan(
                terminate=True,
                next_action=FinishAction(
                    result={"processed_count": len(self._state.get(PROCESSED_KEY) or [])}
                ),
            )

        return Plan(
            next_action=ToolAction(
                tool="process_item",
                arguments={"item": queue[0]},
                rationale=f"{len(queue)} items remaining",
            )
        )


# -------- Drive --------------------------------------------------------------


def _build_coordinator(session_dir: Path, iter_budget: int) -> Coordinator:
    registry = Registry()
    registry.register_tool(ProcessItemTool())
    planner = ResumablePlanner()
    coord = Coordinator(
        planner=planner,
        registry=registry,
        budget=Budget(max_iterations=iter_budget, max_tokens=100_000, max_wall_seconds=10),
        session_dir=session_dir,
    )
    planner.bind(coord.state)  # state handle only exists after Coordinator __init__
    return coord


def _seed_queue_if_empty(coord: Coordinator) -> None:
    if not coord.state.get(QUEUE_KEY):
        coord.state.put(QUEUE_KEY, list(INPUT_ITEMS))
        coord.state.put(PROCESSED_KEY, [])


def main() -> None:
    # Start clean so the example is reproducible.
    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)

    # --- Run 1: deliberately undersized iteration budget. -------------
    # 4 iterations means we'll burn through ~3 items before exhaustion
    # (one iter per plan-then-tool round, plus the final plan that
    # discovers we're out of budget).
    coord1 = _build_coordinator(SESSION_DIR, iter_budget=4)
    _seed_queue_if_empty(coord1)
    print(f"Run 1: queue size = {len(coord1.state.get(QUEUE_KEY) or [])}")
    res1 = coord1.run("Process the queue (run 1).")
    print(json.dumps(res1.model_dump(), indent=2, default=str))
    processed_after_1 = coord1.state.get(PROCESSED_KEY) or []
    remaining_after_1 = coord1.state.get(QUEUE_KEY) or []
    print(f"\nAfter run 1: {len(processed_after_1)} processed, {len(remaining_after_1)} remaining\n")

    assert res1.incomplete, "expected run 1 to exhaust iterations"

    # --- Run 2: resume against the same session_dir. ------------------
    # The planner reads QUEUE_KEY from FileSystemState; the trace is a
    # fresh file (we truncate on each run by design — replay is what
    # historical traces are for) but state IS the continuity.
    coord2 = _build_coordinator(SESSION_DIR, iter_budget=50)
    print(f"Run 2: resumed queue size = {len(coord2.state.get(QUEUE_KEY) or [])}")
    res2 = coord2.run("Process the queue (run 2 — resumption).")
    print(json.dumps(res2.model_dump(), indent=2, default=str))
    processed_after_2 = coord2.state.get(PROCESSED_KEY) or []
    print(f"\nAfter run 2: {len(processed_after_2)} processed total")
    assert not res2.incomplete, "expected run 2 to complete"
    assert len(processed_after_2) == len(INPUT_ITEMS), (
        f"expected all {len(INPUT_ITEMS)} items processed, got {len(processed_after_2)}"
    )
    print("\nFinal processed list (in completion order):")
    for entry in processed_after_2:
        print(f"  {entry['original']:8s} → {entry['transformed']}")


if __name__ == "__main__":
    main()
