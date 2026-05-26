"""Append-only structured trace. See §8 of the spec.

The trace is the substrate for debugging, replay, and evals. Every
coordinator decision and primitive invocation appends exactly one entry.
Entries are immutable once written (the public API gives you no way to
mutate one). The on-disk format is JSONL so you can tail it during a
session and grep it after.

We deliberately do not buffer: each ``append`` flushes. That makes a
crashed session still inspectable.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .types import ActionType, Cost, Outcome


@dataclass(frozen=True)
class TraceEntry:
    step_id: str
    parent_step_id: str | None
    timestamp: float
    action_type: ActionType
    action_input: dict[str, Any]
    action_output: dict[str, Any]
    cost: Cost
    outcome: Outcome
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "parent_step_id": self.parent_step_id,
            "timestamp": self.timestamp,
            "action_type": self.action_type.value,
            "action_input": self.action_input,
            "action_output": self.action_output,
            "cost": self.cost.model_dump(),
            "outcome": self.outcome.value,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEntry":
        return cls(
            step_id=data["step_id"],
            parent_step_id=data.get("parent_step_id"),
            timestamp=data["timestamp"],
            action_type=ActionType(data["action_type"]),
            action_input=data["action_input"],
            action_output=data["action_output"],
            cost=Cost(**data["cost"]),
            outcome=Outcome(data["outcome"]),
            notes=data.get("notes"),
        )


class Trace:
    """Append-only log. Optionally backed by a JSONL file on disk."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._entries: list[TraceEntry] = []
        self._path: Path | None = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate on session start. Resumption is a v2 concern.
            self._path.write_text("")

    @property
    def path(self) -> str | None:
        return str(self._path) if self._path else None

    def append(
        self,
        *,
        action_type: ActionType,
        action_input: dict[str, Any],
        action_output: dict[str, Any],
        cost: Cost,
        outcome: Outcome,
        parent_step_id: str | None = None,
        notes: str | None = None,
    ) -> TraceEntry:
        entry = TraceEntry(
            step_id=uuid.uuid4().hex,
            parent_step_id=parent_step_id,
            timestamp=time.time(),
            action_type=action_type,
            action_input=action_input,
            action_output=action_output,
            cost=cost,
            outcome=outcome,
            notes=notes,
        )
        self._entries.append(entry)
        if self._path is not None:
            with self._path.open("a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def entries(self) -> list[TraceEntry]:
        # Return a copy so callers can't mutate our internal list.
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[TraceEntry]:
        return iter(self.entries())

    # ---- rollups --------------------------------------------------------

    def cost_rollup(self) -> Cost:
        total = Cost()
        for e in self._entries:
            total = total + e.cost
        return total

    def by_type(self, action_type: ActionType) -> list[TraceEntry]:
        return [e for e in self._entries if e.action_type == action_type]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Trace":
        """Load a JSONL trace file for inspection or replay."""
        trace = cls()  # no-path so we don't truncate
        p = Path(path)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            trace._entries.append(TraceEntry.from_dict(json.loads(line)))
        return trace
