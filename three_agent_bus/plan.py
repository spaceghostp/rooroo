from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import Constraint


@dataclass
class PlanChange:
    version: int
    op: str
    constraint_id: str


class Plan:
    def __init__(self) -> None:
        self._constraints: dict[str, Constraint] = {}
        self._history: list[PlanChange] = []
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def history(self) -> list[PlanChange]:
        return list(self._history)

    def add(self, constraint: Constraint) -> int:
        existing = self._constraints.get(constraint.id)
        if existing is not None and existing.is_active():
            raise ValueError(f"constraint {constraint.id} already active")
        self._version += 1
        constraint.version_introduced = self._version
        constraint.version_removed = None
        self._constraints[constraint.id] = constraint
        self._history.append(PlanChange(self._version, "add", constraint.id))
        return self._version

    def remove(self, constraint_id: str) -> int:
        c = self._constraints.get(constraint_id)
        if c is None or not c.is_active():
            raise ValueError(f"constraint {constraint_id} not active")
        self._version += 1
        c.version_removed = self._version
        self._history.append(PlanChange(self._version, "remove", constraint_id))
        return self._version

    def is_active(self, constraint_id: str) -> bool:
        c = self._constraints.get(constraint_id)
        return c is not None and c.is_active()

    def get(self, constraint_id: str) -> Optional[Constraint]:
        return self._constraints.get(constraint_id)

    def active_constraints(self) -> list[Constraint]:
        return [c for c in self._constraints.values() if c.is_active()]

    def scopes(self) -> set[str]:
        return {c.scope for c in self.active_constraints()}
