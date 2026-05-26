"""External, typed, observable working state. See §6 of the spec.

Two backends ship in v1:

- ``InMemoryState`` — for tests and ephemeral sessions
- ``FileSystemState`` — JSON files under a session directory

Both go through the same typed accessor API. Every mutation is appended to
a change-log so the trace can show a diff per step (per §8: 'Diff view').

Important: state is *not* a free-form dict from the coordinator's
perspective. The coordinator and primitives only manipulate state via
``get``, ``put``, ``update``, ``delete`` — there is no "raw" mode. This
keeps the change-log honest.
"""

from __future__ import annotations

import abc
import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class StateChange:
    """One entry in the change-log."""

    timestamp: float
    op: str  # put | update | delete
    key: str
    before: Any
    after: Any


class State(abc.ABC):
    """Abstract typed key-value store with a change-log."""

    def __init__(self) -> None:
        self._changes: list[StateChange] = []

    # ---- typed accessors -------------------------------------------------

    @abc.abstractmethod
    def _read(self, key: str) -> Any | None: ...

    @abc.abstractmethod
    def _write(self, key: str, value: Any) -> None: ...

    @abc.abstractmethod
    def _erase(self, key: str) -> None: ...

    @abc.abstractmethod
    def keys(self) -> Iterable[str]: ...

    def get(self, key: str, default: Any = None) -> Any:
        value = self._read(key)
        return default if value is None else copy.deepcopy(value)

    def put(self, key: str, value: Any) -> None:
        before = self._read(key)
        self._write(key, value)
        self._changes.append(
            StateChange(
                timestamp=time.time(),
                op="put",
                key=key,
                before=copy.deepcopy(before),
                after=copy.deepcopy(value),
            )
        )

    def update(self, key: str, patch: dict[str, Any]) -> None:
        """Merge ``patch`` into the dict at ``key`` (creates if missing)."""
        current = self._read(key) or {}
        if not isinstance(current, dict):
            raise TypeError(
                f"state.update on key '{key}' requires a dict value, got {type(current).__name__}"
            )
        before = copy.deepcopy(current)
        current = {**current, **patch}
        self._write(key, current)
        self._changes.append(
            StateChange(
                timestamp=time.time(),
                op="update",
                key=key,
                before=before,
                after=copy.deepcopy(current),
            )
        )

    def delete(self, key: str) -> None:
        before = self._read(key)
        self._erase(key)
        self._changes.append(
            StateChange(
                timestamp=time.time(),
                op="delete",
                key=key,
                before=copy.deepcopy(before),
                after=None,
            )
        )

    def changes_since(self, index: int) -> list[StateChange]:
        return list(self._changes[index:])

    def change_count(self) -> int:
        return len(self._changes)

    def summary(self) -> dict[str, Any]:
        """Cheap structured view of the whole store, for planner prompts."""
        return {k: self.get(k) for k in self.keys()}


class InMemoryState(State):
    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._store: dict[str, Any] = copy.deepcopy(initial or {})

    def _read(self, key: str) -> Any | None:
        return self._store.get(key)

    def _write(self, key: str, value: Any) -> None:
        self._store[key] = copy.deepcopy(value)

    def _erase(self, key: str) -> None:
        self._store.pop(key, None)

    def keys(self) -> Iterable[str]:
        return list(self._store.keys())


class FileSystemState(State):
    """One JSON file per key under ``root``.

    The keys are URL-encoded so arbitrary characters are safe; the on-disk
    layout is meant for humans to grep, not for programs to traverse.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        from urllib.parse import quote

        return self.root / f"{quote(key, safe='')}.json"

    def _read(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def _write(self, key: str, value: Any) -> None:
        self._path(key).write_text(json.dumps(value, indent=2, sort_keys=True))

    def _erase(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def keys(self) -> Iterable[str]:
        from urllib.parse import unquote

        return [unquote(p.stem) for p in self.root.glob("*.json")]
