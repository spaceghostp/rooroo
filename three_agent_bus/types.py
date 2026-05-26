from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Constraint:
    id: str
    kind: str
    scope: str
    description: str
    version_introduced: int = 0
    version_removed: Optional[int] = None

    def is_active(self) -> bool:
        return self.version_removed is None


@dataclass
class Message:
    id: str
    topic: str
    sender: str
    timestamp: str
    seq: int
    type: str
    payload: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Message":
        return cls(**json.loads(s))
