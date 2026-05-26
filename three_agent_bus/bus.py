from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import Message


class Bus:
    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._messages: list[Message] = []
        self._seq_per_topic: dict[str, int] = {}
        self._log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def post(self, topic: str, sender: str, type: str, payload: dict) -> Message:
        seq = self._seq_per_topic.get(topic, 0) + 1
        self._seq_per_topic[topic] = seq
        msg = Message(
            id=str(uuid.uuid4()),
            topic=topic,
            sender=sender,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seq=seq,
            type=type,
            payload=payload,
        )
        self._messages.append(msg)
        if self._log_path is not None:
            with self._log_path.open("a") as f:
                f.write(msg.to_json() + "\n")
        return msg

    def all(self) -> list[Message]:
        return list(self._messages)

    def for_topic(self, topic: str) -> list[Message]:
        return [m for m in self._messages if m.topic == topic]

    def by_type(self, type: str) -> list[Message]:
        return [m for m in self._messages if m.type == type]

    def find(self, message_id: str) -> Optional[Message]:
        for m in self._messages:
            if m.id == message_id:
                return m
        return None
