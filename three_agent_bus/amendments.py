from __future__ import annotations

from .bus import Bus
from .types import Message


def pending(bus: Bus) -> list[Message]:
    voted_ids = {m.payload["amendment_id"] for m in bus.by_type("vote")}
    return [a for a in bus.by_type("amendment") if a.id not in voted_ids]


def ratified(bus: Bus) -> list[tuple[Message, Message]]:
    by_id = {m.id: m for m in bus.by_type("amendment")}
    out: list[tuple[Message, Message]] = []
    for v in bus.by_type("vote"):
        if v.payload.get("vote") != "accept":
            continue
        a = by_id.get(v.payload.get("amendment_id"))
        if a is not None:
            out.append((a, v))
    return out


def per_topic(bus: Bus) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in bus.by_type("amendment"):
        counts[m.topic] = counts.get(m.topic, 0) + 1
    return counts
