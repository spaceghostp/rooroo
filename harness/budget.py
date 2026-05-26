"""Budget tracking. See §7 of the spec.

A ``Budget`` is consumed in three dimensions — tokens, iterations, wall-clock.
It is also the unit of carving: a sub-agent gets a child budget whose caps
are a strict slice of the parent's remaining budget, and consumption on the
child propagates up to the parent so the session-level totals stay accurate.

Termination on budget is a first-class outcome, not an exception, at the
session boundary. Internally we raise ``BudgetExhausted`` for control flow
and the coordinator catches it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .errors import BudgetExhausted


@dataclass
class Budget:
    """Mutable counter across three dimensions.

    Construction is by caps (the ``max_*`` fields). Counters (``tokens_used``
    etc) increment as work happens. ``check()`` is the canonical "do we still
    have headroom?" probe — the coordinator calls it before every action,
    not just at iteration boundaries, as the spec requires.
    """

    max_tokens: int = 1_000_000
    max_iterations: int = 50
    max_wall_seconds: float = 600.0

    tokens_used: int = 0
    iterations_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    parent: "Budget | None" = None

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iterations_used)

    def remaining_wall_seconds(self) -> float:
        return max(0.0, self.max_wall_seconds - (time.monotonic() - self.started_at))

    def exhausted(self) -> str | None:
        """Return the name of the first dimension that is out, or ``None``."""
        if self.remaining_tokens() == 0:
            return "tokens"
        if self.remaining_iterations() == 0:
            return "iterations"
        if self.remaining_wall_seconds() <= 0:
            return "wall_clock"
        return None

    def check(self) -> None:
        """Raise ``BudgetExhausted`` if any dimension is empty.

        The exception carries the dimension name in ``args[0]`` so the
        coordinator can populate ``incomplete_reason`` cleanly.
        """
        dim = self.exhausted()
        if dim is not None:
            raise BudgetExhausted(dim)

    def consume_tokens(self, n: int) -> None:
        self.tokens_used += n
        if self.parent is not None:
            self.parent.consume_tokens(n)

    def consume_iteration(self) -> None:
        self.iterations_used += 1
        # Iterations are local — a sub-agent's loop doesn't burn the parent's
        # iteration count. Tokens and wall-time DO propagate because they
        # represent shared resources.

    def carve(
        self,
        *,
        max_tokens: int | None = None,
        max_iterations: int | None = None,
        max_wall_seconds: float | None = None,
    ) -> "Budget":
        """Carve a child budget for a sub-agent.

        Caps are clamped to the parent's remaining headroom — you cannot
        grant a sub-agent more budget than you currently have available.
        """
        child_tokens = min(
            max_tokens if max_tokens is not None else self.remaining_tokens(),
            self.remaining_tokens(),
        )
        child_iters = (
            max_iterations
            if max_iterations is not None
            else max(1, self.remaining_iterations() // 2)
        )
        child_wall = min(
            max_wall_seconds
            if max_wall_seconds is not None
            else self.remaining_wall_seconds(),
            self.remaining_wall_seconds() if math.isfinite(self.remaining_wall_seconds()) else max_wall_seconds or 60.0,
        )
        return Budget(
            max_tokens=child_tokens,
            max_iterations=child_iters,
            max_wall_seconds=child_wall,
            parent=self,
        )

    def snapshot(self) -> dict[str, float | int]:
        """Serializable view, useful for trace metadata."""
        return {
            "tokens_used": self.tokens_used,
            "iterations_used": self.iterations_used,
            "wall_elapsed": round(time.monotonic() - self.started_at, 3),
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
            "max_wall_seconds": self.max_wall_seconds,
        }
