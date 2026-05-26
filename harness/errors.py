"""Exceptions raised by the harness machinery itself.

Primitive failures (a tool that crashed, a verifier that found a problem)
do NOT use exceptions — they return ``StructuredError`` or ``Verdict``.
These exceptions are reserved for things that should never happen at
runtime if the harness is wired up correctly: budget exhaustion at the
machinery layer, depth-cap violations, schema-contract violations.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all harness-internal errors."""


class DepthLimitExceeded(HarnessError):
    """Raised when a sub-agent invocation would exceed the depth cap (P3).

    This is intentionally a hard refusal — not a recoverable error — because
    a misbehaving planner trying to climb the ladder is exactly the kind of
    drift the spec is designed to prevent.
    """


class BudgetExhausted(HarnessError):
    """Raised internally when a sub-budget runs out mid-call.

    Caught and converted to ``Outcome.BUDGET`` by the coordinator; never
    surfaced to the caller. ``SessionResult.incomplete`` is the user-facing
    representation.
    """


class SchemaContractError(HarnessError):
    """Raised at registration time when a primitive lacks required schemas.

    Acceptance criterion 7: 'reject (at design-review time) any sub-agent spec
    without a typed input/output schema'. We can't review designs at runtime,
    but we can refuse to register primitives that don't carry their schemas.
    """


class UnknownPrimitive(HarnessError):
    """Planner referenced a primitive name that isn't registered."""


class GroundingError(HarnessError):
    """A verifier returned a verdict but reported zero evidence items.

    Per §4.3 / P4, a 'verifier' with no external evidence is commentary, not
    verification. We refuse to count its verdict toward retry decisions.
    """


class RetryBudgetExhausted(HarnessError):
    """A verifier produced more failed verdicts in this session than
    ``Verifier.max_retries`` permits.

    Per §4.3: "Verifier failures trigger at most N retries (default N=2),
    then escalate to coordinator. No unbounded refine loops." The
    coordinator catches this internally and converts it into an
    ``incomplete=True`` SessionResult — it never bubbles to the caller.
    """


class AllowlistViolation(HarnessError):
    """A sub-agent attempted to expose a tool outside its declared
    ``tool_allowlist`` to its inner coordinator (§4.2).

    Raised by ``SubAgent.build_inner_registry`` when an allowlist entry
    references a tool the parent registry doesn't own, or when extra
    tools are requested beyond the allowlist.
    """
