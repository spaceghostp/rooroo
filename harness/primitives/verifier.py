"""Verifiers — grounded checks. See §4.3 and P4.

A verifier MUST touch external reality. Subclasses declare the kinds of
evidence they consume (``evidence_sources``) and the harness audits each
verdict: a verdict with no evidence is flagged ``passed=False`` with a
``GroundingError`` recorded, and is never used to retry a generator.

The retry policy lives on the *verifier*, not the coordinator: the
coordinator just calls ``run`` and respects whatever ``Verdict`` comes
back. A bounded retry loop wrapping the generator-of-record is the
coordinator's responsibility (see ``coordinator.execute`` for VerifyAction).
"""

from __future__ import annotations

import abc
import time
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from ..errors import SchemaContractError
from ..types import Cost, Outcome, StructuredError, Verdict


class VerifierResult(BaseModel):
    """What ``Verifier.run`` returns. Distinct from ``Verdict``: this also
    carries the harness-level outcome (did the verifier itself crash?).
    """

    ok: bool  # did the verifier RUN successfully (regardless of pass/fail)?
    verdict: Verdict | None = None
    error: StructuredError | None = None
    cost: Cost = Cost()
    outcome: Outcome = Outcome.OK


# The set of evidence kinds the harness recognizes as grounded.
# A verifier must declare at least one. Free-form strings are permitted
# beyond this list (subclasses can extend), but at least one must be
# listed so the grounding audit has something to point at.
GROUNDED_EVIDENCE_KINDS: frozenset[str] = frozenset(
    {
        "execute_code",
        "fetch_source",
        "query_db",
        "typecheck",
        "schema_validate",
        "diff_canonical",
        "filesystem_read",
        "http_fetch",
    }
)


class Verifier(abc.ABC):
    """Subclasses MUST declare ``evidence_sources`` (P4) and implement
    ``_verify(target)`` returning a ``Verdict`` whose ``evidence`` list is
    non-empty if ``passed`` is set."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    target_schema: ClassVar[type[BaseModel]]
    evidence_sources: ClassVar[tuple[str, ...]] = ()
    max_retries: ClassVar[int] = 2  # retry policy per §4.3

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        if not cls.name:
            raise SchemaContractError(f"Verifier {cls.__name__} missing 'name'")
        if not hasattr(cls, "target_schema"):
            raise SchemaContractError(
                f"Verifier {cls.__name__} must declare target_schema"
            )
        if not cls.evidence_sources:
            raise SchemaContractError(
                f"Verifier {cls.__name__} must declare at least one "
                "evidence_source (P4: verifiers must touch reality)"
            )

    @abc.abstractmethod
    def _verify(self, target: BaseModel) -> Verdict: ...

    def run(self, target: dict[str, Any]) -> VerifierResult:
        started = time.monotonic()
        try:
            parsed = self.target_schema(**target)
        except ValidationError as e:
            return VerifierResult(
                ok=False,
                error=StructuredError(
                    kind="target_validation",
                    message=str(e),
                    details={"errors": e.errors()},
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        try:
            verdict = self._verify(parsed)
        except Exception as e:
            return VerifierResult(
                ok=False,
                error=StructuredError(
                    kind="verifier_internal_error",
                    message=f"{type(e).__name__}: {e}",
                ),
                cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
                outcome=Outcome.ERROR,
            )

        # Grounding audit: a verifier verdict with no evidence is commentary,
        # not verification. We rewrite it to passed=False so downstream retry
        # logic won't escalate, and we annotate the verdict explicitly so the
        # trace shows why we ignored the verifier's claim.
        if not verdict.evidence:
            verdict = Verdict(
                passed=False,
                evidence=[],
                suggested_revisions=verdict.suggested_revisions,
                notes=(
                    "GROUNDING_FAILED: verifier returned no evidence; "
                    "treating verdict as commentary (P4). Original notes: "
                    f"{verdict.notes!r}"
                ),
            )

        return VerifierResult(
            ok=True,
            verdict=verdict,
            cost=Cost(wall_ms=int((time.monotonic() - started) * 1000)),
            outcome=Outcome.OK,
        )

    @classmethod
    def manifest(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "target_schema": cls.target_schema.model_json_schema(),
            "evidence_sources": list(cls.evidence_sources),
            "max_retries": cls.max_retries,
        }
