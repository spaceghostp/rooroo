"""Verifier contract: must touch reality (P4); commentary is rejected."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness import Verdict, Verifier
from harness.errors import SchemaContractError


def test_grounded_pass(nonempty_verifier):
    res = nonempty_verifier.run({"text": "hi"})
    assert res.ok is True
    assert res.verdict.passed is True
    assert res.verdict.evidence  # at least one evidence item


def test_grounded_fail(nonempty_verifier):
    res = nonempty_verifier.run({"text": ""})
    assert res.ok is True
    assert res.verdict.passed is False
    assert res.verdict.evidence
    assert res.verdict.suggested_revisions


def test_commentary_verifier_demoted_to_failed(commentary_verifier):
    """A verifier returning no evidence is commentary, not verification.

    The harness MUST refuse to count its 'passed' verdict.
    """
    res = commentary_verifier.run({"text": "anything"})
    assert res.ok is True
    assert res.verdict.passed is False
    assert "GROUNDING_FAILED" in res.verdict.notes


def test_missing_evidence_sources_rejected_at_class_creation():
    """P4: a verifier without declared evidence_sources cannot exist."""

    class _T(BaseModel): pass

    with pytest.raises(SchemaContractError):
        class _Bad(Verifier):
            name = "bad"
            description = "Has no evidence_sources."
            target_schema = _T
            # evidence_sources NOT set

            def _verify(self, target):  # pragma: no cover
                return Verdict(passed=True)


def test_target_validation_returns_structured_error(nonempty_verifier):
    res = nonempty_verifier.run({"not_text": 1})
    assert res.ok is False
    assert res.error.kind == "target_validation"


def test_internal_exception_is_caught():
    class _T(BaseModel): pass

    class _Crashy(Verifier):
        name = "crashy"
        description = "Raises in _verify."
        target_schema = _T
        evidence_sources = ("schema_validate",)

        def _verify(self, target):
            raise RuntimeError("boom")

    res = _Crashy().run({})
    assert res.ok is False
    assert res.error.kind == "verifier_internal_error"
