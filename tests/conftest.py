"""Shared fixtures: a tiny vocabulary of primitives used across tests.

Keep these minimal — each one exercises exactly one harness concept.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness import (
    SubAgent,
    SubAgentResult,
    SubAgentTask,
    Tool,
    ToolFailure,
    Verdict,
    Verifier,
)


# -- a pure tool --

class _EchoIn(BaseModel):
    text: str


class _EchoOut(BaseModel):
    echoed: str


class EchoTool(Tool):
    name = "echo"
    description = "Return the input string unchanged."
    input_schema = _EchoIn
    output_schema = _EchoOut

    def _execute(self, payload: _EchoIn) -> _EchoOut:
        return _EchoOut(echoed=payload.text)


# -- a tool that raises ToolFailure --

class _BoomIn(BaseModel):
    pass


class _BoomOut(BaseModel):
    pass


class BoomTool(Tool):
    name = "boom"
    description = "Always fails with a structured error."
    input_schema = _BoomIn
    output_schema = _BoomOut

    def _execute(self, payload: _BoomIn) -> _BoomOut:
        raise ToolFailure("intentional", "intended failure", reason="test")


# -- a sub-agent --

class _UppercaseTask(SubAgentTask):
    text: str


class _UppercaseResult(SubAgentResult):
    text: str


class UppercaseSubAgent(SubAgent):
    name = "uppercase"
    description = "Uppercase the input text."
    task_schema = _UppercaseTask
    result_schema = _UppercaseResult

    def _run(self, task: _UppercaseTask, scratch: dict) -> _UppercaseResult:
        return _UppercaseResult(text=task.text.upper())


# -- a grounded verifier --

class _NonEmptyTarget(BaseModel):
    text: str


class NonEmptyVerifier(Verifier):
    name = "nonempty"
    description = "Pass if text is non-empty (grounded on schema_validate)."
    target_schema = _NonEmptyTarget
    evidence_sources = ("schema_validate",)

    def _verify(self, target: _NonEmptyTarget) -> Verdict:
        if target.text:
            return Verdict(
                passed=True,
                evidence=[f"schema_validate: len(text)={len(target.text)} > 0"],
            )
        return Verdict(
            passed=False,
            evidence=["schema_validate: text is empty"],
            suggested_revisions=["provide a non-empty string"],
        )


# -- a 'verifier' that returns no evidence (commentary, not verification) --

class _CommentaryVerifier(Verifier):
    name = "commentary"
    description = "Bad verifier: returns no evidence."
    target_schema = _NonEmptyTarget
    evidence_sources = ("schema_validate",)  # declared but ignored

    def _verify(self, target):
        return Verdict(passed=True, evidence=[], notes="vibes only")


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


@pytest.fixture
def boom_tool() -> BoomTool:
    return BoomTool()


@pytest.fixture
def uppercase_subagent() -> UppercaseSubAgent:
    return UppercaseSubAgent()


@pytest.fixture
def nonempty_verifier() -> NonEmptyVerifier:
    return NonEmptyVerifier()


@pytest.fixture
def commentary_verifier() -> _CommentaryVerifier:
    return _CommentaryVerifier()
