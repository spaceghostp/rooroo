"""Tool contract: typed in/out, structured errors, never throws."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness import Tool


def test_happy_path(echo_tool):
    res = echo_tool.call({"text": "hello"})
    assert res.ok is True
    assert res.output == {"echoed": "hello"}
    assert res.error is None


def test_input_validation_returns_structured_error(echo_tool):
    res = echo_tool.call({"wrong_key": "x"})
    assert res.ok is False
    assert res.error is not None
    assert res.error.kind == "input_validation"


def test_tool_failure_returns_structured_error(boom_tool):
    res = boom_tool.call({})
    assert res.ok is False
    assert res.error.kind == "intentional"
    assert "test" in res.error.details["reason"] if isinstance(res.error.details, dict) else True


def test_internal_exception_is_caught_not_raised():
    """A misbehaving tool that raises a non-ToolFailure must not bubble."""

    class _BadIn(BaseModel):
        pass

    class _BadOut(BaseModel):
        pass

    class _Bad(Tool):
        name = "bad"
        description = "Raises ValueError."
        input_schema = _BadIn
        output_schema = _BadOut

        def _execute(self, payload):
            raise ValueError("uncaught")

    res = _Bad().call({})
    assert res.ok is False
    assert res.error.kind == "tool_internal_error"


def test_manifest_includes_schemas(echo_tool):
    m = echo_tool.manifest()
    assert m["name"] == "echo"
    assert "input_schema" in m and "output_schema" in m
    assert m["side_effects"] == []


def test_missing_name_rejected_at_class_creation():
    from harness.errors import SchemaContractError

    class _In(BaseModel): pass
    class _Out(BaseModel): pass

    with pytest.raises(SchemaContractError):
        class _Nameless(Tool):
            # no name
            description = "x"
            input_schema = _In
            output_schema = _Out

            def _execute(self, payload):  # pragma: no cover - never reached
                return _Out()
