"""Playbook Pattern 1 — Pure Tool Pipeline.

Extract structured contact records from a free-text blob using two
deterministic tools (regex extractor + email normalizer). No sub-agent,
no verifier — the tools' typed outputs are the gate.

The planner here is scripted (no LLM), but the *shape* is exactly what a
real LLM planner would produce: pick tool, pass typed args, observe,
pick next tool, finish. Tools have side-effects declared and a wall_ms
cost budget so registration-time review can audit them.

Run:

    python -m examples.extract_session
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, EmailStr, Field

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Registry,
    ScriptedPlanner,
    Tool,
    ToolAction,
    ToolFailure,
)


# -------- Tool: extract_emails ------------------------------------------------


class ExtractEmailsIn(BaseModel):
    text: str = Field(..., min_length=1)


class ExtractEmailsOut(BaseModel):
    emails: list[str]


class ExtractEmailsTool(Tool):
    name = "extract_emails"
    description = "Pull e-mail addresses out of a free-text blob via regex."
    input_schema = ExtractEmailsIn
    output_schema = ExtractEmailsOut
    side_effects = ()  # pure
    cost_budget_wall_ms = 50

    _RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def _execute(self, payload):
        return ExtractEmailsOut(emails=self._RE.findall(payload.text))


# -------- Tool: normalize_email ---------------------------------------------


class NormalizeIn(BaseModel):
    email: str


class NormalizeOut(BaseModel):
    canonical: str
    domain: str


class NormalizeEmailTool(Tool):
    name = "normalize_email"
    description = "Lower-case and split an e-mail into canonical form + domain."
    input_schema = NormalizeIn
    output_schema = NormalizeOut
    side_effects = ()
    cost_budget_wall_ms = 10

    def _execute(self, payload):
        if "@" not in payload.email:
            raise ToolFailure("malformed_email", f"missing '@' in {payload.email!r}")
        local, _, domain = payload.email.lower().partition("@")
        return NormalizeOut(canonical=f"{local}@{domain}", domain=domain)


# -------- Drive ---------------------------------------------------------------


def main() -> None:
    registry = Registry()
    registry.register_tool(ExtractEmailsTool())
    registry.register_tool(NormalizeEmailTool())

    blob = (
        "Reach the team at Hello@Example.com or sales@example.com. "
        "Escalations: oncall@infra.example.com."
    )

    # A real LLM planner would emit these actions iteratively after
    # observing each tool's output. We script the sequence so the example
    # has no API dependency, but the action shape is identical.
    actions = [
        ToolAction(tool="extract_emails", arguments={"text": blob}),
        ToolAction(tool="normalize_email", arguments={"email": "Hello@Example.com"}),
        ToolAction(tool="normalize_email", arguments={"email": "sales@example.com"}),
        ToolAction(tool="normalize_email", arguments={"email": "oncall@infra.example.com"}),
        FinishAction(
            result={
                "contacts": [
                    {"canonical": "hello@example.com", "domain": "example.com"},
                    {"canonical": "sales@example.com", "domain": "example.com"},
                    {"canonical": "oncall@infra.example.com", "domain": "infra.example.com"},
                ]
            }
        ),
    ]

    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
        budget=Budget(max_iterations=10, max_tokens=10_000, max_wall_seconds=10),
        session_dir=Path(".sessions/extract"),
    )

    result = coord.run("Extract and normalize contacts from the team blurb.")
    print(json.dumps(result.model_dump(), indent=2, default=str))
    print(f"\nTrace: {result.trace_path}")
    print(f"Steps: {len(coord.trace)}")


if __name__ == "__main__":
    main()
