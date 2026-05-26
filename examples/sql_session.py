"""Playbook Pattern 2 — Tools + Grounded Verifier.

A miniature analytics agent: structured query intent → composed SQL →
EXPLAIN-style dry-run against an in-memory schema. No sub-agent —
the planner's job is to translate intent into typed arguments for the
``compose_query`` tool, then gate the result through ``sql_explain``.

The verifier is grounded in two evidence sources:

  - ``schema_validate``: every projected column exists in the table schema
  - ``query_db``: the (mock) planner can EXPLAIN the SQL and reports a
    plausible row estimate

A real planner would retry the tool on a failed verdict (up to
``Verifier.max_retries``); we script a happy-path + a deliberately-bad
path so the trace shows both branches.

Run:

    python -m examples.sql_session
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Registry,
    ScriptedPlanner,
    Tool,
    ToolAction,
    ToolFailure,
    Verdict,
    Verifier,
    VerifyAction,
)


# -------- Mock database -----------------------------------------------------

SCHEMA: dict[str, dict[str, str]] = {
    "users": {"id": "int", "email": "text", "plan": "text"},
    "orders": {"id": "int", "user_id": "int", "amount_cents": "int", "created_at": "date"},
}
ROW_COUNTS: dict[str, int] = {"users": 12_400, "orders": 218_900}


# -------- Tool: compose_query -----------------------------------------------


class ComposeIn(BaseModel):
    table: str
    projections: list[str] = Field(..., min_length=1)
    where: dict[str, str | int] = Field(default_factory=dict)


class ComposeOut(BaseModel):
    sql: str
    table: str
    projections: list[str]


class ComposeQueryTool(Tool):
    name = "compose_query"
    description = "Build a SELECT statement from a typed query intent."
    input_schema = ComposeIn
    output_schema = ComposeOut
    side_effects = ()  # pure string building
    cost_budget_wall_ms = 5

    def _execute(self, payload):
        if payload.table not in SCHEMA:
            raise ToolFailure("unknown_table", f"no table {payload.table!r}")
        cols = ", ".join(payload.projections)
        sql = f"SELECT {cols} FROM {payload.table}"
        if payload.where:
            preds = " AND ".join(
                f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}"
                for k, v in payload.where.items()
            )
            sql += f" WHERE {preds}"
        return ComposeOut(sql=sql, table=payload.table, projections=list(payload.projections))


# -------- Verifier: sql_explain (grounded: schema_validate + query_db) ------


class ExplainTarget(BaseModel):
    sql: str
    table: str
    projections: list[str]


class SQLExplainVerifier(Verifier):
    name = "sql_explain"
    description = "Dry-run the query against the schema and report estimated rows."
    target_schema = ExplainTarget
    evidence_sources = ("schema_validate", "query_db")
    max_retries = 2

    def _verify(self, target: ExplainTarget) -> Verdict:
        schema = SCHEMA.get(target.table)
        if schema is None:
            return Verdict(
                passed=False,
                evidence=[f"schema_validate: table {target.table!r} not in catalog"],
                suggested_revisions=[f"Use one of {sorted(SCHEMA)}"],
            )

        missing = [c for c in target.projections if c not in schema]
        if missing:
            return Verdict(
                passed=False,
                evidence=[
                    f"schema_validate: {target.table} has columns {sorted(schema)}",
                    f"schema_validate: projected {missing} not present",
                ],
                suggested_revisions=[
                    f"Drop columns {missing}; available: {sorted(schema)}"
                ],
            )

        rows = ROW_COUNTS.get(target.table, 0)
        return Verdict(
            passed=True,
            evidence=[
                f"schema_validate: all {len(target.projections)} projected columns exist in {target.table}",
                f"query_db: EXPLAIN ok; estimate ~{rows} rows scanned for {target.sql!r}",
            ],
        )


# -------- Drive --------------------------------------------------------------


def main() -> None:
    registry = Registry()
    registry.register_tool(ComposeQueryTool())
    registry.register_verifier(SQLExplainVerifier())

    # First branch: a bad query (projects a column that doesn't exist).
    # The verifier reports grounded evidence pointing at the schema, and
    # the planner re-emits with corrected projections.
    actions = [
        ToolAction(
            tool="compose_query",
            arguments={
                "table": "orders",
                "projections": ["id", "amount"],  # 'amount' is wrong
                "where": {"user_id": 7},
            },
            rationale="first attempt",
        ),
        VerifyAction(
            verifier="sql_explain",
            target={
                "sql": "SELECT id, amount FROM orders WHERE user_id = 7",
                "table": "orders",
                "projections": ["id", "amount"],
            },
            rationale="dry-run before shipping",
        ),
        # Verifier failed; planner corrects and retries.
        ToolAction(
            tool="compose_query",
            arguments={
                "table": "orders",
                "projections": ["id", "amount_cents"],  # corrected
                "where": {"user_id": 7},
            },
            rationale="retry with verifier-suggested column",
        ),
        VerifyAction(
            verifier="sql_explain",
            target={
                "sql": "SELECT id, amount_cents FROM orders WHERE user_id = 7",
                "table": "orders",
                "projections": ["id", "amount_cents"],
            },
            rationale="re-verify",
        ),
        FinishAction(
            result={"sql": "SELECT id, amount_cents FROM orders WHERE user_id = 7"}
        ),
    ]

    coord = Coordinator(
        planner=ScriptedPlanner(actions=actions),
        registry=registry,
        budget=Budget(max_iterations=15, max_tokens=20_000, max_wall_seconds=10),
        session_dir=Path(".sessions/sql"),
    )

    result = coord.run("Get the order amounts for user 7.")
    print(json.dumps(result.model_dump(), indent=2, default=str))

    # Show the two grounded verdicts so the pattern's value is visible.
    from harness.types import ActionType

    print("\nVerifier verdicts (in order):")
    for entry in coord.trace.by_type(ActionType.VERIFY):
        verdict = entry.action_output.get("verdict")
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
