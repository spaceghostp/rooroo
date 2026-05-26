"""Playbook Pattern 2 — Tools + Grounded Verifier.

A miniature analytics agent: structured query intent → composed SQL →
EXPLAIN-style dry-run against an in-memory schema. No sub-agent —
the planner's job is to translate intent into typed arguments for the
``compose_query`` tool, then gate the result through ``sql_explain``.

The verifier is grounded in two evidence sources:

  - ``schema_validate``: every projected column exists in the table schema
  - ``query_db``: the (mock) planner can EXPLAIN the SQL and reports a
    plausible row estimate

The planner here is a small custom class so the feedback loop is
*visible*: on a failed verdict it parses ``suggested_revisions`` and
re-emits ``compose_query`` with the corrected projection list. The
coordinator caps consecutive failed verdicts at
``Verifier.max_retries`` (§4.3) — no unbounded refine loops.

Run:

    python -m examples.sql_session
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness import (
    Budget,
    Coordinator,
    FinishAction,
    Plan,
    PlanRequest,
    Planner,
    Registry,
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


# -------- Custom planner: feedback-driven correction loop -------------------


class _SQLPlanner(Planner):
    """Compose → verify → finish. On a failed verdict, parse
    ``suggested_revisions`` for the columns the verifier reports missing
    and rewrite the projection list. Concedes after two failed verdicts
    in a row, matching ``SQLExplainVerifier.max_retries``.
    """

    _DROP_RE = re.compile(r"^Drop columns (\[[^\]]+\])")

    def __init__(self, intent_table: str, intent_projections: list[str],
                 intent_where: dict[str, Any]) -> None:
        self._table = intent_table
        # Start from the planner's interpretation of the intent. In real
        # life this is an LLM guess; here we pass in a deliberately-wrong
        # column ('amount') so the verifier has something to correct.
        self._projections = list(intent_projections)
        self._where = dict(intent_where)
        self._last_sql: str | None = None
        self._last_compose: dict[str, Any] | None = None
        self._phase = "compose"

    def plan(self, request: PlanRequest) -> Plan:
        obs = request.last_observation or {}

        if self._phase == "compose":
            # Compose first; verify on the next turn.
            self._phase = "await_compose"
            return Plan(
                next_action=ToolAction(
                    tool="compose_query",
                    arguments={
                        "table": self._table,
                        "projections": list(self._projections),
                        "where": dict(self._where),
                    },
                    rationale="translate query intent into typed SQL",
                )
            )

        if self._phase == "await_compose":
            out = obs.get("output") or {}
            if not out:
                # Tool failed — bail with the structured error.
                return Plan(
                    terminate=True,
                    next_action=FinishAction(
                        result={"error": obs.get("error", "compose_query failed")}
                    ),
                )
            self._last_sql = out["sql"]
            self._last_compose = out
            self._phase = "await_verify"
            return Plan(
                next_action=VerifyAction(
                    verifier="sql_explain",
                    target=out,
                    rationale="dry-run before shipping",
                )
            )

        # self._phase == "await_verify"
        verdict = (obs.get("verdict") or {}) if obs else {}
        if verdict.get("passed"):
            return Plan(
                terminate=True,
                next_action=FinishAction(result={"sql": self._last_sql}),
            )

        # Failed verdict — apply the verifier's suggested fix.
        revisions = verdict.get("suggested_revisions") or []
        corrected = self._apply_revisions(self._projections, revisions)
        if corrected == self._projections:
            # No actionable correction; concede.
            return Plan(
                terminate=True,
                next_action=FinishAction(
                    result={"error": "no actionable revision",
                            "evidence": verdict.get("evidence", [])}
                ),
            )
        self._projections = corrected
        self._phase = "compose"
        return self.plan(request)  # re-enter immediately with the new state

    @staticmethod
    def _apply_revisions(projections: list[str], revisions: list[str]) -> list[str]:
        """Best-effort: read 'Drop columns [...]; available: [...]'.

        Drops every column named in the first bracketed list, and for
        every dropped column, picks the first column from the 'available'
        list that doesn't already appear in projections.
        """
        if not revisions:
            return projections
        msg = revisions[0]
        m_drop = _SQLPlanner._DROP_RE.match(msg)
        m_avail = re.search(r"available: (\[[^\]]+\])", msg)
        if not m_drop or not m_avail:
            return projections
        # The evidence uses Python-list repr (single quotes); ast.literal_eval handles it.
        import ast
        try:
            drop: list[str] = ast.literal_eval(m_drop.group(1))
            available: list[str] = ast.literal_eval(m_avail.group(1))
        except (SyntaxError, ValueError):
            return projections
        out = [c for c in projections if c not in drop]
        for d in drop:
            for cand in available:
                if cand not in out and cand != "id":  # 'id' is usually already there
                    # Heuristic: pick the column whose name shares a substring
                    # with the dropped one. Falls back to the first new one.
                    if d in cand or cand.startswith(d):
                        out.append(cand)
                        break
            else:
                # nothing fancy matched; add the first available not already present
                for cand in available:
                    if cand not in out:
                        out.append(cand)
                        break
        return out


# -------- Drive --------------------------------------------------------------


def main() -> None:
    registry = Registry()
    registry.register_tool(ComposeQueryTool())
    registry.register_verifier(SQLExplainVerifier())

    planner = _SQLPlanner(
        intent_table="orders",
        # Deliberately-wrong projection: 'amount' isn't in the schema, so
        # the first verdict fails and the planner has to apply the
        # verifier's suggestion. The replay-friendly path is exactly the
        # feedback loop the pattern exists to teach.
        intent_projections=["id", "amount"],
        intent_where={"user_id": 7},
    )

    coord = Coordinator(
        planner=planner,
        registry=registry,
        budget=Budget(max_iterations=15, max_tokens=20_000, max_wall_seconds=10),
        session_dir=Path(".sessions/sql"),
    )

    result = coord.run("Get the order amounts for user 7.")
    print(json.dumps(result.model_dump(), indent=2, default=str))

    # Show the verdict trail so the feedback loop is visible.
    from harness.types import ActionType

    print("\nVerifier verdicts (in order):")
    for entry in coord.trace.by_type(ActionType.VERIFY):
        verdict = entry.action_output.get("verdict")
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
