# Harness Playbook

Six configurations, ordered by where they sit on the P2 capability ladder
(tools → tools+verifier → tools+sub-agent+verifier → depth-2). Each entry
states what the configuration is *for*, what primitive shape it uses, and
which anti-patterns it's most likely to drift into. The spec (`SPEC.md`)
is the contract; this is the menu.

Two rules cut across every entry:

- Climb the ladder only when the rung below provably can't do the job (P2).
- If you can't write the input/output schema for a sub-agent, the
  abstraction isn't ready — don't promote (P3, AC7).

---

## Pattern 1 — Pure Tool Pipeline

**Use it for:** structured data extraction, API orchestration, ETL,
deterministic transforms. Anything where the LLM's job is to *pick which
tool to call and with what arguments*, not to reason inside a loop.

**Primitive shape:**

| Tools | Sub-agents | Verifiers |
|------:|-----------:|----------:|
|   3-8 |          0 |       0-1 |

The verifier is optional; if the tools' own typed outputs already gate
the result, you don't need one. Add a verifier only if there's an
external grounding check (schema, downstream API echo) the tools don't
already perform.

**Skeleton:**

```python
class FetchUserIn(BaseModel):
    user_id: str

class FetchUserOut(BaseModel):
    id: str
    email: str
    plan: Literal["free", "pro", "team"]

class FetchUserTool(Tool):
    name = "fetch_user"
    description = "Read one user row by ID."
    input_schema = FetchUserIn
    output_schema = FetchUserOut
    side_effects = ("network", "db:read")
    cost_budget_wall_ms = 200

    def _execute(self, payload):
        row = db.users.get(payload.user_id)
        if row is None:
            raise ToolFailure("not_found", f"no user with id {payload.user_id}")
        return FetchUserOut(**row)
```

**When NOT to use:** when the planner needs more than a few turns of
*reasoning between tool calls*. If you find yourself wishing the planner
had its own working memory, you've outgrown this pattern — promote one
piece of the work to a sub-agent.

**Anti-patterns to watch:**
- Sneaking an LLM call into `_execute`. Tools are deterministic by
  definition (§4.1); if you need a model call, that's a sub-agent.
- Letting one tool return arbitrary JSON "for flexibility". Schemas
  earn their keep at *every* boundary.

---

## Pattern 2 — Tools + Grounded Verifier

**Use it for:** generating an output whose correctness has a cheap,
external check. SQL queries (run the query, check the result schema),
generated code snippets (typecheck), structured form fills (downstream
validation API), copy generation (length + vocabulary rubric).

**Primitive shape:**

| Tools | Sub-agents | Verifiers |
|------:|-----------:|----------:|
|   2-5 |          0 |       1-2 |

**Skeleton:**

```python
class SQLTarget(BaseModel):
    sql: str

class SQLDryRunVerifier(Verifier):
    name = "sql_dry_run"
    description = "EXPLAIN the query and confirm it returns the expected columns."
    target_schema = SQLTarget
    evidence_sources = ("query_db", "schema_validate")
    max_retries = 2

    def _verify(self, target):
        try:
            plan = db.explain(target.sql)
        except DBError as e:
            return Verdict(
                passed=False,
                evidence=[f"query_db: EXPLAIN failed: {e}"],
                suggested_revisions=["fix the syntax error reported above"],
            )
        cols = plan["output_columns"]
        if set(cols) != {"user_id", "total_spend"}:
            return Verdict(
                passed=False,
                evidence=[f"schema_validate: returned columns {cols}, expected user_id+total_spend"],
                suggested_revisions=["project exactly user_id, total_spend"],
            )
        return Verdict(
            passed=True,
            evidence=[f"query_db: EXPLAIN ok ({plan['row_estimate']} rows est)",
                      f"schema_validate: columns match"],
        )
```

**Coordinator loop is responsible for the retry:** the verifier returns
a verdict; the *planner* sees the verdict on the next iteration and
either re-emits a fixed `ToolAction` or finishes. Cap retries at
`Verifier.max_retries`. No unbounded refine loops (§4.3).

**Anti-patterns to watch:**
- A "verifier" that just asks an LLM "does this look right?" — that's
  the critic-only pattern explicitly rejected (P4, §9). The harness
  will demote its verdict to `passed=False` with `GROUNDING_FAILED`.
- A verifier that touches reality but returns `evidence=[]` because
  "it passed cleanly". Always cite what you checked, even on the
  happy path — that's how the audit works.

---

## Pattern 3 — Tool + Sub-agent + Verifier (the canonical depth-1)

**Use it for:** anything where a chunk of the work needs *its own
reasoning loop and tool subset* but the overall task is still one
intent. The bug-fix-PR agent, the research-and-cite agent, the
"summarize this 50-page doc into a structured brief" agent.

**Primitive shape:**

| Tools | Sub-agents | Verifiers |
|------:|-----------:|----------:|
|   3-8 |        1-2 |       1-2 |

This is the rung most production agents should sit on. Above it, you're
paying compounding-error tax for each additional layer.

**When the sub-agent earns its keep:**
- The parent's context would be polluted by the exploration (e.g.,
  reading 12 files to find one symbol)
- The work benefits from a *different system prompt* (a careful
  fact-extractor vs. a confident planner)
- The work needs a *restricted tool subset* you don't want the parent
  to use freely

**Skeleton:**

```python
class LocateBugTask(SubAgentTask):
    symptom: str
    repo_root: str

class LocateBugResult(SubAgentResult):
    file_path: str
    line_range: tuple[int, int]
    hypothesis: str

class LocateBugAgent(SubAgent):
    name = "locate_bug"
    description = "Given a symptom, find the file + line range most likely to host the bug."
    task_schema = LocateBugTask
    result_schema = LocateBugResult
    tool_allowlist = ("read_file", "grep", "list_dir")
    system_prompt = "You are a careful bug-locator. Cite file:line for every claim."
    default_max_iterations = 8
    default_max_tokens = 30_000

    def _run(self, task, scratch):
        # Sub-agent's own coordinator loop runs here (its own planner,
        # its own budget, its own tool subset). Returns a typed result.
        ...
```

See `examples/bugfix_session.py` for a runnable end-to-end version.

**Anti-patterns to watch:**
- The sub-agent is just a wrapper around one LLM call with no
  internal loop. That's not a sub-agent (§4.2 explicitly rejects it) —
  it should be a tool.
- The sub-agent returns "raw conversation" instead of a typed result.
  Same rejection. Its schema is its contract.
- The parent passes the sub-agent its full state. Sub-agents get a
  typed task, not a transcript (§6).

---

## Pattern 4 — Depth-2 Sub-agent (recursive search space)

**Use it for:** problems where the *search space itself is recursive*
and you've already proven depth-1 can't cover it. Per P3, depth 2
requires an explicit `depth2_justification` on the sub-agent class
that goes into design review. Three canonical cases:

- **Codebase walk:** root → directory → file → symbol
- **Document tree exploration:** corpus → doc → section → claim
- **Multi-step planning:** goal → milestone → task → action

**Primitive shape:**

| Tools | Sub-agents | Verifiers |
|------:|-----------:|----------:|
|   4-10 |       1-3 |       1-3 |

At least one sub-agent has `depth2_justification` set. **No depth 3.**

**Skeleton:**

```python
class WalkDirTask(SubAgentTask):
    root: str
    interest: str  # what we're looking for

class WalkDirResult(SubAgentResult):
    hits: list[dict]

class WalkDirAgent(SubAgent):
    name = "walk_dir"
    description = "Recursively walk a directory tree, gathering hits matching `interest`."
    task_schema = WalkDirTask
    result_schema = WalkDirResult
    tool_allowlist = ("list_dir", "read_file")
    depth2_justification = (
        "Codebase tree is recursive; we descend dir→subdir→file. "
        "Bounded by depth=2 and per-call iteration budget."
    )

    def _run(self, task, scratch):
        # Inside _run, this sub-agent may invoke `extract_symbols`
        # (another sub-agent) at depth=2. Anything beyond depth=2
        # is refused at SubAgent.run with DepthLimitExceeded.
        ...
```

**Anti-patterns to watch:**
- Using depth-2 because "the parent context is too long" — that's not
  a recursive search space, it's a context-management issue. Solve it
  with state, not with another layer.
- Discovering you want depth 3. The spec rejects this categorically
  (§9). If you genuinely need it, the conversation is "should we lift
  the principle in §2, and what's the cost?" — *not* an
  implementation tweak.

---

## Pattern 5 — Multi-Verifier Gate (use with care)

**Use it for:** outputs that must satisfy *several* independent grounded
checks before they ship. Generated code: passes typecheck **and** unit
tests **and** style lint. Generated SQL: parses **and** dry-runs **and**
returns the expected schema. Generated copy: passes the rubric **and**
the brand-vocabulary check.

**Primitive shape:**

| Tools | Sub-agents | Verifiers |
|------:|-----------:|----------:|
|   2-5 |        0-1 |       2-4 |

**Status (per §11):** verifier composition is an open question — the
spec doesn't lock in conflict-resolution semantics. Until it does, use
this pattern with an explicit rule that *all* verifiers must pass
(AND-composition). The planner emits N `VerifyAction`s in sequence and
only emits `FinishAction` when each prior verdict is `passed=True`.

**Sketch of the planner's contract for this pattern:**

```python
# Pseudo: the planner watches `last_observation` across iterations.
verifiers = ["typecheck", "unit_tests", "style_lint"]
pending = list(verifiers)
if last_observation and last_observation.get("verdict", {}).get("passed"):
    pending.pop(0)
if pending:
    return Plan(next_action=VerifyAction(verifier=pending[0], target={...}))
return Plan(terminate=True, next_action=FinishAction(result={...}))
```

**Anti-patterns to watch:**
- OR-composition without justification. If any single verifier is
  enough to ship, you don't need the others — drop them.
- A "rollup verifier" that runs the other verifiers internally and
  composes their results. The trace becomes opaque. Keep each verifier
  as its own step so the trace shows which check failed.

---

## Pattern 6 — Resumable Long-Run

**Use it for:** sessions whose work doesn't fit in a single budget
window — large-scale refactors, multi-stage migrations, long research
runs. The trick is that the spec already gives you everything you need:
external state (§6) is durable, the trace is append-only (§8), and
budget exhaustion is a first-class outcome (§7) not an error.

**Configuration:**

- `FileSystemState` rooted at a stable `session_dir`
- Trace at `session_dir/trace.jsonl`
- Coordinator carries a generous wall-clock cap but a *modest* iteration
  cap, so each run terminates cleanly on `incomplete=True` and the
  caller can re-launch with the same `session_dir`
- Planner consults `state.summary()` first; it should re-derive "where
  am I" from state, not from the trace

**Skeleton:**

```python
coord = Coordinator(
    planner=ResumablePlanner(),       # consults state, picks up where it left off
    registry=registry,
    session_dir=Path(f".sessions/{session_id}"),
    budget=Budget(max_iterations=25, max_tokens=200_000, max_wall_seconds=300),
)
result = coord.run(intent)

if result.incomplete:
    # Schedule a re-launch; FileSystemState already holds the working set.
    schedule(session_id, retry_in_seconds=60)
```

**Anti-patterns to watch:**
- Stashing the trace into state so the resumed run "remembers
  everything". The trace is a record, not a memory. State is what the
  next run reads.
- Relying on in-memory caches that vanish between runs. If it matters
  for resumption, it goes in `state`.
- Forgetting that sub-agents have *their own* (in-memory) scratch — a
  resumable parent does not automatically resume a sub-agent's
  in-flight loop. Sub-agents should be small enough to complete within
  one parent iteration.

---

## Decision table — which pattern should I start with?

| If your task...                                                  | Start with     |
|------------------------------------------------------------------|----------------|
| ...is "pick the right API call and pass typed args"              | Pattern 1      |
| ...generates an artifact you can check externally                | Pattern 2      |
| ...needs reasoning over many sources/files within one intent     | Pattern 3      |
| ...explores a tree-shaped or planning-shaped search space        | Pattern 4      |
| ...must satisfy multiple independent gates before shipping       | Pattern 5      |
| ...won't fit in a single budget window                           | Pattern 6      |

Default bias: pick the *lowest* row that fits. P2 exists because every
rung costs you reliability, latency, cost, and debuggability — and
those costs compound.

---

## Cross-pattern checklist (run before you ship a configuration)

- [ ] Every primitive declares its schemas (AC7)
- [ ] Every verifier declares ≥1 `evidence_sources` and actually consults
      external evidence (P4)
- [ ] Every sub-agent has a `tool_allowlist` that is a *real subset* of
      the parent's tools (§4.2)
- [ ] Budget caps are set on the parent and carved (or default-set) for
      every sub-agent (P6, §7)
- [ ] If any sub-agent runs at depth 2, it has `depth2_justification`
      documented (P3)
- [ ] The planner does not reach for the raw trace — only `state_summary`
      and `last_observation` (§5)
- [ ] You can articulate, in one sentence, what `incomplete=True` means
      for this configuration and what the caller should do
