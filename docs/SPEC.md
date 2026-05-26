# Agent Harness Abstraction — Spec / PRD

**Status:** Draft v0.1
**Type:** Architecture spec with locked-in design reasoning
**Audience:** Implementers, reviewers, future-self

---

## 1. Orientation

A harness for LLM-driven agent systems. The thesis: most agent complexity comes from the wrong primitives. We commit to **one coordinator, three composable primitives, bounded depth, grounded verification, and externalized state.** Every other design decision falls out of those commitments.

This document locks in the *why* alongside the *what*. Section by section, each spec carries its reasoning so the design is not silently relitigated during implementation.

### Scope

- Single-tenant agent runtime: one user-facing intent per session
- Synchronous and async task execution
- Tool use, sub-task delegation, validation, memory

### Out of scope (explicitly deferred)

- Multi-agent negotiation / market-style coordination
- Long-lived autonomous agents with no human in the loop
- Agent-to-agent recursion beyond depth 2
- Emergent role assignment

---

## 2. Locked Design Principles

These are invariants. Implementations that violate these are not conformant. Each carries the reasoning so the constraint travels with it.

### P1 — One coordinator per session

**Spec:** Every session has exactly one top-level coordinator agent that owns the intent and the loop.
**Reasoning:** Multiple coordinators means multiple places for intent to drift, multiple debugging surfaces, and emergent coordination overhead. One coordinator = one source of truth for "what are we trying to do."

### P2 — Tools before sub-agents, sub-agents before recursion

**Spec:** Capability promotion ladder. Add a tool first. Promote to a single-call sub-agent only if the work requires its own reasoning loop. Permit a second level of delegation only if the search space itself is recursive (codebases, document trees, multi-step planning).
**Reasoning:** Each rung up costs reliability (compounding error), latency (serial LLM calls), cost (multiplicative tokens), and debuggability (more layers to inspect). Climb only when the rung below provably can't do the job.

### P3 — Bounded depth, typed contracts

**Spec:** Default max delegation depth is 1. Depth 2 requires an explicit justification recorded in the spec for that sub-agent. No depth 3+. Every sub-agent boundary has a structured input schema and structured output schema.
**Reasoning:** Three layers at 90% reliability compose to 73% end-to-end. Typed contracts force the abstraction to be explicit; if you can't write the schema, the abstraction isn't ready.

### P4 — Verifiers must have epistemic advantage

**Spec:** A validator that doesn't touch reality (run code, fetch sources, query data, check schema, hit a typechecker) is not a verifier — it's commentary. The harness distinguishes the two and only the former participates in retry loops.
**Reasoning:** Same-model self-critique without external signal degrades quality about as often as it improves it. Validation only earns its cost when the validator has information the generator didn't.

### P5 — State is external, typed, and observable

**Spec:** Working state lives in a filesystem, scratchpad, or database — not buried in conversation history. The coordinator reads and writes state via typed accessors. State changes are logged.
**Reasoning:** Cramming state into context bloats prompts, loses information at every boundary, and makes resumption impossible. Externalized state is debuggable, resumable, and shareable across sub-agents.

### P6 — Explicit budget and termination

**Spec:** Every loop carries a budget (tokens, wall-clock, iterations). Termination conditions are declarative, not "until satisfied." Budget exhaustion is a first-class outcome, not an error.
**Reasoning:** Implicit termination is how agents melt down. A loop without a stopping rule is a bug.

### P7 — Every step is observable

**Spec:** Every coordinator decision, tool call, sub-agent invocation, and verifier outcome is recorded to a structured trace. The trace is queryable post-hoc.
**Reasoning:** Without traces, debugging is divination. Traces are also the substrate for evals.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────┐
│              Coordinator (1)                │
│  ┌────────────────────────────────────────┐ │
│  │  Plan → Act → Observe → Verify loop    │ │
│  └────────────────────────────────────────┘ │
│         │           │            │          │
│         ▼           ▼            ▼          │
│      Tools      Sub-agents    Verifiers     │
│   (typed fns)   (typed fns)  (grounded)     │
└─────────────────────────────────────────────┘
              │                │
              ▼                ▼
        External State    Trace Log
        (FS / DB)         (structured)
```

**Three primitives, one loop, two stores.** Sub-agents may locally instantiate the same pattern (their own coordinator + tools + verifiers) but cannot themselves spawn sub-agents.

---

## 4. Primitive Specs

### 4.1 Tools

**Definition:** Deterministic or near-deterministic functions with typed input and output. No LLM call inside.

**Required properties:**

- JSON-schema input and output
- Idempotent where semantically possible
- Side effects declared in metadata
- Failures return structured error, never throw to the coordinator
- Cost and latency budget declared

**When to use:** Default choice. Anything that doesn't need a reasoning loop.

### 4.2 Sub-agents

**Definition:** A bounded reasoning unit invoked as a typed function. Receives a structured task, returns a structured result. Has its own context window, its own system prompt, its own tool subset.

**Required properties:**

- JSON-schema input and output
- Independent context (no leakage from parent)
- Tool allowlist (subset of harness tools) — enforced by constructing the
  inner Registry via ``SubAgent.build_inner_registry(parent_registry)``;
  an allowlist entry not present in the parent raises
  ``AllowlistViolation`` at construction time, not at runtime
- Own budget (tokens, iterations, wall-clock) — carved from the parent's
  remaining headroom via ``SubAgent.carve_budget(parent_budget, ...)``
  so token / wall-time consumption propagates back to the parent (P6)
- Cannot invoke other sub-agents unless explicitly justified per P3
- Returns either a typed result or a typed failure — never raw conversation

**When to use:** When the work needs its own reasoning loop, the parent's context would be polluted by the exploration, or the task benefits from a different system prompt or tool set.

**When NOT to use:** When a tool would suffice. When the parent could do the work with one more turn. When the "sub-agent" is just a wrapper around a single LLM call with no internal loop.

### 4.3 Verifiers

**Definition:** Primitives that check generator output against grounded evidence.

**Required properties:**

- Must touch external reality (one or more of: execute code, fetch sources, query DB, run typechecker, validate schema, diff against canonical)
- Return structured verdict: `{passed: bool, evidence: [...], suggested_revisions: [...]?}`
- Failures are detectable from output alone, not vibes
- Verifier is on a different epistemic footing than the generator (different tools, different evidence, different prompt frame)

**Anti-pattern (rejected):** "Critic agent reads generator output and says if it's good." This is not a verifier under this spec.

**Retry policy:** A given verifier may produce at most ``max_retries``
failed verdicts in a single session (default N=2). The coordinator
counts failed verdicts per verifier name and, on the ``(N+1)``-th
failure, terminates the loop with
``incomplete_reason="retry_budget_exhausted:<verifier>"``. Planners
cannot work around this — the cap is enforced inside the coordinator's
dispatch path, not inside the planner. No unbounded refine loops.

---

## 5. Coordinator Loop Spec

**Pseudocode:**

```
while not done and budget_remaining():
    plan = coordinator.plan(intent, state, trace)
    if plan.terminate: break
    action = plan.next_action  # one of: tool_call, subagent_call, verify, finish
    result = execute(action)
    state.update(result)
    trace.append(action, result)
done = plan.terminate or budget_exhausted
return synthesize(state, trace)
```

**Required behaviors:**

- Single decision per iteration (no parallel planning at the coordinator level in v1)
- State is updated *before* the next plan step
- Budget checked before every action, not just at iteration boundaries
- Trace is append-only and never read by the coordinator's plan step except as explicitly summarized state

**Optional in v1, deferred:**

- Parallel action dispatch (requires careful state-merge semantics)
- Speculative execution
- Mid-loop replanning on partial results

---

## 6. State & Memory Spec

**Two layers:**

1. **Working state** — typed, mutable, scoped to the session. Lives in a structured store (filesystem with JSON files, or KV store). Accessed via typed read/write primitives. Diff-able.
2. **Trace log** — append-only, structured, immutable. One entry per coordinator decision and primitive invocation. Includes inputs, outputs, timing, cost. Append-only also *across runs sharing the same ``session_dir``*: a resumed session opens the existing JSONL file and continues appending to it, so the historical record survives resumption (P6 / §7's first-class ``incomplete=True`` outcome). To start fresh, the caller deletes the session directory.

**Forbidden:**

- Stuffing arbitrary blobs into context as a state substitute
- Sub-agents reading parent's full trace (they get a typed task, not a transcript)
- Mutable trace entries

**Memory across sessions (v1 stance):** Not in scope. Cross-session memory is a separate primitive deferred to v2 — it has different consistency and privacy semantics and shouldn't be conflated with in-session state.

---

## 7. Budget & Termination Spec

**Every loop carries:**

- Token budget (input + output)
- Iteration cap
- Wall-clock budget
- Per-sub-agent sub-budgets (carved from parent's budget)

**Termination conditions (declarative):**

- Coordinator emits `finish` with synthesized result
- Verifier passes on the designated deliverable
- Budget exhausted in any dimension
- Unrecoverable error from a primitive

**Budget exhaustion is a first-class outcome:** the coordinator returns the best partial result with an explicit `incomplete` flag and the reason. Not an exception.

---

## 8. Observability Spec

**Trace schema (minimum):**

```
{
  step_id, parent_step_id, timestamp,
  action_type,            // plan | tool | subagent | verify | finish
  action_input,           // structured
  action_output,          // structured
  cost: {tokens_in, tokens_out, wall_ms, dollars},
  outcome: ok | error | budget,
  notes?
}
```

**Required tooling:**

- Trace viewer (per-session timeline)
- Diff view (state before/after each step)
- Cost rollup per session and per sub-agent
- Replay: re-run a session from any trace point with modified inputs

**Reasoning:** Traces are not just for debugging — they're the substrate for evaluation, regression testing, and iterating on the coordinator's planning prompt.

---

## 9. Anti-Patterns (Explicitly Rejected)

| Anti-pattern                                            | Why rejected                                               |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| Agents that can dispatch to arbitrary other agents      | Emergent coordination doesn't work; debug surface explodes |
| Critic-only validation loops (LLM reads sibling output) | No epistemic advantage; degrades as often as helps         |
| Deep context-passing instead of typed contracts         | Telephone game; intent compresses at every hop             |
| Implicit termination ("loop until satisfied")           | No stopping rule = no termination guarantee                |
| Stateless re-runs instead of resumable checkpoints      | Wastes budget; defeats observability                       |
| Sub-agents that return raw conversation                 | Not a function; not testable; not swappable                |
| Shared mutable context across siblings                  | Race conditions; non-determinism without value             |
| Recursion without a base case                           | This is just a bug in agent clothing                       |

---

## 10. Acceptance Criteria

A conformant implementation must:

1. Run a session with one coordinator, return a structured result with full trace
2. Successfully invoke a tool, a sub-agent, and a verifier
3. Terminate cleanly on budget exhaustion with a partial result
4. Refuse to invoke a sub-agent that would exceed depth limits
5. Produce a trace that can be replayed deterministically given the same primitive outputs
6. Pass a verifier-grounding audit: every retry-eligible verifier must demonstrably touch external evidence
7. Reject (at design-review time) any sub-agent spec without a typed input/output schema

---

## 11. Open Questions / Deferred Decisions

Things deliberately not locked in v1. Decide these in v2 or in implementation-specific extensions.

- **Parallelism at the coordinator level:** Currently serial. Parallel dispatch with state-merge semantics is plausible but requires more spec work.
- **Cross-session memory:** Different consistency and privacy properties; deserves its own spec.
- **Human-in-the-loop checkpoints:** How does a coordinator pause and request human input? Likely a special primitive, but not yet specified.
- **Streaming partial results to the caller:** Useful for UX, but not yet defined in the result contract.
- **Cost-based planning:** Coordinator currently plans without a cost model in the prompt. Plausible v2 addition.
- **Verifier composition:** Can multiple verifiers gate a single output? If so, with what conflict resolution?

---

## 12. Synthesis

The harness gets its leverage from refusing options. No arbitrary recursion. No vibes-based validation. No implicit state. No implicit termination. What remains is small, debuggable, and composable: a coordinator running a tight loop over three typed primitives, with state and traces in plain view.

If a future change proposal violates a locked principle in §2, it needs to either justify lifting the principle (and accept the cost recorded in its reasoning) or find another way. The reasoning is the contract.
