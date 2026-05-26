# Three-Agent Message Bus Architecture

**Status:** Substrate (plan, bus, amendment protocol, supervisor) implemented as a Python prototype in `three_agent_bus/`, driven by scenarios in `three_agent_bus/drivers/scenarios.py`. The LLM executor and adversary are not built — per the design's own build order, the substrate must survive stupid drivers first.

## Premise

A live adversarial/executor pair, coordinated through an async typed message bus, supervised by a deterministic control plane. The aim is autonomous correctness on long-running tasks where step-by-step human supervision isn't practical.

Three roles, but only two are LLM agents:

- **Executor** — does the work.
- **Adversary** — live, plan-aligned, monitors the executor continuously and raises objections grounded in the plan.
- **Supervisor** — *not* a third deliberator. A control plane: deterministic, no LLM calls, enforces protocol.

The original sketch named the third role "mediator." That label biases implementation toward more LLM calls (deliberation, judgment, weighing arguments). "Supervisor" / "control plane" steers toward what the role actually needs: bookkeeping, escalation, and protocol enforcement. The rename is load-bearing — every other piece of the design follows from it.

Three is also a suspicious number in agent designs. Two often works (executor + critic); N works (swarm); three usually means a coordinator got smuggled in because two didn't compose cleanly. Naming the third role as control plane rather than agent is what dissolves that smell — it isn't a third deliberator, it's the substrate.

## The Plan as Typed Rule Base

The plan is not a prose document. It is a structured rule base:

- **Typed constraints** with stable IDs (`#C-1`, `#C-2`, …). Each constraint carries type, scope (artifact / subtask / global), and grounds.
- **Versioned**, append-only. Every change is a new version; history is preserved.
- A human-readable prose narrative can sit alongside for human readers, but agents key off the structured form.

This is the load-bearing piece. Without stable constraint IDs, "ground citation" is fuzzy quotation; with them, citations are *references* the supervisor can mechanically validate. Every other mechanism in this doc depends on this representation.

## Message Contract

The bus carries typed messages. Type determines blocking semantics, not just content:

| Type | Blocking | Use |
|---|---|---|
| `Observation` | Non-blocking | "I notice X." Logged. No required response. |
| `Question` | Soft-block | "Why X instead of Y?" Must be acknowledged; can be deferred with reason. |
| `Objection` | Hard-block on the cited artifact/action | "X violates constraint #C-N." Executor must fix or override with logged reason. |
| `Halt` | Full stop, escalate | "This is going to break something irrecoverable." |
| `Amendment` | Proposal — see Amendment Protocol | "Add / change / remove constraint." |

Rules:

- **Scoped blocking.** Objections block the cited artifact/action, not the executor globally. Work on unrelated artifacts continues.
- **Required grounds.** Objections must cite a constraint by ID. Stale citations (constraint amended out) auto-stale the objection.
- **Per-topic ordering.** The bus is partitioned per artifact / subtask. Ordered semantics where they matter; concurrency where they don't.
- **Override is logged, not blocked.** Executor can override an Objection if it provides a written reason. The reason becomes a first-class bus message; the supervisor can count overrides per artifact as a stuck signal.

## Amendment Protocol

The plan must evolve — real work uncovers facts that invalidate the initial plan.

- **Both agents may propose.** Asymmetric in practice:
  - Executor amendments tend to be **relaxations** (constraint too strict or factually wrong).
  - Adversary amendments tend to be **tightenings** (gap the plan should cover).
- **Two-sided ratification.** Proposer drafts; counterparty accepts, rejects, or counters. Counter becomes a normal disagreement loop, bounded by the supervisor's amendment-per-artifact cap.
- **Supervisor-bounded.** Per-artifact amendment cap is visible to both agents. Unresolved past budget escalates to human.

The asymmetry is a diagnostic signal the supervisor can act on:

- Many relaxations → executor is rewriting around problems.
- Many tightenings → adversary is goalpost-moving.

## Adversary Trigger Model

Autonomous, goal-driven, ranked. On each tick:

> "What is the strongest objection I could raise right now that I haven't already raised?"

If the answer clears a threshold, post. Otherwise stay silent. **Silence is a valid output.** Forced ranking is the difference between signal and nitpicking — without it, adversaries default to nitpicking and signal density collapses.

Two channels run in parallel:

- **Per-action ratification.** Adversary clears each executor action before it commits. Local correctness.
- **Continuous review.** Adversary keeps scanning accumulated state for emergent violations — five locally-fine actions composing into a plan violation. This is what makes "live" meaningful instead of just "incremental linter."

Polling vs reacting is the trap: pure polling burns tokens; pure reaction recreates synchrony in disguise. Autonomous goal-driven (with both channels above) is the working compromise.

## Write Authority

Three non-overlapping write surfaces:

| Role | Writes |
|---|---|
| Executor | Artifacts (code, files, etc.) |
| Adversary | Bus messages only |
| Supervisor | Run state (budgets, halts, escalations, plan version pointer) |

No agent races another for the same resource. This is what makes async safe.

## Supervisor (Control Plane)

Deterministic rules over the bus log. No LLM calls:

- Track unanswered Objections per artifact. Escalate above threshold.
- Detect Objection ping-pong (same grounds, same artifact, repeated) — stuck signal.
- Enforce token / turn budgets per agent.
- Diff executor output against current plan version. Drift = action outside plan scope.
- Cap amendment frequency per artifact.
- Validate constraint citations (does `#C-17` still exist in the current plan version?).
- Own the human escalation channel.

Every supervisor rule is a small program over the bus log. None requires judgment. That is the payoff of the relabel — every rule writes itself, mechanically, once the message contract is fixed.

## The Reframe

Once typed constraints + versioned amendments + ratification are in place, the system is no longer really "three agents on a bus." It is a **version-controlled typed rule base with two agents holding amendment rights and a control plane enforcing the protocol.**

Most of the design problems (conflict resolution, history, blame, rollback) are already solved in version control and consensus literature; this design just puts two LLMs on top of that substrate.

The recommended build order is therefore *inverted*: build the rule base, amendment protocol, and supervisor first, with dumb scripts driving them. Add the executor and adversary only once the substrate survives being driven by stupid drivers. If the substrate cannot survive scripts, two LLMs will not save it — they will hide the failures.

## Smell-checks / Unresolved

- **Politeness loops.** Two LLM agents on an async bus can drift into "good point, let me reconsider" cycles. Forced-ranking on the adversary side helps; not proven sufficient.
- **Bus log volume.** Continuous review generates a lot of `Observation` traffic. The supervisor's drift diff might be enough to discard most of it, but storage and replay design is open.
- **Initial plan quality.** The whole design assumes a useful initial typed plan. If the plan is wrong or empty, amendments dominate from the start. Plan authoring is out of scope here but is the upstream dependency.
- **Adversary capture / collusion drift.** If executor and adversary share too much prior context, they converge on the same blind spots. Periodic context resets help; the right cadence is open.
- **Human-in-the-loop fallback.** Designed for autonomous runs. No equivalent of rooroo's Principle of Least Assumption (ask the user when ambiguous). Collaborative work would need one.

---

## Comparison to the Existing Rooroo Architecture (v0.5.10)

| Dimension | Rooroo (v0.5.10) | Three-Agent Bus |
|---|---|---|
| **Topology** | Hub-and-spoke. Navigator is central orchestrator and UI; experts are dispatched and report back. | Lateral peers (executor, adversary) + deterministic control plane. No central LLM orchestrator. |
| **Communication** | Synchronous JSON Output Envelopes returned from `attempt_completion`. One expert active at a time. | Async typed messages over a partitioned bus. Both agents running concurrently. |
| **Critic / adversary** | None. Navigator triages and Planner plans, but no role second-guesses the executor mid-run. | First-class live autonomous role. |
| **Plan format** | Prose Markdown `context.md` per task + queued task list (`queue.jsonl`). | Typed constraints with stable IDs. Versioned. |
| **Plan mutability** | Implicit. Planner may revise the queue; no formal amendment record. | Explicit versioned amendments with ratification and audit trail. |
| **Stuck detection** | The user notices and intervenes. | Supervisor detects ping-pong and budget exhaustion mechanically. |
| **Drift detection** | None as a defined concept. Tasks either complete or fail. | Mechanical diff of executor actions against current plan version. |
| **Write authority** | Navigator writes queue / log / context; experts write artifacts. Convention, not formally bounded. | Three non-overlapping surfaces, formally bounded. |
| **Audit trail** | `.rooroo/logs/activity.jsonl` — event stream of dispatch and completion. | Bus log + plan version history — every disagreement and amendment recorded. |
| **LLM cost profile** | One expert active at a time. Tier optimization per role (cheap Navigator, smart Planner, varied experts). | Two LLMs running continuously. ~2–3× token spend per unit of work. |
| **Human-in-the-loop** | Pervasive. Principle of Least Assumption — ask the user when ambiguous. | Minimal. Supervisor escalates only at hard limits. Designed for autonomous runs. |
| **Implemented?** | Yes, shipping. | No, design proposal. |

### What rooroo does well that this design does not (yet) capture

- **Principle of Least Assumption.** Rooroo's first move on ambiguity is to ask the user. The three-agent bus has no equivalent — its ambiguity resolution is *internal* (executor and adversary disagree, then ratify or escalate). For autonomous runs this is correct; for collaborative work it would be a regression.
- **Cost discipline.** Rooroo's tier system (cheap Navigator, smart Planner) is real engineering. The three-agent bus is more expensive by construction.
- **Inspectable failure modes.** Rooroo's failure modes are mostly synchronous and traceable through a single envelope chain. Async multi-agent systems have a longer and weirder list of failure modes (politeness loops, stale objections, amendment storms, ordering bugs).
- **"Link, don't embed" context discipline.** Rooroo's context briefings stay terse by linking to source-of-truth files instead of duplicating them. The three-agent bus has no equivalent rule yet — bus message volume could grow unchecked without one.

### What this design captures that rooroo does not

- **Live adversarial pressure.** Rooroo has no mid-execution critic; correctness is whatever the executor produces, judged after the fact (or by the user).
- **Plan as ground truth.** Rooroo's plan is implicit in the queue and the prose context. There is no structured object the system can mechanically check against. The three-agent bus makes the plan a first-class, queryable, diffable artifact.
- **Mechanical drift and stuck detection.** Rooroo relies on the user. The three-agent bus does not.
- **Survives autonomous long-running work.** Rooroo's loop expects user check-ins at every decision point.

### The honest framing

These two architectures are not interchangeable. They optimize for different points:

- **Rooroo** — cost-efficient, user-in-the-loop, synchronous orchestration of well-scoped tasks.
- **Three-agent bus** — autonomous correctness on long-running work where the user can't supervise step-by-step.

This design is *not* a successor to rooroo. It is a different shape suited to a different operating regime. The cleanest reading: the three-agent bus is what rooroo would need to become if you wanted to run it overnight with no user present and trust the result in the morning.

A weaker reading is also true and worth naming: if rooroo's Planner were to emit typed constraints instead of prose contexts, and an Adversary role were introduced alongside the existing experts, and the Navigator's stuck/drift handling were promoted from "user notices" to mechanical rules — you would have most of this design, layered on rooroo's existing substrate. The pieces are compatible. They have not been combined here, and combining them is its own design problem.
