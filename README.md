# harness

> One coordinator. Three composable primitives. Bounded depth. Grounded
> verification. Externalized state.

A Python implementation of the agent-harness abstraction. The full design
contract lives in [`docs/SPEC.md`](docs/SPEC.md); this README is the quick
tour.

## What it is

The harness commits to a small set of primitives and refuses options that
make LLM agent systems hard to reason about:

- **Tools** — deterministic typed functions, no LLM inside
- **Sub-agents** — bounded reasoning units invoked as typed functions, with
  their own context, prompt, and tool subset
- **Verifiers** — checks that touch external reality and return a structured
  verdict with evidence

A single **Coordinator** runs the `plan → act → observe` loop over those
three primitives. **State** is external and typed (filesystem or in-memory).
The **Trace** is append-only JSONL and is what you debug, replay, and eval
against.

See `docs/SPEC.md` §2 for the seven locked principles (P1–P7) and §10 for
the acceptance criteria the implementation must satisfy.

For copy-pasteable configurations — pure-tool pipelines, tools+verifier,
the canonical depth-1 stack, depth-2 recursive search, multi-verifier
gating, resumable long-runs — see [`docs/PLAYBOOK.md`](docs/PLAYBOOK.md).

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. The only runtime dependency is Pydantic.

## Run the example

```bash
python -m examples.echo_session     # all three primitives, smallest possible
python -m examples.extract_session  # Pattern 1 — pure tool pipeline
python -m examples.bugfix_session   # Pattern 3 — sub-agent + grounded test verifier
```

Each example writes a session directory under `.sessions/` containing
the JSONL trace and the typed state — both are meant to be inspected
after the run.

## Run the tests

```bash
pytest
```

The acceptance criteria from §10 of the spec are each pinned by a test in
`tests/test_acceptance.py`.

## Project layout

```
harness/
  coordinator.py     §5 — the single coordinator loop
  planner.py         Planner interface + ScriptedPlanner (test/replay)
  budget.py          §7 — token / iteration / wall-clock budgets
  state.py           §6 — typed external state (in-memory + filesystem)
  trace.py           §8 — append-only JSONL observability
  types.py           Action discriminated union + shared types
  errors.py          Harness-internal exceptions
  primitives/
    tool.py          §4.1
    subagent.py      §4.2 (depth caps enforced)
    verifier.py      §4.3 (grounding audit enforced)
docs/SPEC.md         the binding spec
examples/            runnable end-to-end demo
tests/               unit + acceptance tests
```

## Bringing your own LLM

The harness has no provider dependency by design. To wire up a real LLM
planner, subclass `LLMPlannerBase` and implement `_call_model` to return a
dict matching the `Plan` schema. Concrete provider planners belong outside
this package so the core stays small and dependency-free.
