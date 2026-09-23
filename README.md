# deliberation-judge

Multi-agent deliberation judge: multiple blind framings of one state, evaluated by a local calibrated decision model (Laya), aggregated in code (vote / weight / veto) with explicit abstention.

**Status:** planning complete (`REFERENCE.md`); scaffold only — not yet functional.

## Layout

- `src/deliberation_judge/` — library
- `tests/` — pytest (default CI: `-m "not live"`)
- `REFERENCE.md` — pre-build design reference (theory, API, decision tables)

## Not yet implemented

State builder, backend wrapper, jurors, aggregators, CLI behavior. See milestones M1–M7 in `REFERENCE.md`.
