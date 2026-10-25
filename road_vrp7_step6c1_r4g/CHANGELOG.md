# R4-G changelog

## v1.1 — environment isolation fix

- Fixed `run_r4g.sh` unintentionally inheriting generic `TASK_IDS`,
  `SCENARIO_IDS`, `SEEDS`, `MAX_ITERATIONS`, and `SNAPSHOTS` from preceding
  (potentially Validation) experiments. This produced the legitimate
  `R4G only accepts original TRAIN tasks 0000..0003` rejection during Smoke.
- Introduced explicit scoped overrides: `R4G_TASK_IDS`, `R4G_SCENARIO_IDS`,
  `R4G_SEEDS`, `R4G_MAX_ITERATIONS`, `R4G_SNAPSHOTS`, and
  `R4G_MAX_SNAPSHOTS`, plus read-only `show-config` / `R4G_VIEW_MODE=pilot`.
- Print effective configuration before invoking data/ALNS; proactively reject
  task IDs outside task_0000..task_0003. Preserve unchanged Python TRAIN role,
  eligibility, and downstream SHA checks.
- Added four subprocess regressions simulating inherited Validation variables
  and scoped overrides; total 17 local tests pass.
- No changes to ALNS, Section/Repair, Road.replay, Train capture Python,
  checkpoint schema, original R4-A/B proof logic, or experiment data.

## v1.0

- Original Train-only capture, Plain/Capture audit and exact continuation code.

## v1.2 — 2026-10-24: Correct cross-module R4-A hash path

- FIX: `collect_train_r4g.py` wrote `core_sha256` using `Path(__file__).with_name('r4a_core.py')`, wrongly looking inside `road_vrp7_step6c1_r4g`. Now reads `ROOT/'road_vrp7_step6c1_r4a/r4a_core.py'`, exactly matching the verifier's existing provenance check.
- ADD: two isolated end-to-end collector regression tests (`plain` and `capture`) exercising TRAIN manifest/provenance writing and checking the actual R4-A core path.
- KEEP: v1.1 R4G-prefixed environment variable isolation, strict TRAIN checks, frozen ALNS operators and checkpoint schema unchanged.
- Verification: 19 local tests pass. Chicago/ALNS production Smoke still must run on server.

## v1.3 — 2026-10-24: Persist strict-failure evidence

- Fix audit-order defect: previously, `--require-strict` threw before creating a report. Audit now writes full `report.md/json` before failing closed.
- Add read-only `diagnose` command for already completed Plain/Capture collections (no repeat search).
- Add per-pair reasons distinguishing 20k iteration cap, iteration mismatch, route mismatch, and Road.replay mismatch. Keep TRAIN role and SHA pinning.
- Add five regression tests for wall-limited equal-result pairs, route/road divergences, strict failure report persistence, and non-strict diagnostics.
- No modification to ALNS, R4-A/B, Section, Repair, original checkpoint data, or strict proof semantics.
