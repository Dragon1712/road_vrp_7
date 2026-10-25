# Step6-C1-R4-G v1 — TRAIN-only live ALNS checkpoint acquisition and exact continuation

## Why R4-G, not a new network?

Verified source: GitHub `Dragon1712/road_vrp_7` main commit `5014866fc4d2428b9b603959ea3dffa0c8761c9d`, R4-F `r4f_reversal_16snaps_v1`.

R4-F has 125 physical action choices at each K, 16 scenario checkpoints but only 8 distinct physical ALNS states; K200->K500 shows 15 positive-to-negative reversals from **two** distinct physical states. These are July Validation *development* cases. They must NOT become supervised train labels.

The July `tasks_10` split is **TRAIN July 2–21**, **VALIDATION July 22–25**, and **TEST July 26–31**, regardless of any older prose notes. The actual per-scenario `split` and `eligible` fields decide data eligibility. `R4-G` deliberately rejects non-TRAIN scenarios and limits task IDs to `task_0000...task_0003` to match prior Train-group supervision.

R4-G delivers **data acquisition only**, not R4-H counterfactual labels or a learning model:

- `collect_train_r4g.py`: original frozen `v3.solve('C3_topk')`; imports **unchanged** R4-A `instrument_official_alns` to collect read-only pre-Section checkpoints, or original uninstrumented Plain. 30sec cap / 20k iteration cap for Pilot. No oracle and no intervention.
- `audit_r4g.py`, `r4g_core.py`: check source SHA, TRAIN role, strict exact-K paired-route and Road equality; report unique physical preselection states (including RNG, selector weights and SA temperature) separate from traffic scenarios.
- `verify_train_r4g.py`: independently validates capture provenance and TRAIN role, then reuses **unchanged** R4-B `reference_recording`, `exact_resume` and original ALNS 7.0.0. Check K=1/50/500 event-by-event. Stratify checkpoints across early/middle/late milestones instead of selecting only earliest. Verify only on Pilot with max_iterations=20000. Each case writes an atomic proof unit; `RESUME=1` can continue an existing proof study.

## Install and syntax/tests

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4g_v1.zip
bash road_vrp7_step6c1_r4g/run_r4g.sh test
```

The provided 13+ tests are synthetic/local unit tests; they **do not** claim Chicago/official ALNS end-to-end proof. The old R4-A/B original source files are not edited by this package.

## Stage 1: 2k deterministic paired smoke

```bash
STUDY_NAME=r4g_train2k_smoke_v1 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh smoke

cat runs/chicago_july_v1/step6c1_r4g/r4g_train2k_smoke_v1_audit/report.md
```

One task × one TRAIN scenario × seed 60, Plain and Capture, K=2000. Requires strict fixed-iteration route and Road equality. Any wall-capped case fails strict audit. **Never use these 2k checkpoints in a 20k continuation proof.**

## Stage 2: 20k / 30-sec TRAIN acquisition

```bash
STUDY_NAME=r4g_train16_v1 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh pilot

cat runs/chicago_july_v1/step6c1_r4g/r4g_train16_v1_audit/report.md
```

Default: 4 original TRAIN tasks × 2 eligible TRAIN scenarios (`2025-07-02_0800`, `2025-07-09_1700`) × 2 seeds (60,61) = 16 paired instances (32 solver runs total); milestone requests at 1000,4000,8000,12000,16000,19000, and a snapshot is captured at the next actual Section draw. Six snapshots/run is a *target*, not a guaranteed count.

Pair outputs:
`runs/chicago_july_v1/step6c1_r4g/r4g_train16_v1_plain/`,
`.../r4g_train16_v1_capture/`,
`.../r4g_train16_v1_audit/`.

If the strict audit fails due to wall-cap or changed final route, do NOT call capture non-invasive; inspect which case failed and reduce scope/recording frequency using **new** study names. Pilot acquisition must run to completion before proof. Use a fresh `STUDY_NAME` for each capture; never write into a previous study directory.

## Stage 3: validate true continuous fixed-K on TRAIN

One-case official ALNS Smoke:

```bash
STUDY_NAME=r4g_train16_v1 \
PROOF_STUDY_NAME=r4g_proof_1case_v1 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh proof-smoke
cat runs/chicago_july_v1/step6c1_r4g/r4g_proof_1case_v1/report.md
```

When all horizons K=1,50,500 are exact on the first case, expand:

```bash
STUDY_NAME=r4g_train16_v1 \
PROOF_STUDY_NAME=r4g_proof_16cases_v1 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh proof-pilot
cat runs/chicago_july_v1/step6c1_r4g/r4g_proof_16cases_v1/report.md
```

By default, select max 3 checkpoints/case: early, mid, late among observed checkpoints; expected up to 16×3×3 = 144 precise K comparisons if all snapshots exist and cases reach their full iteration cap. Fail closed if the uninterrupted reference search hits 30-sec wall prior to 20k. Proof is about **fixed K**, not remaining wall-clock 30 seconds.

Interrupted proof continuation **only** (newly created capture studies are not resumed):

```bash
STUDY_NAME=r4g_train16_v1 PROOF_STUDY_NAME=r4g_proof_16cases_v1 RESUME=1 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh proof-pilot
```

`units/*.json` are per-case source-checked evidence, and `paired_rows.jsonl`/`report.md/json` summarize verified K. No supervised label is computed yet.

## Further optional controlled expansion

Expand independent solver seeds after first batch, not by relabeling scenario duplicates:

```bash
STUDY_NAME=r4g_train_moreseeds_v1 \
SEEDS=62,63,64 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh pilot
```

Group physical states by task, seed, checkpoint counter, current+best route, RNG, RouletteWheel and anneal. Scenario repeats are not independent. No Test use, no training on previous July Validation R4-F labels, and no claims of online gain.

## Next stage (NOT included): R4-H

Only after R4-G strict paired checks and TRAIN proof succeed: adapt the R4-C first-Section counterfactual intervention and C3/SHAM control **with the new TRAIN evidence schema**, explicitly train-only; label horizons K=50,200,500 and aggregate to independent physical states. Then fit simple, conservative selectors and keep an untouched validation evaluation where possible. Do not drop role checks or relax existing frozen solver SHA checks.

## R4-G v1.1: prevent inherited shell-variable contamination (Smoke correction)

**Why the earlier Smoke failed:** R4-G v1 used generic `${TASK_IDS:-...}`,
`${SCENARIO_IDS:-...}`, `${SEEDS:-...}`, etc. If those variables had
previously been `export`ed for a July Validation study, the Smoke unintentionally
received task_0008/task_0009. `collect_train_r4g.py` correctly rejected these
because this collector allows only task_0000--task_0003.

**What changed:** `run_r4g.sh` now **ignores the generic inherited variables**
`TASK_IDS`, `SCENARIO_IDS`, `SEEDS`, `MAX_ITERATIONS`, and `SNAPSHOTS`.
For intentional overrides, use the namespaced `R4G_TASK_IDS`,
`R4G_SCENARIO_IDS`, `R4G_SEEDS`, `R4G_MAX_ITERATIONS`,
`R4G_SNAPSHOTS`, and `R4G_MAX_SNAPSHOTS`. The strict Python TRAIN-only task,
role and `eligible` checks remain unchanged. No ALNS, road-cost, or checkpoint
schema code was modified.

First inspect the **effective** configuration (no solver/data access):

```bash
bash road_vrp7_step6c1_r4g/run_r4g.sh show-config
R4G_VIEW_MODE=pilot bash road_vrp7_step6c1_r4g/run_r4g.sh show-config
```

Then use a new study name (the failed v1 guard executes before output creation,
but a distinct name also keeps experiments separate):

```bash
bash road_vrp7_step6c1_r4g/run_r4g.sh test
STUDY_NAME=r4g_train2k_smoke_fix1 bash road_vrp7_step6c1_r4g/run_r4g.sh smoke
cat runs/chicago_july_v1/step6c1_r4g/r4g_train2k_smoke_fix1_audit/report.md
```

Optional intended pilot overrides, **using namespaced env vars**:

```bash
STUDY_NAME=r4g_train_moreseeds_fix1 R4G_SEEDS=62,63,64 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh pilot
```

**Old examples earlier in this README using `SEEDS=` for pilot overrides
must now use `R4G_SEEDS=`.** Likewise `TASK_IDS=` is not an R4-G override.
Use `R4G_TASK_IDS=` for specific allowed TRAIN task IDs.
This fix does not establish ALNS/Chicago end-to-end validity; run Smoke and
inspect the strict non-interference audit before the 20k Pilot.

## v1.2 hotfix: correct R4-A core file SHA path

The preceding v1.1 Smoke can fail before creating an output directory with:

```
FileNotFoundError: .../road_vrp7_step6c1_r4g/r4a_core.py
```

Root cause: collector tries to hash `r4a_core.py` adjacent to the R4-G collector, but this file lives under `road_vrp7_step6c1_r4a/`. The original downstream verifier already expects the latter. v1.2 fixes the collector without copying, moving, or editing R4-A source. It also adds synthetic collector-entrypoint tests for both Plain and Capture.

Install v1.2 in the **repository root** (not inside the R4-G subfolder):

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4g_v1_2.zip
bash road_vrp7_step6c1_r4g/run_r4g.sh test
STUDY_NAME=r4g_train2k_smoke_fix2 bash road_vrp7_step6c1_r4g/run_r4g.sh smoke
cat runs/chicago_july_v1/step6c1_r4g/r4g_train2k_smoke_fix2_audit/report.md
```

Inspect that `strict_pass == pairs`, that both Plain/Capture completed their iteration cap, and that they have identical final routes and Road.replay costs. Only then start the 20k Pilot with a separate `STUDY_NAME`.

The failed v1.1 invocation normally stops before `out.mkdir`, so the old study name can be free, but a distinct name is recommended for auditability. No previously completed R4-A/B/C/D/E/F result is modified. This is not a verified end-to-end Chicago run; the local 19 tests rely on stubs for real Road and ALNS.

## v1.3 — Strict Pilot failure: read-only diagnosis without rerunning ALNS

If the 20k/30sec Train Pilot successfully writes all Plain/Capture data, but exits with
`Fixed-iteration noninterference NOT fully verified`, **do not delete or redo** its
32 solver runs before inspecting the failure. In v1.2 `audit_r4g.py` raised
**before writing** an explanation. This does not imply a broken collector:
some cases may have hit the 30-second wall before reaching 20,000 iterations;
others may genuinely have different terminal routes. These outcomes must not
be conflated.

v1.3 makes `audit_r4g.py` write `report.md/json` with each offending
(task, scenario, seed) and category **before** failing strict mode. It adds
`diagnose`, which loads finished artifacts using the same TRAIN-only role and
SHA verification but **does not run ALNS, repair, or Road.replay**.

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4g_v1_3.zip
bash road_vrp7_step6c1_r4g/run_r4g.sh test

# Replace STUDY_NAME with the exact prior completed Pilot name.
STUDY_NAME=r4g_train16_fix2 \
  bash road_vrp7_step6c1_r4g/run_r4g.sh diagnose
cat runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_audit_diagnostic_v1_3/report.md
```

To repeat a diagnostic use a **new** `R4G_DIAG_NAME=...` (outputs are
never silently overwritten). The existing `_plain`, `_capture` and their
`provenance.json` **must not** be edited. With v1.3, a regular `audit` still
exits nonzero if strict verification fails, but writes the report first.

Interpret failure categories:

- `ITERATION_CAP_NOT_MET`: Plain or Capture did not finish 20,000 iterations.
  It is not a fixed-iteration noninterference proof; may be normal for
  30-second wall-limited tasks.
- `ITERATION_COUNT_DIFFERS`: different completed iterations, often indicating
  a wall-clock effect.
- `FINAL_ROUTE_MISMATCH`: final route hashes differ; even equal iteration
  counts need a closer investigation (e.g. deadline-sensitive operator code).
- `ROAD_COST_MISMATCH`: final Road.replay cost differs.

**Important:** v1.3 does **not** relax `--require-strict`. An incomplete
20k/30sec pair or differing final route must not be marked VERIFIED. In
addition, `verify_train_r4g.py` as shipped currently requires the original
reference to reach 20,000 iterations. Do not proceed to full Train checkpoint
proof from a failed Pilot without diagnosing and addressing this limitation.
If a smaller fixed-iteration cap is needed, it requires a separate new
capture study and a separately validated proof protocol, not merely setting
`R4G_MAX_ITERATIONS=10000` and reusing the 20k-only proof code.

v1.3 changes only `audit_r4g.py`, `r4g_core.py`, the R4-G shell script,
documentation and tests. Frozen C3, Repair, ALNS, R4-A/B code, checkpoint
format, and existing Train capture artifacts are unchanged.
