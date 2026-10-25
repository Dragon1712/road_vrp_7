# R4-O v1

- Reuses frozen R4-M source_preflight and checks the R4-N original model/report/provenance SHA.
- Creates three nonleaking Section-action feature sets (15, 38, 50 dimensions).
- Separately fits within-snapshot pairwise ranker and state-level candidate-opportunity gate.
- Performs four TRAIN-task LOTO folds and two grouped seed-transfer folds, with whole-state target shuffle null.
- Adds per-state random-Top1 expected positive-rate comparator, seven strategies including clearly marked oracle-only bounds.
- Fail-closed source identity checks, no ALNS execution, immutable new results, source and code SHA registry.
- Synthetic regression suite: 14 tests. Real Chicago dataset not run inside this packaging environment.
