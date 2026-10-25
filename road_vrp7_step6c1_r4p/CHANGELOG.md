# R4-P v1

- Compare prior route-context-38 vs new static directed road OD-50 and static-road-conditioned interaction-62 under exactly the R4-O grouped pairwise/gate protocol.
- Add read-only source-pinned Train role + edge index SHA checks; do not load temporal FIFO history as features.
- Generate 4 deterministic static OD matrices and guard customer IDs at every captured route state.
- All six grouped outer splits, C3/SHAM prior controls, training-only whole-state shuffled labels, oracle policy nondeployable flags.
- Original ALNS, C3 operators, R4-H/K labels and R4-O modules on disk are never modified.
- 12 local unit tests (including full synthetic task/seed LOTO training) pass; actual Chicago server run pending.
