# R4-P STATIC Road Section Fusion (Train-only)

**Not dynamic traffic and NOT a trained neural FiLM.** K=500 fixed-step offline diagnostic only.

Mode: route_context / road_concat / road_conditioned; only static directed OSM shortest paths enter features.

| Holdout | Road features | Learned gate + Pairwise gain (sec/snapshot) | Delegations | Positive / Negative |
|---|---|---:|---:|---|
| hold_task_task_0000 | route_context | -3.800 | 18 | 3/5 |
| hold_task_task_0000 | road_concat | -1.851 | 18 | 3/5 |
| hold_task_task_0000 | road_conditioned | -0.096 | 18 | 4/4 |
| hold_task_task_0001 | route_context | -5.525 | 20 | 0/7 |
| hold_task_task_0001 | road_concat | -5.525 | 20 | 0/7 |
| hold_task_task_0001 | road_conditioned | -5.525 | 20 | 0/7 |
| hold_task_task_0002 | route_context | +4.542 | 12 | 6/3 |
| hold_task_task_0002 | road_concat | +3.824 | 12 | 6/3 |
| hold_task_task_0002 | road_conditioned | +3.824 | 12 | 6/3 |
| hold_task_task_0003 | route_context | -2.019 | 14 | 4/4 |
| hold_task_task_0003 | road_concat | -2.642 | 14 | 3/4 |
| hold_task_task_0003 | road_conditioned | -2.873 | 14 | 4/4 |
| old_to_new | route_context | +3.239 | 98 | 17/19 |
| old_to_new | road_concat | +2.045 | 98 | 17/22 |
| old_to_new | road_conditioned | +2.045 | 98 | 17/22 |
| new_to_old | route_context | +0.000 | 0 | 0/0 |
| new_to_old | road_concat | +0.000 | 0 | 0/0 |
| new_to_old | road_conditioned | +0.000 | 0 | 0/0 |

- Each outer Train task/seed split is fitted separately; no heldout label or preprocessing optimization.
- Oracle-conditioned rows are diagnostic upper bounds and are never deployable.
- Static road-network distance is a distinct geographical modality, **not** measured live traffic.
- Random/shuffled training label control and baseline route-only mode must be inspected.
- No real 30-second ALNS and no independent Validation/Test.
