# R4-M v1 (2026-10-25)

- Based on reviewed R4-L commit `62caafdac72b55053b7a932005f6e549aac00199` and its completed Train-only 200-state LOTO/seed-transfer experiment.
- New nested leave-one-Train-task-out empirical overprediction quantile estimates, with fail-closed calibration support gate; conservative deferral and dual K200/K500 risk gate.
- Original R4-L Ridge reference, full-vs-geometry ablation, shuffled-training-label negative control, 6 strictly grouped holdouts.
- Pinned SHA/source role check, full result provenance, no ALNS or Road calculations, immutable experiment output directory.
- Local 8 regression tests and synthetic full evaluator pass. Chicago server real evaluation still pending.
