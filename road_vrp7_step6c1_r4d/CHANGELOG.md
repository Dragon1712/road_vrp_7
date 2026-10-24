# Step6-C1-R4-D v1

- New immutable posthoc funnel auditor for source-pinned R4-C complete pilot.
- Reads original R4-C unit events, per-action road/static reward, original C3 and SHAM controls.
- Recomputes all candidate labels and K prefixes, rejects mismatched source files/rows and duplicated keys.
- Identifies candidate change, acceptance, current/best divergence, sparse realized best gain and per-state sampled oracle upper bound.
- Separates physical search state repetition across scenarios from road evaluation events.
- No new ALNS search, no new learned model, no changes to frozen source files.
- Synthetic comprehensive tests for rejection cases and full audit report generation.
