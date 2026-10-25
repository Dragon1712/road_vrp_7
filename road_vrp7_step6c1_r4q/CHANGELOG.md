# R4-Q v1.1 — Stage2 left-boundary historical availability fix

- Confirmed Stage2 time_index begins at 2025-07-02 00:00:00; July 2 08:00 minus 24 hours predates this dataset.
- Added `resolve_historical_source`: no-data-before-start => explicit full static fallback, *not* a substitute history slot; missing slots inside/after the range still fail.
- Historical matrices for absent history are asserted equal to static OD; masks and coverage are zero.
- Preflight reports requested/actual slots and availability; July 9 still requires a real exact 24-hour slot.
- Report adds per-scenario availability and stratified non-Oracle learned-policy metrics; saved OD contains the availability flag.
- Added source-boundary, leakage, missing-interior and scenario-stratification regressions. No changes to prior ALNS/CF sources, R4-O model or labels.
