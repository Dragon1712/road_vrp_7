# VQ4 v1 changelog

- Added read-only comparison for nested 48→192 original training state selections.
- Added validated copying of old 48-state completed unit files with full original-source SHA identity checks.
- Added exact candidate/seed/baseline and true-road cost consistency checks when an initialization unit exists in target.
- Added immutable nested collection views for 48/96/144/192 states without rerunning a single extra search.
- Reused unmodified frozen VQ3 training and aggregation algorithms for sample-size learning curves.
- Added cross-size/cross-seed source SHA consistency and frozen original VQ3 48-state reproduction check.
- Added 9 synthetic positive and negative contract tests; no real-world optimization gains claimed.
