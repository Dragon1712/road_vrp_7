# R4-I v1, 2026-10-25

- 首次以R4-H完整Train反事实结果分析任务/搜索阶段收益和跨任务可学习性。
- 固定输入来源验证：R4-H Full、R4-G Capture、R4-G-Prefix SHA与TRAIN角色、真实C3/SHAM、动作K配对、道路成本等式。
- 原始快照的纯决策前特征、按物理状态加权的纯Python岭回归、任务级留一探索评估。
- K500和保守min(K200,K500)两种目标、4组预设门槛、打乱训练标签负对照、永远C3/随机/早期随机/事后Oracle参照。
- 独立新结果目录、逐折记录、README与本地合成正反向完整测试。
- 不改动ALNS/R4-A至H、不需要新ALNS求解、不将Train离线评估称作30秒在线效果。

## R4-I v1.1 — 2026-10-25: physical search-state identity fix

- FIX: R4-I v1 rounded `anneal_temperature_before_accept` to nine decimal places,
  incorrectly merging 15 R4-H full-precision physical-state groups in actual
  72-snapshot TRAIN source data (R4-H exact 51 vs R4-I v1 rounded 36).
- MATCH SOURCE: physical IDs now use the original unrounded finite anneal
  temperature and SHA256 canonical JSON RouletteWheel weights, matching
  R4-H's `r4a_core.canonical_hash` semantics for captured JSON arrays.
- KEEP the `n_distinct_physical_states` strict source check and no-validation/test
  role checks. Model/features/labels/train-task splits and thresholds unchanged.
- ADD four regressions: real-case near-equal temps, source cardinality mismatch
  detection, weight hash equivalence and invalid number fail-closed.
- Local synthetic suite: 18/18 tests pass; original server TRAIN probe not yet
  re-run in this environment.
