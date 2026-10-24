# R4-E v1

- 读取并分析最新R4-D `a2a915d` 结果（固定K历史最优收益稀疏）。
- 在独立R4-E实验目录中扩展 ALNS 逐步恢复验证至 K=100、200、500，冻结R4-B代码，不覆盖源码。
- 复用原R4-C首步Section替换，增加已完成R4-E长K精确证明门槛、R4-C历史K50逐事件前缀负对照、SHAM_C3和跨K前缀检查。
- 增加样本独立性统计：物理搜索起点与不同道路情境分开统计。
- 含SHA/配置来源验证、独立目录、atomic JSON结果、可断点续跑。
- 本地12项测试通过；未在真实Chicago服务器上执行。

# R4-E v1.1 — R4B schema and resume hotfix

- Fix `verify_r4b_evidence()` using R4B `paired_rows.jsonl` with `snapshot_stop_count` (R4A/R4E use `stop_count`). Both names are accepted only when present counters agree.
- Fix `verify_long.py` JSON checkpoint resume identity by comparing JSON lists to JSON lists rather than lists to tuples.
- Strengthen regression tests to use the *actual R4-B paired_rows schema* and reject conflicting/missing counters; add resume JSON round-trip test.
- No changes to ALNS, R4-A/B/C sources, frozen Section solver, or experiment definition.
