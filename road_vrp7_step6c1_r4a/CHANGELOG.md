# R4-A v1.1 — 2026-10-24

- **修复实测Smoke阻断错误**：`collect_r4a.py` 的 `plain` 分支原来没有调用冻结C3求解器，导致 `UnboundLocalError: local variable 'ans' referenced before assignment`。现与捕获组使用相同 `v3.solve('C3_topk', ... )` 参数，plain无hook。
- 增加 `tests/test_collector.py`：无需真实Chicago输入，分别覆盖plain/capture，验证两组调用冻结求解器、能形成完整结果、并可通过固定迭代配对审计。
- Runner在开始搜索前拒绝已存在的Plain/Capture/Audit目录，防止与失败部分结果混用。
- 仍要求固定迭代Smoke通过后才进行原20k/30秒Pilot。真实Chicago服务器上的无干预一致性仍待验证。

---

# Step6-C1-R4-A v1

- 原始VQ6在线pilot负收益后停止继续VQ离线调参，转向连续搜索状态忠实捕获。
- 非侵入式只读hook官方 `ALNS.iterate`，在选择算子前捕获 current/best、roulette权重、真实SA温度、RNG状态与停止进度。
- `plain` C3独立负对照，`capture`只读观测，`compare`配对SHA与固定迭代无干预审计。
- Pilot支持Section-event捕获：保留里程碑后真实选中section时的**选择前**RNG与路线；非Section的临时候选快照立即丢弃。
- Smoke中 `--require-exact` 对提前wall截断、路线不一致、OD成本不同均显式拒绝。
- 10项合成/单元测试：读状态不消费RNG、更新只转发一次、hook最终恢复、来源约束、统计守护。
- 声明限制：R4A只捕获真实状态，不包含精确续接与反事实；R4B必须进一步通过回放一致性验证。
