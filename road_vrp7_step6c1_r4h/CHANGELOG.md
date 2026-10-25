# R4-H v1 — 2026-10-25

- R4-G-Prefix完整16案例报告298/298后，将实验转入原Train的first-Section反事实标签阶段。
- 严格验证R4-G-Prefix源paired_rows/manifest/excluded_windows/capture source SHA；只有全部请求K都准确验证的状态才允许入选。
- 直接调用**未修改的**R4-C `shadow_action`, `run_one`, R4-C `compare_sham`, 冻结C3算子/Repair/SA/静态目标。
- Train角色、求解器/Section/SHA、防止验证/测试泄漏与后验Road评价独立性检查。
- 加入按4个Train任务轮转、早晚状态抽样、可选最大新单位数及严格resume。
- 新增Road当前与历史最优标签、物理起点去重、固定K正负零比例、跨K收益转移/C3追赶。
- 13项本地测试通过；无真实Chicago端到端验证，不声称在线改进。
