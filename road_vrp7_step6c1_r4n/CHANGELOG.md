# R4-N v1

- 新增Train-only的Section Pairwise排序与状态级Gate解耦诊断，使用原R4-L任务/种子分组及R4-M完整SHA来源校验。
- 配对差分标签只用于训练，不作为输入特征；同任务物理搜索起点与跨场景重复进行分组权重处理。
- 增加`C3`、原Ridge、Pairwise强制委托、Ridge Gate+Pairwise，以及三组显式Oracle/Perfect实验，严格标记未来标签策略不可部署。
- 新增外层任务留一/种子迁移、全特征/几何消融、任务内Shuffle负对照、独立结果目录及不可变来源追踪。
- 8项合成回归测试覆盖任务/种子隔离、排序无标签泄漏、打乱保留分布、无信息回退、Oracle上界及完整报告输出。
- 未修改官方ALNS、Repair、冻结Road或原R4-G至R4-M实验产物；本地测试不等于真实Chicago运行成功。
