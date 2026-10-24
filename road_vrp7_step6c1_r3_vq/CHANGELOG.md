# R3-VQ v1 Change Log

- 基于SV2结果新开发Train-only候选级多步价值标签生成工具。
- C3 vs 每个有效Section候选，统一Static Repair、相同重启随机种子与同一道路事后回测。
- 采集H20、H50每候选3个重复搜索收益的均值与样本标准差，保留C3零锚点。
- 新增Train任务内部Task-heldout四折评价：一步ExtraTrees、VQ-mean、VQ-lower-confidence-bound 0.5/1.0。
- 保留原R2/ALNS/SV源码；SHA锁定分片，允许只追加同配置续跑。
- 8项合成与单元测试。服务器上的真实Chicago端到端执行**尚未完成**。
