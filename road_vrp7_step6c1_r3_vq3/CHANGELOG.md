# R3-VQ3 v1

- 新增候选+C3 anchored pairwise logistic 排序，普通及Regret加权两路，状态归一权重。
- 新增Train任务内OOF大额损失分类模型、Platt校准及Brier/AUPRC/ECE评价。
- 新增风险概率0.20/0.35门控、残差风险门控、排序风险门控。
- 新增与Risk35训练OOF覆盖率匹配的E0静态分数阈值负对照。
- 严格外层原Train四任务LOTO，复用已存在VQ2标签与五个冻结模型参考。
- 新增状态级Pairwise准确率、完整逐候选预测审计、多种子汇总与SHA锁定。
- 限制：不是在线ALNS、不提供形式化概率保证、不用Validation/Internal Test调参。
