# VQ6 v1

- 基于 GitHub 5450d9cb2654317c6884d7486d59747331b42755 的VQ5固定面板结果选定在线评估范围。
- 复用VQ4 192 Train状态和VQ3 ExtraTrees / Task-OOF残差训练逻辑；固定H50、seed42、alpha0.5、threshold0。
- 接入官方R2 real-C3-shadow 30秒搜索，不改冻结ALNS或Road.replay。
- 加入DEFER_RESID负对照：完整预测但必定提交真实C3。
- 加入严格四臂配对、错误工件身份/基线未评分/负对照覆盖审计。
- 8项测试，Python编译与Shell语法检查。
- 限制：真实Chicago端到端尚未执行；July Validation是开发数据，不是盲测试。
