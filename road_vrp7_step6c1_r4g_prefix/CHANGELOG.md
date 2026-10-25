# R4-G-Prefix v1

- 使用 GitHub commit `3934546e17c03fe0544ab0fefab68581cf2f2a9e` 的R4-G 16任务场景/76 Section状态和R4-G已通过单case 9/9训练续接结果设计。
- 修复方法层面的不匹配：旧R4-G `verify_train_r4g.py` 对每条参考轨迹硬要求20k终止，导致30秒墙钟截断的七组状态无法参与证明；新代码**没有改变**旧严格失败报告，而是建立独立原30s前缀长度证据、90秒原生参考证据及逐事件恢复证据。
- K=1/50/200/500的独立资格审计，已上传样本的理论窗口数76/76/74/72，共298；这只是eligibility，不是事先宣称通过。
- 复用冻结 R4-G Train-only SHA/role preflight 与官方ALNS 7.0.0 R4-B reference_recording/exact_resume/check_start/verify_events。
- 每Section只恢复最长K一次；新长K真实事件与短K前缀逐项比较。
- 独立manifest/units/完整报告/被排除窗口、断点续跑及SHA检查。
- 包含13项合成测试（含事件篡改必须失败），未在真实服务器Chicago环境完成End-to-End。
