# R4-C v1

- 基于 GitHub `a8d2d947e03469fb675ca32d39e8bb3b367495e7` R4-B 64/64 固定K精确续接结果开发。
- 新增首步真实C3 Destroy随机数等量消耗 + Section Partial精准替换。
- 新增 SHAM_C3 全轨迹等价负对照及跨K前缀一致性审计。
- 仅使用R4-B已验证16个快照白名单，禁止擅自用42个未全部审计的快照。
- 单状态静态候选、冻结分层Section采样、Road.replay事后收益（current/best分别记录）。
- 独立结果目录、逐动作checkpoint、显式RESUME、原始数据/源码SHA保护。
- 不改变R4-A/R4-B/官方C3任何源文件，不声称真实30秒墙钟反事实已恢复。
- 9项本地单元测试、Python与Bash语法检查。
