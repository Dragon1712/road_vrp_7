# R4-K v1 — 2026-10-25

- 基于GitHub 95b3fbf 精确确认R4-J新seed快照128、源prefix 512/512，R4-H pilot 8快照的K500 2正28负32零，Full仅480个unit（PARTIAL）。
- 新增read-only partial审计和来源SHA、Train角色、全K/全arms、RNG/首事件/SHAM与官方R4C unit digest核验，缺失unit不当作零收益。
- Full完成门禁：源report COMPLETE、计划128快照完整、逐unit/跨K审计与原始labels收益一致；未通过强制失败。
- 汇总搜索轨迹粗粒度去重与精确SA状态计数、C3追赶、收益符号反转及分任务标签情况。
- 新增固定原R4J/R4H的断点续跑编排器，并为大型Full做单快照级事件内存释放。
- 本地10项合成和篡改负对照测试通过；还没有在用户的真实Chicago Full目录实跑。
