# unresolved_items.md — 现有证据仍不能核对的事实

按任务要求：只列证据缺口；不列"补跑新算例"（用户已拒绝增加计算，
S2 不列为待办，也不声称已验证堆芯空间收敛）。

## 1. OpenMC 参考标准差 ±3.3–3.6 pcm 的原始出处

正文多处引用"same-geometry pure-XS OpenMC 参考（各 ≈±3.3–3.6 pcm）"
与 Study I 的 ±3.6 / 1.1 / 2.4 pcm。仓库内只有 `examples/c5g7/c5g7-mgxs.h5`
（截面库）与结果日志（dref 值），**OpenMC 侧的运行日志/统计误差输出未在
仓库中**。该数值来自此前的会话工作流（OpenMC 均质化/纯 XS 参考运行），
本轮按"导入已有结果"处理，未重新取证。→ 投稿前建议把对应 OpenMC 运行
的统计误差输出归档进仓库（不影响本次改稿的任何数值）。

## 2. §3.1 局部构造 240 例检验的证据脚本

"Maximum normalized differences over 240 cases: 4.5e-12 / 8.5e-12 /
1.0e-15" 对应的独立数值积分检验脚本**不在仓库**（`tools_ho/accept_local.py`
+ `acceptance_local.json` 是另一组 A3/A4 极角验收数据，不是这 240 例）。
该段文字与数值自冻结稿保留。→ 检验脚本若仍在作者本地，建议入仓归档。

## 3. psn2027（Chao ANE 2027）DOI 的公开可检索性

`10.1016/j.anucene.2026.112707`（Ann. Nucl. Energy 240 (2027) 112707）
为 2027 卷在刊文章，公开数据库当前检索不到（轮询无结果属预期）。修订包
`added_references.bib` 注释标记该 DOI 已核验；本轮**未改动**该条目
（按包内指示"Keep the manuscript's existing psn2027 entry"）。

## 4. 两条新增 Multi-PN 文献：已核验，但仅到卷/页/DOI 级

- Ghazaie/Abbasi/Zolfaghari, Prog. Nucl. Energy **110** (2019) 64–74,
  DOI 10.1016/j.pnucene.2018.09.001 —— ScienceDirect 条目 + JST JGLOBAL
  卷/页（110:64–74, 2019）交叉实证通过。
- Falabino/Sciannandrone/Masiello/Vidal, Ann. Nucl. Energy **177** (2022)
  109301, DOI 10.1016/j.anucene.2022.109301 —— ResearchGate/ScienceDirect
  Vol 177 (Nov 2022) 交叉实证通过。
- 两篇**全文未获取**，正文对它们的引用限于一句话的方法族定位（角子域
  独立展开 / 接口角通量分段多项式），未引用其中任何具体公式或数值。
  按任务边界（不虚构 DOI、不夸大等价性），A3 与两方法**未做逐项代数
  等价声明**，仅表述为同属 multi-P_N-type 家族。

## 5. 棋盘 I=60 单点的日志完整性

weak I60 点（−0.9 pcm，`tools_ho/cb_iXX_weak.out`）与 strong I60（+3.7
pcm，`tools_ho/cb_iXX_strong.out`）日志在仓；正文只引用 weak 侧
"I30→I60 移动 +2.9 pcm"（−3.8→−0.9，与 `cb_i30_weak_ext.log`
S20 点 −3.8 逐位一致）。strong I60 点未入正文，无出处问题。

## 6. 组件 S=2 行 M 全点的 npz 与日志一致性

`r17_asm_final.log` 中 UO₂/MOX S2 行全 6 点与
`tools_ho/data/r17_core4/R17*_M*_S2.npz` 一一对应（12 个 npz 齐全）。
正文引用的 S2 点值（−0.5 / −9.8 / 行末值）逐点核对一致。无缺口。

## 7. Core S2 仅有 M=2 一点（−150.1 pcm）

数据在 `core4_sweep.log` 与 `CORE4_M2_S2.npz`（在仓）。按用户决定：
**S2 不再补充、不列待办、不主张堆芯空间收敛**。图 9 core 面板中的
S2 M2 单点已加图注声明为参考点。若日后 HPC 上 S2 M4–24 作业
（job 44381502 等）完成，是否入稿由用户届时决定，不在本任务范围。

## 8. 复现表"within plot reading"行（BWR 含 Gd）

tab:repro 最后一行最大值写 "within plot reading"（对照原文 Fig. 12
图读取），非逐点数值差。该表述自冻结稿保留，未改。

## 9. 代码可用性

论文首页与 §6 后声明代码位于 `github.com/xycjscs/OpenPSN`。Study IV 本轮把
A3/扇区分支的驱动、数据、图脚本与严格校验包一并推到该仓库的 `psn-4` 分支
（含 `paper/data/`、`examples/psn_repro/logs/`、
`examples/c5g7/study4_sector_core/`、`examples/psn_strict_checks/`）。
公开分支上 legacy PSN（Study I/II/复现）继续可复现；扇区闭包需要
`--opt sector-cholmod` 的本地构建，构建步骤在
`docs/psn4-sector-diffusion.md`。

## 10. 堆芯 OpenMC 参考的运行侧输出未入仓

四分之一堆芯的同几何参考取批次均值 `1.187217554586068`（标准误 3.41 pcm，
20 000 活跃批次）与 `k_combined = 1.187220349151050`（标准误 1.75 pcm），
逐棒参考值与 `best_fission.*` 一起入仓，但 **OpenMC 的输入脚本、statepoint
与批次统计输出不在仓库内**。论文引用的是数值与标准误本身；批次级统计
（10–500 批次分块给 3.39–3.91 pcm）无法由仓内文件复核。→ 建议把 OpenMC 侧
输入与 statepoint 归档进仓库后再投稿。

## 11. 本轮从论文删除的两个无归档出处数值

- 节点响应消去"25 个测点上 $k_\text{eff}$ 最大差 $6.4\times10^{-14}$"：
  仓内无对应脚本或日志，已改为引用 `tests/test_sector_diffusion.py` 实际断言
  的两条判据（压缩解对直接稀疏解 $2\times10^{-11}$、同本征对 $2\times10^{-9}$）。
- 无散射真空板的逐点平均通量误差阶梯（96 节点、`I=8, M=4` 时 `A=0,1,2` 为
  `9.42e-3 / 1.95e-3 / 1.30e-3`，`A=2` 加密为 `3.60e-4`）：无归档脚本，论文
  已删除该句，`docs/psn4-sector-diffusion.md` 保留为"试算记录、不作判据"。

两处都不影响其他数值，也不改变结论。

## 12. `--balance-pin` 的 `subdivide` 记号与 SM 扫表不同

同一物理 pin 网格在 `study4_balance_pin.jsonl` 里记为 `subdivide=1`，在
`study4_assembly_points.csv` 里记为 `S=2`（`--balance-pin` 重排胞内胞数后
`suddivide` 字段的含义改变）。论文中该对照只写配对点的 Δk 数值，不用 `S`
标记两侧，避免读者按同一 `S` 去查两个文件。

## 13. 堆芯矩阵的墙钟时间不是吞吐测量

30 个测点在同机上最多两点并发，部分点复用已存分解因子（表中带 `*`），
墙钟包含装配、因子、外迭代与独立复扫。因此 `tab:study4-core-time` 描述
负载而非吞吐，不能用于成本结论；匹配成本的对比仍在本任务证据之外。

## 14. `docs` 里的 `A=3` 棋盘格 40 格表没有仓内出处

`docs/psn4-sector-diffusion.md` 的方形棋盘格一节曾给出记录闭包在 `A=3` 下的
40 格矩阵（`−268.51 / −49.83 / … / −11.98 / −3.68`）。仓内没有任何脚本或日志
产生这些数：`tools_ho/cb_a3_*.log` 是**方向矩**（`directional_matrix`）闭包的
`A=3` 扫表，闭合不同，数也对不上。文档现把它降级为"试算记录、不作判据"，并
在正文以 `paper/tables/study4-checker-sector.tex`（`A=0,1,2`，314 点，
`closure=sector_diffusion`）为记录口径。记录闭包的 `A=3` 网格尚未归档：
`scripts/study4_uniform_grid.py` 的计划只覆盖 `A=0,1,2`，若要补需要重跑该
计划的 `A=3` 扩展，而不是引用现有日志。
