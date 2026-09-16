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

论文首页与 §6 后声明代码位于 `github.com/xycjscs/OpenPSN`。截至本次
改稿，**A3 驱动/数据/图脚本的 commit（本地 f6caac2）尚未推送远端**
（push 由用户最终决定）。公开仓库当前可复现的是 legacy PSN 分支
（Study I/II/复现）；A3 分支的可复现性以本地仓库 + 本包交付的
`tools_ho/data/r17_core4/` 数据为准。→ push 之前，投稿信中如需强调
代码可得性，措辞应与用户确认。
