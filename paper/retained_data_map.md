# retained_data_map.md — 既有结果在新稿中的位置、方法与配置

原则：只列位置与条件，不改任何数值。所有数值与改稿前逐点核对一致；
A3 各行已对照 `tools_ho/r17_asm_final.log`、`tools_ho/core4_sweep.log`、
`tools_ho/data/r17_core4/*.npz` 实证。"新稿位置"以修订后 PDF 30 页的章节号计
（main.tex §1–§6，studyIII.tex 挂为 §4.4、studyIV.tex 挂为 §4.5，文献景观段顺延为 §4.6；章节号以最终 `main.pdf` 为准）。

---

## §3.1 局部构造代数检验（验证章）

| 结果 | 方法/配置 | 数值 |
|---|---|---|
| 240 例局部构造检验 | A3 节点响应矩阵 vs 直接组装，逐算例机器精度比对 | 最大归一差见正文 §3.1（原样保留） |

## §3.2 已发表 PSN 结果复现（验证章，Table tab:repro + 附录 A）

| 问题 | 方法/配置 | 点数 | 最大偏差 |
|---|---|---|---|
| 棋盘 weak（Fig. 3 原文） | legacy PSN, A=0, I=30, S=1–10 × M=2,4,8,12,16,24, kref=1.12974 | 60 | 1.2 pcm |
| 棋盘 strong | 同上, kref=0.51673 | 60 | 1.4 pcm |
| BWR 12 棒, 无 Gd（Fig. 10） | legacy PSN, 2 群, kref=1.18797 | 60 | 3.8 pcm |
| BWR 12 棒, 含 Gd（Fig. 12） | 同上 + Gd pin（base[3,3]）, kref=0.86688 | 60 | within plot reading |
| 合计 248 点 | 240 表格点（附录 Tables appA–appD）+ 8 个受限 M=24 点 | 248 | — |

逐点数据随代码交付：`docs/fig3ref.json`（棋盘 120 点）、
`docs/bwr_ref.json`（BWR 120 点）。

## §4.2 Study I — pinwise 均质化模型（评估章）

方法：legacy PSN（含矩形节点选项），TY3 极角约定，flux-weighted
self-shielded pin XS（pin-homogenised 模型）。

| 几何 | kref（OpenMC 均质化） | 最优 S×M 点 | Δk |
|---|---|---|---|
| UO₂ 组件 | 1.3336285 | S=1, M=24 | +0.0 pcm |
| MOX 组件 | 1.1853993 | 见表 | −1.5 pcm |
| 1/4 core | 1.1868230 | 见表 | 见正文 |

扫表 S=1–6 × M=2,4,8,12,16,24（36 点/几何）。图：fig 1–3 区域
（pinwise-homogenised Δk 图 + 最佳点场图）。

## §4.3 Study II — 3×3 矩形平铺显式材料模型（评估章，Table tab:rectsm）

方法：legacy PSN 矩形节点，3×3 pin tiling，TY3，纯 XS OpenMC 参考。

| 几何 | kref（纯 XS OpenMC） | 最优 S×M 点 | Δk |
|---|---|---|---|
| UO₂ 组件 | 1.3347260 | S=1, M=24 | +1.2 pcm |
| MOX 组件 | 1.1851053 | S=2, M=24 | −33.5 pcm |
| 1/4 core | 1.1872176 | S=1, M=24 | −10.2 pcm |

扫表：组件 S=1–6 × 6M（36 点/几何）；core S=1,2 × 6M + S=3 × M=2,4,8
（15 点）。S 单调恶化段（UO₂ −28.1@S2 → −56.5@S6）与 MOX 单内峰
（S=2）表述保留，限定"within this tiling and the tested range"。
图：rect_b4_sm_heatmap（绿环=最佳点）、c5g7_rect33_best_fission（三行
场图）。

## §4.4 Study III — A=3 局部角矩（2×2 fuel-split tiling，评估章）

方法：A=3 segmentwise even-parity angular-moment closure
（directional_matrix），TY3，GMRES 内扫（生产驱动 rtol=1e-7，
maxiter=800 restart cycles，restart=20），1e-7 双门（dlam & src_res）。
节点数：组件 S1=4624 / S2=18496 矩形；core S1=41616 / S2=166464。

### A3 组件角分辨率扫描（S=1,2 × M=2,4,8,12,16,24 = 24 点，全收敛 1e-7）

| 几何 | S=1 行 (M=2,4,8,12,16,24) | S=2 行 |
|---|---|---|
| UO₂ 组件（kref 1.3347260, ±3.6） | −119.2, −35.0, −2.8, +5.3, +8.3, +10.3 | −128.7, −50.2, −14.7, −6.3, −2.8, **−0.5**（best） |
| MOX 组件（kref 1.1851053, ±3.3） | −274.8, −74.1, −14.5, −0.9, +4.1, +7.4 | −303.3, −103.2, −33.9, −19.6, −13.7, −9.8 |

两组件 S=1 行自 M=2 起按符号 Δk 单调（正文表述保留）。

### A3 1/4 core 角分辨率扫描（S=1 全 6 点已收口）

kref 1.1872176：M=2,4,8,12,16,24 → **−135.6, −34.2, −1.7, +6.2, +9.1,
+11.0 pcm**。M8 进入 MC 标准差带；M24（+11.0）为本次从
`CORE4_M24_S1.npz`/`core4_sweep.log` 导入的已完成点（未重跑）。
core S2 仅 M=2 一点（−150.1 pcm）在图 9 中作为参考点，图注声明不据此
主张堆芯空间收敛；S2 其余点不补、不列待办（用户决定）。

### A3 归一化裂变场比较（fig:a3fission，同几何纯 XS OpenMC 功率图）

| 配置 | Δk | RMS（max） |
|---|---|---|
| UO₂ 组件 S=2, M=24 | −0.5 pcm | 0.055%（max 0.19%），264 棒 |
| MOX 组件 S=2, M=24 | −9.8 pcm | 0.075%（max 0.22%），264 棒 |
| 1/4 core S=1, M=8 | −1.7 pcm | 0.286%（max 0.64%），1056 棒（2×2 燃料块裁剪） |

限定：选定配置的归一化场比较，非完整 S×M 空间场误差调查。

### A3 棋盘复现（fig:cbA3，原文 Fig. 3 几何）

方法：A=3 closure，generic even I=30（15 独立扇区），kref 1.12974 /
0.51673。S=1–20 × M=2,4,8,12,16,24 = 每吸收体 120 点、两吸收体共 240
点，全收敛 1e-7。最佳点：weak S=12, M=16/24（−0.1 pcm）；strong
S=14, M=24（−0.5 pcm）。I=60 单点（S=20, M=24）：相对 I=30 同点
移动 +2.9 pcm（−3.8 → −0.9），表述保留。

### 三扇区角图（fig:pinangular）

pin 内水区角表示示意（静态图 `pin_angular_three_sectors_en.pdf`，无
独立生成脚本，直接入仓）；图注含 A=0,1,2,3 各段 A+1 个局部偶角系数的
表述（"A degrees of freedom" 已更正为 A+1 local even angular
coefficients per segment）。

## §4.5 Study IV — 有限扇区扩散闭包（评估章，sec:study4）

原则同上：只列位置与条件。本轮新算的只有四分之一堆芯的 5 × 6 矩阵；其余
Study IV 数值来自此前的 campaign，逐点核对后写入，源数据一并入仓
（`paper/data/`、`paper/tables/`、`examples/psn_repro/logs/`）。

| 论文位置 | 数据 | 方法/配置 | 仓内出处 |
|---|---|---|---|
| 表 tab:study4-checker | 棋盘 130 点（$A{=}0$ 44 / $A{=}1$ 40 / $A{=}1$ 双线性 40 / $A{=}2$ 6） | `sector_average`，$I{=}30$，$k_\text{ref}$ 0.51673 / 1.12973 | `examples/psn_repro/logs/*.log` → `paper/data/study4-checkerboard.csv`、`paper/tables/study4-checker.tex`（由 `scripts/study4_checkerboard_report.py` 重建，并复核同号与单调性） |
| §4.5 谷线定位 | legacy 变号点（$S{=}2$：$-790.22$ → $+807.92$ pcm 等） | 原法 $A{=}0$，$I{=}30$ | `docs/fig3ref.json`（与附录 A 同源） |
| §4.5 谷线定位 | `strict_phase_space` 对照 $-1363.03$ / $+76.41$ pcm | 只换体内张量、保留原面闭合 | `examples/psn_repro/logs/strong_strict_sector_valley.log` |
| §4.5 计时句 | 双线性 12.81 s 对 legacy 37.44 s（$S{=}12,M{=}24$） | 同网格两点，日志未记线程数/负载 | `timing_bilinear_full_strong.log`、`timing_final_legacy_strong.log` |
| 表 tab:study4-assembly 与三张 studyIII 图 | 384 点六矩阵与最佳点逐棒偏差 | $A{=}1$ 双线性角系数，组件 $S{=}1\ldots6$、棋盘 $S{=}1\ldots20$ | `paper/data/study4_assembly_points.csv`、`paper/data/study4_assembly_summary.csv`，图 `figures/studyIII_*.pdf` |
| 表 tab:study4-checker-sector、tab:study4-closure-ratio | 记录口径棋盘 314 点（发表网格 114 + $A{=}1$ 宽矩阵 200）与 416 个同格配对 | `sector_diffusion`，$I{=}30$，逐点独立复扫（最大源残差 3.93e-9） | `paper/data/uniform/points.jsonl` → `paper/data/study4-checkerboard.csv`、`paper/data/study4_closure_pairs.csv`、`paper/tables/study4-checker-sector.tex`（`scripts/study4_uniform_grid.py merge` + `scripts/study4_closure_report.py`） |
| 表 tab:study4-assembly-sector 与四张 $S\times M$ 矩阵图 | 记录口径六矩阵 440 点（棋盘 314、组件 126；两个矩形组件各缺同样 9 格） | $A{=}1$，$I{=}8$（组件）/ $I{=}30$（棋盘），逐点耗时与因子池入档 | `paper/data/study4_assembly_points.csv`、`paper/data/study4_assembly_summary.csv`、`paper/data/uniform/plan_unreachable.json`，图 `figures/study4_{checker_sm,sm6_sm,assembly_grid2x2,assembly_wall,sm_overview}.{png,pdf}` 由 `scripts/study4_sm_figures.py` 从上述文件重画 |
| §4.5 矩形 MOX 段 | $M{=}24,A{=}2$ 下 $-185.20\to-52.81$ pcm | 显式水隙空间加密 | `paper/data/study4_rectangular_controls.csv`（series=Control） |
| §4.5 $M{=}2\to4$ 段 | 57.33 → 40.69 pcm、$A{=}4$ 对照 $-71.17\to-33.21$ pcm | `--balance-pin` 重排胞内胞数 / 提高 $A$ 与空间基 | `paper/data/study4_balance_pin.jsonl`、`paper/data/study4_rectangular_controls.csv` |
| 表 tab:study4-core-pcm / -time / -best 与两张堆芯图 | 四分之一芯 30 点 | TY3、$A{=}1$、60 线程、$k_\text{ref}=1.187217554586068$ | `examples/c5g7/study4_sector_core/`（30 个日志 + 两张矩阵 CSV + `sm_matrix.md`），论文表由 `scripts/study4_core_report.py` 生成 |
| §4.5 严格校验段 | 两层板解析量、四反射均质七群 $k_\infty$、LP 上界 $k<1.894$ | 符号/有理精确 | `examples/psn_strict_checks/`（`unittest discover`，14 项） |
| §4.5 节点响应段 | 压缩面迹解 vs 直接稀疏解 $2\times10^{-11}$、同本征对 $2\times10^{-9}$ | 4 个小矩形 + 强棋盘 | `tests/test_sector_diffusion.py` |

## §4.6 文献景观对比（Table tab:lit + NEA 20 码）

- OpenPSN 行：k=1.1872011，方法栏现为 "Local angular-moment nodal
  (A=3; square-fuel model, S=1, M=8)"；对同几何 OpenMC 1.1872176 差
  −1.7 pcm；对圆棒文献参考 1.18646 差 +74.1 pcm（含几何模型差，脚注
  已声明非同几何比较）。
- 同带：pinwise-homogenised 点在 MOC 参考簇（±37 pcm）内。
- 离群：COHINT −1125、HELIOS +675（保留，归因各自近似）。
- 矩形节点行锚定 rectangular-rod 理想化，其 OpenMC 参考 1.1872176
  位于 1.18646 上方 +75.8 pcm（保留）。

## 图/表 ID 与公式标签

全部保留原 ID：tab:repro、tab:rectsm、tab:lit、tab:appA–appD；本轮新增
tab:study4-checker、tab:study4-assembly、tab:study4-core-pcm、
tab:study4-core-time、tab:study4-core-best、fig:study4-c5g7、
fig:study4-fission、fig:study4-walltime、fig:study4-core、
fig:study4-core-fission、eq:study4-even、sec:study4。
fig:rectsm、fig:rect33best、fig:cbA3、fig:a3mconv、fig:a3fission、
fig:pinangular 等；公式 \label{eq:*} 未动，交叉引用按新章节号更新。
