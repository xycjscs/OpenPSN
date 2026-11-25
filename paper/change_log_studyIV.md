# change_log_studyIV.md — Study IV（扇区扩散闭包）本轮新增记录

输入：`sector_diffusion` / `sector_average` 闭包分支（`psn-4` 分支既有实现）、
`validation/` 下此前与本轮的实测数据树、`validation/psn-strict-verification/`
的严格代数校验包。与 `change_log.md` 不同，**本轮包含新的输运/本征值计算**
（C5G7 矩形四分之一堆芯 5 × 6 = 30 点完整矩阵），其余 Study IV 数值来自此前
已归档的 campaign，逐点核对后写入论文。

**总原则：论文里每个数值都能追溯到仓库内的日志或结果文件；生成物由脚本重建，
不手抄；达不到的结论（收敛性证明、普遍成本优势、极限输运精度）不写。**

---

## 1. 新增章节 `paper/studyIV.tex`

- 新增 `\subsection{Study IV --- the finite-sector diffusion closure}`
  （`\input{studyIV.tex}`，挂在 §4 末尾，标签 `sec:study4`）。
- 五个小标题：Formulation / The diagonal valley of the square checkerboard /
  Assemblies: six $S\times M$ matrices / C5G7 rectangular quarter core /
  Symbolic and exact checks，末尾 Scope。
- 公式 (eq:study4-even)：每个方向（或每个有限角胞）的偶宇称二阶方程
  $-\nabla\cdot(\Omega\Omega^{\mathsf T}u/\Sigma_t)+\Sigma_t u=q$，真空项带
  $|\Omega\cdot n|$，镜面反射方向共用面上迹；空间用连续双二次 Lagrange 节点，
  体项、材料界面法向流、真空项出自同一弱形式（不引入额外面流模型）；每材料节点
  9 个空间矩。
- 明确写出两处"代数而非数值"的验证：`angular.node_response` 的内部自由度精确消去
  （改为引用仓内回归：4 个小矩形上压缩面迹解对直接稀疏解 $2\times10^{-11}$、强棋盘
  上消去与直接系统同本征对相对差 $2\times10^{-9}$，`tests/test_sector_diffusion.py`；
  原稿的"25 点最大差 $6.4\times10^{-14}$"无归档脚本/日志，已删除，见
  `unresolved_items.md`），以及有限角闭包不可能精确
  传递角分布的两个反例（纯吸收板出射映射的单射性 → 任意有限矩存在同矩不同出射；
  余弦源反射板"先解后平均"给 $\pi/4$、"先平均张量再解"给 $3/4$）。

## 2. 棋盘对角谷线（§4.5，表 tab:study4-checker）

- 记录原文误差沿对角线变号的位置随 $S$ 移动（$S{=}2$ 行在 $M{=}4\!\to\!8$ 之间
  由 $-790.22$ 到 $+807.92$ pcm 等）。
- 记录"只换体内张量、保留原面闭合"的对照仍变号（$-1363.03$ / $+76.41$ pcm），
  据此把变号定位到面闭合。
- 扇区闭包 130 点（$A{=}0$ 44 点、$A{=}1$ 二次 40 点、$A{=}1$ 双线性 40 点、
  $A{=}2$ 6 点）全部同号（负），两轴加密单调趋零；并披露两点求积的失败试验
  （$M{=}4$ 若干点重新变号，因扇区张量欠积分）。
- 表 tab:study4-checker 改由 `scripts/study4_checkerboard_report.py` 生成：
  9 个逐点日志（$A{=}0/1/2$、双线性、strict_phase_space 对照、legacy 计时对照）
  入仓 `examples/psn_repro/logs/`，脚本逐点校验 closure/$A$/求积/$k_\text{ref}$/
  残差门槛与 pcm 自洽，断言同号与单调性，再输出
  `paper/tables/study4-checker.tex` 与 `paper/data/study4-checkerboard.csv`
  （140 行，含每个数的日志出处）。
- 双线性表示原写"只在 $S{=}12,M{=}24$ 测过"，核对日志后更正：它在两个吸收体上
  都跑过完整 $S{=}2\ldots12\times M{=}4,8,12,24$ 的 20 点网格，弱吸收最佳
  $-8.94$ pcm。计时对照改为同源日志（双线性 12.81 s 对 legacy 37.44 s），并写明
  这两份日志未记线程数与负载，不构成受控计时比较。
- 变分下界只在一群、正移除、精确求积、嵌套加密时成立，正文同时写明七群问题不受
  该界约束、部分加密非嵌套。

## 3. 六个组件/棋盘矩阵（表 tab:study4-assembly、图 studyIII_*）

- 384 点（$A{=}1$ 双线性角系数）覆盖两棋盘与四方组件，逐点通过自身复扫；
  图 `figures/studyIII_c5g7_sm.pdf`、`studyIII_best_fission.pdf`、
  `studyIII_wall_time.pdf`（PNG 同名入 `figures/` 供 markdown 版引用）。
- 矩形 MOX 单列一段：固定 $M{=}24,A{=}2$ 时误差随 $S$ 由 $-185.20$ 降到
  $-52.81$ pcm，说明瓶颈是水隙的空间分辨率而非方位角。
- 矩形 UO$_2$ 的 $M{=}2\!\to\!4$ 反常单独成段：均匀划分下 $S{=}1$ 变差 $57.33$ pcm；
  `--balance-pin` 把胞内 2/2/2 改 1/4/1（同一批材料边界、同一总宽度）降到 $40.69$ pcm；
  均匀网格上改 $A{=}4$ 则 $-71.17\to-33.21$ pcm，反常消失。结论写成"粗扇区 × 粗
  胞内空间分辨率的耦合"，不宣称角求积被修复。

## 4. 四分之一堆芯完整 $5\times6$ 矩阵（§4.5）

- 表 `tables/study4-core-pcm.tex`、`study4-core-time.tex`、`study4-core-best.tex`
  与图 `figures/sector_diffusion_sm.pdf`、`study4_core_fission.pdf` 全部由
  `scripts/study4_core_report.py` 从 30 个逐点日志重建（逐格断言 closure/model/
  $(S,M,A,\text{threads})$/节点数/`kref`/残差门槛与 pcm 自洽），不做手抄。
- 写明参考值口径：同几何 OpenMC 批次均值 $1.187217554586068\pm3.41$ pcm
  （20 000 活跃批次）、`k_combined` $1.187220349151050\pm1.75$ pcm；并写明
  OpenPSN 读入的 $\sum_g\chi_g=1.00000917761$ 与 OpenMC 归一化之差为 $1.0896$ pcm，
  与目标精度同量级，故按格报告而非给单一数字。
- 同几何、同 $S{=}1,M{=}24$、同 60 线程、均冷启动的闭包对照：原版 $-10.12$ pcm /
  128.08 s，扇区 $-54.09$ pcm / 340.68 s —— 最粗角分辨率下扇区闭包不总更好。
- 方位阶次外推（$I{=}8,A{=}1$：局部阶 1.32 / 1.47，$M\to\infty$ 外推
  $-54.64$ / $-58.90$ pcm）与空间对照（$M{=}16$ 下 172→244 网格仅 3.24 pcm）
  一并写明，结论限制为"在该网格族上残差受胞内材料与极角分辨率限制"。

## 5. 严格代数校验（§4.5 末）

- 两层真空板（单位光学厚度、源项 1 与 2）的界面通量 $3[1-E_2(1)]/2$、界面净流
  $-1/4+E_3(1)/2$、吸收+泄漏 $=3$；镜像矩形解析通量代入一阶与偶宇称二阶方程的
  符号残差恒零；斜特征线跨两界面并反射时加细分不改响应系数；C5G7 截面按精确有理数
  读入的四反射均质七群本征值 4 个；堆芯逐群加权守恒 + 有理线性规划给
  $k_\text{eff}<1.894$。
- 14 项校验入仓 `examples/psn_strict_checks/`（`unittest discover`，约 2 s，依赖
  `sympy` 1.13.3 与 PyYAML），正文同时写明"约束算子，不约束生产网格的截断误差"。

## 6. 正文其余位置

- §1 贡献由 fourfold 改 fivefold，新增第四条（有限扇区闭包 + 棋盘谷线在全部实测点消失）。
- §1 组织结构段、§2 框架段（`angular.closure: sector_average|sector_diffusion`）、
  §5 讨论段（新增"扇区研究同标准：实测网格非收敛证明、最佳点仍在 MC 标准误之外、
  同 $M$ 下更贵"）、§6 结论各补一段。
- `paper/main.md` / `revised_main.md`：同步上述段落，并插入与 LaTeX 同源的 Study IV
  小节与三张表（markdown 表由同一 CSV 生成）；图按现有 markdown 约定引用 `figures/`
  下同名 PNG。

## 7. 仓库其余同步

- `README.md`：§3 增加 Study IV 命令块；§5 对照表增加 4 行（棋盘 130 点、384 点组件
  矩阵、四分之一芯 SM 矩阵、严格校验包）；§6 增加扇区/严格测试命令；§8 目录树补
  实现、example、paper、tests、scripts、docs 条目。
- `docs/psn4-sector-diffusion.md`：新增"完整 5 × 6 SM 矩阵"小节（TY3 与 $I{=}8$
  的区别、30 点网格、example 目录与生成脚本）。
- `examples/c5g7/study4_sector_core/`：新目录，含协议说明、30 个逐点日志、
  两张矩阵 CSV + `sm_matrix.md`、色板图、最佳点裂变图与逐棒统计。
- `examples/psn_strict_checks/`：新目录，14 项符号/有理精确校验与 README。
- `scripts/study4_core_report.py`：日志 → 矩阵 CSV / markdown / 色板图 / 论文表的
  唯一生成入口。

## 8. 已知未做

- 记录口径的补测已收敛到这台主机的上限：`scripts/study4_uniform_grid` 计划的 458
  点里 440 点收敛并并入 `paper/data/` 两个逐点 CSV（棋盘发表网格 114/114、`A=1`
  宽矩阵新增 200/200、方形组件 72/72、矩形组件 54/72）。剩 18 点是两个矩形组件
  各同样的 9 格（`S=4` 的 `M=16,24`、`S=5` 的 `M=12,16,24`、`S=6` 的 `M≥8`），
  实测稀疏因子池 49.77–182.85 GiB，超过主机 58 GiB，逐格记在
  `paper/data/uniform/plan_unreachable.json`；这两例的最佳点因此是下界不是上限，
  论文与 `docs` 都按这个口径写。换更大的机器之前这 18 点跑不出来。
- 新闭合那侧没有逐棒裂变图：`study4_assembly_summary.py` 派生的汇总行只有本征值
  与耗时，`fuel_rms_percent` / `fuel_max_percent` 留空，所以
  `tab:study4-assembly-sector` 的 Pin 两列是 `---`，两篇论文的正文也写明这一点。
  对照闭合那侧的棒图仍出自 Study III 的逐棒裂变输出。
- 新五组矩阵图只进了两篇 LaTeX 论文；markdown 镜像（`paper/studyIV.md` 与
  `main.md` / `revised_main.md`）仍只有文字与表，沿用镜像里既有的取舍。
- `paper/main.docx` 未重生成：本机无 pandoc（只有 tectonic），无法从改后的
  LaTeX/Markdown 复原 docx。`main.pdf` 与两份 markdown 已更新。
- OpenMC 堆芯参考的 statepoint / 统计误差原始输出未入仓（见
  `unresolved_items.md`）。
- 堆芯矩阵的墙钟时间不是吞吐测量：同机多点并发，`*` 标注复用分解因子的格子。

## 9. 独立成文的 companion 论文（本轮追加）

- 新增 `paper/sector_method.tex`（14 页，已随仓提交 `paper/sector_method.pdf`）：
  有限扇区扩散闭包的独立文章，结构与 OpenPSN 论文同型（引言—方法—实现与复现—
  测试算例—结果—讨论与范围—结论—Reproducibility—附录），正文以
  `\cite{openpsn2026}` 引用 OpenPSN 框架论文，可脱离 `main.tex` 单独编译：
  `tectonic paper/sector_method.tex` 或 `make -C paper companion`。
- 三张结果表不再手抄：新增 `scripts/study4_assembly_report.py`，从
  `paper/data/study4_assembly_points.csv` 重建 `paper/tables/study4-assembly.tex`
  （六组件最佳点、逐点误差、棒功率 RMS/max、逐点耗时求和，并断言 planned ==
  complete、最佳点与最佳 pcm 自洽、时间列等于逐点之和）。`paper/studyIV.tex` 与
  companion 论文 `\input` 同一文件；重生成时发现手抄表的一处末位舍入
  （矩形 MOX 求和 2957.8 s → 2957.7 s），以生成值为准。棋盘与堆芯表沿用
  `study4_checkerboard_report.py` / `study4_core_report.py`。
- companion 论文里新增的测试算例总表（问题 / 输入 / 参考 / 约束什么）逐项对齐
  仓库参考值；与 `main.tex` 一样，表内数值全部来自生成物。
- 数值口径校正：实测网格点数按存档记录写成 540（棋盘 130 + 组件 384 + 堆芯
  已收敛 26），组件+堆芯的"其余点数"相应为 410；`544`/`414` 是把未收敛的
  4 格与 10 个旧闭包对照点混计进去的结果。
- 两处措辞收紧：（1）正方形 UO₂ 上 $M{=}2\to4$ 的台阶不再写成"没有台阶"，改为
  已归档的 $S{=}1,A{=}1$ 双线性 $-10.84$ pcm / 双二次 $-0.06$ pcm；（2）"扇区闭包在
  同 $M$ 下更贵"改为分场景：四分之一芯 $S{=}1,M{=}24$ 冷启动 340.68 s 对 128.08 s
  （约 2.7 倍），组件网格上双线性扇区点反而快于同 $M$ 原版、双二次更慢，且两条路径
  在该表里用的是不同分解核。
- 删除一处无源码可追的常数（真空平板角扩散平均通量 `0.718691069833`），改为直接
  指向 `tests/test_sector_diffusion.py::test_vacuum_slab_matches_analytic_angular_diffusion`
  （32 格、1.6 mfp、$M{=}4$、$A{=}1$，六位一致）。同时把节点响应等价性的两条断言
  改写为测试实际做的事（三材料矩形在四个空间/角向子空间、其中两个在反射不变子空间，
  对 splu 直解 $2\times10^{-11}$；强棋盘消元与直解同本征对 $2\times10^{-9}$）。
- `paper/Makefile` 增加 `companion` 目标并入 `all`，`clean` 一并清理；`.gitignore`
  增加 `paper/sector_method.{aux,log,out}`。`README.md` §3 与目录树、`studyIV.tex`
  开头各加一处指向 companion 论文的说明。
- 本机校验：`tests/test_sector_diffusion.py` 22 项 11.8 s 全过；
  `examples/psn_strict_checks` 在本机只能收集 8 项（`test_rectangle.py` 因未安装
  `sympy` 无法导入，已收集的 7 项通过），本机无网络、无法按
  `requirements.txt` 补装依赖，故 14 项全量需在有依赖的环境里跑。
- companion 论文的表与 `main.tex` 共用生成物，因此本轮把 markdown 镜像驱动也入仓：
  `scripts/finalize_markdown.py` + `paper/studyIV.md`（原先是两个带绝对路径的本地脚本，
  读者无法重建 `main.md` / `revised_main.md`）。入仓后 `python -m scripts.finalize_markdown
  --force` 与本地版输出逐字节相同（两份 markdown 各改 2 行）。

## 10. 明确每张表用的是哪个扇区闭合（应审阅提问追加）

- 两篇论文此前只说"the same closure"，读者无法判断哪张表是 `sector_diffusion`、哪张是
  `sector_average`。现按归档记录逐处写明（`points.csv` / `study4-checkerboard.csv` 的
  `closure` 字段）：棋盘 130 点与六组件 384 点是 `sector_average`（$A{=}0,1,2$，双二次或
  双线性角系数子空间，组件 $I{=}8$ 有限极角带）；四分之一芯 $5	imes6$ 是 `sector_diffusion`
  （TY3，$A{=}1$）。
- 同时更正一处容易被误读的口径：`study4-checkerboard.csv` 里那 10 行 `strict_phase_space`
  既不是新闭合也不是发表版复现，而是"把精确扇区张量插回原四模体匹配 + 面平均"的反例
  （$S{=}2$ 的 $-1363.03$ / $+76.41\,\mathrm{pcm}$，出处
  `examples/psn_repro/logs/strong_strict_sector_valley.log`）；发表版谷线值出自
  `docs/fig3ref.json`（$-790.2$ / $+807.9$）。
- `paper/studyIV.tex` 开头与两段测量描述、`paper/sector_method.tex` §5 开头各写明这一点；
  companion 论文的测试算例表 `tab:tests` 增加 "Ran with" 列，逐行标 `sector_average` /
  `sector_diffusion` / "algebra only"（14 项符号与有理检查只约束两闭合共用的算子与精确关系）。
  镜像正文 `paper/studyIV.md` 同步，`main.md` / `revised_main.md` 由
  `python -m scripts.finalize_markdown --force` 重建。
- 重编译结果：`main.pdf` 39 页、仍只有 189 / 543 / 1163 三处既有 overfull；
  `sector_method.pdf` 14 页、0 overfull、无未定义引用。

## 11. 记录口径改为逐方向闭合（本轮追加）

§10 只回答"哪张表用了哪个闭合"。本轮把它换成一个正向口径：**逐方向闭合
`sector_diffusion` 是方法记录，`sector_average` 是等节点数对照**，两篇论文、
`README.md`、`docs/psn4-sector-diffusion.md` 与本节都按这个顺序叙述；对照侧的
数值不删，改为标注归属。两者共用同一空间弱式、同一镜面迹共享、同一门槛
（外层 `1e-9`、复扫源残差 `5e-9`、扇区线性求解 `5e-12`）和同一每扇区自由度
`K = A + 1`（`psn2d/__main__.py:107`），所以同格逐点差只来自闭合本身。

### 11.1 新增的生成物（不手抄）

- `scripts/study4_closure_report.py`：读两个逐点 CSV 的 `closure` 列 →
  `paper/tables/study4-checker-sector.tex`（记录口径棋盘矩阵）、
  `study4-checker-average.tex`（对照矩阵）、`study4-closure-ratio.tex`
  （逐点 $|\Delta k|$ 比值，配对数/中位/极值/变号写在 caption 里，随数据自动同步）
  与 `paper/data/study4_closure_pairs.csv`。发表网格的每个已发表点都必须有配对点，
  补测未完时 `--allow-missing` 会打印留空格数。
- `scripts/study4_closure_cost.py`：同机背靠背跑配对的两闭合 →
  `study4-closure-cost.tex` 与 `paper/data/study4_closure_cost.csv`（同 $M$、$A$、
  $I$、线程预算、分解核与内存上限，逐条记录开始时负载、空闲内存与 CPU 亲和位掩码）。
- `scripts/study4_uniform_grid.py`（`plan` / `run` / `merge`）：按已发表网格逐点补测
  `sector_diffusion`，收敛点入 `paper/data/uniform/points.jsonl`（逐点出处随码发布，
  `paper/data/` 两个 CSV 的 `source` 字段指向它），`merge` 按 `closure` + 网格键并入，
  已发表行不改写，脚本幂等。
- `scripts/study4_assembly_report.py` 与 `scripts/study4_checkerboard_report.py`
  改成闭合感知：前者加 `--closure`，后者重写自己那 9 份日志的行、其余行原样保留，
  因此重生成一种闭合不会抹掉另一种。`study4_assembly_points.csv` 增加 `closure` 列。

### 11.2 记录口径下的数值变化（旧 → 新）

- 棋盘：原写"扇区闭合 130 点全部同号（负），任一轴加密单调趋零"——那是
  `sector_average` 的 130 点，仍成立但降级为对照。记录口径改为按发表网格逐点重测的
  **187 点**（`A=0,1,2`，`I=30`，全部通过独立复扫，最大源残差 `3.93e-9`、最大本征值
  残差 `9.68e-10`），符号结构写成"187 点中 172 点为负；15 个正偏差全部在 `A=1`
  （每扇区两个 Gauss 方向）的 `M=4` 一列，强吸收 $S{=}3,4,5,6,7,8,9,12$、弱吸收
  $S{=}4,5,6,7,8,9,12$，最大 $+123.97$ / $+17.55\,\mathrm{pcm}$"，这 15 格同格的角区
  平均值仍为负，是两种闭合唯一的符号分歧。
- 单调性："两轴均单调趋零" → 固定 `M>4` 加密 S 时 187 点无一例 $|\Delta k|$ 上升；
  `M` 方向饱和（强吸收 `A=1`、`S=12` 在 `M=8,12,24` 为 −5.73 / −10.79 / −12.25 pcm）。
- 棋盘最佳点：`sector_average` `A=1` 的 −34.13 / −7.85 pcm（`S=12,M=24`）→
  记录口径 `A=1` 的 −5.73 / −2.68 pcm（`S=12,M=8`）；`A=0` −8.38 / −2.90
  （`S=12,M=12`）、`A=2` −12.08（`S=12,M=24`）/ −3.67（`S=12,M=8`）。
  同格提高 `A` 不再有系统收益：(4,8)、(8,12)、(12,24) 三行里 `A=0` 在两个吸收体上
  都略优，这一点此前没有写过。
- 闭合对比：`sector_diffusion`/`sector_average` 的 $|\Delta k|$ 比值，本轮写表时
  189 个配对点中 184 点更小、中位 0.488、最好 0.016、最差 8.394；分案例是强吸收
  85/85（中位 0.445）、弱吸收 78/78（0.472）、方形 MOX 12/12（0.570）、方形 UO$_2$
  8/12（0.937），矩形组件各 1 点（0.710 与 8.394）不作趋势。补测跑完后这些计数会
  变，caption 由脚本同步，正文只引用比值表。
- 成本：删掉"扇区闭合同 $M$ 更贵"的未归档探针值（1.1–1.5×），换成入仓的配对计时
  0.52–1.45、中位 0.98（10 组，8 线程、10 GiB 上限、同一 `sector-cholmod`）。
  `2.7×`（`S=1,M=24` 冷启动 340.68 s 对 128.08 s）只作为**对 legacy PSN 路径**的
  代价保留；`docs` 里 `60.21/82.32` 对扇区列那张旧计时表也补写明它比的是 legacy
  路径，且两侧分解核不同。
- 一处口径错误更正：双线性方案的顺序计时写的是 12.68 s，日志
  `examples/psn_repro/logs/timing_bilinear_full_strong.log` 的 `time_s` 是 **12.81 s**，
  与 `main.tex` 一致（legacy 同侧 37.44 s，出自 `timing_final_legacy_strong.log`）。
- `docs/psn4-sector-diffusion.md` 的"方形棋盘格"小节原以默认 `A=3` 的 40 格联合加密
  （`S=12,M=24` 处 −11.98 / −3.68 pcm）为表值，**仓内无对应脚本或日志**，按
  `unresolved_items.md` §11 的同类条目降级为试算记录；表值改由已归档的 `A=0,1,2`
  187 点给出，复现命令按方位阶逐条写出。
- `README.md` §5 的 Study IV 行拆成"记录口径 187 点"与"对照 130 点"两行，各自的
  结论只写自己那份数据支持的；§3 写明 `slope_space` / `polar_degree` /
  `azimuth_grid` / `node_response` / `reflection_reduction` 只在
  `sector_average` 下被读入（`psn2d/model.py:197`），逐方向闭合没有 `slope_space`
  变体；§6 补上三个新脚本与 `--closure` 的用法与前置条件。

### 11.3 仍不是结论的东西

- 变号例外只在 `M=4`、`A=1` 一列，但没有把它证成两点求积的必然结果，只写实测分布
  与对照侧的同格行为。
- 矩形组件的配对样本各 1 点，方形 UO$_2$ 有 4/12 点变大，因此"逐方向闭合更准"
  在论文里限定为"在已配对的实测点上"。
- 配对计时是负载记录（主机同时承载其余扫表），不是吞吐测量；`main.docx` 仍未重生成。

## 12. 补测跑满后把全部正文重算一遍（本轮追加）

- 数据合并：`merge` 之后 `paper/data/uniform/points.jsonl` 的收敛点写进
  `study4-checkerboard.csv`（`sector_diffusion` 314 行）与
  `study4_assembly_points.csv`（`sector_diffusion` 440 行 = 棋盘 314 + 组件 126），
  已发表的 `sector_average` 行逐字未动（棋盘 130、组件与宽棋盘 384），
  `study4_closure_pairs.csv` 从 189 对涨到 416 对。
- 表：`python -m scripts.study4_assembly_summary --closure sector_diffusion` 先派生
  汇总行（补 `closure` 列），再
  `python -m scripts.study4_assembly_report --closure sector_diffusion --allow-partial
  --table paper/tables/study4-assembly-sector.tex --label tab:study4-assembly-sector`
  才出得了表；新增的 `--label` 专为这第二张表，两篇论文各自 `\input`、标签不撞。
  已发表的 `tables/study4-assembly.tex` 重跑后逐字节不变（`git diff` 为空）。
- 图的入口：新增 `scripts/study4_sm_figures.py`，只读 `paper/data/` 两个逐点 CSV、
  `paper/data/uniform/plan.json` 与 `examples/c5g7/study4_sector_core/sm_error_pcm.csv`，
  一次画五组图（PNG + 同名 PDF）落 `paper/figures/`：棋盘六格
  `study4_checker_sm`、`A=1` 宽矩阵两格 `study4_sm6_sm`、组件 2×2 误差图
  `study4_assembly_grid2x2`（上排方形、下排矩形，左 UO₂、右 MOX）、同布局耗时图
  `study4_assembly_wall`、13 格总览 `study4_sm_overview`。三条口径固定在代码里：
  每格独立色标（跨格只比印出的 pcm 数）、绿环标该格 $|\Delta k|$ 最小点（耗时图也环
  同一个点，两图对着读就是"最佳点花多少"）、灰斜线 = 排程里就没有的格、`n/a` =
  排了但主机算不出的格。宽矩阵那两格把 `checker` 与 `sm6_checker` 两个族并成一张
  120 格矩阵 —— 那是同一道题的两份扫表清单，不是两道题，分开画会把 20 个已测点画成
  缺口。
- 正文按合并后的实数重算（旧 → 新）：棋盘 187 → 314 点；同号 172 → 279；正偏差
  15 → 35 个，仍全在 `A=1`、`M=4` 一列（强吸收 $S{=}3\ldots20$、最大 +127.38 pcm，
  弱吸收 $S{=}4\ldots20$、最大 +18.24 pcm）；沿 $S$ 的加密步 596 → 280 步（口径改为
  314 点里同 $M$ 的相邻点对），趋零 540 → 246、反例 56 → 34，措辞从"无一例上升"
  改成"34 例全在该列内"；方位饱和判据从 `M≥8` 收到 `M≥12`（55 条列、最宽 6.79 pcm），
  `M=8` 那条最宽首步单写（`A=0`、`S=12` 强吸收 −22.40 pcm，下一格 −8.38 pcm）；
  配对 189 → 416、更小 184 → 398、中位 0.488 → 0.437、最好 0.016 → 0.011、最差仍是
  8.394；18 个反向配对全部是粗角向格（方形 UO₂ 的 12 个 $M{\le}8$ 格、矩形 UO₂ 的
  6 个 $M{=}2$ 格）。改动落在 `main.tex` 引言与结论、`studyIV.tex`、
  `sector_method.tex`（摘要 757 → 984 点、贡献 (iv)(v)、结果开头、棋盘段、组件段、
  范围与结论）、`README.md`、`docs/psn4-sector-diffusion.md`、
  `paper/unresolved_items.md`、`paper/retained_data_map.md`。
- 新写进去的东西：`tab:study4-assembly-sector`（新闭合六矩阵）与
  `fig:study4-checker` / `fig:study4-checker-wide` / `fig:study4-assembly-grid` /
  `fig:study4-assembly-wall` / `fig:study4-overview` 进 `studyIV.tex`，
  `tab:study4-assembly-sector` 与 `fig:assembly-grid` 进 `sector_method.tex`；
  两个矩形案例的"新闭合最佳点较差"写明是覆盖受限（对照侧最佳格
  $S{=}6,M{=}24$ 正在这 9 格里），共享的 27 格全部偏向新闭合。
- markdown 镜像：`paper/studyIV.md` 三段正文 + 新六组件表同步（顺手把该表最后一格
  的 2957.8 s 改回生成值 2957.7 s）；`scripts/finalize_markdown.py` 里三处字面量
  （引言 EDITS、结论 EDITS、棋盘表题）同步；`main.md` 已落盘的旧数字单独改写后再
  `--force` 重建 —— EDITS 的 marker 命中就跳过，只改脚本是刷不掉镜像里的旧句子的。
- 重编译：`main.pdf` 41 页（多了 5 图 2 表），overfull 仍是原来那四处（`main.tex`
  194 / 548 / 1168 与 `studyIV.tex` 209，行号随插入平移），无未定义引用；
  `sector_method.pdf` 17 页、0 overfull、无未定义引用。
