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

- 四分之一堆芯矩阵当前 26/30 格：`S=4,M=24` 与 `S=5,M=8,12,24` 在写表时仍在运行，
  两张表里留空并在 caption 里标注；其余 26 格已入仓。补齐后重跑
  `scripts/study4_core_report.py`（去掉 `--allow-missing`）即填上并去掉标注；
  若新的最佳格改变，需同时重跑 `validation/sector-diffusion-sm/fission_map.py`
  更新逐棒裂变对照图。

- `paper/main.docx` 未重生成：本机无 pandoc（只有 tectonic），无法从改后的
  LaTeX/Markdown 复原 docx。`main.pdf` 与两份 markdown 已更新。
- OpenMC 堆芯参考的 statepoint / 统计误差原始输出未入仓（见
  `unresolved_items.md`）。
- 堆芯矩阵的墙钟时间不是吞吐测量：同机多点并发，`*` 标注复用分解因子的格子。
