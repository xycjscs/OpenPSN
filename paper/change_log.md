# change_log.md — 本轮改稿记录（不新增计算）

输入：`OpenPSN_no_new_runs_revision_package`（快照 `source_main_2.md`，SHA256
`ed00e0fa…9ee8`）。实际排版源为 LaTeX（`paper/main.tex` + `paper/studyIII.tex`
+ `paper/appendix.tex`），先确认其内容与快照对应后修改源文件。备份文件
`*.bak_rev` 已于本轮末尾清理（改动全部落盘并编译验证）。

**总原则：不运行任何输运/本征值求解器；不改任何数值；只重组结构、改定位、
改措辞、改图注、补文献引用。Core S=1 六个点中缺稿的 M24（+11.0 pcm）来自
已有结果文件 `tools_ho/data/r17_core4/CORE4_M24_S1.npz` 与
`tools_ho/core4_sweep.log` 的导入，未重跑。S2 不再补充，也不列为待办。**

---

## 1. 标题 / 摘要 / 首页（main.tex）

- 标题：`OpenPSN: Rectangular Phase-Space Nodal Transport with Higher-Order
  Angular Moments` → **`OpenPSN: Implementation, Verification, and Assessment
  of Nodal Transport Methods`**（按手册推荐标题）。
- 摘要整体重写为"框架含两类离散"的定位：PSN 实现 + 独立局部角矩形式；
  明确 A3 属 multi-P_N-type 家族、与原 PSN 构造的更高阶推广相区别；声明
  两类 campaign 用不同 pin tiling，比较的是整套配置而非角阶次本身；
  core 研究限定于 S=1 角分辨率扫描；结尾明确不承诺普遍成本优势、联合单调
  收敛或极限输运精度。
- 首页代码行、署名、日期不变。

## 2. 引言（§1）

- 第二段（PSN 介绍）按替换稿改写：保留 Chao et al. 的 PSN 定义与
  "SP3-based development 指更高 streaming 算子幂次"的引文语境，删除把
  A3 说成 PSN 自然推广的暗示。
- 新增一段本地角展开文献定位：Ghazaie 2019（Multi-PN，角子域独立展开 +
  空间有限元）与 Falabino 2022（短特性法内接口角通量分段多项式），
  并引 Chao 原文区分 PSN 与"独立选择的局部角展开"——两篇 bib 条目
  （`ghazaie2019mpn`、`falabino2022mpn`）已加入 `thebibliography`，
  内容与修订包 `added_references.bib` 逐字一致。
- 贡献段重写为 fourfold：实现 → 验证（实现检查 + 复现）→ 文档化独立角矩
  分支 → 用与材料/几何模型相称的参考做评估；明确不声称新的局部角展开
  原理，也不做与所引方法的普遍精度/效率排名。
- "The remainder is organised as follows" 段落按新六章结构更新。

## 3. 方法章重组（§2，原 "The PSN method and the OpenPSN implementation"）

- 章名 → **Numerical formulations and implementation**。
- §2.1 Legacy PSN：身份固定为"已发表 PSN 闭包的原样实现"，原法 A=0 不再
  称 "original closure"（改为 legacy PSN / the $A{=}0$ member of the
  directional family 等无混淆措辞）。
- §2.2 实现与高精度形式：实现清单第 (iv) 项由 "High-order directional
  moment closure" 改为 **Local angular-moment closure (the A=3 variant)**，
  定位为"同一节块框架内的另一种角离散"，不是对原法的精度升级；
  9×9 等实现细节归位到原分支语境；GMRES 表述按 scipy 1.17.1 源码核实
  修正：`callback_type='pr_norm'` 下 `maxiter` 计 **restart cycles**
  （源码 `for iteration in range(maxiter): for col in range(restart)`，
  restart 默认 20），故写 "restart length of 20 inner iterations, at
  most 800 restart cycles"，并标注这是接口参数口径而非 Krylov 步数；
  生产 sweep 驱动 `tools_ho/hpc_point.py` 实参 `rtol=1e-7, maxiter=800`，
  与外门 1e-7 相同；CLI 独立运行默认 1e-11——均与代码一致。源监控判据
  `‖s^{n+1}−s^n‖₂/max(‖s^n‖₂,1)` 与 `psn2d/highsweep.py` 逐字核对保留。
- **§2.3 The segmentwise angular-moment variant**：A3 理论推导
  （局部偶宇称角矩展开、矩阵值二阶角矩耦合、分离节点响应、A=0 退化关系、
  每段 A+1 个局部偶角系数）整体从原 studyIII 前移至此，公式标签与图表
  ID 全部保留；A=0 关系段补限定：本构张量一致 ≠ 节点响应一致，方向 A=0
  是独立离散，不与 legacy PSN 合并为同一阶次序列。
- 交叉引用（Section~2 / 2.2 / 4 等硬编码编号）按新章节号逐处核对更新。

## 4. 验证章（§3，新增独立章）

- **§3.1 Local algebra and the A=0 regression**：240 例局部构造代数检验
  （最大归一差）归入验证章。
- **§3.2 Reproduction of the published PSN results**：棋盘 weak/strong
  （各 60 点）+ BWR 无 Gd / 含 Gd（各 60 点），共 248 点；Table tab:repro
  与附录 A 数值全部原样保留，仅文字定位改为"实现验证"。

## 5. 评估章（§4，原 "Numerical results" 系列）

- 章名 → **Numerical assessment**；§4.1 Configuration 明确三组参考
  （pin-homogenised OpenMC kref 1.3336285 / 1.1853993 / 1.1868230；
  纯 XS 1.3347260 / 1.1851053 / 1.1872176；各 ≈±3.3–3.6 pcm）与
  极角规则（Study I/II 用 TY3 极角约定，来自 yaml 实证）。
- §4.2 Study I、§4.3 Study II：数值、表格（tab:rectsm 全表）、图全部
  原样保留；文字按"指定配置的评估"口径微调，删除对"收敛"的过度解读。
- **§4.4 Study III**（`studyIII.tex`）：
  - 小节标题 → "Study III — the segmentwise angular-moment variant at
    A=3"；理论段已前移，此处只留配置与结果。
  - **Core S=1 收口**：正文 S1 行六点全齐（−135.6 / −34.2 / −1.7 / +6.2 /
    +9.1 / +11.0 pcm，M2→M24），Figure 9 两排补点，图注与正文
    "in progress / will be calculated" 全部删除；S2 M2（−150.1）在图中
    仅作参考点，图注明确"报告的核心研究是已完成的 S=1 角分辨率扫描，
    不据此点主张堆芯空间收敛"；S2 其余点不补、不列待办。
  - 修正笔记点名的事实错误："the assemblies first enter it at M=12–24"
    与日志矛盾（UO₂ S1 M8 = −2.8 pcm 已在 ±3.6 带内，日志
    `r17_asm_final.log` 实证），改为逐题点值表述（UO₂ 组件 S1 M8 −2.8、
    MOX 组件 S1 M12 −0.9 均已在带内）。
  - 棋盘 A3 复现：I30 每吸收体 120 点（S=1–20×6M）共 240 点、1e-7 收敛；
    I60 单点（S20 M24）移动 +2.9 pcm 的表述保留并限定范围。
  - 组件 S=2 行全 6 点（UO₂ −128.7…−0.5；MOX −303.3…−9.8）在日志
    `r17_asm_final.log` 与 npz 中实证齐全，正文引用值逐点核对一致。
  - 裂变场图（fig:a3fission）：UO₂ S2M24 RMS 0.055%（max 0.19%）、
    MOX S2M24 RMS 0.075%（max 0.22%）、core S1M8 RMS 0.286%
    （max 0.64%，1056 根棒）——数值原样保留，图注限定为"选定配置的
    归一化场比较，非完整空间场收敛调查"。
  - "Where the method now stands" 段删除，内容并入 §5 讨论。
- §4.5 文献景观对比表（tab:lit）：OpenPSN 行标签
  `PSN nodal (rect. node, A=3, S=1, M=8)` → **`Local angular-moment nodal
  (A=3; square-fuel model, S=1, M=8)`**（列宽改 p{} 防超宽）；表下脚注
  新增：对同几何 OpenMC（1.1872176）差 −1.7 pcm，对圆棒文献参考 1.18646
  的 +74.1 pcm 含几何模型差，不是同几何离散误差比较；文献值仅作基准
  背景，非同网格/同算法/同资源的排名。k 值 1.1872011 未动。

## 6. 讨论章（§5，新增）

- 新增 **Discussion and scope of comparisons**：PSN 与 A3 结果应解释为
  共同软件框架内指定离散的结果；矩形 campaign 同时改角形式与 pin
  tiling，不能归因角阶次单独作用；本征值精度与空间分布精度互补、谷区
  点不因"落在谷区"而否定；现有扫描以本征值为主，场比较只在选定配置，
  不构成完整 S×M 场误差调查，也不主张联合单调收敛；迭代收敛 ≠
  离散误差量化，与 OpenMC 标准差内的一致是统计相容而非确定性误差上界；
  角矩分支核心研究限于 S=1 角扫描，无堆芯空间收敛主张；等成本比较与
  角阶次单独归因超出本文证据。

## 7. 结论章（§6）

- 整体重写为替换稿口径：两类形式的框架文档 + 验证证据 + 指定配置评估；
  A3 属 multi-P_N-type，A=3 不等于原法更高 streaming 阶次；列出具体
  点值（UO₂ S2M24 −0.5、MOX S1M12 −0.9、场 RMS 0.055/0.075%、core
  S1M8 Δk=−1.7 / RMS 0.286%）；结尾限定"支持指定配置的精度，不是极限
  输运解的证明"。

## 8. 图

- **Figure 9（c5g7_rect4_a3_m_conv.png）重生成**（仅标签/标题，数据未动）：
  脚本 `tools_ho/plot_a3_mconv.py` 中 "S=2 row: in progress" 两处注释 →
  "completed S=1 angular sweep (fixed spatial resolution)"；suptitle
  "directional closure" → "local angular-moment closure"。数据仍读
  `tools_ho/data/r17_core4/*.npz`，S=1 六点全在。
- 全部图注重写，每张注明：方法身份（legacy PSN / A=3 local
  angular-moment）、网格（tiling、节点数）、极角规则（TY3 / generic
  I=30 / I=60）、参考模型（同几何纯 XS OpenMC / 复现参考 / 文献参考）、
  误差量（Δk pcm / 归一化裂变 Σfφ RMS 或 max）、取值 S/M。
- 其余图片文件未改。

## 9. 参考文献

- 新增 2 条：`ghazaie2019mpn`（Prog. Nucl. Energy 110 (2019) 64–74，
  DOI 10.1016/j.pnucene.2018.09.001）、`falabino2022mpn`（Ann. Nucl.
  Energy 177 (2022) 109301，DOI 10.1016/j.anucene.2022.109301），
  与修订包 added_references.bib 逐字一致。
- 既有 `psn2027` 条目（Ann. Nucl. Energy 240 (2027) 112707，DOI
  10.1016/j.anucene.2026.112707）按包内注释保留不改。
- 无删除、无重编号之外的变动。

## 10. 编译与交付

- tectonic 重编译：30 页，0 error；3 处 overfull（main L177 hbox 2.2pt、
  L531 hbox 10.0pt、L1150 vbox 37.7pt）经对照旧版构建确认**全部为旧版
  既有**（行号平移），本轮零新增 overfull。
- 三格式交付：`main.pdf`（30 页）、`main.md` / `revised_main.md`
  （pandoc markdown 目标，引用为 [@key] 形式，与旧版 main.md 一致）、
  `main.docx`。
- 交付文档：`change_log.md`（本文件）、`retained_data_map.md`、
  `unresolved_items.md`。

## 数值未动声明

本轮所有 keff、pcm、RMS、max 误差、kref、S/M/I/A、节点数、样本数、
迭代/容差参数均与改稿前逐点核对一致（A3 组件/core 行对照
`tools_ho/r17_asm_final.log`、`tools_ho/core4_sweep.log` 及
`tools_ho/data/r17_core4/*.npz`；Study I/II 与复现表对照既有表格与
`docs/fig3ref.json`、`docs/bwr_ref.json`）。唯一新增正文点值为 core
S1 M24 = +11.0 pcm，来源为已完成结果文件导入（用户已确认该点完成）。
