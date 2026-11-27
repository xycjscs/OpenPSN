# PSN-4 扇区扩散验证

## 记录口径

本文记录两种有限扇区闭合，二者在相同的空间弱式、相同的 `I`、`M` 与相同的
每扇区角向自由度数（`K = A + 1`，`psn2d/__main__.py:107`）下逐点对照，
外层容差 `1e-9`、独立复扫源残差 `5e-9`、扇区线性求解 `5e-12` 门槛完全共用。
**逐方向闭合 `sector_diffusion` 是本文与论文的记录口径**，
`sector_average`（角区平均）保留为等节点数对照。

| | `sector_diffusion`（记录口径） | `sector_average`（对照） |
|---|---|---|
| 每扇区角向自由度 | `A+1` 个 Gauss 方向，逐方向解偶宇称二阶方程 | 角平均场 + `A` 个扇区内方位形函数，解一个耦合扩散块 |
| 扩散张量 | 每个保留方向直接给 `ΩΩᵀ/Σₜ` | 先对"角区 × 形函数乘积"平均成 `⟨b_a b_b ΩΩᵀ⟩` |
| 专属选项 | — | `slope_space`、`polar_degree`、`azimuth_grid`、`node_response`、`reflection_reduction` |
| 默认 `A` | 3（`psn2d/model.py:195`） | 0 |

后五个选项只在 `closure: sector_average` 下被读入（`psn2d/model.py:197`），
逐方向闭合的方位系数是 Gauss 方向而非多项式，因此没有 `slope_space` 变体。
下文标"对照"的小节属于 `sector_average`，其数值仍是该闭合自己的实测。

配对结果与代价全部由脚本生成，不手抄：

- `python -m scripts.study4_closure_report` → `paper/tables/study4-checker-sector.tex`
  （记录口径的棋盘矩阵）、`study4-checker-average.tex`（对照矩阵）、
  `study4-closure-ratio.tex`（逐点比值），逐点配对数据在
  `paper/data/study4_closure_pairs.csv`；
- `python -m scripts.study4_closure_cost` → `study4-closure-cost.tex` 与
  `paper/data/study4_closure_cost.csv`（同机背靠背配对计时）。

已发表的棋盘与组件矩阵出自 `sector_average` 扫表。
`python -m scripts.study4_uniform_grid plan|run|merge` 按同一网格逐点补测
`sector_diffusion`（`paper/data/uniform/points.jsonl` 追加收敛点，`merge`
按 `closure` 一并并入 `paper/data/` 的两个逐点 CSV，不改写已发表行）。
458 个计划点均已通过复扫。棋盘发表网格 114/114，`A=1` 宽矩阵新增 200/200，
方形与矩形组件各 72/72。矩形组件的最后 18 格按照
`paper/data/uniform/plan_packed.json` 用 `chol-packed-rs` 补算，线程上限为 60。
旧直解因子的内存投影保留在历史文件 `paper/data/uniform/plan_unreachable.json`。
新后端用压缩 Cholesky 预条件器求解同一 FP64 扇区算子。`OPENPSN_PACKED_RTOL`
和 `OPENPSN_PACKED_MAXITER` 默认 `5e-12`、`80`。补测中出现 FP64 线性残差下限后，
后续运行使用 `1e-11`、`160`，外层本征值与复扫源的阈值仍为 `1e-9`、`5e-9`。
四组组件逐棒图均取已计算的最细网格 `S=6, M=24`，与四分之一堆芯的
`S=5, M=24` 逐棒图共用 −0.35% 至 +0.35% 的相对误差色标。
[组件示例和重绘命令](../examples/c5g7/study4_sector_assembly/README.md)

在 `A=1` 的 UO₂ 组件矩阵中，`sector_average` 从 `M=2` 加密到 `M=4` 时，
方形、矩形的 12 行 `k_eff` 均下降。`sector_diffusion` 的 12 行均上升。以 `S=2`
为例，矩形偏差从 −204.06 收窄到 −122.77 pcm，方形从 +1.46 增至 +6.25 pcm。

共有 434 个等自由度配对点，416 格的 `|Δk|` 较小，中位比 0.437，39 格变号。
矩形 UO₂ 在 `S=6, M=24` 为 −0.19 pcm，30/36 格的绝对偏差较小。
矩形 MOX 在 `S=6, M=24` 为 −92.55 pcm，36/36 格的绝对偏差较小。
逐格结果、变号位置和比值由 `scripts.study4_closure_report` 生成。
代价上逐方向闭合
不是一律更贵：10 组配对（8 线程、10 GiB 上限、同一 `sector-cholmod` 分解核）
比值 0.52–1.45，中位 0.98。`## 扇区扩散性能` 里那张端到端计时表的 1.95 倍是
**`sector_average` 对 legacy PSN 路径**的比较（两侧分解核还不同），不是两种扇区
闭合之间的成本比；四分之一堆芯的 2.7 倍同属这类 legacy 路径对照。

## 对照：有限角区平均闭合（`sector_average`）

`sector_average` 的默认 `A=0` 在每个有限极角带与方位扇区的交集上保留一个偶宇称场，
直接积分二阶张量 `⟨Ω Ωᵀ⟩` 和真空边界系数 `⟨|Ω·n|⟩`，
角区之间通过标量通量源和镜面反射边界耦合。极角带沿 θ 等宽，
积分权重按 `d(cos θ) dφ` 计算。体扩散和材料界面法向流共用完整张量的
双二次弱式。有限宽角区的二维扩散张量有两个正特征值。

### 显式节点响应

`angular.node_response: true` 将有限角区弱式转写为共享面迹系统。节点内部与面流使用同一
张量 `D_ab = ⟨b_a ΩΩᵀ b_b⟩ / Σₜ`，其中 `b_a` 是有限扇区内的角向基函数。
取 `A=0` 时只保留角平均项，取 `A≥1` 时保留角矩之间的交叉项。真空和镜面反射
仍分别使用弱式边项和共享面迹。

将自由度记为节点内部 `I` 和节点边界 `B`，双二次中心基函数在所有边上为零，故
`K_II` 在每个节点内独立，且保留该节点内部不同角矩之间的耦合。消元得

```text
R = K_II⁻¹ K_IB
T_B = K_BB − K_BI R
g_B = f_B − Rᵀ f_I
T_B u_B = g_B
u_I = K_II⁻¹ f_I − R u_B
```

`T_B u_B − g_B` 给出按面迹基测试的边界残差，材料界面的两侧贡献在组装后相消。
源项进入同一个局部逆，求出共享面迹后恢复节点内部通量。双线性角向系数的四个
角点都在节点边界，因此它们的面耦合由共享迹系统保留。这个开关对同一试验空间作
精确代数消元，原弱式与节点响应在舍入误差内给出同一个通量与本征值。改变 `A`
会改变角向试验空间，可以独立测量其精度收益。

```bash
python -m scripts.sector_diffusion_benchmark strong --closure sector_average \
  --polar-order 30 --azimuth-degree 1 --slope-space bilinear --node-response --sm 2,8 6,16 10,24
```

原版节点的体内形函数使用标量扩散，界面电流使用扇区各向异性系数。
`strict_phase_space` 将体内形函数改为同一张量的精确模态，保留原有四模态
及面平均匹配。强吸收棋盘格固定 `I=30, S=2`，原版从 `M=4` 的
−790.22 pcm 到 `M=8` 的 +807.92 pcm，`strict_phase_space` 则分别为
−1363.03 和 +76.41 pcm。这个对照显示，修正体内张量之后，原有的空间
节点闭合仍允许误差随 M 改变符号。

`sector_average` 固定 `I=30`，计算 `S=2,4,6,8,12` × `M=4,8,12,24`
以及 `S=12, M=48,96`。强、弱吸收共 44 个测点的误差均为负，
固定 S 增大 M，或固定 M 增大 S，误差均单调向零靠近。所有测点通过独立
复扫，最大源残差为 `1.81e-9`。

| S | M | 强吸收 Δk (pcm) | 弱吸收 Δk (pcm) |
|--:|--:|-----------------:|-----------------:|
| 2 | 4 | −2138.16 | −393.13 |
| 4 | 8 | −625.76 | −119.01 |
| 8 | 12 | −309.02 | −59.51 |
| 12 | 24 | −107.43 | −21.61 |
| 12 | 48 | −51.66 | −11.16 |
| 12 | 96 | −36.71 | −8.37 |

在具有正去除截面的单群问题中，偶宇称弱式给出 `k_eff` 的广义 Rayleigh 商。
二分空间单元或平分原角区产生嵌套试验空间，在精确积分下，细化所得的
`k_eff` 非减并从下方逼近连续问题的特征值。七群问题的散射耦合一般不对称，
表中观测也包含非嵌套细化，两者均超出这条变分结论。粗角区的横向平均扩散
造成较大的低估，表中的绝对偏差随着角区进一步划分而下降。

在输入文件中设 `angular.model: generic`、`angular.closure: sector_average`，
`A` 使用默认值 0。棋盘格复现命令如下。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
samples=(2,4 2,8 2,12 2,24 4,4 4,8 4,12 4,24 6,4 6,8 6,12 6,24
         8,4 8,8 8,12 8,24 12,4 12,8 12,12 12,24 12,48 12,96)
python -m scripts.sector_diffusion_benchmark strong --sm "${samples[@]}" --closure sector_average --threads 8
python -m scripts.sector_diffusion_benchmark weak --sm "${samples[@]}" --closure sector_average --threads 8
python -m scripts.sector_diffusion_benchmark strong --sm 2,4 2,8 --closure strict_phase_space --threads 8
```

七群 C5G7 矩形四分之一堆芯固定 `I=8, S=1`，使用下文同一个 OpenMC
参考值。三点均通过独立复扫，固定网格上的角向加密减小了低估。

| M | k_eff | Δk (pcm) | 复扫源残差 |
|--:|------:|---------:|-----------:|
| 4 | 1.184351995607 | −286.56 | 9.44e−11 |
| 8 | 1.184846881742 | −237.07 | 4.66e−11 |
| 16 | 1.185759916068 | −145.76 | 4.53e−10 |

当前七群测试固定 S，验证了这三个方位角分辨率上的趋势，距约 1 pcm 的目标仍有差距。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENPSN_PREFACTOR_WORKERS=5
python -m scripts.sector_diffusion_benchmark rect --closure sector_average \
    --opt chol-packed-rs --sm 1,4 1,8 1,16 --threads 60
```

### 扇区内方位形函数

`A=1` 和 `A=2` 在每个有限角区分别加入一次和二次方位形函数。归一化的局部
Legendre 基只定义在各自方位扇区上。每两个基函数 `b_i, b_j` 的扩散和真空
算子分别用 `⟨b_i b_j ΩΩᵀ⟩` 和 `⟨b_i b_j |Ω·n|⟩` 组装。保留完整的跨基
耦合项，使每个扇区求解一个耦合扩散块。极角带系数仍解析积分，方位系数用
12 点 Gauss 积分。各向同性源只加载平均模态，散射和裂变的标量通量也只读取
平均模态。空间离散和镜面反射配对沿用同一个弱式。

在上述单群正去除问题中，`A=0,1,2` 的近似空间逐阶嵌套，和角网格细化一样
可提高变分下界。强、弱棋盘格在 `A=1, I=30` 上重复 40 个测点，所有偏差为
负，分别加密 S 和 M 时均单调趋近零。最大独立复扫源残差为 `2.02e-9`。

| S | M | 强吸收 Δk (pcm) | 弱吸收 Δk (pcm) |
|--:|--:|-----------------:|-----------------:|
| 2 | 4 | −418.44 | −77.84 |
| 4 | 8 | −126.04 | −24.30 |
| 8 | 12 | −56.54 | −11.83 |
| 12 | 24 | −34.13 | −7.85 |

当 `S=12, M=24` 时，增加一次形函数将强、弱吸收的绝对偏差分别从
107.43 和 21.61 pcm 降至 34.13 和 7.85 pcm。对每个强、弱算例，上节
命令加 `--azimuth-degree 1` 即可复算。强吸收在相同网格上继续升至 `A=2`
得 −32.05 pcm。`S=4, M=4` 升阶的收益更大，`A=0,1,2` 分别给出
−2051.74、−271.34、−149.71 pcm。

本节的角向形函数属于对照闭合。同样的 `A`（即同样的每扇区自由度数 `A+1`）改用
逐方向张量后，`S=12, M=24` 上强、弱吸收的偏差分别是 −12.25 与 −3.73 pcm，
即上表同格的 0.36 与 0.48 倍；逐点比值见
`paper/tables/study4-closure-ratio.tex`，棋盘上的完整矩阵见
`## 方形棋盘格`。

无散射真空板另用解析输运解验证。板厚 1.6、总截面和各向同性均匀源均为 1，
离散扫掠必须复现解析的单元平均通量，该判据作为回归固定在
`tests/test_sector_diffusion.py::test_vacuum_slab_matches_analytic_angular_diffusion`
（`A=1`、32 单元、`I=4, M=4`，六位一致）。

此前的会话试算还记录过逐点平均通量的绝对误差阶梯（96 节点、`I=8, M=4`
时 `A=0,1,2` 分别 `9.42e-3` / `1.95e-3` / `1.30e-3`，`A=2` 加密到
`I=16, M=8` 为 `3.60e-4`）。**该阶梯没有归档的脚本或日志**，因此只作为
试算记录保留在这里，不写入论文，也不作为验收判据。

### C5G7 矩形的角向和空间加密

同一 OpenMC 参考值为 `1.187217554586068`。以 60 线程上限计算，空间网格
保留原材料界面。原网格为 153×153，核心区最大步长为 0.32 和 0.2393 cm 的
重划网格分别为 172×172 和 244×244。逐材料面积与原输入的最大差异分别为
`5.44e-10 cm²` 和 `1.83e-10 cm²`。

| 每轴节点数 | I | M | A | k_eff | Δk (pcm) |
|--:|--:|--:|--:|--:|--:|
| 153 | 8 | 4 | 0 | 1.184351995607 | −286.56 |
| 153 | 8 | 4 | 1 | 1.184952640791 | −226.49 |
| 153 | 8 | 4 | 2 | 1.185861788867 | −135.58 |
| 153 | 8 | 8 | 0 | 1.184846881742 | −237.07 |
| 153 | 16 | 8 | 0 | 1.184914823315 | −230.27 |
| 153 | 8 | 8 | 1 | 1.185982099944 | −123.55 |
| 153 | 8 | 16 | 0 | 1.185759916068 | −145.76 |
| 153 | 8 | 16 | 1 | 1.186394856871 | −82.27 |
| 153 | 8 | 32 | 1 | 1.186544069791 | −67.35 |
| 172 | 8 | 8 | 1 | 1.186251986768 | −96.56 |
| 172 | 8 | 16 | 1 | 1.186744381522 | −47.32 |
| 244 | 8 | 16 | 1 | 1.186776801976 | −44.08 |

在原网格和 `M=8` 上，`A=0` 将 `I` 从 8 增至 16 改善 6.79 pcm；固定
`I=8` 增加一次方位形函数改善 113.52 pcm。加入一次形函数后，同一网格将
`M` 从 8 增至 16 改善 41.28 pcm，再增至 32 改善 14.92 pcm。在
`M=8` 和 `M=16` 时重划空间网格
分别改善 26.99 和 34.95 pcm。交叉差为 7.96 pcm，这个范围内把空间和角向
误差独立相加会带来数 pcm 的偏差。固定 `M=16` 再把 172×172 重划到 244×244，
偏差仅减少 3.24 pcm，复扫源残差为 `1.26e-9`。244×244 算例的因子和原算子
模板常驻池为 18.64 GiB。以上结果距约 1 pcm 的目标仍有差距。

原网格上用 `M=4,8,16` 和 `M=8,16,32` 的相邻增量估计局部方位收敛阶，
分别为 1.32 和 1.47。两个窗口按各自阶次估出的原网格角向极限偏差分别为
−54.64 和 −58.90 pcm。这是固定空间和极角网格的诊断，窗口差异体现了
有限分辨率下外推的敏感性。

从 OpenMC 的 HDF5 存档复算得到 `k_combined` 标准误 1.75 pcm。将 20,000 个
活跃批次的 `k_generation` 按 10 至 500 批成组后，均值标准误估计为
3.39–3.91 pcm。核验约 1 pcm 的绝对误差需要统计误差更小的参照。

固定能群和方位扇区时，极角带只改变体扩散项和真空项的非负系数。各算子的
二次型之比被系数比的最小值和最大值夹住，因而可以用系数比筛选共享因子的极角带。
在 244×244 网格上另用一个热群、同方位的相邻极角带检验因子复用。两带的体扩散
系数比为 1.370，真空系数比为 1.173。目标扩散块有 954,530 个未知量。用前一
极角带的因子作为预条件器后，目标块的额外存储量
从 171.4 MB 降到 12.0 MB。以目标块的 FP64 矩阵重算得到相对残差 `4.65e-13`，
与独立因子解的相对差为 `1.26e-13`。单次线性求解耗时约增加到 2.5 倍。
在 153×153 全堆芯验证中，固定 `A=1, I=8, M=8`，每个能群与方位组合的最高
极角带复用前一带的因子。因子与算子常驻池从 3.85 GB 降到 2.89 GB，减少约 25%。
独立复扫源残差为 `6.94e-10`，两种构造所得 `k_eff` 相差 `6.97e-8 pcm`。
这两项因子复用试验使用验证驱动。

矩形加密算例可用同一个基准入口复算。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
rect=(rect --closure sector_average --opt chol-packed-rs --threads 60 --mem-limit-gb 31)
python -m scripts.sector_diffusion_benchmark "${rect[@]}" --azimuth-degree 1 --sm 1,4 1,8 1,16 1,32
python -m scripts.sector_diffusion_benchmark "${rect[@]}" --azimuth-degree 2 --sm 1,4
python -m scripts.sector_diffusion_benchmark "${rect[@]}" --polar-order 16 --sm 1,8
python -m scripts.sector_diffusion_benchmark "${rect[@]}" --azimuth-degree 1 --core-step 0.32 --sm 1,8 1,16
python -m scripts.sector_diffusion_benchmark "${rect[@]}" --azimuth-degree 1 --core-step 0.2393 --sm 1,16
```

## 扇区扩散性能

本节前半段（`polar_degree`、`slope_space`、`azimuth_grid`、
`reflection_reduction` 与双线性/双二次角系数对照）都是**对照闭合
`sector_average`** 的选项与实测：逐方向闭合没有这些开关（见 `## 记录口径`）。
后半端的分解核与端到端计时对两种扇区闭合都适用。

有限扇区方位角可用局部 Legendre 多项式从 A=0 加到 A=4。设
`angular.polar_degree=1` 可在每个极角带加入沿 cos θ 的一次形函数。它与方位形函数
通过体扩散和真空边界积分耦合，角区仍使用同一个偶宇称弱式。`polar_degree=0`
只保留带内极角平均量。

`angular.slope_space` 可取 `quadratic`、`bilinear` 或 `quadratic-linear`。前两种分别
用双二次或双线性基表示全部非常数角系数。混合选项对第一方位角系数用双二次基，
对更高方位角系数和一次极角系数用双线性基。角平均量采用双二次基。

`angular.azimuth_grid=axis` 在坐标轴附近缩窄方位扇区，并按各扇区的实际角宽积分。
在矩形 MOX 的 S=1、M=24、I=8、A=1 和一次极角设置上，均匀网格的偏差为
−189.82 pcm，近轴加密为 −178.29 pcm。两套设置的方位扇区数相同。

对于四边反射、材料和节点宽度均关于两条中心轴对称的几何，可设置
`angular.reflection_reduction=true`。空间反射交换混合导数的两个符号场，求解器在
反射不变子空间内直接组装，并将源矩及标量通量还原到整幅网格。这个选项适用于
全反射组件。混合边界的 C5G7 四分之一堆芯使用其原边界上的相空间矩阵。

在 C5G7 矩形组件 S=1、I=8 的同几何比较中，A=4 加全双二次空间基使 UO₂
在 M=2、4、8、12、16、24 的偏差依次为 −71.17、−33.21、−17.57、−14.53、−13.55、
−12.80 pcm。相同配置下 MOX 的偏差依次为 −298.84、−220.61、−194.99、−190.09、
−188.54、−187.40 pcm。UO₂ 正方形 S=1 在 A=1 下从双线性改为双二次空间基，
M=2 至 4 的本征值变化由 −10.84 pcm 缩至 −0.06 pcm。关闭互余方位折叠独立
复算后，旧闭合的 M=2、4 结果与折叠值在 10⁻⁶ pcm 内一致。

`sector_average` 可将平均场留在双二次空间，把一次或二次方位形函数的
系数放到同一网格的连续双线性空间。配置为 `angular.slope_space: bilinear`。
单元体积分和真空面项都在这一子空间上组装，保留跨形函数的角区张量耦合。
双线性空间嵌入双二次空间，独立矩阵对照验证了组装结果等于 `Pᵀ A P`。
以 `I=30, A=1` 复算 `S=2,4,6,8,12` × `M=4,8,12,24` 的强、弱棋盘格。
40 个双线性测点的误差均为负，固定 S 或 M 加密另一轴时均单调向零接近。
`S=12, M=24` 时，强、弱吸收偏差分别为 −40.24、−8.94 pcm。
另以同样的 8 线程预算和 `--opt off` 顺序计时强吸收 `S=12, M=24`，
原版用时 37.44 秒，双线性方案用时 12.81 秒（两份日志
`examples/psn_repro/logs/timing_final_legacy_strong.log` 与
`timing_bilinear_full_strong.log` 的 `time_s` 字段）。

对于支持 NUMA 内存绑定的 Linux 主机，`OPENPSN_NUMA=1` 将同一个能群和角区
的矩阵分解与扫掠分配到相同的节点。每群先按固定次序累加角区的空间解，再投影
到标量通量矩。分解和空间求解器只影响速度与内存，闭合张量和复扫判据沿用原定义。

扇区闭合可选 FP64 超节点 Cholesky。它复用稀疏矩阵的符号分析和每个数值因子的
求解工作区。Linux 上安装带 METIS 的 SuiteSparse 开发库后，可用以下命令构建。

```bash
cc -O3 -fPIC -shared -I/usr/include/suitesparse psn2d/sector_cholmod.c \
    -lcholmod -o psn2d/libsector_cholmod.so
```

可用 `OPENPSN_SECTOR_CHOLMOD` 指定共享库的其他位置。

同机端到端计时使用 C5G7 矩形 153×153 网格、`I=8`、`S=1`、60 线程上限、
45 GiB 因子池上限及 `OPENPSN_NUMA=1`。`k_eff` 和独立复扫的源残差门槛分别为
`1e-9` 和 `5e-9`。原版使用 `chol-packed-rs`，两个扇区列使用
`sector-cholmod`；两列的扇区闭合都是 `sector_average`（下表复现命令里的
`--closure`）。各点依次运行，时间包含组装、分解、外层求解和复扫。

| M | 原版耗时 / 偏差 | `sector_average` A=1 双线性 | `sector_average` A=1 双二次 |
|--:|:--|:--|:--|
| 8 | 60.21 s / −203.73 pcm | 50.26 s / −200.95 pcm | 59.12 s / −123.55 pcm |
| 16 | 82.32 s / −78.52 pcm | 73.12 s / −114.13 pcm | 160.49 s / −82.27 pcm |

双二次方案在 M=8 与原版耗时相近，M=16 耗时约为原版的 1.95 倍，因子池约
41.64 GiB。双线性方案在两点均更快，M=16 的因子池为 22.56 GiB，偏差较原版多
35.61 pcm。双二次方案的两个本征值都复现了已有扇区扩散结果。
**这张表量的是扇区闭合对 legacy PSN 路径的代价**（两条路径还用了不同分解核），
不能读成两种扇区闭合之间的成本比。

两种闭合等自由度的配对计时另由 `python -m scripts.study4_closure_cost` 生成，
10 组配对、8 线程、10 GiB 上限、同一 `sector-cholmod` 分解核，一组内两个闭合
背靠背运行，结果在 `paper/tables/study4-closure-cost.tex` 与
`paper/data/study4_closure_cost.csv`：比值 0.52–1.45，中位 0.98，
即等角向自由度数下逐方向闭合不是一律更贵。该表的 `pcm` 列同时复现了
棋盘与方形组件发表格的对应格，说明两侧跑的确实是同一测点。主机当时还承载
其余扫表，每组开始时的负载与空闲内存随记录入档，因此这些是负载记录而非吞吐
测量。

计时桥接库用 SuiteSparse 4.4.4 头文件构建，连接 CVXOPT 1.3.3 提供的
CHOLMOD 3.0.6。一次 CHOLMOD 5.3.1 的双二次 M=8 运行在四分钟后终止，
该版本的整程性能未得到验证。复现命令如下。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENPSN_NUMA=1 OPENPSN_PREFACTOR_WORKERS=12
args=(--sm 1,8 1,16 --threads 60 --mem-limit-gb 45)
python -m scripts.sector_diffusion_benchmark rect --closure legacy --opt chol-packed-rs "${args[@]}"
python -m scripts.sector_diffusion_benchmark rect --closure sector_average --opt sector-cholmod \
    --azimuth-degree 1 "${args[@]}"
python -m scripts.sector_diffusion_benchmark rect --closure sector_average --opt sector-cholmod \
    --azimuth-degree 1 --slope-space bilinear "${args[@]}"
```

## 离散

在材料常数的空间节点内，把输运方程写成方向反转下的偶、奇宇称方程。
各向同性散射和裂变源的奇宇称分量为零，因此沿 `Ω` 的奇宇称通量为
`−(Ω·∇u)/Σₜ`。消去它后，偶宇称通量 `u` 满足

```text
−∇·[(Ω Ωᵀ/Σₜ)∇u] + Σₜu = q.
```

空间以双二次 Lagrange 节点离散。体扩散项、材料间法向流和真空边界项
使用同一个弱式。真空边界的系数是 `|Ω·n|`。镜面反射方向在对应边界共享迹值，
从弱式自然得到净流守恒。源和通量在每个材料节点上用九个正交空间矩投影。
方位扇区内用 `A+1` 点 Gauss 积分，极角用 `I/2` 点 Gauss 积分或既有 TY3。
每个小角区的扩散张量由方向直接给出。对称几何上，互余方位角只分解一侧，应用
空间转置恢复角积分。PSN-3 的 Ritz 投影仍由细网扩散扫描组装本征问题，并用新扫描复核。

上一段就是记录口径 `sector_diffusion` 的离散。对照闭合 `sector_average` 共用
同一空间弱式、同一镜面迹共享与同一组门槛，只把张量与真空系数先对"角区 ×
方位形函数乘积"积分成 `⟨b_a b_b ΩΩᵀ⟩` 与 `⟨b_a b_b |Ω·n|⟩`，再解一个耦合
扩散块；`A=0` 时它就是角区平均。两套离散共用同样的 `I`、`M`、`A` 与自由度
数，所以同一格上的逐点差只来自闭合本身。

### 等节点数空间加密

`geometry.interface_grading` 可在 `S>1` 时将原有子节点向材料界面和真空边界移动。
取值范围 `[0, 1)`，默认 0 对应均匀划分，也可用同名 case 参数覆盖。每个原区间
仍含 S 个子区间，原有物理界面坐标保持不动。棋盘格上可以用以下命令比较。

```bash
python -m scripts.sector_diffusion_benchmark strong --closure sector_average \
  --polar-order 30 --azimuth-degree 1 --slope-space bilinear --sm 6,16 10,24
python -m scripts.sector_diffusion_benchmark strong --closure sector_average \
  --polar-order 30 --azimuth-degree 1 --slope-space bilinear --sm 6,16 10,24 \
  --interface-grading 0.65
```

C5G7 显式材料网格也可以在固定预算内重新分配子节点。矩形组件的 `--balance-pin`
保持每个棒元每轴的 `3S` 个子区间，优先减小最宽的子区间。四分之一堆芯的
`--core-reflector-cells 18` 将均质外反射层的 34 个区间转给堆芯中的宽区间，保留
原 153×153 网格预算、材料面积和材料界面。空间重分配会改变离散本征对和 Ritz
收敛所需的扫掠次数。可用下列命令复算同角区预算的测点。

```bash
python -m scripts.sector_diffusion_benchmark rect_mox --closure sector_average \
  --azimuth-degree 1 --slope-space bilinear --balance-pin --sm 2,4 2,8 2,16
python -m scripts.sector_diffusion_benchmark rect --closure sector_average \
  --azimuth-degree 1 --slope-space bilinear --azimuth-grid axis --core-reflector-cells 18 \
  --opt sector-cholmod --threads 60 --mem-limit-gb 43 --sm 1,8 1,16
```

## 方形棋盘格

用论文 Fig. 3 对应的二维棋盘格几何，强、弱两种吸收，固定 `I=30`。参考是论文中的
强吸收 `0.51673`、弱吸收 `1.12974`。误差定义为 `(k_eff - k_ref) × 10⁵`。
强吸收的 legacy 路径（`closure: legacy`）过零位置随 S 移动：S=2 在 M=4–8 间
（−790.22 → +807.92 pcm），S=4 在 M=8–12 间，S=6 与 S=8 在 M=12–24 间，
这就是论文 Fig. 3 的对角谷线。

记录口径 `sector_diffusion` 已按发表网格逐点重测并归档：`A=0,1,2` 三个方位阶、
强与弱共 314 个测点（`paper/data/study4-checkerboard.csv` 里
`closure=sector_diffusion` 的行，逐点出处 `paper/data/uniform/points.jsonl`）：
发表网格 114 点全齐，另有 `A=1` 宽矩阵新增的 200 点。全部通过独立复扫，
最大源残差 `3.93e-9`、最大本征值残差 `9.96e-10`。下表与论文
表 `tab:study4-checker-sector`（`paper/tables/study4-checker-sector.tex`）同源；
"—" 是 `sector_average` 当时未发表、因而不在这轮镜像补测计划里的格。

| S | M | 强 A=0 | 强 A=1 | 强 A=2 | 弱 A=0 | 弱 A=1 | 弱 A=2 |
|--:|--:|-------:|-------:|-------:|-------:|-------:|-------:|
| 2 | 4 | −746.69 | −146.66 | — | −120.55 | −31.05 | −50.61 |
| 4 | 8 | −72.92 | −74.01 | −76.46 | −14.44 | −14.80 | −15.21 |
| 8 | 12 | −16.47 | −22.38 | −23.13 | −4.42 | −5.57 | −5.66 |
| 12 | 24 | −9.97 | −12.25 | −12.08 | −3.35 | −3.73 | −3.70 |
| 12 | 48 | −11.68 | — | — | −3.63 | — | −3.68 |
| 12 | 96 | −11.99 | — | — | −3.68 | — | −3.68 |

对角谷线消失，但符号结构比"全部同号"更细：314 点中 279 点为负，35 个正偏差
全部落在 `A=1`（每扇区只有两个 Gauss 方向）的 `M=4` 一列 —— 强吸收
S=3…20、弱吸收 S=4…20，最大分别 +127.38 与 +18.24 pcm，宽矩阵把这一列一路
推到 `S=20` 也没有翻回负号。同一 `A,S,M` 下的角区平均值在这 35 格上仍为负。
对照闭合在角区积分点不足时也出现过变号。逐方向闭合直接构造各方向的张量，
现有结果只能确认变号对方位采样敏感。除 `M=4` 这一列外，固定 `M` 加密 S 的
280 步里 246 步 `|Δk|` 趋零，34 个反例全在该列内。M 方向的余差已饱和：强吸收
`A=1`、`S=12` 随 M=8、12、24 得到 −5.73、−10.79、−12.25 pcm，55 条至少两点、
`M≥12` 的 `(A,S)` 列里最宽的也只有 6.79 pcm，本征值误差由角向设置主导；最宽的
首步是 `A=0`、`S=12` 强吸收那条（`M=8` 处 −22.40 pcm，之后收到 −8.38 pcm）。

同一格上提高 `A` 不再有系统收益：(4,8)、(8,12)、(12,24) 三行里 `A=0` 在两个
吸收体上都略优于 `A=1` 与 `A=2`。三个方位阶各自的最佳点落在不同网格：`A=0`
强 −8.38 / 弱 −2.90（`S=12, M=12`）、`A=1` 强 −5.73 / 弱 −2.68（`S=12, M=8`）、
`A=2` 强 −12.08（`S=12, M=24`）/ 弱 −3.67（`S=12, M=8`）。此外 `A=1` 的宽矩阵
（`S=1…20 × M=2,4,8,12,16,24`，每侧 100 格新增点）已跑满，每侧连同发表网格的
`A=1` 点共 120 格，最强点在 `S=20, M=8`（强 −0.90、弱 −1.79 pcm），`M=4`
一列仍整列异号。`python -m scripts.study4_sm_figures` 从同一批数据重画这些矩阵
（棋盘、宽矩阵、组件 2×2 误差图与耗时图、13 格总览）。等自由度的
`sector_average` 对照逐点比值见论文表 `tab:study4-closure-ratio`。

把每扇区的 Gauss 方向从 `A=1` 的两个增至 `A=2` 的三个后，强、弱吸收的
`M=4, S=1–20` 共 40 格偏差均为负。`A=2, S=20, M=2` 两例仍出现正偏差。
默认 `A=3` 使用四个方向。在 `S=1–20 × M=2、4、8、12、16、24` 的 240 格中，
两种吸收体全为负偏差。228 个空间加密步全部缩小误差，200 个方位加密步中有 155 个
缩小误差。各自最佳格均为 `S=20, M=8`，强、弱吸收分别为 −6.19、−2.65 pcm。
逐点结果和独立复扫残差见[方位阶对照数据](../paper/data/study4_checker_azimuth_controls.csv)。

本节此前记录的是默认 `A=3`（每扇区 4 个方向）的 40 格联合加密（`S=12, M=24`
处 −11.98 / −3.68 pcm），**仓内没有对应的脚本或日志**，按
`paper/unresolved_items.md` §11 的同类条目只作试算记录，不再作为表值；上表
改由已归档的 `A=0,1,2` 测点生成。按方位阶复现发表网格：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
samples=(2,4 2,8 2,12 2,24 4,4 4,8 4,12 4,24 6,4 6,8 6,12 6,24
         8,4 8,8 8,12 8,24 12,4 12,8 12,12 12,24)
for a in 0 1 2; do
  python -m scripts.sector_diffusion_benchmark strong --closure sector_diffusion \
      --azimuth-degree $a --sm "${samples[@]}" --threads 8
  python -m scripts.sector_diffusion_benchmark weak --closure sector_diffusion \
      --azimuth-degree $a --sm "${samples[@]}" --threads 8
done
python -m scripts.study4_uniform_grid run     # 镜像发表网格逐点补齐，可中断续跑
python -m scripts.study4_closure_report       # 重建两张棋盘矩阵与比值表
python -m scripts.sector_diffusion_benchmark strong --sm "${samples[@]}" --closure legacy --threads 1
```

原论文为 [Chao、Li、Chen, Annals of Nuclear Energy 240, 112707](https://www.sciencedirect.com/science/article/pii/S0306454926005955)。

## C5G7 矩形四分之一堆芯

使用 `c5g7_rect_quarter_core.yaml`，方形燃料中心和外围水层显式建模，
七群材料截面保持原输入。已逐项核对 51×51 pin 布局、1.26 cm 节距、
反射与真空边界以及总截面、散射、裂变产生截面和裂变谱。

OpenMC 0.11 同几何参考取自 `omc_rect_core/statepoint.20200.h5`，
200 个非活跃批次后，20,000 个活跃批次的 `k_generation` 均值为
`1.187217554586068`，标准误 3.41 pcm。Statepoint 的 `k_combined` 为
`1.187220349151050`，标准误 1.75 pcm。表中偏差沿用 PSN-3 的批次均值口径。

本节的堆芯算例全部用记录口径 `sector_diffusion`（`scripts/sector_diffusion_benchmark.py`
的默认闭合），固定 `I=8, A=1`。表中所有结果均通过独立复扫。

| S | M | 节点 | k_eff | Δk (pcm) | 复扫源残差 |
|--:|--:|-----:|------:|---------:|-----------:|
| 1 | 4 | 23,409 | 1.185612543298 | −160.50 | 3.66e−9 |
| 1 | 8 | 23,409 | 1.186419133929 | −79.84 | 6.66e−10 |
| 2 | 4 | 93,636 | 1.185746014252 | −147.15 | 8.96e−11 |
| 2 | 8 | 93,636 | 1.186619765186 | −59.78 | 3.50e−11 |

在这个 2×2 交叉表中，单独增大 S 或 M 均降低误差。联合从 S=1、M=4
加密到 S=2、M=8，绝对误差从 160.50 降到 59.78 pcm。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.sector_diffusion_benchmark rect --sm 1,4 1,8 --opt chol-rs --threads 32
python -m scripts.sector_diffusion_benchmark rect --sm 2,4 2,8 --opt chol-packed-rs --threads 32
```

压缩分解的 PCG 用原 FP64 矩阵验算线性残差。扇区扩散使用 `5e-12`
线性相对残差门槛；完整 SM 矩阵的 30 格里只有 `S=5、M=24` 一点是记录的例外，
它的一对配对解在自己的邻锚预条件下停在 `5.10e-12`（250 步），把上限提到 400 步
只到 `5.087e-12`，属预条件子的地板，故该点门槛记为 `3e-11`，设置随该点写进
`points.csv`。矩形本征计算的外层容差为 `1e-9`，独立复扫的源残差
门槛为 `5e-9`。直接 Cholesky 的 S=2、M=4 试算在大量换页下终止，
同离散的压缩 Cholesky 重跑收敛。S=2、M=8 的首次压缩试算在首扫中
触及旧线性门槛，调整后通过复扫。

### 完整 5 × 6 SM 矩阵

上表固定 `I=8`。在此之外把极角换成 TY3 三点规则、保持 `A=1` 与互余方位折叠，
扫 `S=1,2,3,4,5` × `M=2,4,8,12,16,24` 的全部 30 格。每格独立求解并通过
独立复扫（本征值相对变化 ≤ `1e-9`，复扫源残差 ≤ `5e-9`），30 格全部收敛，
两张矩阵都没有空格，重建命令不再需要 `--allow-missing`。误差在整张网格上同号，
49 个相邻加密步（每行 5 个方位步 × 5 行加每列 4 个空间步 × 6 列）全部把误差推向
零，最小格为 `S=5, M=24` 的 `−12.98 pcm`。逐点日志、CSV、
色板图与最佳点裂变率图收录在
[`examples/c5g7/study4_sector_core/`](../examples/c5g7/study4_sector_core/README.md)，
论文表 `tab:study4-core-pcm` 与 `tab:study4-core-time` 由
`scripts/study4_core_report.py` 从这些日志逐格重建。口径与边界的完整说明在该
目录的 README，要点是：部分测点热启动并复用分解因子，墙钟只反映负载而非吞吐；
`S=5` 的七群因子模板另计约 2144 秒；`M=16` 一列取自此前的同配置算例。

同一目录还记录了 legacy PSN 路径在同几何、同 `S=1, M=24`、同 60 线程冷启动下的
`−10.12 pcm` / 128.08 s 对扇区扩散的 `−54.09 pcm` / 340.68 s（约 2.7 倍）：最粗
角分辨率下扇区闭合不总是更好。这是**记录口径对 legacy 路径**的对照，与上文
两种扇区闭合之间的等自由度配对计时（中位 0.98）是两回事，不能互换引用。

<!-- SM_MATRIX_BEGIN -->

### Δk 矩阵（pcm，k_ref=1.187217555）

| S \ M | 2 | 4 | 8 | 12 | 16 | 24 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -271.73 | -149.28 | -68.90 | -60.26 | -57.21 | -54.09 |
| 2 | -263.55 | -137.20 | -49.66 | -40.31 | -37.10 | -33.95 |
| 3 | -260.06 | -133.29 | -39.46 | -28.91 | -25.68 | -23.05 |
| 4 | -257.30 | -131.26 | -33.57 | -22.58 | -19.44 | -16.77 |
| 5 | -254.69 | -130.03 | -29.34 | -18.52 | -15.85 | **-12.98** |

### 墙钟时间矩阵（秒，`*` 复用分解因子）

| S \ M | 2 | 4 | 8 | 12 | 16 | 24 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77.8 | 113.4 | 166.4 | 176.1 | 237.8 | 340.7 |
| 2 | 625.3 | 790.3 | 548.5 | 1299.0 | 996.6 | 1810.2 |
| 3 | 1199.5 | 1757.3 | 1559.8 | 1787.4 | 2790.4 | 2518.8 |
| 4 | 3263.4 | 1647.5 | 3486.0 | 2665.9 | 2163.1\* | 4360.9 |
| 5 | 5787.4 | 3161.0 | 3440.0 | 5084.0 | 4841.3\* | 10930.6\* |

两张矩阵由本脚本从 `logs/S{S}_M{M}.log` 逐格读回；`logs/timing_basis.csv` 记录每格的因子来源（六格取自此前的同配置算例，其原始运行名保留在各日志末行的 `name` 字段中），`*` 只标给复用了已存分解因子的格子。

<!-- SM_MATRIX_END -->
### 保持材料界面的加密试算

在同一几何上合并两侧材料完全相同的网格面，再把堆芯区域按不超过
0.2393 cm 的步长剖分。外部水反射层根据距堆芯的距离放宽步长。
重划网格为 244×244，逐材料面积核对的最大差异为 `1.83e-10 cm²`。
仍固定 `I=8, A=1`，在 `M=16` 得到 `k_eff=1.186955204039`，
偏差 **−26.24 pcm**。33 次扫描包含最终复扫，复扫源残差为 `1.28e-9`。
压缩因子和原算子模板合计 12.01 GiB。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENPSN_PREFACTOR_WORKERS=10
python -m scripts.sector_diffusion_benchmark rect --core-step 0.2393 --sm 1,16 \
    --opt chol-packed-rs --threads 20 --mem-limit-gb 15
```

上述界面对齐网格的试算偏差为 −26.24 pcm。矩形堆芯均匀网格的最小绝对偏差为 12.98 pcm，
距约 1 pcm 的目标仍有差距。
当前 OpenMC 存档的两个估计量的标准误也都高于 1 pcm。

更细网格的内存抽测使用 `S=3, M=16, I=8, A=1` 的一个高热群镜像方位块。
该块有 1,687,285 个未知量，压缩因子和原算子模板共 188,911,315 字节，
原 FP64 方程复核的相对残差为 `8.12e-14`。若 224 块平均采用该存储量，
常驻池约为 39.4 GiB。该数字是按单块外推的估计，未计 Ritz 缓存和分解临时数组。
