# OpenPSN

**Transport accuracy, diffusion-code simplicity.**

OpenPSN is licensed under the [Apache License 2.0](LICENSE).

OpenPSN is a deterministic neutron transport solver built on the **Phase
Space Nodal method (PSN)**: it solves the transport equation with the same
machinery as a diffusion code, yet delivers transport-grade accuracy — with
**no ray effects**, by construction. Multi-group (1 / 2 / 7 groups
verified), 2-D Cartesian cores with square **and rectangular** nodes, a
faithful, modernized reproduction of the PSN nodal method (Chao, Li & Chen,
*Annals of Nuclear Energy* 240 (2027) 112707) **extended to arbitrary
per-node x/y widths** (explicit un-homogenised intra-node fuel/water
structure) and to **higher-order directional angular-moment closure**
(A = 0–3), rebuilt as a clean, YAML-driven Python package **plus a pure-JS
in-browser engine**.

🌐 **Try it live:** <https://openpsn-ai.com> — the checkerboard benchmark
runs entirely in your browser (no server, no dependencies), converging
k<sub>eff</sub> to 10⁻¹⁰ in a few seconds, with the converged scalar-flux
field rendered on screen.

📄 **Paper (preprint):** *OpenPSN: Rectangular Phase-Space Nodal Transport
with Higher-Order Angular Moments* —
<https://www.researchgate.net/publication/414038054_OpenPSN_a_multi-group_Phase_Space_Nodal_solver_and_its_validation_on_the_C5G7_MOX_benchmark>

代码仓库：<https://github.com/xycjscs/OpenPSN>

**Python 启动即可**：`python -m psn2d run <problem.yaml>`。任意群数（1 / 2 / 7
群已验证），任意几何（材料网格 + 边界 + 矩形节点），角向/空间离散、并行度、
内存预算与收敛参数全部用户可配。`examples/` 下的输入文件**逐一对应论文各
实验**（输入 → 网格 → 参考值 → 结果，见 §5 对照表）。

---

## 0. 在线演示（in-browser engine）

`docs/` 目录下是一个自包含的静态站点（GitHub Pages，<https://openpsn-ai.com>）：

- **psn.js** — PSN 通用模型的纯 JavaScript 移植（641 行，无依赖），与 Python
  参考实现逐点咬合（24/24 点，max Δk<sub>eff</sub> = 4.8×10⁻¹⁰）。
- 浏览器内实时求解 1 群棋盘基准：选吸收体强弱 / 空间细分 S / 方位角段 M /
  收敛容差，点 Run，几秒收敛到 10⁻¹⁰，并渲染收敛曲线与收敛后的标量通量场。
- 页面 05 节展示 C5G7 结果：pinwise 均质化（BWW）最佳 S×M 点的 3×3 裂变对比
  图（论文 Fig.2）+ 矩形节点研究（节点布局 / M 收敛 / 裂变对比）+ 原论文
  （Chao et al. 2027）图 1/3/6/7/8/10，均标注出处。
- 本地预览：`cd docs && python3 -m http.server 8899` →
  <http://127.0.0.1:8899/>（需 http，因 `fig3ref.json` 用 `fetch` 加载）。

---

## 1. 功能

- **任意群数多群输运**：1 群（Fig.3 棋盘问题）、2 群（BWR 束组件，含 Gd 钉
  变体）、7 群（C5G7 MOX 基准）全部复现并与原文对齐（248 个已发表本征值点
  逐点对照，几 pcm 内咬合；逐点数据见 `docs/fig3ref.json` 与
  `docs/bwr_ref.json`）。
- **PSN 节点法**：抛物线形节点内插（quartic parabola），节点内消元
  （intra-nodal elimination），面流耦合；通用角度模型（generic）与 TY 受限
  模型（ty3）两套参数共用同一闭式族。
- **高序方向角向矩闭包（A = 0–3）**：`angular.A: 1..3` +
  `angular.closure: directional_matrix`，每个 (i,g) 系统 21 方向矩（A=3
  TY3 协议）或 A·(I+1) 方向矩（generic I 协议）；内层 LU 直解或
  GMRES + 进程池（§3）。论文 Study III 的 248 点扫表全部由该路径生成。
- **矩形节点**：节点算子扩展到任意 x/y 宽度（`widths`），非均质 7 群材料
  截面直接赋给矩形节点，无需 pin 均质化 —— 首个带显式节点内燃料/水结构的
  PSN 实现。
- **YAML 输入**：几何、截面、边界、矩形宽度、算法参数、批量 case（M×S×A
  扫参）一个文件描述；per-case 可覆盖 `M / S(subdivide) / A / closure /
  model / I`。
- **稀疏 + 向量化**：节点耦合矩阵稀疏求解，多群循环向量化；继承快照版的
  107× 提速与 OOM 根治（见 `snapshot/`）；内存优化后端族（`--opt`，§3）。

## 2. 安装

```bash
pip install -r requirements.txt   # numpy, scipy, pyyaml
```

Python ≥ 3.9。纯 Python（无 C 扩展、无 MPI），单机即可。numba 可选
（GMRES matvec 加速，缺省自动回退 NumPy 路径，结果位级一致）。

## 3. 运行

```bash
# §3 复现（原法 A=0，generic I=30）
python -m psn2d run examples/psn_repro/checkerboard_1g.yaml --case S6_M24    # 1 群棋盘（Fig.3，弱吸收）
python -m psn2d run examples/psn_repro/bwr_bundle_2g.yaml --case N4_M12      # 2 群 BWR 束（Fig.8/10，无 Gd）
python -m psn2d run examples/psn_repro/bwr_bundle_2g_with_gd.yaml --case N4_M12  # 2 群 BWR 束（含 Gd 钉）

# Study I（pinwise 均质化，BWW 自屏蔽 XS，ty3）
python -m psn2d run examples/c5g7/study1_homogenised/c5g7_uo2_assembly_bww_om.yaml --case M12_S5
python -m psn2d run examples/c5g7/study1_homogenised/c5g7_mox_assembly_bww.yaml --case M24_S6
python -m psn2d run examples/c5g7/study1_homogenised/c5g7_2d_quarter_core_bww_om.yaml --case M24_S6

# Study II（矩形节点 3×3 平铺，非均质 7 群）
python -m psn2d run examples/c5g7/study2_rectangular/c5g7_rect_uo2_assembly.yaml --case M24
python -m psn2d run examples/c5g7/study2_rectangular/c5g7_rect_quarter_core.yaml --case M24_S1

# Study III（A=3 方向闭包，2×2 燃料分割铺装）
python -m psn2d run examples/c5g7/study2_rectangular/c5g7_rect17_split4_uo2_assembly.yaml --case M24_S2
python -m psn2d run examples/c5g7/study2_rectangular/c5g7_core_split4_uo2_mox_quarter_core.yaml --case M8_S1

# Study III 棋盘（A=3，TY3 / generic I=30 / I=60 协议）
python -m psn2d run examples/psn_repro/checkerboard_1g_rect.yaml --case i30_M24_S20

# Study IV（扇区扩散闭包：每个有限角区解偶宇称二阶扩散方程）
python -m scripts.sector_diffusion_benchmark strong --sm 12,24 --threads 8
python -m scripts.sector_diffusion_benchmark rect --sm 1,16 --opt chol-packed-rs --threads 60
python -m scripts.study3_bilinear rect_uo2 --output <dir> --s 6 --m 24 --threads 8

# 通用开关
python -m psn2d run <file>.yaml --case <name>        # 只跑指定 case
python -m psn2d run <file>.yaml --json                # 结果输出 JSON
python -m psn2d run <file>.yaml --opt auto            # 内存优化后端（见下）
python -m psn2d run <file>.yaml --threads 24 --mem-limit-gb 32   # 并行/内存预算
```

**可选：内存优化后端（`--opt`，默认 `off`，默认路径位级不变）**

```bash
python -m psn2d run examples/c5g7/study1_homogenised/c5g7_2d_quarter_core.yaml --case M16_S2 --opt auto
```

| 值 | 后端 | 说明 |
|---|---|---|
| `off`（默认） | 原始路径 | COLAMD SuperLU，位级可复现 |
| `auto` | 共享 Cholesky → 角度 Schur → tile → 紧凑 MMD-LU → 原路径 | 内存感知：按 `--mem-limit-gb` 估算各后端因子池，选装得下且最小的；全部装不下才 fail-loud（不构造装不下的因子） |
| `chol` | 紧凑组装 + SPD 行缩放 + xy 转置因子共享 + Eigen 稀疏 Cholesky（METIS） | 需编译 `psn2d/memopt_cxx/libeigen_chol.so`；C5G7 1/4 芯 M16 实测峰值 RSS ≈ 1/9，keff 不变（≤1e-15） |
| `chol-rs` | 共享稀疏 Cholesky，使用 Rust 求解核 | 需要编译仓库内的 Rust 内核，依赖 Rust 工具链和 METIS |
| `chol-packed-rs` | 共享稀疏 Cholesky，使用 int16 压缩预条件器和 FP64 PCG | 支持同一共享几何条件。每次 PCG 都检查原 FP64 块残差，未过门限即报错 |
| `angschr` | 角度 Schur 剖分：系统按方位角方向块对角化（M/2 个 2D 网格 Cholesky，填充与 M 无关）+ 镜反射行的小角度 Schur 分量（M/2 个 544² 级连通分量） | 因子池 ∝ M 线性（tile 缝合面 ∝ M²）：rect M192 组件 21 系统全池 ~48GB，62GB 机器可行（tile 缝合面 185GB）；keff 与共享 Cholesky 逐位一致（M16 实测 Δk=2.2×10⁻¹⁶）；S 形成 8 路并行（实测 6.5×，位级一致）；池超预算或几何非方向块对角时 fail-loud 证书 |
| `tile` | Schur 分块 + 每块 Eigen Cholesky（METIS）+ 稀疏多右端核 | 内存回退后端：按 `--mem-limit-gb` 自动选块大小（几何周期 P0），用户无需指定 tile 或 assembly；纯几何块划分，无 assembly 概念也能用（如棋盘基准）；keff 与原路径位级一致（实测 Δ=0.0000 pcm）；最细块仍超预算时 fail-loud（不静默 OOM） |
| `mmd` | 紧凑组装 + 对称型 MMD 排序 SuperLU | 无 C 依赖 |
| `lu` | 紧凑组装 + COLAMD SuperLU | 无 C 依赖 |

共享 Cholesky 的几何闸门在安装期检查，匹配精度为 1e-12。
要求材料图转置对称（`mat == mat.T`）、方格网（nx==ny）、x/y 边界匹配、
半平面 M%4==0、vacuum alpha 有限非零；**矩形节点（`rect: true`）额外要求
宽度序列转置匹配（hx[j,i] == hy[i,j]）**——行缩放按面长加权（x 面乘
hy、y 面乘 hx，方形退化为常数 h），asym 压到 ~1e-16 后与方形同一链路；
另有每系统随机 RHS 残差 ≤1e-11 的原方程校验。
若条件不满足，显式指定的共享 Cholesky 后端会报错，`auto` 会评估后续后端。

C++ 桥编译（Eigen 头文件必须；METIS 可选，缺则用 AMD 排序）：

```bash
python psn2d/memopt_cxx/build.py \
    --eigen-include /path/to/eigen3 \
    --metis-include /path/to/include --metis-lib /path/to/lib/libmetis.so
```

**C5G7 四分之一堆芯加速**

在 Linux x86-64 上编译 Rust 内核。若 METIS 安装在自定义目录，
用 `OPENPSN_METIS_LIB` 指定共享库，再运行对应的几何和后端。

```bash
PSN_CHOL_CPU=native sh rust/build.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENPSN_NUMA=pool PSN_PAR=16 OPENPSN_PREFACTOR_WORKERS=5 \
  OPENPSN_TABLE_WORKERS=4 PSN_METIS_SEED=1 OPENPSN_TRACE=1 \
  python -m psn2d run examples/c5g7/study2_rectangular/c5g7_rect_quarter_core.yaml \
  --case M24_S3 --threads 60 --mem-limit-gb 32 --opt chol-packed-rs --ritz --quiet --json
```

`--ritz` 用粗网扩散预条件构造搜索方向，并用细网输运扫描计算和验证本征对。
它支持标量扫描的真空及反射边界，可替代 Anderson、Dombey 或 CMFD 外迭代。
`sweeps_executed` 包含最终验证扫描，`OPENPSN_TRACE=1` 记录逐次扫描时间。
压缩后端保留 Cholesky 对角项和原算子的 FP64 精度，仅量化预条件器的非对角项。
扫描逐方位角构造面源与累加矩，减少每个进程的工作内存。

**矩形 `M24_S3` 实测。** 459×459 节点，60 个并行单位，同一输入的第 2–11 次
细网扫描中位耗时从 160.16 秒降到 19.17 秒，**提速 8.35 倍**。
旧混合 IC/PCG 路径在 45 分钟时已完成 11 次扫描，随后达到时限。
标准 CLI 的压缩 Cholesky + Ritz 在 1261.48 秒内完成 22 次扫描，其中 1 次为最终验证。
预分解与算子准备耗时 691.31 秒。
整机峰值使用量 28.83 GiB，压缩因子和 FP64 算子池 9.419 GiB。
`--mem-limit-gb` 检查常驻因子池，实测整机峰值来自独立监控。

同一矩形几何和纯 7 群截面的 OpenMC 参考为
`1.187217554586068`（标准误约 3.41 pcm）。
OpenPSN `k_eff = 1.1867671564869706`，偏差 **−45.04 pcm**
（`(k_eff - k_ref) × 10⁵`），独立细网验证的源残差为 `2.57e-12`。
逐次用时和测量口径见 [矩形基准记录](examples/c5g7/study2_rectangular/m24_s3_benchmark.json)。

**均质 `M16_S2`。** 仓库原始均质输入上的 60 并行单位求解计时为
99.03 → 30.25 秒（**3.27 倍**，319 → 34 次扫描），Ritz 运行的端到端
时间为 31.31 秒。基线是 Dombey + Anderson，源停止条件较宽。
该输入的结果用于性能回归。
做 OpenMC 误差对照时使用 Study I 的 BWW 均质输入和其同几何参考
`1.1868230110346432`。该 BWW 输入本次复测为 30.31 秒，偏差 −23.60 pcm。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENPSN_NUMA=pool PSN_METIS_SEED=1 \
  python -m psn2d run examples/c5g7/study1_homogenised/c5g7_2d_quarter_core_bww_om.yaml \
  --case M16_S2 --threads 60 --mem-limit-gb 32 --opt chol-rs --ritz --quiet --json
```

**扇区扩散闭合（`sector_diffusion`）**

该闭合对每个方位扇区内的 Gauss 节点求解偶宇称二阶扩散方程
`−∇·[(Ω Ωᵀ/Σₜ) ∇u] + Σₜu = q`。空间采用连续双二次节点，
材料常数逐节点积分。扩散体项和面法向流来自同一个弱式。反射面共享镜像方向
的迹值，真空面使用 `|Ω·n| u` 的 Robin 项。散射、裂变和最终复扫使用同一离散。
角基只在各自扇区内定义，外层沿用 PSN-3 的扩散预条件 Ritz。

- `S` 是每个原几何节点每轴的等分数。`M` 是半平面的方位扇区数。
  `A+1` 是每扇区 Gauss 点数，默认 `A=3`。
- `model: generic` 使用 `I/2` 个正半轴 Gauss 极角点。
  `model: ty3` 使用既有三点极角积分。
- 支持真空和镜面反射边界。可用 `--opt off`、`--opt chol-rs` 或
  `--opt chol-packed-rs`，其中 Rust 后端的编译方式见上文。
  `factor_bytes` 是常驻因子池的大小，`sweeps_executed` 包含独立复扫。
- 固定 `model: generic` 时，可选 `closure: sector_average`。默认 `A: 0`
  只保留角区平均场。`A: 1` 或 `A: 2` 在每个扇区增加局部方位形函数，
  在同一个扩散弱式中同时求解。
- 开启 `A: 1` 或 `A: 2` 时，可选 `slope_space: bilinear`，以较少空间
  自由度表示高阶方位系数。平均场仍用双二次节点。多路 NUMA 机器可设
  `OPENPSN_NUMA=1` 让每个扇区扩散块的分解和扫描访问本地内存。
- [`--opt sector-cholmod` 构建与计时说明](docs/psn4-sector-diffusion.md#扇区扩散性能)
  适用于这两种扇区扩散闭合。
- 两种扇区闭合的比较见 [PSN-4 收敛记录](docs/psn4-sector-diffusion.md)。

**高序方向闭合的内层求解器（`solver.inner`，YAML `solver:` 段，默认 `lu`）**

A>0 的方向矩闭合（`closure: directional_matrix`）下，每个 (i,g) 系统可以
LU 直解，也可以 GMRES + 红黑节点扫描 matvec + fork 进程池并行（21 个系统
各一个 worker，worker 内 BLAS 钉死 1 线程）：

```yaml
solver:
  inner: gmres          # lu（默认）| gmres
  gmres_rtol: 1.0e-11   # 可选，系统残差判据（论文生产扫表用 1e-7，与外迭代门一致）
  gmres_maxiter: 800    # 可选（内部 Krylov 迭代上限，每 20 次重启）
  gmres_workers: 21     # 可选，默认 = 块数（system 模式=系统数，pair 模式=系统数×M/2）
  gmres_pairs: on       # 可选，off|on，默认 auto（M=2 走 system，M≥4 走 pair）
```

`gmres_pairs`（2026-09-14，杠杆 1）：把每个 (i,g) 系统按**镜像角度对**
{p, M-1-p} 再劈成 M/2 个独立块——(i,g) 的面方程矩阵 H 对这些对严格块
对角（internal/vacuum 行只在同一 m 内、reflect 行只耦合镜像对），每个块
各跑自己的 GMRES（对子局部维度，Krylov 空间减半），矩汇总 Σ_块 精确还原
全角度求和。M8 实测 system→pair 提速 1.53×（1056s→691s，keff 差
5.7e-14）；M 越大收益越大（48 核从 21 用到满，最慢系统被劈散）。
matvec 的 b-assembly（占 matvec ~87%）另有 numba prange 内核（杠杆 3，
自动启用，`OPSN_NO_NUMBA=1` 可关）：去 GIL 后 worker 内可用
T=核数/worker数 线程（BLAS 仍钉 1）。

| 值 | 说明 |
|---|---|
| `lu`（默认） | 每系统一张 SuperLU 因子常驻：M8 峰值 ~54 GB（21 因子），小 M 最快（0.7 s/外迭代） |
| `gmres` | 仅驻留 KB 级节点因子表：M8 3.65 GB、M12 6.41 GB（~15× 内存削减），keff 与 LU 路径全收敛位级一致（A=1/2/3 实测 Δ≤2e-14）；M≥12 或 S≥2 时是 62 GB 机器上唯一可行路径 |

几何约束：矩形节点、半平面偶数 M、整网格统一 A、reflect/vacuum 边界；
纯 Python matvec 的线程路线不可用（实测 GIL：b-assembly 持锁，24 线程
效率 ~2% 比串行慢），并行走进程池 + 块分片，worker 内 b-assembly 走
numba prange。

## 4. 输入格式（YAML）

```yaml
geometry:
  grid:        # (ny, nx) 材料索引网格，0 基；值对应 materials 顺序
    - [0, 0, 1]
    - [0, 0, 1]
  unit_size: 1.26        # cm，每个网格单元的边长
# widths: [hx_grid, hy_grid]   # 可选：矩形节点每节点 x/y 宽度（rect: true）
boundaries:              # 四条边：reflective 或 vacuum
  left: reflective
  bottom: reflective
  right: reflective
  top: reflective
materials_file: c5g7_materials_pure.yaml   # 或内联 materials/chi
angular:                   # 角向模型（全局默认；per-case 可覆盖）
  model: generic           # generic（默认）| ty3
  M: 12                    # 方位角段数（偶数）
  I: 30                    # generic 模型的极向段数（ty3 忽略）
  A: 0                     # 高序方向矩阶数 0..3（A>0 需 closure）
  closure: directional_matrix   # A>0 时的闭包类型
solver:
  max_outer: 3000
  keff_tol: 1e-10
  threads: 24            # 可选：进程树总并行度预算（父因子化 BLAS / 预因子线程池 / fork 扫算池之和恒 ≤ N）
  mem_limit_gb: 32       # 可选：--opt auto/chol/angschr/tile 的因子池预算（默认 32）
cases:                   # 批量：per-case 覆盖 M / subdivide / A / closure / model / I
  - { name: M8_S1,  model: generic,  M: 8,  subdivide: 1 }
  - { name: M12_S1, model: generic,  M: 12, subdivide: 1 }
  - { name: M12_S2, model: generic,  M: 12, subdivide: 2 }
  - { name: A3_M24_S2, A: 3, closure: directional_matrix, M: 24, subdivide: 2, kref: 1.3347260 }
```

- **model**：角度模型，`generic`（默认，论文 Sec.2 通用 PSN）或 `ty3`
  （TY 受限二维模型）。
- **M**：方位角离散段数（角向精度，须为偶数）。
- **S**：空间细分（每个材料网格再分为 S×S 子节点 → 空间精度；`subdivide`）。
- **A / closure**：高序方向矩闭包；`A: 0`（默认）= 原法（论文 §3 复现）。
- **kref**（可选）：参考 k 值，给出后自动打印 Δ（pcm）。

内存模型：ty3 下 21 个 (i,g) 系统（3 TY 极线 × 7 群）各一次一次性因子化并
常驻缓存，外迭代纯回代；**S 主导内存，M 只改块数不改填充**（1/4 芯 S=1 2601
节点 ~4 GB，S=2 10404 节点峰值 ~14 GB；优化后端把 S=3..6 全收回 4–5 GB
量级）。

## 5. 输入 ↔ 论文实验对照（examples/ 覆盖论文全部实验）

`examples/` 下每个输入文件的 `cases` 块覆盖论文对应表格的完整扫表网格；
`kref` 即论文所用参考值（同几何 OpenMC / 原文发表值），运行时自动打印
Δk（pcm）。

| 论文实验 | 输入文件 | 网格（cases） | 参考 kref | 结果（论文） |
|---|---|---|---|---|
| §3 复现：1 群棋盘（原法 A=0，generic I=30） | `psn_repro/checkerboard_1g.yaml` | S=1–10 × M=2–24（60 点） | 1.12974 | 与原文 Fig.3 逐点咬合（`docs/fig3ref.json`） |
| §3 复现：1 群棋盘强吸收 | `psn_repro/checkerboard_1g_strong.yaml` | S=1–10 × M=2–24（60 点） | 0.51673 | 同上（附录 B） |
| §3 复现：2 群 BWR 束（无 Gd） | `psn_repro/bwr_bundle_2g.yaml` | N=1–10 × M=4–24（60 点） | 1.18797 | 原文 Fig.8/10（附录 C） |
| §3 复现：2 群 BWR 束（含 Gd 钉） | `psn_repro/bwr_bundle_2g_with_gd.yaml` | N=1–10 × M=4–24（60 点） | 0.86688 | 附录 D |
| Study I：pinwise 均质化 UO₂ 组件 | `c5g7/study1_homogenised/c5g7_uo2_assembly_bww_om.yaml` | S=1–6 × M=2–24（36 点） | 1.3336285 | 最佳 S=5, M=12：+0.0 pcm（进 MC 误差带 ±3.5） |
| Study I：pinwise 均质化 MOX 组件 | `c5g7/study1_homogenised/c5g7_mox_assembly_bww.yaml` | S=1–6 × M=2–24（36 点） | 1.1853993 | 最佳 S=6, M=24：−1.5 pcm（±3.3） |
| Study I：pinwise 均质化 1/4 芯 | `c5g7/study1_homogenised/c5g7_2d_quarter_core_bww_om.yaml` | S=1–6 × M=2–24（36 点） | 1.1868230 | 最佳 S=6, M=24：−7.7 pcm（±3.4） |
| Study II：3×3 平铺 UO₂ 组件（矩形节点） | `c5g7/study2_rectangular/c5g7_rect_uo2_assembly.yaml` | S=1–6 × M=2–24（36 点） | 1.3347260 | S=1, M=24：+1.2 pcm（S=1 行 M 收敛到参考带内） |
| Study II：3×3 平铺 MOX 组件 | `c5g7/study2_rectangular/c5g7_rect_mox_assembly.yaml` | S=1–6 × M=2–24（36 点） | 1.1851053 | 同上网格（表 tab:rectsm） |
| Study II：3×3 平铺 1/4 芯 | `c5g7/study2_rectangular/c5g7_rect_quarter_core.yaml` | S=1–2 全 M + S=3 M=2,4,8（15 点） | 1.1872176 | S=1, M=24：−10.2 pcm（最佳点） |
| Study III：A=3 组件 UO₂（2×2 燃料分割） | `c5g7/study2_rectangular/c5g7_rect17_split4_uo2_assembly.yaml` | S=1,2 × M=2–24（12 点） | 1.3347260 | S=1：−119.2→+10.3 pcm（M2→M24）；S=2：−128.7→−0.5 |
| Study III：A=3 组件 MOX | `c5g7/study2_rectangular/c5g7_rect17_split4_mox_assembly.yaml` | S=1,2 × M=2–24（12 点） | 1.1851053 | S=1：−274.8→+7.4；S=2：−303.3→−9.8 |
| Study III：A=3 1/4 芯 | `c5g7/study2_rectangular/c5g7_core_split4_uo2_mox_quarter_core.yaml` | S=1,2 × M=2–24（12 点） | 1.1872176 | S=1 全行：−135.6 / −34.2 / −1.7 / +6.2 / +9.1 / +11.0 pcm（M2→M24）；S=2 行在产（M=2：−150.1） |
| Study III：A=3 棋盘弱吸收（TY3 / I30 / I60） | `psn_repro/checkerboard_1g_rect.yaml` | TY3 S=1–12；I30 S=1–20；I60 单点（194 case） | 1.12974 | I30 最佳 S=12：−0.1 pcm；I60 S=20 M=24：−0.9（vs I30 同点 −3.8，+2.9 pcm） |
| Study III：A=3 棋盘强吸收 | `psn_repro/checkerboard_1g_rect_strong.yaml` | 同上（194 case） | 0.51673 | 同上网格 |
| Study IV：扇区扩散闭包 棋盘弱/强吸收 | `psn_repro/checkerboard_1g.yaml`、`psn_repro/checkerboard_1g_strong.yaml` | S=2–12 × M=4–96，共 130 点（A=0 44、A=1 二次 40、A=1 双线性 40、A=2 6） | 1.12974 / 0.51673 | 130 点全部同号（负），任一轴加密单调趋零，对角谷线消失 |
| Study IV：扇区扩散闭包 4 组件 + 2 棋盘 | `c5g7/study1_homogenised/c5g7_{uo2,mox}_assembly_*.yaml`、`c5g7/study2_rectangular/c5g7_rect_{uo2,mox}_assembly.yaml` | 组件 S=1–6、棋盘 S=1–20 × M=2–24（384 点），A=1 双线性 | 同 Study I / II | 论文表 tab:study4-assembly；矩形 MOX 最佳 −118.55 pcm（材料/平铺受限） |
| Study IV：扇区扩散闭包 矩形 1/4 芯 SM 矩阵 | `c5g7/study2_rectangular/c5g7_rect_quarter_core.yaml` | S=1–5 × M=2,4,8,12,16,24（TY3，A=1，逐点独立复扫） | 1.1872176 | 矩阵与最佳点见 `examples/c5g7/study4_sector_core/`（`sm_matrix.md` 由 `scripts/study4_core_report.py` 从逐点日志重建） |
| 严格代数校验（14 项，符号/有理精确） | `examples/psn_strict_checks/` | 两层真空板、镜像矩形、四反射均质 7 群、堆芯守恒 LP | 精确解析值 | k∞ 四位精确小数、`k_eff<1.894` 上界，`unittest` 约 2 s |

（棋盘 A3 的 `subdivide` 约定：case 名中的 S = 论文 S，2×2 基网格再细分
S×S → 2S×2S 节点，即 `subdivide: S`。）

旧快照版输入（`c5g7_2d_quarter_core.yaml`、`c5g7_uo2_assembly.yaml`、
`study2_rectangular` 的 rect4x4 / rect6 / pin 系列）保留作历史对照与回归
基线，不再对应论文主表。

**完整 248 点复现数据**：`docs/fig3ref.json`（棋盘 weak/strong 各 60 点，
S×M 键控，ours/paper/Δ 三列）与 `docs/bwr_ref.json`（BWR 无 Gd/含 Gd 各
60 点，this-work 值）。原始快照仓库见 `snapshot/`。

## 6. 复现验证与回归测试

回归套件 `tests/run_examples.py` 覆盖 §5 各问题的代表点（fast 层 ~3 min，
full 层另含 C5G7 1/4 芯均质化与矩形节点重 case），对照
`tests/baseline_keff.json` 的位级基线（Δ 容差 1.0 pcm，`--record` 合并
记录新基线，`--only <yaml 片段>` 过滤）：

```bash
python tests/run_examples.py          # fast 层（当前 7/7 全过，Δ=0.000 pcm）
python tests/run_examples.py --full   # 含 full 层重 case
```

扇区扩散闭包（论文 Study IV）的算子与本征对回归，以及严格闭包门禁：

```bash
python tests/test_sector_diffusion.py  # 22 项，~13 s：扇区张量、节点响应精确消元、界面与对称性
python tests/test_strict_closure.py    # 严格闭包锚点/一致性门禁（M=2 处严格基 = 原法基）
cd examples/psn_strict_checks && python -m unittest discover -s . -p 'test_*.py'
                                       # 14 项有理/符号精确检查（sympy + PyYAML，~2 s）
```

C5G7 20 码 NEA 景观对照（高保真参考 1.18646，论文 §4.5）：

```
MOC 集群 ±28 pcm 内：CHAPLET +10, MCCG3D +11, DeCART +14, APOLLO2 -28
SN / Pn 节点法       -150 … +99
P1 扩散              -290 … -323
OpenPSN（矩形节点 M=16）  1.1866067  +14.7   ← 落在高保真 MOC 集群内
```

## 7. 论文扫表驱动与数据（tools_ho/）

论文 Study III 各表的生产扫表（HPC，SLURM）由 `tools_ho/` 下的单点驱动
+ 批处理脚本生成；**逐点日志与 dump 数据随码发布**（`tools_ho/data/`），
图可由仓库内脚本一键重生成：

```
tools_ho/
├── hpc_point.py            # 单点驱动（生产协议：GMRES 池, rtol=1e-7, pairs, dump）
├── hpc_flux_dump.py        # 单点裂变场 dump 驱动（Figure-2 风格对比图用）
├── rect17_run.py           # Study III 组件 24 点扫表协调器（2×2 燃料分割）
├── make_core_split4.py     # 1/4 芯 2×2 燃料分割 YAML 生成器
├── verify_core_split4.py   # 分割几何一致性校验
├── cb_a3_table.py          # 棋盘 A3 TY3 表 + checkerboard_a3 图（自日志）
├── cb_i30_table.py         # 棋盘 A3 I30/I60 表 + checkerboard_a3_i30_sm 图
├── cb_fit_rms.py           # 棋盘 A3 RMS 拟合
├── plot_a3_mconv.py        # 论文 Fig.（Δk + fission-max vs M，2×3 合并图）
├── plot_a3_fission.py      # 论文 Fig.（3 行 pin 平均裂变对比热力图）
├── plot_rect4_pin_geom.py  # 论文 Fig.（2×2 燃料分割节点几何示意图）
├── plot_r17_m_conv*.py     # 组件 M 收敛图（历史版本）
├── smoke_*.py              # 小网格冒烟测试（A3 各协议）
├── slurm_*.sh              # SLURM 提交脚本（asm/core 串行批、棋盘 A3/I30/I60、redump）
├── *.log / *.out           # 生产扫表逐点日志（assembly 24 点、core 7 点、棋盘 TY3/I30/I60、
│                           #   Study II 4 点对照、DIR 早期 sweep）
└── data/
    ├── r17_core4/          # 组件/芯 A3 点 fission dump（31 × .npz，pin 平均 + 全节点 + 元数据）
    └── omc_*.npy           # 同几何 OpenMC 参考功率场（组件 UO₂/MOX、1/4 芯）
```

论文图 → 生成脚本：`c5g7_rect4_a3_m_conv.png` ← `plot_a3_mconv.py`；
`c5g7_rect4_a3_fission.png` ← `plot_a3_fission.py`；
`checkerboard_a3_i30_sm.png` ← `cb_i30_table.py`；
`rect4_pin_geometry.png` ← `plot_rect4_pin_geom.py`。

## 8. 目录结构

```
OpenPSN/
├── psn2d/            # 主包：现代多群 PSN（YAML 驱动）
│   ├── __main__.py   #   CLI 入口 (run, --opt/--threads/--mem-limit-gb, per-case A/closure)
│   ├── model.py      #   YAML 解析（expand_cases：per-case 默认填充）
│   ├── solver.py     #   多群 PSN 节点求解器（稀疏+向量化，fork 扫算池）
│   ├── node.py       #   方形节点闭式解（generic / ty3 两模型）
│   ├── node_rect.py  #   矩形节点闭式解（任意 x/y 宽度）
│   ├── highorder.py  #   高序方向矩闭合（A=0..3，directional/original 分支，LU 内层）
│   ├── highsweep.py  #   GMRES 内层 + fork 进程池（solver.inner: gmres，静态分片 warm-start）
│   ├── dump.py       #   fission 场 dump（全节点 + pin 块平均 + 元数据，生产扫表落盘用）
│   ├── runtime.py    #   线程/内存预算（用户输入，进程树恒 ≤ N）
│   ├── memopt.py     #   内存优化后端 auto 分派（--opt auto）
│   ├── angschr.py    #   角度 Schur 剖分后端（方向块对角化, --opt angschr）
│   ├── tile.py       #   Schur 分块内存回退后端（--opt tile）
│   ├── sector_diffusion.py  # 扇区扩散闭包（sector_diffusion / sector_average，论文 Study IV）
│   ├── phase_response.py    # 有限角区节点响应的精确消元（node_response）
│   ├── phase_reflection.py  # 镜面反射方向的迹共享与互余方位折叠
│   ├── sector_cholmod.c/.py # FP64 超节点 Cholesky 桥（--opt sector-cholmod，按需编译）
│   └── memopt_cxx/   #   C++ Cholesky 桥（eigen_chol.cpp + build.py, 按需编译）
├── docs/             # 在线演示站（GitHub Pages → openpsn-ai.com）
│   ├── index.html    #   产品页 + 浏览器内求解器 + C5G7 结果
│   ├── psn.js        #   纯 JS PSN 引擎（无依赖）
│   ├── fig3ref.json  #   论文 Fig.3 棋盘 120 点逐点数据（weak/strong 各 60）
│   ├── bwr_ref.json  #   论文附录 C/D BWR 120 点逐点数据（无 Gd/含 Gd 各 60）
│   ├── psn4-sector-diffusion.md # Study IV 收敛记录（方法推导、SM 矩阵、计时与后端）
│   └── img/          #   内嵌插图（原论文 Fig.1/3/6/7/8/10 + C5G7 图）
├── examples/         # 论文问题输入（按论文结构分文件夹，见 §5 对照表）
│   ├── psn_repro/    #   §3 复现 + Study III 棋盘：1 群棋盘（A=0 / A=3 rect）+ 2 群 BWR（±Gd）
│   ├── c5g7/
│   │   ├── c5g7-mgxs.h5             # 原始 7 群截面库
│   │   ├── study1_homogenised/      # Study I：pinwise 均质化（含 BWW 自屏蔽）
│   │   ├── study2_rectangular/      # Study II：3×3 矩形节点 + Study III：2×2 燃料分割（A=3）
│   │   └── study4_sector_core/      # Study IV：1/4 芯 SM 矩阵（逐点日志 + CSV + 图）
│   ├── psn_strict_checks/  # 14 项有理/符号精确检查（论文 Study IV 的符号对照）
│   ├── tools/        #   probe/chain/绘图等辅助脚本
│   └── logs/         #   历史扫参日志
├── tools_ho/         # 论文扫表驱动 + 逐点日志 + dump 数据 + 图脚本（见 §7）
├── paper/            # 论文：main.tex + studyIII.tex + studyIV.tex + appendix.tex
│                     #   + figures/ + tables/ + 三格式（pdf/md/docx）
├── tests/            # 回归套件：run_examples.py + test_memopt.py + test_tile.py
│                     #   + test_sector_diffusion.py + test_strict_closure.py
│                     #   + verify_rect_node.py + baseline_keff.json（位级基线）
├── scripts/          # C5G7 规格/截面生成脚本 + 扇区扩散基准与 Study IV 表图驱动
├── snapshot/         # 原始复现快照（core/ + drivers/ + 数据/图/日志，248 点）
├── requirements.txt
└── README.md
```

## 9. 方法说明

PSN（Phase Space Nodal，相空间节点法）节点法在每个节点内以四次抛物线插值
表示通量，通过节点内消元得到面流与节点源之间的显式关系，再在节点边界
耦合。相比传统 SN 差分，PSN 用更少的空间网格达到同等精度；相比 MOC，PSN
不依赖追踪方向在网格上的投影，对复杂网格更稳健。三条能力线：

1. **原法复现（A=0）**：generic / ty3 两套角度模型的忠实复现（论文 §3，
   248 点逐点咬合）；
2. **矩形节点**：把每节点的 x/y 宽度作为独立几何参数进入全部闭式（衰减
   常数、面流、面通量、体矩），使非均质 7 群材料截面可直接赋给显式
   燃料/水矩形节点（论文 §4 Study II）；
3. **高序方向角向矩闭包（A=1–3）**：在节点面流耦合上引入方向矩修正，
   把角向误差从 O(Δθ²) 压低到 O(Δθ^{2A+2}) 量级；A=3 + 2×2 燃料分割
   铺装在 C5G7 组件/芯/棋盘上全部进入同几何 OpenMC 参考误差带（论文
   §5，248 点）。

细节见论文（`paper/`）与 `psn2d/solver.py`、`psn2d/node_rect.py`、
`psn2d/highorder.py` 注释（含论文式号 B.3 的抛物线源更新）。

## 10. 并行与部署

- **线程/内存是用户输入**（YAML `solver.threads` / `solver.mem_limit_gb`，
  CLI `--threads` / `--mem-limit-gb`）：`threads` 是**整个进程树的总并行度
  预算**——父进程因子化 BLAS ≤ N、预因子线程池 W×(N//W) ≤ N、fork 扫算池
  ≤ N 进程 × 1 BLAS 线程，严格不超；缺省 = 机器核数的一半（48 核机实测 24
  比 48 快）。`mem_limit_gb` 是 `--opt` 后端因子池预算（默认 32 GB），
  装不下即 fail-loud（不静默 OOM）。
- **sweep 并行 = fork 进程池**（SuperLU 回代不释放 GIL，线程池实测更慢）。
  并行度自适应（cgroup 感知），`PSN_PAR=1` 强制串行；
  小问题（ncol < 15 万）自动串行。
- **BLAS 确定性钉扎**：`psn2d/__init__.py` 在 import 时把 OpenBLAS/OMP
  钉到默认线程数（DYNAMIC_ARCH 构建最高 64 线程，不钉则回归漂移）；
  fork worker 与父进程走同一 BLAS 代码路径。
- **sweep 共享缓冲 = POSIX shm 专用（无磁盘 fallback）**：源项/归约走
  `/dev/shm`（父进程付 ~MB 级拷贝）。两段（q、r）整个运行期共存，
  故安装时按 **q+r 合计** ≤ tmpfs 空闲判（tmpfs 按实际触碰页记账，
  逐段检查有竞态）。**不足直接 fail-loud**（`MemoryError`，附扩容指引）：
  部署机必须配足 `/dev/shm`（core 最大点 S=6 需 ~100 MiB；64 MiB
  的 Docker 默认装不下 core S≥5，建议 `--shm-size=4g`）。确实要在这
  类小 shm 机器上跑，用 `PSN_PAR=1` 串行（跳过进程池及其共享段）。
- **内存预算按机器总占用算**：父进程 RSS + fork worker 私有状态
  （COW 只共享读因子，私有部分是角度通量/局部归约）。62 GB 机器上
  core M12_S2 plain 全程约 41 GB，优化后端（chol/angschr/tile）低数倍至
  一个数量级（M192 组件 angschr 全池 ~48 GB 可行，tile 缝合面 185 GB 不可行）。

---

论文：[OpenPSN: Rectangular Phase-Space Nodal Transport with Higher-Order Angular Moments](https://www.researchgate.net/publication/414038054_OpenPSN_a_multi-group_Phase_Space_Nodal_solver_and_its_validation_on_the_C5G7_MOX_benchmark)（ResearchGate 预印本，2026）
