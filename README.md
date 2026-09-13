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
structure), rebuilt as a clean, YAML-driven Python package **plus a pure-JS
in-browser engine**.

🌐 **Try it live:** <https://openpsn-ai.com> — the checkerboard benchmark
runs entirely in your browser (no server, no dependencies), converging
k<sub>eff</sub> to 10⁻¹⁰ in a few seconds, with the converged scalar-flux
field rendered on screen.

📄 **Paper (preprint):** *OpenPSN: a multi-group Phase Space Nodal solver
with rectangular nodes, validated on the C5G7 MOX benchmark* —
<https://www.researchgate.net/publication/414038054_OpenPSN_a_multi-group_Phase_Space_Nodal_solver_and_its_validation_on_the_C5G7_MOX_benchmark>

**Python 启动即可**：`python -m psn2d run <problem.yaml>`。任意群数（1 / 2 / 7
群已验证），任意几何（材料网格 + 边界 + 矩形节点），角向/空间离散、并行度、
内存预算与收敛参数全部用户可配。

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

- **任意群数多群输运**：1 群（Fig.3 棋盘问题）、2 群（BWR 束组件）、7 群
  （C5G7 MOX 基准）全部复现并与原文对齐（248 个已发表本征值点，附录 A 逐点
  对照，几 pcm 内咬合）。
- **PSN 节点法**：抛物线形节点内插（quartic parabola），节点内消元（intra-nodal
  elimination），面流耦合；通用角度模型（generic）与 TY 受限模型（ty3）两套
  参数共用同一闭式族。
- **矩形节点**：节点算子扩展到任意 x/y 宽度（`widths`），非均质 7 群材料
  截面直接赋给矩形节点，无需 pin 均质化 —— 首个带显式节点内燃料/水结构的
  PSN 实现。
- **YAML 输入**：几何、截面、边界、矩形宽度、算法参数、批量 case（M×S 扫参）
  一个文件描述。
- **稀疏 + 向量化**：节点耦合矩阵稀疏求解，多群循环向量化；继承快照版的
  107× 提速与 OOM 根治（见 `snapshot/`）；内存优化后端族（`--opt`，§3）。

## 2. 安装

```bash
pip install -r requirements.txt   # numpy, scipy, pyyaml
```
Python ≥ 3.9。纯 Python（无 C 扩展、无 MPI），单机即可。

## 3. 运行

```bash
python -m psn2d run examples/psn_repro/checkerboard_1g.yaml        # 1 群棋盘（Fig.3）
python -m psn2d run examples/psn_repro/bwr_bundle_2g.yaml          # 2 群 BWR 束（Fig.8/10）
python -m psn2d run examples/c5g7/study1_homogenised/c5g7_2d_quarter_core.yaml   # 7 群 C5G7-2D 1/4 芯（均质化 XS）
python -m psn2d run examples/c5g7/study1_homogenised/c5g7_2d_quarter_core_bww_om.yaml   # Study I：BWW 自屏蔽均质化
python -m psn2d run examples/c5g7/study2_rectangular/c5g7_rect_uo2_assembly.yaml  # 7 群矩形节点单 UO2 组件（非均质）
python -m psn2d run examples/c5g7/study2_rectangular/c5g7_rect6_quarter_core.yaml # 7 群矩形节点 1/4 芯（6×6 准方划分）
python -m psn2d run <file>.yaml --case M12_S1            # 只跑指定 case
python -m psn2d run <file>.yaml --json                   # 结果输出 JSON
python -m psn2d run <file>.yaml --opt auto               # 内存优化后端（见下）
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
| `angschr` | 角度 Schur 剖分：系统按方位角方向块对角化（M/2 个 2D 网格 Cholesky，填充与 M 无关）+ 镜反射行的小角度 Schur 分量（M/2 个 544² 级连通分量） | 因子池 ∝ M 线性（tile 缝合面 ∝ M²）：rect M192 组件 21 系统全池 ~48GB，62GB 机器可行（tile 缝合面 185GB）；keff 与共享 Cholesky 逐位一致（M16 实测 Δk=2.2×10⁻¹⁶）；S 形成 8 路并行（实测 6.5×，位级一致）；池超预算或几何非方向块对角时 fail-loud 证书 |
| `tile` | Schur 分块 + 每块 Eigen Cholesky（METIS）+ 稀疏多右端核 | 内存回退后端：按 `--mem-limit-gb` 自动选块大小（几何周期 P0），用户无需指定 tile 或 assembly；纯几何块划分，无 assembly 概念也能用（如棋盘基准）；keff 与原路径位级一致（实测 Δ=0.0000 pcm）；最细块仍超预算时 fail-loud（不静默 OOM） |
| `mmd` | 紧凑组装 + 对称型 MMD 排序 SuperLU | 无 C 依赖 |
| `lu` | 紧凑组装 + COLAMD SuperLU | 无 C 依赖 |

几何闸门（`chol`/`auto` 安装期检查，1e-12 精度，不过即 `ValueError`）：
材料图转置对称（`mat == mat.T`）、方格网（nx==ny）、x/y 边界匹配、
半平面 M%4==0、vacuum alpha 有限非零；**矩形节点（`rect: true`）额外要求
宽度序列转置匹配（hx[j,i] == hy[i,j]）**——行缩放按面长加权（x 面乘
hy、y 面乘 hx，方形退化为常数 h），asym 压到 ~1e-16 后与方形同一链路；
另有每系统随机 RHS 残差 ≤1e-11 的原方程校验。不符合的几何自动落回紧凑
MMD-LU。

C++ 桥编译（Eigen 头文件必须；METIS 可选，缺则用 AMD 排序）：

```bash
python psn2d/memopt_cxx/build.py \
    --eigen-include /path/to/eigen3 \
    --metis-include /path/to/include --metis-lib /path/to/lib/libmetis.so
```

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
solver:
  max_outer: 3000
  keff_tol: 1e-10
  threads: 24            # 可选：进程树总并行度预算（父因子化 BLAS / 预因子线程池 / fork 扫算池之和恒 ≤ N）
  mem_limit_gb: 32       # 可选：--opt auto/chol/angschr/tile 的因子池预算（默认 32）
cases:                   # 批量：M=方位角段数, S=空间细分, model=generic|ty3
  - { name: M8_S1,  model: generic,  M: 8,  subdivide: 1 }
  - { name: M12_S1, model: generic,  M: 12, subdivide: 1 }
  - { name: M12_S2, model: generic,  M: 12, subdivide: 2 }
```

- **model**：角度模型，`generic`（默认，论文 Sec.2 通用 PSN）或 `ty3`（TY 受限二维模型）。
- **M**：方位角离散段数（角向精度，须为偶数）。
- **S**：空间细分（每个材料网格再分为 S×S 子节点 → 空间精度）。
- **kref**（可选）：参考 k 值，给出后自动打印 Δ（pcm）。

内存模型：ty3 下 21 个 (i,g) 系统（3 TY 极线 × 7 群）各一次一次性因子化并
常驻缓存，外迭代纯回代；**S 主导内存，M 只改块数不改填充**（1/4 芯 S=1 2601
节点 ~4 GB，S=2 10404 节点峰值 ~14 GB；优化后端把 S=3..6 全收回 4–5 GB 量级）。

## 5. 复现验证

| 问题 | 输入 | 结果 | 对照 |
|------|------|------|------|
| Fig.3 棋盘（1 群） | `psn_repro/checkerboard_1g.yaml` | 与快照逐点咬合 | 原文 Fig.3（<4 pcm） |
| BWR 2 群束 | `psn_repro/bwr_bundle_2g.yaml` | Δ=+28.3 pcm（N=2, M=8） | 原文 Fig.8/10 |
| C5G7-2D 1/4 芯（7 群，均质化 XS） | `c5g7/study1_homogenised/c5g7_2d_quarter_core.yaml` | k=1.18861（M12_S1） | 1.18646（McGraw PHYSOR 2014 高保真），+215 pcm |
| C5G7 单 UO2 组件（7 群，均质化 XS） | `c5g7/study1_homogenised/c5g7_uo2_assembly.yaml` | k=1.34030 | nTRACER 1.33367 |
| Study I 1/4 芯（BWW 自屏蔽均质化） | `c5g7/study1_homogenised/c5g7_2d_quarter_core_bww_om.yaml` | 最佳点 S=6, M=24：−7.7 pcm | 自算 OpenMC 同均质化参考 1.1867462（±3.4 pcm）；对高保真 1.18646 为 +28.6 pcm |
| C5G7 单 UO2 组件（7 群，矩形节点非均质） | `c5g7/study2_rectangular/c5g7_rect_uo2_assembly.yaml` | k=1.3347384（M24, S=1） | 自算 OpenMC 显式几何参考 1.3347260 ±3.6 pcm，+12.4 pcm |
| C5G7-2D 1/4 芯（7 群，矩形节点非均质） | `c5g7/study2_rectangular/c5g7_rect_quarter_core.yaml` | k=1.1871157（M24, S=1） | 自算 OpenMC 同几何参考 1.1872176 ±3.4 pcm，−10.2 pcm |
| rect6 6×6 准方划分 1/4 芯 | `c5g7/study2_rectangular/c5g7_rect6_quarter_core.yaml` | M=2/4/8/12/16：−278.8…−126.2 pcm（S=1） | 同上同几何参考；core M≥24 超 62 GB 机器预算 |

完整 248 点复现数据与图见 `snapshot/`（原始快照仓库）。

回归套件 `tests/run_examples.py` 覆盖上表问题（fast 层 ~3 min，full 层另含
C5G7 1/4 芯均质化与矩形节点重 case），对照 `tests/baseline_keff.json` 的位级
基线（Δ 容差 1.0 pcm，`--record` 合并记录新基线，`--only <yaml 片段>` 过滤）：

```bash
python tests/run_examples.py          # fast 层
python tests/run_examples.py --full   # 含 full 层重 case
```

## 6. C5G7 基准结果（论文 §4）

C5G7-2D MOX 1/4 芯（51×51 pin，2×2 燃料块 + L 形水反射，7 群）。两个研究
把节点法的两类误差源分开：

### Study I — pinwise 均质化（BWW 自屏蔽均质化截面）

S×M 扫参，对照**同一均质化数据**上直接计算的 OpenMC 参考
（100k×20200 批，collision，SEED 20260909）：

```
  UO2 组件  最佳 S=5, M=12   +0.0 pcm   （进入 MC 统计误差带 ±3.5 pcm）
  MOX 组件  最佳 S=6, M=24   -1.5 pcm   （±3.3 pcm）
  1/4 芯    最佳 S=6, M=24   -7.7 pcm   （±3.4 pcm；对高保真 1.18646 为 +28.6 pcm
                                         —— 均质化模型本身的残余）
最佳点 pin 通量场 vs 同配方 OpenMC：0.07% RMS（max 0.25%），1056 根燃料棒
  —— 平滑方法级偏差，不是局部误差对消。
```

### Study II — 矩形节点（非均质 7 群材料截面，无均质化模型）

**3×3 平铺**（153×153 节点，UO2 组件 S=1 全 M 收敛）：

```
1/4 芯（S=1）
  M=8    -184.3 pcm
  M=12   -110.9
  M=16    -61.1
  M=24    -10.2   ← 最佳点（同几何 OpenMC 参考 1.1872176 ±3.4）
pin 级裂变 map（keff 最优配置）：UO2 组件 0.17% RMS；1/4 芯 1056 棒
  0.54% RMS（max 1.50%，MOX 单元最大——局部梯度最陡处）
```

**6×6 准方划分**（单 pin 水隙节点长宽比 1.05，8×8 基网格；136×136 组件 /
408×408 芯）：单 pin 验证最佳 M=192×S=6 达 −8.7 pcm（进参考 ±8.2 pcm 带），
节点级裂变 RMS 0.05%/max 0.10%；全几何 pin 平均裂变分布 M≥8 即达 0.06–0.07%
RMS 并保持到 M=48（先于 keff 收敛）。

### 20 码 NEA 景观对照（高保真参考 1.18646）

```
MOC 集群 ±28 pcm 内：CHAPLET +10, MCCG3D +11, DeCART +14, APOLLO2 -28
SN / Pn 节点法       -150 … +99
P1 扩散              -290 … -323
OpenPSN（矩形节点 M=16）  1.1866067  +14.7   ← 落在高保真 MOC 集群内
```

- 空间细化 S=1→2（2601→10404 节点）k 仅动 ~4 pcm → 离散已收敛。
- 独立代数 k∞ = 1.32936 与 SPHINCS/nTRACER 无自屏蔽值**逐位一致**（数据接线铁证）。
- 与 20 个确定性代码的完整对照见论文 §4.5（NEA/NSC/DOC(2003)16 Table 17）。

论文：[OpenPSN: a multi-group Phase Space Nodal solver with rectangular nodes, validated on the C5G7 MOX benchmark](https://www.researchgate.net/publication/414038054_OpenPSN_a_multi-group_Phase_Space_Nodal_solver_and_its_validation_on_the_C5G7_MOX_benchmark)（ResearchGate 预印本，2026）

## 7. 目录结构

```
OpenPSN/
├── psn2d/            # 主包：现代多群 PSN（YAML 驱动）
│   ├── __main__.py   #   CLI 入口 (run, --opt/--threads/--mem-limit-gb)
│   ├── model.py      #   YAML 解析
│   ├── solver.py     #   多群 PSN 节点求解器（稀疏+向量化，fork 扫算池）
│   ├── node.py       #   方形节点闭式解（generic / ty3 两模型）
│   ├── node_rect.py  #   矩形节点闭式解（任意 x/y 宽度）
│   ├── runtime.py    #   线程/内存预算（用户输入，进程树恒 ≤ N）
│   ├── memopt.py     #   内存优化后端 auto 分派（--opt auto）
│   ├── angschr.py    #   角度 Schur 剖分后端（方向块对角化, --opt angschr）
│   ├── tile.py       #   Schur 分块内存回退后端（--opt tile）
│   └── memopt_cxx/   #   C++ Cholesky 桥（eigen_chol.cpp + build.py, 按需编译）
├── docs/             # 在线演示站（GitHub Pages → openpsn-ai.com）
│   ├── index.html    #   产品页 + 浏览器内求解器 + C5G7 结果
│   ├── psn.js        #   纯 JS PSN 引擎（无依赖）
│   ├── fig3ref.json  #   论文 Fig.3 60 点复现数据（页面比对用）
│   └── img/          #   内嵌插图（原论文 Fig.1/3/6/7/8/10 + C5G7 图）
├── examples/         # 论文问题输入（按论文结构分文件夹）
│   ├── psn_repro/    #   §3 复现旧 PSN：1 群棋盘 + 2 群 BWR 束
│   ├── c5g7/
│   │   ├── c5g7-mgxs.h5       # 原始 7 群截面库
│   │   ├── study1_homogenised/    # §4.3 Study I：pinwise 均质化（含 BWW 自屏蔽）
│   │   └── study2_rectangular/    # §4.4 Study II：矩形节点 3×3 / 6×6（非均质 XS）
│   ├── tools/        #   probe/chain/绘图等辅助脚本
│   └── logs/         #   历史扫参日志
├── paper/            # 论文：main.tex + appendix.tex + figures/ + 三格式交付（pdf/md/docx）
├── tests/            # 回归套件：run_examples.py + test_memopt.py + test_tile.py
│                     #   + verify_rect_node.py + baseline_keff.json（位级基线）
├── scripts/          # C5G7 规格/截面生成脚本
├── snapshot/         # 原始复现快照（core/ + drivers/ + 数据/图/日志，248 点）
├── requirements.txt
└── README.md
```

## 8. 方法说明

PSN（Phase Space Nodal，相空间节点法）节点法在每个节点内以四次抛物线插值表示通量，
通过节点内消元得到面流与节点源之间的显式关系，再在节点边界耦合。
相比传统 SN 差分，PSN 用更少的空间网格达到同等精度；相比 MOC，PSN 不依赖
追踪方向在网格上的投影，对复杂网格更稳健。矩形节点扩展把每节点的 x/y 宽度
作为独立几何参数进入全部闭式（衰减常数、面流、面通量、体矩），使非均质
7 群材料截面可直接赋给显式燃料/水矩形节点。细节见论文与 `psn2d/solver.py`、
`psn2d/node_rect.py` 注释（含论文式号 B.3 的抛物线源更新）。

## 9. 并行与部署

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
