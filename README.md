# OpenPSN

**Transport accuracy, diffusion-code simplicity.**

OpenPSN is licensed under the [Apache License 2.0](LICENSE).

OpenPSN is a deterministic neutron transport solver built on the **Phase
Space Nodal method (PSN)**: it solves the transport equation with the same
machinery as a diffusion code, yet delivers transport-grade accuracy — with
**no ray effects**, by construction. Multi-group (1 / 2 / 7 groups
verified), 2-D Cartesian cores, a faithful, modernized reproduction of the
PSN nodal method (Chao, Li & Chen, *Annals of Nuclear Energy* 240 (2027)
112707), rebuilt as a clean, YAML-driven Python package **plus a pure-JS
in-browser engine**.

🌐 **Try it live:** <https://openpsn-ai.com> — the checkerboard benchmark
runs entirely in your browser (no server, no dependencies), converging
k<sub>eff</sub> to 10⁻¹⁰ in a few seconds, with the converged scalar-flux
field rendered on screen.

📄 **Paper (preprint):** *OpenPSN: a multi-group Phase Space Nodal solver
and its validation on the C5G7 MOX benchmark* —
<https://www.researchgate.net/publication/414038054_OpenPSN_a_multi-group_Phase_Space_Nodal_solver_and_its_validation_on_the_C5G7_MOX_benchmark>

**Python 启动即可**：`python -m psn2d run <problem.yaml>`。任意群数（1 / 2 / 7
群已验证），任意几何（材料网格 + 边界），角向/空间离散与收敛参数全部用户可配。

---

## 0. 在线演示（in-browser engine）

`docs/` 目录下是一个自包含的静态站点（GitHub Pages，<https://openpsn-ai.com>）：

- **psn.js** — PSN 通用模型的纯 JavaScript 移植（641 行，无依赖），与 Python
  参考实现逐点咬合（24/24 点，max Δk<sub>eff</sub> = 4.8×10⁻¹⁰）。
- 浏览器内实时求解 1 群棋盘基准：选吸收体强弱 / 空间细分 S / 方位角段 M /
  收敛容差，点 Run，几秒收敛到 10⁻¹⁰，并渲染收敛曲线与收敛后的标量通量场。
- 页面同时内嵌原论文（Chao et al. 2027）的图 1/3/6/7/8/10（角度收敛趋势、
  误差热图、射线效应 MOC vs PSN 对比、BWR 几何）与 OpenPSN 的 C5G7 功率分布，
  标注出处。
- 本地预览：`cd docs && python3 -m http.server 8899` →
  <http://127.0.0.1:8899/>（需 http，因 `fig3ref.json` 用 `fetch` 加载）。

---

## 1. 功能

- **任意群数多群输运**：1 群（Fig.3 棋盘问题）、2 群（BWR 束组件）、7 群
  （C5G7 MOX 基准）全部复现并与原文对齐。
- **PSN 节点法**：抛物线形节点内插（quartic parabola），节点内消元（intra-nodal
  elimination），面流耦合。
- **YAML 输入**：几何、截面、边界、算法参数、批量 case（M×S 扫参）一个文件描述。
- **稀疏 + 向量化**：节点耦合矩阵稀疏求解，多群循环向量化；继承快照版的
  107× 提速与 OOM 根治（见 `snapshot/`）。

## 2. 安装

```bash
pip install -r requirements.txt   # numpy, scipy, pyyaml
```
Python ≥ 3.9。纯 Python（无 C 扩展、无 MPI），单机即可。

## 3. 运行

```bash
python -m psn2d run examples/checkerboard_1g.yaml        # 1 群棋盘（Fig.3）
python -m psn2d run examples/bwr_bundle_2g.yaml          # 2 群 BWR 束（Fig.8/10）
python -m psn2d run examples/c5g7_2d_quarter_core.yaml   # 7 群 C5G7-2D 1/4 芯
python -m psn2d run examples/c5g7_uo2_assembly.yaml      # 7 群单 UO2 组件
python -m psn2d run <file>.yaml --case M12_S1            # 只跑指定 case
python -m psn2d run <file>.yaml --json                   # 结果输出 JSON
python -m psn2d run <file>.yaml --opt auto               # 内存优化后端（见下）
```

**可选：内存优化后端（`--opt`，默认 `off`，默认路径位级不变）**

```bash
python -m psn2d run examples/c5g7_2d_quarter_core.yaml --case M16_S2 --opt auto
```

| 值 | 后端 | 说明 |
|---|---|---|
| `off`（默认） | 原始路径 | COLAMD SuperLU，位级可复现 |
| `auto` | 共享 Cholesky → 角度 Schur → tile → 紧凑 MMD-LU → 原路径 | 内存感知：先按 `--mem-limit-gb` 估算各后端因子池，选装得下且最小的；全部装不下才 fail-loud（不构造装不下的因子） |
| `chol` | 紧凑组装 + SPD 行缩放 + xy 转置因子共享 + Eigen 稀疏 Cholesky（METIS） | 需编译 `psn2d/memopt_cxx/libeigen_chol.so`；C5G7 1/4 芯 M16 实测峰值 RSS ≈ 1/9，keff 不变（≤1e-15） |
| `angschr` | 角度 Schur 剖分：系统按方位角方向块对角化（16/192 个 2D 网格 Cholesky）+ 镜反射行的小角度 Schur 分量（544² 级） | 因子池 ∝ M 线性（tile 缝合面是 ∝ M²）：M192 组件 21 系统全池 ~48GB，62GB 机器可行；keff 与共享 Cholesky 逐位一致（M16 实测 Δk=2.2×10⁻¹⁶）；S 形成 8 路并行（实测 6.5×）；池超预算或几何非方向块对角时 fail-loud 证书 |
| `tile` | Schur 分块 + 每块 Eigen Cholesky（METIS）+ 稀疏多右端核 | 内存回退后端：按 `--mem-limit-gb` 自动选块大小，用户无需指定 tile 或 assembly；纯几何块划分，无 assembly 概念也能用（如棋盘基准）；keff 与原路径位级一致（实测 Δ=0.0000 pcm） |
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
boundaries:              # 四条边：reflective 或 vacuum
  left: reflective
  bottom: reflective
  right: reflective
  top: reflective
materials:
  - name: UO2
    Sigma_t:  [ ... ]    # (ng) 总宏观截面
    Sigma_s:  [ ... ]    # (ng, ng) 群散射矩阵
    nu_Sf:    [ ... ]    # (ng) 每群 νΣf
  - name: Water
    ...
  chi: [ ... ]           # (ng) 归一化中子源能谱
solver:
  max_outer: 3000
  keff_tol: 1e-10
cases:                   # 批量：M=方位角段数, S=空间细分, model=generic|ty3
  - { name: M8_S1,  model: generic,  M: 8,  subdivide: 1 }
  - { name: M12_S1, model: generic,  M: 12, subdivide: 1 }
  - { name: M12_S2, model: generic,  M: 12, subdivide: 2 }
```

- **model**：角度模型，`generic`（默认，论文 Sec.2 通用 PSN）或 `ty3`（TY 受限二维模型）。
- **M**：方位角离散段数（角向精度）。
- **S**：空间细分（每个材料网格再分为 S×S 子节点 → 空间精度）。
- **kref**（可选）：参考 k 值，给出后自动打印 Δ（pcm）。

## 5. 复现验证

| 问题 | 输入 | 结果 | 对照 |
|------|------|------|------|
| Fig.3 棋盘（1 群） | `checkerboard_1g.yaml` | 与快照逐点咬合 | 原文 Fig.3（<4 pcm） |
| BWR 2 群束 | `bwr_bundle_2g.yaml` | Δ=+28.3 pcm（N=2, M=8） | 原文 Fig.8/10 |
| C5G7-2D 1/4 芯（7 群，均质化） | `c5g7_2d_quarter_core.yaml` | k=1.18861（M12_S1） | 1.18646（McGraw PHYSOR 2014 高保真），+215 pcm |
| C5G7 单 UO2 组件（7 群，均质化） | `c5g7_uo2_assembly.yaml` | k=1.34030 | nTRACER 1.33367 |
| C5G7 单 UO2 组件（7 群，矩形节点非均质） | `c5g7_rect_uo2_assembly.yaml` | k=1.3343536（M16） | 自算 OpenMC 显式几何参考 1.3334947，+86 pcm |
| C5G7-2D 1/4 芯（7 群，矩形节点非均质） | `c5g7_rect_quarter_core.yaml` | k=1.1866067（M16, tol 1e-10） | 自算 OpenMC 显式几何参考 1.1864955（±3.4 pcm），+11 pcm |

完整 248 点复现数据与图见 `snapshot/`（原始快照仓库）。

回归套件 `tests/run_examples.py` 覆盖上表全部问题（fast 层 ~3 min，full 层另含
C5G7 1/4 芯均质化与矩形节点重 case），对照 `tests/baseline_keff.json` 的位级基线
（Δ 容差 1.0 pcm，`--record` 合并记录新基线，`--only <yaml 片段>` 过滤）：

```bash
python tests/run_examples.py          # fast 层
python tests/run_examples.py --full   # 含 full 层重 case
```

## 6. C5G7 基准结果

C5G7-2D MOX 燃料组件（51×51 pin，2×2 燃料块 + L 形水反射，7 群）：

```
矩形节点非均质（纯材料 XS，153×153 节点，3×3 每 pin）
  M=8    1.1853742   -112.1 pcm   (vs 自算 OpenMC 显式几何参考 1.1864955 ±3.4 pcm)
  M=12   1.1861084    -38.7 pcm
  M=16   1.1866067    +11.1 pcm   （进入 MC 统计误差带）

参考    1.18646    McGraw et al. (PHYSOR 2014) LDG 高保真值
                  (Rattlesnake 收敛极限 1.186446)
参考    1.18655    MCNP5，NEA 2003 原文 Table 3（偏高约 10 pcm）
```

pin 级 fission map 对比（M16，自算 OpenMC 显式几何参考场，20000 批 × 100k 粒子，collision）：
单组件 264 棒 mean −0.002% / RMS 0.070%；1/4 芯 1056 棒组内形状
RMS 0.26%（UO2 组 0.14–0.19%，MOX 组 0.32%）。

- 空间细化 S=1→2（2601→10404 节点）k 仅动 ~4 pcm → 离散已收敛。
- 独立代数 k∞ = 1.32936 与 SPHINCS/nTRACER 无自屏蔽值**逐位一致**（数据接线铁证）。
- 与 20 个确定性代码的官方对比见论文（NEA/NSC/DOC(2003)16 Table 17）。

论文：[OpenPSN: a multi-group Phase Space Nodal solver and its validation on the C5G7 MOX benchmark](https://www.researchgate.net/publication/414038054_OpenPSN_a_multi-group_Phase_Space_Nodal_solver_and_its_validation_on_the_C5G7_MOX_benchmark)（ResearchGate 预印本，2026）

## 7. 目录结构

```
OpenPSN/
├── psn2d/            # 主包：现代多群 PSN（YAML 驱动）
│   ├── __main__.py   #   CLI 入口 (run, --opt 内存优化后端)
│   ├── model.py      #   YAML 解析
│   ├── solver.py     #   多群 PSN 节点求解器（稀疏+向量化）
│   ├── node.py       #   节点内插/消元
│   ├── node_rect.py  #   矩形节点闭式解
│   ├── memopt.py     #   可选内存优化后端（紧凑组装/共享Cholesky, --opt）
│   ├── angschr.py    #   角度 Schur 剖分后端（方向块对角化, --opt angschr）
│   ├── tile.py       #   Schur 分块内存回退后端（--opt tile）
│   └── memopt_cxx/   #   C++ Cholesky 桥（eigen_chol.cpp + build.py, 按需编译）
├── docs/             # 在线演示站（GitHub Pages → openpsn-ai.com）
│   ├── index.html    #   产品页 + 浏览器内求解器
│   ├── psn.js        #   纯 JS PSN 引擎（无依赖）
│   ├── fig3ref.json  #   论文 Fig.3 60 点复现数据（页面比对用）
│   └── img/          #   内嵌插图（原论文 Fig.1/3/6/7/8/10 + C5G7 功率）
├── examples/         # 论文问题输入（1群/2群/7群）+ C5G7 数据与脚本
├── tests/            # 回归套件：run_examples.py + test_memopt.py + baseline_keff.json（位级基线）
├── snapshot/         # 原始复现快照（core/ + drivers/ + 数据/图/日志）
├── requirements.txt
└── README.md
```

## 8. 方法说明

PSN（Phase Space Nodal，相空间节点法）节点法在每个节点内以四次抛物线插值表示通量，
通过节点内消元得到面流与节点源之间的显式关系，再在节点边界耦合。
相比传统 SN 差分，PSN 用更少的空间网格达到同等精度；相比 MOC，PSN 不依赖
追踪方向在网格上的投影，对复杂网格更稳健。细节见原文与 `psn2d/solver.py`
注释（含论文式号 B.3 的抛物线源更新）。

## 9. 并行与部署

- **sweep 并行 = fork 进程池**（SuperLU 回代不释放 GIL，线程池实测更慢）。
  并行度默认 = 可用核数的一半（cgroup 感知），`PSN_PAR=1` 强制串行；
  小问题（ncol < 15 万）自动串行。
- **BLAS 恒单线程**：`psn2d/__init__.py` 在 import 时钉 OPENBLAS/OMP=1
  （bit-identical 回归的前提）。
- **sweep 共享缓冲 = POSIX shm 专用（无磁盘 fallback）**：源项/归约走
  `/dev/shm`（父进程付 ~MB 级拷贝）。两段（q、r）整个运行期共存，
  故安装时按 **q+r 合计** ≤ tmpfs 空闲判（tmpfs 按实际触碰页记账，
  逐段检查有竞态——C5G7 core M4_S5：q=52 MiB + r=17 MiB 各自过检、
  合计 69 MiB 超 64 MiB 上限，worker 尾写 SIGBUS、进程池挂起）。
  **不足直接 fail-loud**（`MemoryError`，附扩容指引）：扫算是重活，
  部署机必须配足 `/dev/shm`（core 最大点 S=6 需 ~100 MiB；64 MiB
  的 Docker 默认装不下 core S≥5，建议 `--shm-size=4g`）。确实要在这
  类小 shm 机器上跑，用 `PSN_PAR=1` 串行（跳过进程池及其共享段）。
- **内存预算按机器总占用算**：父进程 RSS + fork worker 私有状态
  （COW 只共享读因子，私有部分是角度通量/局部归约）。62 GB 机器上
  core M12_S2 plain 全程约 41 GB，mmd/chol 后端低 2.6–8.8×。
