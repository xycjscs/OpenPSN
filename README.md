# OpenPSN

**Transport accuracy, diffusion-code simplicity.**

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
cases:                   # 批量：M=方位角段数, S=空间细分, model=psn|generic
  - { name: M8_S1,  model: psn,      M: 8,  subdivide: 1 }
  - { name: M12_S1, model: psn,      M: 12, subdivide: 1 }
  - { name: M12_S2, model: psn,      M: 12, subdivide: 2 }
```

- **M**：方位角离散段数（角向精度）。
- **S**：空间细分（每个材料网格再分为 S×S 子节点 → 空间精度）。
- **kref**（可选）：参考 k 值，给出后自动打印 Δ（pcm）。

## 5. 复现验证

| 问题 | 输入 | 结果 | 对照 |
|------|------|------|------|
| Fig.3 棋盘（1 群） | `checkerboard_1g.yaml` | 与快照逐点咬合 | 原文 Fig.3（<4 pcm） |
| BWR 2 群束 | `bwr_bundle_2g.yaml` | Δ=+28.3 pcm（N=2, M=8） | 原文 Fig.8/10 |
| C5G7-2D 1/4 芯（7 群） | `c5g7_2d_quarter_core.yaml` | k=1.18861（M12_S1） | 1.18646（McGraw PHYSOR 2014 高保真），+215 pcm |
| C5G7 单 UO2 组件 | `c5g7_uo2_assembly.yaml` | k=1.34030 | nTRACER 1.33367 |

完整 248 点复现数据与图见 `snapshot/`（原始快照仓库）。

## 6. C5G7 基准结果

C5G7-2D MOX 燃料组件（51×51 pin，2×2 燃料块 + L 形水反射，7 群）：

```
PSN2D   1.18861   +215 pcm   (vs 1.18646)
OpenMOC 1.18582   -64  pcm   (本机实测，官方 c5g7-2d.py 几何)
参考    1.18646    McGraw et al. (PHYSOR 2014) LDG 高保真值
                      (Rattlesnake 收敛极限 1.186446)
参考    1.18655    MCNP5，NEA 2003 原文 Table 3（偏高约 10 pcm）
```

- 空间细化 S=1→2（2601→10404 节点）k 仅动 ~4 pcm → 离散已收敛。
- 独立代数 k∞ = 1.32936 与 SPHINCS/nTRACER 无自屏蔽值**逐位一致**（数据接线铁证）。
- 与 20 个确定性代码的官方对比见论文（NEA/NSC/DOC(2003)16 Table 17）。

## 7. 目录结构

```
OpenPSN/
├── psn2d/            # 主包：现代多群 PSN（YAML 驱动）
│   ├── __main__.py   #   CLI 入口 (run)
│   ├── model.py      #   YAML 解析
│   ├── solver.py     #   多群 PSN 节点求解器（稀疏+向量化）
│   └── node.py       #   节点内插/消元
├── docs/             # 在线演示站（GitHub Pages → openpsn-ai.com）
│   ├── index.html    #   产品页 + 浏览器内求解器
│   ├── psn.js        #   纯 JS PSN 引擎（无依赖）
│   ├── fig3ref.json  #   论文 Fig.3 60 点复现数据（页面比对用）
│   └── img/          #   内嵌插图（原论文 Fig.1/3/6/7/8/10 + C5G7 功率）
├── examples/         # 论文问题输入（1群/2群/7群）+ C5G7 数据与脚本
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
