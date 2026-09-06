# PSN 复现快照 (2026-09-06)

Chao, Li & Chen, "Diffusion-based phase space nodal method (PSN): Solving the
neutron transport equation with a diffusion code", Annals of Nuclear Energy 240
(2027) 112707 —— 独立 Python 实现的论文结果复现快照。

## 复现结论

| 基准 | 点数 | 平均\|Δ\| | 最大\|Δ\| |
|---|---|---|---|
| Fig.3-A 棋盘弱吸收 (generic I=30) | 60 | 0.7 pcm | 1.2 pcm |
| Fig.3-B 棋盘强吸收 (generic I=30) | 60 | 0.3 pcm | 1.4 pcm |
| Fig.4-A restricted 弱 M=24 | 4 | — | ≤70 pcm (图读误差内) |
| Fig.5-A restricted 强 M=24 | 4 | — | ≤30 pcm (图读误差内) |
| Fig.10 BWR 束组件无 Gd (2 组) | 60 | 3.3 pcm | 3.8 pcm |
| Fig.12 BWR 束组件含 Gd (2 组) | 60 | 谷线形态 + Sec 4.4 判据点 +17.4 pcm | — |

口径: pcm = (k − kref) × 10⁵（绝对差）。全部 348 点单进程串行完成，
峰值内存 ≤ 2.5 GB，无并行无 GPU。

## 目录

```
core/
  psn_node.py      节点模型（generic βᵢ 修正闭式 + TY 3 点 restricted，闭式解，附录 A/B 全公式）
  psn_solver.py    PSN2D 全局求解器（稀疏 COO 组装、向量化节点状态、多群 keff 幂迭代、
                   generic/restricted、reflect/vacuum 边界、2.16 源更新）
drivers/
  run_checkerboard.py    Fig.3 棋盘问题（1 组，2×2 基本块 × S 细分）
  run_fig3_fill.py       Fig.3 补行 S=3,5,6,7,8,9 串行驱动（可续跑）
  run_bwr_seq.py         BWR 120 点串行驱动（双路 fsync 落盘、可续跑）
  test_bwr.py            BWR 2 组基准（截面 = 论文 Table 2；sanity/diag/full）
  verify_beta_decisive.py   generic βᵢ ground truth（直接四重积分）
  verify_vectorized_equiv.py 快/慢版等价性验证（5 case, ~1e-15）
data/
  report_data.json    全部复现数据的汇总 JSON（fig3 全表 / bwr 全表 / 参照值 / 速度）
  *.log               各点原始运行日志
```

## 运行

```
python3 -            # 无外部依赖问题: 需 numpy scipy
drivers/test_bwr.py sanity      # BWR N=1 M=4 无 Gd，期望 err ≈ +123 pcm (论文)
drivers/run_checkerboard.py 12  # Fig.3 棋盘 M=12（weak + strong）
```

## 关键实现要点（复现时踩过的坑）

1. **pcm 是绝对差** (k−kref)×10⁵，不是相对差。
2. **generic βᵢ (2.8d)**: 印刷式 1 − ½(cos²θᵢcosΔθᵢ + cos2θᵢcos²(Δθᵢ/2)) 与闭式
   βᵢ = 1 − ¼cosΔθᵢ − ¼cos2θᵢ − ½cos2θᵢcosΔθᵢ 代数恒等（最大差 2.2e-16）；
   早期 PDF 文本层提取的 1−0.5cos2θ(cosΔθ+cos²(Δθ/2)) 是乱码，勿用。
3. **稀疏组装**: 系统矩阵每行仅 ~9 非零；稠密 np.zeros 在 N=10 时需 159 GB
   （OOM 真凶）。COO/CSC 组装后任意 N 安全。
4. **向量化节点状态**: node_state 对 (J4,q) 严格线性，用 9 次基向量探测缓存
   9×9 线性映射后批量矩阵乘。注意 b 项分组必须按 (i,g) 系统分开缓存。
5. **多群源更新 (B.3)**: qmom_g = Σ_g2 [Sgg[g,g2] + χ_g·νΣf_g2/λ]·mom_g2，
   裂变源取局部抛物面矩（与通量同空间形状），按 χ 谱分布。

## 参考值 (kref)

- Fig.3 棋盘: weak = 1.12974, strong = 0.51673 (OpenMC)
- BWR 束组件: 无 Gd = 1.18797, 含 Gd = 0.86688 (OpenMC, Stepanek 1982)
