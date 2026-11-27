## Study IV C5G7 组件

方形 UO₂、方形 MOX、矩形 UO₂、矩形 MOX 均使用 `sector_diffusion`。每个组件有
36 格，取 `S=1–6`、`M=2、4、8、12、16、24`、`I=8`、`A=1`。实测 `k_eff`、
偏差、时间、后端和复扫残差见[逐点 CSV](../../../paper/data/study4_assembly_points.csv)。
[误差矩阵](../../../paper/figures/study4_assembly_grid2x2.pdf)与
[墙钟时间矩阵](../../../paper/figures/study4_assembly_wall.pdf)使用相同布局和最佳点标记。

[四组组件逐棒图](../../../paper/figures/study4_assembly_fission.pdf)统一使用已计算的
最细网格 `S=6, M=24`。四组组件和四分之一堆芯的相对误差共用 −0.35% 至 +0.35%
的色标。四个 `*_finest_pin_fission.csv` 给出图中的完整 17×17 棒元字段。
`*_best_pin_fission.csv` 记录按最小 `|Δk|` 选出的表格数据。相对误差只在每例的
264 根燃料棒统计。方形使用逐棒中子产生率。矩形延续原图口径，用材料平均 `ν`
转换为裂变率。各图扣除最小值并归一化为单位棒元总和。方形 OpenMC 参考取自
Study III 四组逐棒对照归档中的 `OpenMC` 列，矩形直接读取仓内原参考数组。

方形 UO₂ 的 36 格逐棒 RMS 见 [S×M 矩阵 CSV](../../../paper/data/study4_square_uo2_pin_rms.csv)
和[误差图](../../../paper/figures/study4_square_uo2_pin_rms.pdf)。
RMS 在 264 根燃料棒上计算，每个棒元的相对偏差取百分数。
最小 RMS 为 0.02678%，对应 `S=6, M=16`。

在仓库根目录执行以下命令可由入仓数据重建组件表与 SM 图。

```sh
python -m scripts.study4_assembly_summary --closure sector_diffusion
python -m scripts.study4_assembly_report --closure sector_diffusion \
    --table paper/tables/study4-assembly-sector.tex --label tab:study4-assembly-sector
python -m scripts.study4_sm_figures
python -m scripts.study4_pin_maps --output paper/figures
python -m scripts.study4_pin_rms --output paper/figures \
    --latex paper/tables/study4-square-uo2-pin-rms.tex
```

最后 18 个矩形格用 packed Cholesky 预条件器和残差检验的 FP64 PCG 求解，线程上限为
60。外层本征值和独立复扫源的相对残差门槛为 `1e-9`、`5e-9`。队列会从已完成的
逐点记录或检查点续跑。较密网格的内层残差设置见
[方法文档](../../../docs/psn4-sector-diffusion.md)。

```sh
OPENPSN_PREFACTOR_WORKERS=6 OPENPSN_PACKED_RTOL=1e-11 OPENPSN_PACKED_MAXITER=160 \
python -m scripts.study4_uniform_grid run --threads 60 --backend chol-packed-rs \
    --plan paper/data/uniform/plan_packed.json --state runs/study4-uniform \
    --yield-to none --heavy-gib 42
```

部分实测点提高了同时分解的因子数。线程设置和已记录的分解并发度见
[`points.jsonl`](../../../paper/data/uniform/points.jsonl)。

用收敛检查点重建组件逐棒 CSV 时，将检查点目录传给 `--fields`。命令会验证对应
检查点的 `S`、`M`、闭合类型和 `k_eff`。

```sh
python -m scripts.study4_pin_maps --fields runs/study4-uniform/fields \
    --assembly-data runs/study4-assembly-maps --output runs/study4-assembly-maps
```

把同一组检查点通过 `--fields` 传给 `scripts.study4_pin_rms`，可以从 36 个通过复扫的
检查点重新计算矩阵。
