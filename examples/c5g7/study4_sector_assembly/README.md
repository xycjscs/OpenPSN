## Study IV C5G7 组件

方形 UO₂、方形 MOX、矩形 UO₂、矩形 MOX 均使用 `sector_diffusion`。每个组件有
36 格，取 `S=1–6`、`M=2、4、8、12、16、24`、`I=8`、`A=1`。实测 `k_eff`、
偏差、时间、后端和复扫残差见[逐点 CSV](../../../paper/data/study4_assembly_points.csv)。
[误差矩阵](../../../paper/figures/study4_assembly_grid2x2.pdf)与
[墙钟时间矩阵](../../../paper/figures/study4_assembly_wall.pdf)使用相同布局和最佳点标记。

[四组最佳点图](../../../paper/figures/study4_best_fission.pdf)按最小 `|Δk|` 选点。
四个 `*_best_pin_fission.csv` 给出完整的 17×17 棒元字段。相对误差只在每例的
264 根燃料棒统计。方形使用逐棒中子产生率。矩形延续原图口径，用材料平均 `ν`
转换为裂变率。各图扣除最小值并归一化为单位棒元总和。方形 OpenMC 参考取自
Study III 四组逐棒对照归档中的 `OpenMC` 列，矩形直接读取仓内原参考数组。

在仓库根目录执行以下命令可由入仓数据重建组件表与 SM 图。

```sh
python -m scripts.study4_assembly_summary --closure sector_diffusion
python -m scripts.study4_assembly_report --closure sector_diffusion \
    --table paper/tables/study4-assembly-sector.tex --label tab:study4-assembly-sector
python -m scripts.study4_sm_figures
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

用收敛检查点重建逐棒图时，将检查点目录传给 `--fields`。下列命令会验证每张图的
`S`、`M`、闭合类型和 `k_eff`，再从对应检查点取棒元数据。

```sh
python -m scripts.study4_assembly_summary --closure sector_diffusion \
    --fields runs/study4-uniform/fields --pin-output runs/study4-assembly-maps
```
