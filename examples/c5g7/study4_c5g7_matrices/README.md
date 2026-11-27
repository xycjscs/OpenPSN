## Study IV C5G7 完整 S×M 矩阵

[矩阵 PDF](study4_c5g7_matrices.pdf)每例一页。[逐格表](sm_matrices.md)列出全部格点。
数据共 174 格，四组组件分别为 `S=1–6`、每组 36 格，矩形四分之一堆芯为
`S=1–5`、30 格。每组取 `M=2、4、8、12、16、24`，并固定 `A=1`。组件用
`I=8`，堆芯用 TY3 极角规则。

- `*_keff.csv` 给出原始本征值，`*_delta_k_pcm.csv` 给出相对各自同几何 OpenMC
  参考的带符号偏差，`*_pin_rms_percent.csv` 给出逐棒相对误差 RMS（%）。
- 组件 RMS 在各自 264 根燃料棒上计算，沿用逐棒图对场扣除最小值并归一化的口径。
  方形采用中子产生率，矩形采用材料平均中子产额换算的裂变率。
  四分之一堆芯用逐群裂变截面计算裂变率，归一化为单位总和，在 1056 根燃料棒上求 RMS。
  每根燃料棒的相对偏差为 `100 × (OpenPSN / OpenMC − 1)`。
- PDF 中同单位的误差面板共用色标。边框按未舍入值标出该例最小绝对本征值偏差和最小逐棒 RMS。
  [组件耗时](../../../paper/data/study4_assembly_points.csv)和
  [堆芯耗时](../study4_sector_core/sm_wall_seconds.csv)采用各格实测墙钟。

### 重建矩阵

在仓库根目录运行以下命令，即可从已归档的数据重建 15 张 CSV、Markdown 和 PDF。

```sh
python -m scripts.study4_c5g7_matrices --output runs/study4-c5g7-matrices
```

用 `--fields` 指定组件检查点目录可重新计算四组组件 RMS。每格使用对应复扫合格的
检查点和同一个 OpenMC 参考。堆芯逐棒数据的来历见
[`core_pin_provenance.csv`](core_pin_provenance.csv)。24 格使用原检查点的复扫通量，
4 格从已存收敛源反演标量通量，2 格补做同配置求解以保存复扫通量。补算的本征值通过
原归档本征值的 0.001 pcm 一致性检查。堆芯本征值矩阵使用原始计时批次的已归档本征值，
RMS 文件另记实际使用的场对应的本征值。[两个补算的收敛记录](core_field_reruns.jsonl)。

从堆芯检查点重新生成逐棒数据来源表时，准备带 `S,M,checkpoint,run_log` 四列的
清单。路径相对于清单所在目录。带有内嵌求解记录的检查点可留空 `run_log`，
只保存源的检查点须填写对应求解日志。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.study4_core_pin_rms \
    --manifest runs/core-checkpoints.csv --output runs/core_pin_provenance.csv
python -m scripts.study4_c5g7_matrices --core-provenance runs/core_pin_provenance.csv \
    --output runs/study4-c5g7-matrices
```
