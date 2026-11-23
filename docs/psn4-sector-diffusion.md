# PSN-4 扇区扩散验证

## 有限角区平均闭合

`sector_average` 在每个有限极角带与方位扇区的交集上保留一个偶宇称场，
直接积分二阶张量 `⟨Ω Ωᵀ⟩` 和真空边界系数 `⟨|Ω·n|⟩`，
角区之间通过标量通量源和镜面反射边界耦合。极角带沿 θ 等宽，
积分权重按 `d(cos θ) dφ` 计算。体扩散和材料界面法向流共用完整张量的
双二次弱式。有限宽角区的二维扩散张量有两个正特征值。

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

## 方形棋盘格

用论文 Fig. 3 对应的二维棋盘格几何，比较强吸收基线以及强、弱两种吸收的新闭合。
两种材料布局均测试 `S = 2, 4, 6, 8, 12` 与 `M = 4, 8, 12, 24` 的全部交叉点。
新闭合固定 `I=30, A=3`。参考是论文中的强吸收 `0.51673`、弱吸收
`1.12974`。误差定义为 `(k_eff - k_ref) × 10⁵`。

强吸收原版的过零位置随 S 移动。S=2 在 M=4–8 间过零，S=4 在 M=8–12 间，
S=6 和 S=8 在 M=12–24 间。新闭合的强、弱共 40 格均为负误差。在每一列
固定 M 时，两个算例的绝对误差都随 S 单调下降。联合加密如下。

| S | M | 强吸收 Δk (pcm) | 弱吸收 Δk (pcm) |
|--:|--:|-----------------:|-----------------:|
| 2 | 4 | −268.51 | −49.83 |
| 4 | 8 | −75.37 | −15.03 |
| 8 | 12 | −22.62 | −5.58 |
| 12 | 24 | −11.98 | −3.68 |

固定 S 的角向余差很小且存在轻微回摆，例如强吸收 S=12 随 M=4、8、12、24
得到 −12.30、−12.40、−12.20、−11.98 pcm。每扇区只用两个积分点的试算，
强吸收在 M=4 的数个测点仍出现过零。方形结果对应上表的角向设置和测试范围。

复现角网格和空间网格的全部交叉点。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
samples=(2,4 2,8 2,12 2,24 4,4 4,8 4,12 4,24 6,4 6,8 6,12 6,24
         8,4 8,8 8,12 8,24 12,4 12,8 12,12 12,24)
python -m scripts.sector_diffusion_benchmark strong --sm "${samples[@]}" --threads 8
python -m scripts.sector_diffusion_benchmark weak --sm "${samples[@]}" --threads 8
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

矩形测试固定 `I=8, A=1`。表中所有结果均通过独立复扫。

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
线性相对残差门槛。矩形本征计算的外层容差为 `1e-9`，独立复扫的源残差
门槛为 `5e-9`。直接 Cholesky 的 S=2、M=4 试算在大量换页下终止，
同离散的压缩 Cholesky 重跑收敛。S=2、M=8 的首次压缩试算在首扫中
触及旧线性门槛，调整后通过复扫。

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

本轮矩形算例达到的最小绝对偏差为 26.24 pcm，距约 1 pcm 的目标仍有差距。
当前 OpenMC 存档的两个估计量的标准误也都高于 1 pcm。

更细网格的内存抽测使用 `S=3, M=16, I=8, A=1` 的一个高热群镜像方位块。
该块有 1,687,285 个未知量，压缩因子和原算子模板共 188,911,315 字节，
原 FP64 方程复核的相对残差为 `8.12e-14`。若 224 块平均采用该存储量，
常驻池约为 39.4 GiB。该数字是按单块外推的估计，未计 Ritz 缓存和分解临时数组。
