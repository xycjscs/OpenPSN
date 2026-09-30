#!/bin/bash
#SBATCH -J r11b
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=380G
#SBATCH -t 24:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r11b_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
# r11 array-parallel OOM'd (8 fork-workers x up-to-180G peak > 300G cgroup).
# r10 serial pattern proven at M48_S4 with 700G: serialize inside each task,
# cgroup-safe limit under 380G node.
export PSN_PAR=1 MEM_LIMIT_GB=350
cd /work/home/xycjscs/psn_strict
# C5G7 UO2 strict rows S=5,6 x M=24,48 (retry of r11 tasks 2,3,5,6)
T=$SLURM_ARRAY_TASK_ID
S=$((5 + T/2))
M_LIST=(24 48)
M=${M_LIST[$((T%2))]}
$PY -u c5g7_one.py $M $S strict_phase_space
