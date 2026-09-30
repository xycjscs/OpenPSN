#!/bin/bash
#SBATCH -J r11c
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r11c_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
# r11b task0 (M24_S5) OOM'd inside 380G cgroup with PSN_PAR=1: S=5 grid is
# (5/4)^2 = 1.56x the S=4 node count that needed ~<700G at M48. Use the
# proven r10 heavy tier.
export PSN_PAR=1 MEM_LIMIT_GB=700
cd /work/home/xycjscs/psn_strict
# full retry set: S=5,6 x M=24,48 (r11b array cancelled after task0 OOM)
T=$SLURM_ARRAY_TASK_ID
S=$((5 + T/2))
M_LIST=(24 48)
M=${M_LIST[$((T%2))]}
$PY -u c5g7_one.py $M $S strict_phase_space
