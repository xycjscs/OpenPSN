#!/bin/bash
#SBATCH -J r13b
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r13b_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
export OPT=mmd PSN_PAR=1 MEM_LIMIT_GB=700
export MAT=mox
cd /work/home/xycjscs/psn_strict
# 0: M48_S4  1: M48_S5  2: M48_S6  3: M96_S4
M=(48 48 48 96)
S=(4 5 6 4)
$PY -u c5g7_one.py ${M[$SLURM_ARRAY_TASK_ID]} ${S[$SLURM_ARRAY_TASK_ID]} strict_phase_space
