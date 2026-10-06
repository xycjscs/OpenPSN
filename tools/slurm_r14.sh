#!/bin/bash
#SBATCH -J r14
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r14_%A_%a.out
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
export OPT=mmd PSN_PAR=1 MEM_LIMIT_GB=700
export MAT=mox
cd /work/home/xycjscs/psn_strict
# 0: M96_S5   1: M96_S6   2: M192_S4  3: M192_S5  4: M192_S6
M=(96 96 192 192 192)
S=(5 6 4 5 6)
$PY -u c5g7_one.py ${M[$SLURM_ARRAY_TASK_ID]} ${S[$SLURM_ARRAY_TASK_ID]} strict_phase_space
