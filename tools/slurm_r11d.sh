#!/bin/bash
#SBATCH -J r11d
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r11d_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
# M48_S6 = largest UO2 point yet; plain-LU OOM'd at 700G cgroup (r11c task3).
# mmd compact SuperLU: bit-identical (cheq2, |diff|=0), 3.2x lower peak on
# cheq case. Same backend as r12c for cross-case consistency.
export OPT=mmd PSN_PAR=1 MEM_LIMIT_GB=700
cd /work/home/xycjscs/psn_strict
$PY -u c5g7_one.py 48 6 strict_phase_space
