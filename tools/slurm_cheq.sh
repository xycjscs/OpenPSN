#!/bin/bash
#SBATCH -J cheq
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/cheq_%A.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=32 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
cd /work/home/xycjscs/psn_strict
$PY -u chol_equiv_test.py
