#!/bin/bash
#SBATCH -J parab
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=900G
#SBATCH -t 12:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/parab_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=32 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
# A/B vs r11c task0 serial baseline: M24_S5 strict, PSN_PAR=1, t=16415s,
# keff=1.3340875. Here: mmd + process pool (PSN_PAR=8). Doc claims factors
# are prefactorized in the parent and COW-shared (zero copy); re-measure
# peak and time. PSN_PAR=0 -> auto (min(8, jobs)).
export OPT=mmd PSN_PAR=8 MEM_LIMIT_GB=700
cd /work/home/xycjscs/psn_strict
/usr/bin/time -v $PY -u c5g7_one.py 24 5 strict_phase_space 2>&1
