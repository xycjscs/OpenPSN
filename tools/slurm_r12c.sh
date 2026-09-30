#!/bin/bash
#SBATCH -J r12c
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r12c_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
export MAT=mox
# MOX M192_S3: plain LU OOM'd at 865G (r12b, seff measured). compact-MMD
# SuperLU verified bit-identical (cheq2 45492993, |diff|=0) with large peak
# reduction (M24_S2: 13.8G mmd). MMD perm also 2.8x faster on cheq case.
export OPT=mmd PSN_PAR=1 MEM_LIMIT_GB=700
cd /work/home/xycjscs/psn_strict
$PY -u c5g7_one.py 192 3 strict_phase_space
