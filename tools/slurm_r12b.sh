#!/bin/bash
#SBATCH -J r12b
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r12b_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
# r12 task3 (M192_S3) OOM'd at 700G limit inside 900G cgroup with PSN_PAR=8.
# Serialize; proven r10 pattern. MAT=mox -> kref 1.1851053 (post-patch).
export PSN_PAR=1 MEM_LIMIT_GB=700
export MAT=mox
cd /work/home/xycjscs/psn_strict
$PY -u c5g7_one.py 192 3 strict_phase_space
