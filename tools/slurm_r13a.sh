#!/bin/bash
#SBATCH -J r13a
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 40:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/r13a_%A_%a.out
export LC_ALL=C
PY=/work/home/xycjscs/mf/envs/psn/bin/python
export PSN_REPO=/work/home/xycjscs/OpenPSN
export THREADS=64 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
export OPT=mmd PSN_PAR=1 MEM_LIMIT_GB=700
cd /work/home/xycjscs/psn_strict
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  export MAT=mox
  $PY -u c5g7_one.py 96 3 strict_phase_space
else
  export MAT=uo2
  $PY -u c5g7_one.py 16 6 strict_phase_space
fi
