#!/bin/bash
#SBATCH -J pdbench
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /work/home/xycjscs/psn_runs/pdbench_%A.out
export LC_ALL=C
source $HOME/mf/bin/activate pd
$HOME/mf/envs/pd/bin/pip install pypardiso 2>&1 | tail -1
export MKL_NUM_THREADS=32 OMP_NUM_THREADS=32
cd /work/home/xycjscs/psn_strict
$HOME/mf/envs/pd/bin/python -u pd_microbench.py
