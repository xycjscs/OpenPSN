#!/bin/bash
#SBATCH --job-name=psn_mox_s2
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_mox_s2_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_mox_s2_%j.err

# MOX S2 point M=$1 (12/16/24), one node per job, 3 jobs parallel
# (2026-09-15: HPC allows multiple simultaneous jobs; main batch
# 44370803 will be cancelled after its M24_UO2 completes).
# GMRES, 96 workers, pairs+numba, 1e-7 gate, flux+fission dump built in.
set -e
M=$1
mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
D=examples/c5g7/study2_rectangular

echo "=== env (MOX S2 M$M) ==="
hostname; nproc

echo "=== R17MOX S2 M$M ==="
$PY -u tools_ho/hpc_point.py R17MOX $D/c5g7_rect17_split4_mox_assembly.yaml $M 2 gmres 96 1.1851053

echo "=== MOX M$M DONE ==="
