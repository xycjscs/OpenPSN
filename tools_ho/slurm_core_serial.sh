#!/bin/bash
#SBATCH --job-name=psn_core_serial
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_core_serial_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_core_serial_%j.err

# Core rect4 (204x204, 51x51 pins 2x2) full sweep, 2026-09-15.
# Usage: sbatch slurm_core_serial.sh <S> <M...>
#  e.g. sbatch slurm_core_serial.sh 1 2 4 8 12 16 24
# GMRES, 96 workers, pairs+numba, 1e-7 gate, flux+fission dump built in.
# kref = 1.1872176 (OpenMC quarter-core anchor).
set -e
S=$1; shift; MLIST="$@"
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
D=examples/c5g7/study2_rectangular
TAG=CORE4
Y=$D/c5g7_core_split4_uo2_mox_quarter_core.yaml
KREF=1.1872176

echo "=== env ($TAG S$S M:$MLIST) ==="
hostname; nproc; free -g | head -2

for M in $MLIST; do
  echo "=== $TAG S$S M$M ==="
  $PY -u tools_ho/hpc_point.py $TAG $Y $M $S gmres 96 $KREF
done

echo "=== BATCH $TAG S$S DONE ==="
