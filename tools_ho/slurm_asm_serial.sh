#!/bin/bash
#SBATCH --job-name=psn_asm_serial
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_asm_serial_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_asm_serial_%j.err

# Generic assembly batch (2026-09-15: all assembly points moved to HPC).
# Usage: sbatch slurm_asm_serial.sh <uo2|mox> <S> <M...>
#  e.g. sbatch slurm_asm_serial.sh uo2 1 2 4 8 12 16 24
# GMRES, 96 workers, pairs+numba, 1e-7 gate, flux+fission dump built in.
set -e
MAT=$1; S=$2; shift 2; MLIST="$@"
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
D=examples/c5g7/study2_rectangular

if [ "$MAT" = "uo2" ]; then
  TAG=R17UO2; Y=$D/c5g7_rect17_split4_uo2_assembly.yaml; KREF=1.3347260
else
  TAG=R17MOX; Y=$D/c5g7_rect17_split4_mox_assembly.yaml; KREF=1.1851053
fi

echo "=== env ($TAG S$S M:$MLIST) ==="
hostname; nproc

for M in $MLIST; do
  echo "=== $TAG S$S M$M ==="
  $PY -u tools_ho/hpc_point.py $TAG $Y $M $S gmres 96 $KREF
done

echo "=== BATCH $TAG S$S DONE ==="
