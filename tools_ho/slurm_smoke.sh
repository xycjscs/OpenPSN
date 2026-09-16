#!/bin/bash
#SBATCH --job-name=psn_smoke
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=/work/home/xycjscs/psn_runs/%x_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/%x_%j.err

# psn_smoke: verify the HPC stack end-to-end (UO2 assembly, S1 M2)
#   1) gmres path (production config D: pairs+numba, pool)
#   2) lu path    (global face-current SuperLU, all 210 (i,g) factors)
set -e
mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
SPEC=/work/home/xycjscs/OpenPSN/examples/c5g7/study2_rectangular/c5g7_rect17_split4_uo2_assembly.yaml

echo "=== env ==="
hostname; nproc; free -g | head -2
$PY -c "import numpy, scipy, numba, h5py; print('stack', numpy.__version__, numba.__version__)"

echo "=== GMRES S1 M2 (128 threads) ==="
$PY -u tools_ho/hpc_point.py R17UO2 $SPEC 2 1 gmres 128 1.3347260

echo "=== LU S1 M2 (128 threads) ==="
$PY -u tools_ho/hpc_point.py R17UO2 $SPEC 2 1 lu 128 1.3347260

echo "=== SMOKE DONE ==="
