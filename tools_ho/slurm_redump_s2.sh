#!/bin/bash
#SBATCH --job-name=psn_s2_redump
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/%x_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/%x_%j.err

# Redo the 2 points that finished before the dump-capable hpc_point.py
# was synced (UO2 S2 M12/M16): keff values are already in the main batch
# log (1.3346627 / 1.3346979); these re-runs regenerate them AND dump
# flux+fission via psn2d.dump (user 2026-09-15).
set -e
mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
D=examples/c5g7/study2_rectangular

echo "=== env ==="
hostname; nproc
$PY -c "import numpy, numba; print('stack', numpy.__version__, numba.__version__)"

echo "=== R17UO2 S2 M12 (re-dump) ==="
$PY -u tools_ho/hpc_point.py R17UO2 $D/c5g7_rect17_split4_uo2_assembly.yaml 12 2 gmres 96 1.3347260
echo "=== R17UO2 S2 M16 (re-dump) ==="
$PY -u tools_ho/hpc_point.py R17UO2 $D/c5g7_rect17_split4_uo2_assembly.yaml 16 2 gmres 96 1.3347260

echo "=== REDUMP DONE ==="
