#!/bin/bash
#SBATCH --job-name=psn_s2_batch
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/%x_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/%x_%j.err

# psn_s2_batch: the 6 missing S2 points (M12/16/24 x UO2/MOX), GMRES path
# (user decision 2026-09-15: LU too slow, use GMRES).  96 workers,
# pairs+numba, production acceptance gate 1e-7 equalized.
set -e
mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
D=examples/c5g7/study2_rectangular

echo "=== env ==="
hostname; nproc; free -g | head -2
$PY -c "import numpy, numba; print('stack', numpy.__version__, numba.__version__)"

echo "=== R17UO2 S2 M12 ==="
$PY -u tools_ho/hpc_point.py R17UO2 $D/c5g7_rect17_split4_uo2_assembly.yaml 12 2 gmres 96 1.3347260
echo "=== R17UO2 S2 M16 ==="
$PY -u tools_ho/hpc_point.py R17UO2 $D/c5g7_rect17_split4_uo2_assembly.yaml 16 2 gmres 96 1.3347260
echo "=== R17UO2 S2 M24 ==="
$PY -u tools_ho/hpc_point.py R17UO2 $D/c5g7_rect17_split4_uo2_assembly.yaml 24 2 gmres 96 1.3347260
echo "=== R17MOX S2 M12 ==="
$PY -u tools_ho/hpc_point.py R17MOX $D/c5g7_rect17_split4_mox_assembly.yaml 12 2 gmres 96 1.1851053
echo "=== R17MOX S2 M16 ==="
$PY -u tools_ho/hpc_point.py R17MOX $D/c5g7_rect17_split4_mox_assembly.yaml 16 2 gmres 96 1.1851053
echo "=== R17MOX S2 M24 ==="
$PY -u tools_ho/hpc_point.py R17MOX $D/c5g7_rect17_split4_mox_assembly.yaml 24 2 gmres 96 1.1851053

echo "=== BATCH DONE ==="
