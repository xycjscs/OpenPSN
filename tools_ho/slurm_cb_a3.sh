#!/bin/bash
#SBATCH --job-name=psn_cb_a3_$1
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_cb_a3_$1_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_cb_a3_$1_%j.err
#
# Checkerboard (Chao paper Fig.3) A=3 directional S x M sweep.
# $1 = weak | strong ; 36 points each: S=1..6 x M=2 4 8 12 16 24.
# TY3 + A=3 directional matrix closure, GMRES pool 32 workers,
# 1e-7 gate (commit 79fda66), flux+fission dump built in (hpc_point.py).
# Rect formulation (uniform 1 cm nodes) — verified bit-identical to the
# square path at legacy closure (dk ~2e-16); the A=3 high-order closure
# requires rectangular nodes.  block=S so pin dump = 2x2 base cells.
set -e
VARG=$1
case $VARG in
  weak)   YAML=examples/psn_repro/checkerboard_1g_rect.yaml;        TAG=CBW; KREF=1.12974 ;;
  strong) YAML=examples/psn_repro/checkerboard_1g_rect_strong.yaml; TAG=CBS; KREF=0.51673 ;;
  *) echo "usage: slurm_cb_a3.sh weak|strong"; exit 1 ;;
esac

mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python

echo "=== env (checkerboard A3 $VARG) ==="
hostname; nproc; free -g | head -2

for S in 1 2 3 4 5 6; do
  for M in 2 4 8 12 16 24; do
    $PY -u tools_ho/hpc_point.py $TAG $YAML $M $S gmres 32 $KREF $S || echo "POINT FAIL $TAG M${M}_S${S}"
  done
done

echo "=== CB $VARG ALL DONE ==="
