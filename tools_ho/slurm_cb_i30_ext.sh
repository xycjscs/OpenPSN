#!/bin/bash
#SBATCH --job-name=psn_cb_i30x_$1
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_cb_i30x_$1_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_cb_i30x_$1_%j.err
#
# Checkerboard (Chao paper Fig.3) A=3 directional S x M sweep — GENERIC I=30
# EXTENSION: S=13..20 x M=2 4 8 12 16 24 (48 pts each, max grid 40x40).
# Same protocol as slurm_cb_i30.sh.  A=3 directional matrix closure,
# GMRES pool 32 workers, 1e-7 gate, dump with block=S.
set -e
VARG=$1
case $VARG in
  weak)   YAML=examples/psn_repro/checkerboard_1g_rect.yaml;        TAG=CBW30; KREF=1.12974 ;;
  strong) YAML=examples/psn_repro/checkerboard_1g_rect_strong.yaml; TAG=CBS30; KREF=0.51673 ;;
  *) echo "usage: slurm_cb_i30_ext.sh weak|strong"; exit 1 ;;
esac

mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python

echo "=== env (checkerboard A3 $VARG I30 S13-20) ==="
hostname; nproc; free -g | head -2

for S in 13 14 15 16 17 18 19 20; do
  for M in 2 4 8 12 16 24; do
    $PY -u tools_ho/hpc_point.py $TAG $YAML $M $S gmres 32 $KREF $S generic || echo "POINT FAIL $TAG M${M}_S${S}"
  done
done

echo "=== CB $VARG I30 S13-20 ALL DONE ==="
