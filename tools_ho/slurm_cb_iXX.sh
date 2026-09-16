#!/bin/bash
#SBATCH --job-name=psn_cb_iXX_$1
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_cb_iXX_$1_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_cb_iXX_$1_%j.err
#
# Checkerboard (Chao paper Fig.3) A=3 directional — GENERIC polar-set
# convergence at FIXED S=20 / M=24 / A=3: I=60 and I=120 (4 pts total:
# weak x {I60,I120}, strong x {I60,I120}).
# Baseline: I30 S20 M24  weak -3.8 pcm (wall 337s) / strong -11.6 pcm (262s).
# Same protocol as slurm_cb_i30_ext.sh: GMRES pool 40 workers, 1e-7 gate,
# dump with block=S (S=20 = 40x40 nodes, pin=2x2 base grid).
# $1: weak | strong
set -e
VARG=$1
case $VARG in
  weak)   YAML=examples/psn_repro/checkerboard_1g_rect.yaml;        KREF=1.12974 ;;
  strong) YAML=examples/psn_repro/checkerboard_1g_rect_strong.yaml; KREF=0.51673 ;;
  *) echo "usage: slurm_cb_iXX.sh weak|strong"; exit 1 ;;
esac
W=weak
[ "$VARG" = "strong" ] && W=strong
PFX=CB$([ $W = weak ] && echo W || echo S)

mkdir -p /work/home/xycjscs/psn_runs
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python

echo "=== env (checkerboard A3 $VARG I60/I120 S20 M24) ==="
hostname; nproc; free -g | head -2

for I in 60 120; do
  TAG=${PFX}${I}
  $PY -u tools_ho/hpc_point.py $TAG $YAML 24 20 gmres 40 $KREF 20 generic $I || echo "POINT FAIL $TAG M24_S20"
done

echo "=== CB $VARG I60/I120 ALL DONE ==="
