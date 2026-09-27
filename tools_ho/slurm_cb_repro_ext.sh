#!/bin/bash
#SBATCH --job-name=psn_cb_repro_ext
#SBATCH --partition=wzacnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/home/xycjscs/psn_runs/psn_cb_repro_ext_%j.out
#SBATCH --error=/work/home/xycjscs/psn_runs/psn_cb_repro_ext_%j.err
#
# Extend the Fig.3 reproduction tables (appA/appB) with M = 48, 96, 192
# for S = 1..10, legacy Chao-PSN (psn.keff), generic I=30.  $1 = weak|strong
set -e
cd /work/home/xycjscs/OpenPSN
PY=/work/home/xycjscs/mf/envs/psn/bin/python
echo "=== env cb_repro_ext $1 ==="
hostname; nproc
$PY -u tools_ho/cb_repro_ext.py "$1"
