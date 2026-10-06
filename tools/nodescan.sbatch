#!/bin/bash
#SBATCH -J nodescan
#SBATCH -p wzacnormal02
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 06:00:00
set -u
for n in $(sinfo -p wzacnormal02 -h -N -o "%N" | sort); do
  srun --no-kill -N1 -n1 -c32 --mem=8G -w "$n" -t 0:04:00 \
    bash -c 'cd ~/psn_strict && PSN_REPO=~/OpenPSN ~/mf/envs/psn/bin/python -u nodescan_probe.py' \
    2>/dev/null | grep SCAN || echo "SCAN host=$n res=NOEXEC"
done
echo SCAN-DONE
