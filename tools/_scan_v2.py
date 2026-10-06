"""Node typing scan v2: per-node sbatch array for M96_S6 single-system probe."""
import re, time
from hpc import rsh, load_cfg, ledger_load, ledger_save

cfg = load_cfg()

# 1) iparm test v2 (PSN_REPO inline)
rc, out = rsh(cfg,
    "cd ~/psn_strict && sbatch -p wzacnormal02 -N1 -c32 --mem=64G "
    "-x b10r1n29,b10r1n15,b10r3n26 "
    "--wrap='export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PSN_REPO=$HOME/OpenPSN; "
    "$HOME/mf/envs/psn/bin/python -u iparm_test.py' "
    "-o ~/psn_runs/iparmtest_%j.out")
print("iparmtest:", out.strip())

# 2) nodescan v2: array over node list. One task per node, each pinned via
# a generated include file? SLURM arrays cannot pin per-task nodes directly —
# instead submit a modest array (8 parallel lanes) over an index into a node
# list, and each task greps its own node out of the list and uses --ehx? 
# Simplest robust: generate one sbatch per node in a LOOP (109 submits, cheap).
rc, out = rsh(cfg, """
nodes=$(sinfo -p wzacnormal02 -h -N -o '%N' 2>/dev/null | grep -v '^$' | sort)
for n in $nodes; do
  s=$(cd ~/psn_strict && sbatch -p wzacnormal02 -N1 -c32 --mem=8G -t 0:05:00 -w $n -J nscan \
      --wrap="export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PSN_REPO=$HOME/OpenPSN; cd ~/psn_strict && $HOME/mf/envs/psn/bin/python -u nodescan_probe.py" \
      -o ~/psn_runs/nodescan2_%j.out 2>&1 | grep -oE '[0-9]+$')
  echo "$n=$s"
done > /work/home/xycjscs/psn_runs/nodescan2_map.txt 2>/dev/null
wc -l < /work/home/xycjscs/psn_runs/nodescan2_map.txt
tail -3 /work/home/xycjscs/psn_runs/nodescan2_map.txt
""")
print("nodescan2 submits:\n", out[-500:])

m2 = re.search(r"iparmtest: Submitted batch job (\d+)", out) if False else None
led = ledger_load()
ts = time.strftime("%F %T")
import subprocess
# record iparmtest job from first print
led.setdefault("45736045", {"note": "iparm test v1 (env fail)"})
led["iparmtest_v2"] = {"ts": ts, "note": "iparm test v2 submitted (see stdout for id)"}
ledger_save(led)
