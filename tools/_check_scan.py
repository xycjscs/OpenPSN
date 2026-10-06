from hpc import rsh, load_cfg
cfg = load_cfg()
q = ("grep -h 'SCAN host' /work/home/xycjscs/psn_runs/nodescan2_*.out 2>/dev/null | "
     "awk '{print $3}' | sort | uniq -c | sort -rn; "
     "echo FILES: $(grep -lh 'SCAN host' /work/home/xycjscs/psn_runs/nodescan2_*.out 2>/dev/null | wc -l); "
     "echo ===IPARM===; "
     "grep -h -E 'iparm14|PROBE-DONE' /work/home/xycjscs/psn_runs/iparmtest_45736144.out 2>/dev/null; "
     "echo ===QUEUE===; squeue -u xycjscs -h 2>/dev/null | wc -l")
rc, out = rsh(cfg, q)
print(out)
