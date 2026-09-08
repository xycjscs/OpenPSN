#!/bin/bash
# C5G7 串行扫参链条：等 M16_S2 完成 → M16_S1（1/4芯）→ M16_S6（单UO2组件）
# 每条: keff/耗时/峰值RSS 追加到 JSONL + 完整 stdout 到各自 log
set -u
cd /opt/data/workspace/OpenPSN
RES=examples/c5g7_sweep.jsonl
WAITPID=45968

# 0) 等当前 M16_S2 进程退出
while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
echo "TS $(date '+%F %T') M16_S2 done, starting M16_S1" >> examples/c5g7_sweep.log

# 1) M16_S1 1/4 芯
python3 -u examples/c5g7_probe.py 16 1 1e-10 > examples/c5g7_M16_S1.log 2>&1
echo "TS $(date '+%F %T') M16_S1 done" >> examples/c5g7_sweep.log

# 2) 单 UO2 组件 M16_S6（17x17 pin, S=6 → 102x102 = 10404 节点）
python3 -u - <<'EOF' > examples/c5g7_asm_M16_S6.log 2>&1
import resource, sys, time
import numpy as np
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import model
from psn2d import __main__ as M
spec = model.load_spec('/opt/data/workspace/OpenPSN/examples/c5g7_uo2_assembly.yaml')
case = {'name': 'asm_M16_S6', 'model': 'ty3', 'I': 30, 'M': 16,
        'subdivide': 6, 'kref': 1.33367}
t0 = time.time()
psn = M.build_solver(spec, case)
k, phi, qnode = psn.keff(max_outer=6000, outer_tol=1e-10, verbose=True)
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"\nPROBE_DONE M=16 S=6 asm nodes={psn.nodes} keff={k:.7f} "
      f"pcm={(k-1.33367)*1e5:+.1f} wall={dt:.0f}s peak_RSS={peak:.2f}GB", flush=True)
EOF
echo "TS $(date '+%F %T') asm_M16_S6 done, CHAIN COMPLETE" >> examples/c5g7_sweep.log
