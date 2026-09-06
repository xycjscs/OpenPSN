import sys, os, re, time
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
import numpy as np
from psn_solver import PSN2D

FUEL = (1.5, 0.15, 0.24)
ABS_W = (1.0, 0.07, 0.0); ABS_S = (1.0, 0.7, 0.0)
KREF_W = 1.12974; KREF_S = 0.51673
LOGS = ('/tmp/fig3_fill.log', '/opt/data/workspace/psn/fig3_fill.log')
PAT = re.compile(r'(weak|strong) S=(\d+) M=(\d+): keff=([\d.]+)')

done = set()
for f in LOGS:
    if os.path.exists(f):
        for ln in open(f):
            m = PAT.match(ln)
            if m:
                done.add((m.group(1), int(m.group(2)), int(m.group(3))))

def log(msg):
    for f in LOGS:
        with open(f, 'a') as fh:
            fh.write(msg + '\n'); fh.flush(); os.fsync(fh.fileno())

base = np.array([[0, 1], [1, 0]], int)
t0 = time.time(); n = 0
for case in ('weak', 'strong'):
    ABS = ABS_W if case == 'weak' else ABS_S
    KREF = KREF_W if case == 'weak' else KREF_S
    St = np.array([[FUEL[0], ABS[0]]]); Sa = np.array([[FUEL[1], ABS[1]]])
    nu = np.array([[FUEL[2], ABS[2]]])
    Sgg = np.zeros((1, 1, 2)); Sgg[0, 0, :] = St[0, :] - Sa[0, :]
    for S in (3, 5, 6, 7, 8, 9):
        for M in (2, 4, 8, 12, 16, 24):
            if (case, S, M) in done:
                continue
            n += 1
            g = np.kron(base, np.ones((S, S), int)); h = 1.0 / S
            p = PSN2D(g, h, St, Sgg, nu, generic=True, I=30, M=M)
            t1 = time.time()
            k, _, _ = p.keff(max_outer=4000, outer_tol=1e-10, verbose=False)
            dt = time.time() - t1
            abserr = (k - KREF) * 1e5
            log(f'{case} S={S} M={M}: keff={k:.6f}  abs={abserr:+8.1f} pcm  [{dt:5.0f}s]  (elapsed {time.time()-t0:.0f}s)')
log(f'ALL DONE in {time.time()-t0:.0f}s ({n} points)')
