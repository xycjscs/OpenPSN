"""BWR full matrix — SEQUENTIAL single-process driver.

No multiprocessing (memory crashes). Each point: compute -> append+fsync log.
Order: N ascending (cheap points first so progress is checkpointed early).
Re-run to resume: already-logged points are skipped.
"""
import sys, os, re, time
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
import test_bwr as tb

MS = (4, 8, 12, 16, 20, 24)
NS = range(1, 11)
LOG = '/tmp/bwr_seq.log'
MIRROR = '/opt/data/workspace/psn/bwr_seq_mirror.log'

def done_set():
    done = set()
    rx = re.compile(r'(noGd|Gd)\s+N=\s*(\d+)\s+M=\s*(\d+):\s+keff=')
    for lg in (LOG, MIRROR, '/tmp/bwr_inc.log'):
        if os.path.exists(lg):
            for ln in open(lg):
                m = rx.match(ln)
                if m:
                    done.add((int(m.group(2)), int(m.group(3)),
                              1 if m.group(1) == 'Gd' else 0))
    return done

if __name__ == '__main__':
    done = done_set()
    jobs = sorted([(N, M, gd) for gd in (0, 1) for N in NS for M in MS
                   if (N, M, gd) not in done])
    print(f'skipping {len(done)} done, {len(jobs)} to go', flush=True)
    t_all = time.time()
    for i, (N, M, gd) in enumerate(jobs):
        t0 = time.time()
        tag = 'Gd ' if gd else 'noGd'
        try:
            k = tb.run_bwr(N=N, M=M, gd=gd)
            kref = tb.KREF_GD if gd else tb.KREF_NO_GD
            line = (f'{tag} N={N:2d} M={M:2d}: keff={k:.6f}  '
                    f'err={(k-kref)*1e5:+8.1f} pcm  [{time.time()-t0:6.0f}s]')
        except Exception as e:
            line = f'{tag} N={N:2d} M={M:2d}: ERROR {type(e).__name__}: {e}'
        # log to BOTH locations with fsync (crash-proof checkpoint)
        for lg in (LOG, MIRROR):
            with open(lg, 'a') as f:
                f.write(line + '\n')
                f.flush()
                os.fsync(f.fileno())
        print(f'[{i+1}/{len(jobs)}] {line}  (elapsed {time.time()-t_all:.0f}s)',
              flush=True)
    print(f'ALL DONE in {time.time()-t_all:.0f}s', flush=True)
