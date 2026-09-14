import os, time
import numpy as np

print('nproc:', os.cpu_count())
A = np.random.rand(2000, 2000)
B = np.random.rand(2000, 2000)

def bench(tag):
    # warm
    A @ B
    t0 = time.time()
    for _ in range(5):
        A @ B
    dt = (time.time() - t0) / 5
    print(f'{tag}: matmul {dt*1000:.1f} ms')
    return dt

d0 = bench('no-pin')

from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1, user_api='blas'):
    d1 = bench('pin=1')

print(f'ratio no-pin/pin=1 = {d0/d1:.1f}x  '
      f'({"pin works" if d0/d1 > 2 else "PIN INEFFECTIVE"})')

# also scipy lu path
from scipy.linalg import lu_factor, lu_solve
C = np.random.rand(400, 400)
D = np.random.rand(400, 100)
def bench_lu(tag):
    lu, piv = lu_factor(C)
    t0 = time.time()
    for _ in range(20):
        lu_solve((lu, piv), D)
    print(f'{tag}: lu_solve {(time.time()-t0)/20*1000:.3f} ms')

bench_lu('no-pin')
with threadpool_limits(limits=1, user_api='blas'):
    bench_lu('pin=1')
