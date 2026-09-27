# -*- coding: utf-8 -*-
"""Lever-3 kernel test: numba mc_b vs the NumPy _mc_b arithmetic.

For several (pair) microclasses of several (i, g) systems, compare
the packed numba b-assembly against a direct NumPy re-implementation
of the same expressions (gather + sgn + einsum).  Gate: max relative
difference < 1e-12 (summation-order difference only).
Run: python dbg_mckern.py
"""
import os, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep
from psn2d import mckern

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': 'M4_S1', 'model': 'ty3', 'I': 30, 'M': 4, 'subdivide': 1,
        'kref': 1.3347260, 'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000, 'threads': 1})
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
p = psn
rng = np.random.default_rng(1)


def numpy_mc_b(nl, data, qn):
    """Reference NumPy arithmetic, copied from _mc_b (pre-numba)."""
    u = nl.u
    nlist = data["nodes"]
    b = qn[nlist] @ data["SRC"].T
    for lf, parts in data["nb"].items():
        for part in parts:
            idxs = part["idxs"]
            row1d = part["rowidx"].ravel()
            un = u[part["gath"]].reshape(
                len(idxs), 4, nl.M, nl.K).transpose(0, 2, 1, 3)
            prod = part["sgn_nb"] * un
            contrib = np.einsum("mafs,nmfs->nma", part["C0"], prod)
            tmp = b[idxs]
            tmp[:, row1d] += contrib.reshape(len(idxs), -1)
            src = np.einsum("nq,maq->nma", qn[part["nodes_nb"]],
                            part["src5"]).reshape(len(idxs), -1)
            tmp[:, row1d] += part["sigma"] * src
            b[idxs] = tmp
    return b


worst = 0.0
n_checked = 0
for i in (0, 2):
    for g in (0, 6):
        nl = highsweep.HoNodeLocal(p, i, g, 3, 'directional',
                                   gs_tol=1e-11, max_gs=1)
        orders = nl._mc_order()
        keys = []
        for pty in (0, 1):
            for key in orders[pty][0] + orders[pty][1]:
                if key not in keys:
                    keys.append(key)
        qn = rng.standard_normal((p.nodes, 5)) * 0.01
        nl.u = rng.standard_normal(p.ncol * nl.K)
        for key in keys[:4]:
            data = nl._microclass(key)
            pack = data["pack"]
            assert pack is not None, "pack missing"
            b_np = numpy_mc_b(nl, data, qn)
            b_nb = mckern.mc_b(pack, nl.u, qn)
            d = float(np.abs(b_np - b_nb).max())
            rel = d / max(1.0, float(np.abs(b_np).max()))
            worst = max(worst, rel)
            n_checked += 1
        # pair microclasses too
        for pp in range(p.M // 2):
            data = nl._pair_microclass(keys[0], pp)
            # pair pack built from the pair data (same structure)
            packp = mckern.pack_microclass(data)
            qn2 = rng.standard_normal((p.nodes, 5)) * 0.01

            def numpy_mc_b_pair(nl, data, qn):
                u = nl.u
                nlist = data["nodes"]
                b = qn[nlist] @ data["SRC"].T
                for lf, parts in data["nb"].items():
                    for part in parts:
                        idxs = part["idxs"]
                        row1d = part["rowidx"]
                        un = u[part["gath"]].reshape(
                            len(idxs), 4, 2, nl.K).transpose(0, 2, 1, 3)
                        prod = part["sgn_nb"] * un
                        contrib = np.einsum("mafs,nmfs->nma", part["C0"],
                                            prod)
                        tmp = b[idxs]
                        tmp[:, row1d] += contrib.reshape(len(idxs), -1)
                        src = np.einsum("nq,maq->nma",
                                        qn[part["nodes_nb"]],
                                        part["src5"]).reshape(len(idxs), -1)
                        tmp[:, row1d] += part["sigma"] * src
                        b[idxs] = tmp
                return b

            b_np = numpy_mc_b_pair(nl, data, qn2)
            b_nb = mckern.mc_b(packp, nl.u, qn2)
            d = float(np.abs(b_np - b_nb).max())
            rel = d / max(1.0, float(np.abs(b_np).max()))
            worst = max(worst, rel)
            n_checked += 1

print(f"checked {n_checked} microclasses (full+pair), "
      f"worst rel diff = {worst:.3e}")
ok = worst < 1e-12
print("MCKERN", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
