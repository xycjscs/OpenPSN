"""Checkerboard benchmark (Chao et al. ANE 2027, Sec 4.1, Table 1).

2x2 quarter cells (1 cm x 1 cm each), reflective boundaries all around.
1 group.  F A / A F  pattern.
fuel:  St=1.5, Sa=0.15, nuSf=0.24
weak:  St=1.0, Sa=0.07
strong:St=1.0, Sa=0.70
kref (OpenMC): weak=1.12974, strong=0.51673
"""
import numpy as np
from psn_solver import PSN2D

def run(label, Sa_abs, M=12, h=1.0, verbose=True):
    nmat = 2
    St = np.array([[1.5, 1.0]])
    Sgg = np.zeros((1, 1, nmat))
    Sgg[0, 0, 0] = 1.5 - 0.15      # fuel scatter
    Sgg[0, 0, 1] = 1.0 - Sa_abs    # absorber scatter
    nuSf = np.array([[0.24, 0.0]])
    mat_map = np.array([[0, 1],
                        [1, 0]])
    psn = PSN2D(mat_map, h, St, Sgg, nuSf,
                boundary=("reflect",) * 4, M=M)
    keff, phi_bar, qnode = psn.keff(verbose=verbose)
    return keff

if __name__ == "__main__":
    import sys
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    keff_w = run("weak", 0.07, M=M)
    print(f"\nweak absorber   keff = {keff_w:.6f}   (ref 1.12974, "
          f"err {100*(keff_w-1.12974)/1.12974:+.3f}%)")
    keff_s = run("strong", 0.70, M=M)
    print(f"strong absorber keff = {keff_s:.6f}   (ref 0.51673, "
          f"err {100*(keff_s-0.51673)/0.51673:+.3f}%)")
