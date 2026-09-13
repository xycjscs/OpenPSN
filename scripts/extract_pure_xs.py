#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract PURE (unhomogenized) material XS from c5g7-mgxs.h5.

Output: examples/c5g7/study2_rectangular/c5g7_materials_pure.yaml
Materials: Water, UO2, MOX-4.3%, MOX-7%, MOX-8.7%, Control Rod (Gd),
           Fission Chamber — each with 7-group total / scatter / nuSigmaF,
plus the global fission spectrum (identical chi across all fissile mats).

Convention (matches PSN yaml): scatter indexed [source][dest].
"""
import os
import h5py
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5 = os.path.join(HERE, "examples", "c5g7", "c5g7-mgxs.h5")
OUT = os.path.join(HERE, "examples", "c5g7", "study2_rectangular", "c5g7_materials_pure.yaml")

ORDER = [("Water", "water"),
         ("UO2", "uo2"),
         ("MOX-4.3%", "mox43"),
         ("MOX-7%", "mox7"),
         ("MOX-8.7%", "mox87"),
         ("Control Rod", "gd"),
         ("Fission Chamber", "fc"),
         ("Guide Tube", "gt")]

with h5py.File(H5, "r") as f:
    chi = f["material"]["UO2"]["chi"][:].ravel().tolist()
    materials = []
    for h5name, pname in ORDER:
        d = f["material"][h5name]
        total = d["total"][:].ravel().tolist()
        sc = d["scatter matrix"][:].ravel().reshape(7, 7)  # [source][dest]
        nusf = d["nu-fission"][:].ravel().tolist()
        materials.append({"name": pname, "total": total,
                          "scatter": sc.tolist(), "nu_fission": nusf})
    chi_check = []
    for h5name, _ in ORDER:
        c = f["material"][h5name]["chi"][:].ravel()
        if c.max() > 0:
            chi_check.append(c)

# verify all fissile chi identical
import numpy as np
ref = chi_check[0]
for c in chi_check[1:]:
    assert np.allclose(ref, c), "chi differs between materials"

doc = {
    "problem": "c5g7_pure_materials",
    "description": "PURE (unhomogenized) C5G7 7-group XS from c5g7-mgxs.h5. "
                   "scatter [source][dest]; nu_fission = nu*Sigma_f.",
    "fission_spectrum": chi,
    "materials": materials,
}
with open(OUT, "w") as fh:
    fh.write("# C5G7 PURE material cross sections (7-group, c5g7-mgxs.h5).\n")
    fh.write("# NOT pin-homogenized: these are the bare-material XS used by\n")
    fh.write("# the rectangular-rod (square fuel) C5G7 test problem.\n")
    yaml.dump(doc, fh, default_flow_style=False, sort_keys=False)

print("wrote", OUT)
print("fission_spectrum:", chi)
for m in materials:
    print(f"  {m['name']:5s} St_g1={m['total'][0]:.6f} nusf_g1={m['nu_fission'][0]:.6f}")
