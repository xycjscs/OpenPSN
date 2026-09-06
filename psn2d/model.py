# -*- coding: utf-8 -*-
"""PSN2D input model: load & validate a YAML problem specification.

Spec schema (see examples/):

  problem:         name (string)
  description:      free text
  angular:
    model: generic | ty3
    I: 30           # generic: even number of polar segments over 0..pi
    M: 24           # azimuthal segments (must be even)
  geometry:
    grid: [[0, 1], [1, 0]]   # material indices; FIRST ROW = bottom (y-),
                             # columns run left (x-) -> right (x+)
    unit_size: 1.0           # cm, per grid cell
    subdivide: 4             # PSN sub-nodes per grid cell per axis
  boundaries: {left: reflect, bottom: vacuum, right: reflect, top: reflect}
  materials:
    - name: fuel
      total:      [...]      # Sigma_t per group
      scatter:   [[...]]     # [source][dest] isotropic, per group
      nu_fission: [...]      # nu * Sigma_f per group
  fission_spectrum: [...]    # chi per group (default all ones)
  solver: {keff_tol: 1e-10, max_outer: 4000}
  cases:                     # optional sweep; each entry may override
    - {name: ..., M: 16, I: 12, subdivide: 4, kref: 1.18646}

`materials_file: path` pulls materials (+fission_spectrum) from another YAML.
"""
import os
import yaml
import numpy as np

VALID_BND = ("reflect", "vacuum")


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_spec(path):
    path = os.path.abspath(path)
    spec = _load_yaml(path)
    base = os.path.dirname(path)

    mf = spec.get("materials_file")
    if mf:
        p = mf if os.path.isabs(mf) else os.path.join(base, mf)
        mdata = _load_yaml(p)
        spec["materials"] = mdata["materials"]
        if "fission_spectrum" in mdata:
            spec.setdefault("fission_spectrum", mdata["fission_spectrum"])

    ang = spec.get("angular", {})
    model = ang.get("model", "generic")
    if model not in ("generic", "ty3"):
        raise ValueError(f"angular.model must be 'generic' or 'ty3', got {model!r}")
    if int(ang.get("I", 30)) % 2:
        raise ValueError("angular.I must be even (0..pi partition)")
    if int(ang.get("M", 24)) % 2:
        raise ValueError("angular.M must be even (mirror pairs need paired segments)")

    geom = spec["geometry"]
    grid = np.array(geom["grid"], dtype=int)
    if grid.ndim != 2:
        raise ValueError("geometry.grid must be a 2D list of rows")
    nmat = len(spec["materials"])
    if grid.min() < 0 or grid.max() >= nmat:
        raise ValueError(f"grid material index out of range [0, {nmat-1}]")

    ng = len(spec["materials"][0]["total"])
    for m in spec["materials"]:
        if len(m["total"]) != ng:
            raise ValueError(f"material {m['name']}: total must have {ng} entries")
        S = np.asarray(m["scatter"])
        if S.shape != (ng, ng):
            raise ValueError(f"material {m['name']}: scatter must be {ng}x{ng} "
                             f"(indexed [source][dest])")
        if len(m.get("nu_fission", [0.0] * ng)) != ng:
            raise ValueError(f"material {m['name']}: nu_fission must have {ng} entries")

    chi = spec.get("fission_spectrum", [1.0] * ng)
    if len(chi) != ng:
        raise ValueError("fission_spectrum must have one entry per group")
    spec["fission_spectrum"] = chi

    bnd = spec.get("boundaries", {})
    defaults = {"left": "reflect", "bottom": "reflect",
                "right": "reflect", "top": "reflect"}
    for k, v in defaults.items():
        bnd.setdefault(k, v)
        if bnd[k] not in VALID_BND:
            raise ValueError(f"boundaries.{k} must be one of {VALID_BND}")
    spec["boundaries"] = bnd
    spec.setdefault("solver", {})
    spec["solver"].setdefault("keff_tol", 1e-10)
    spec["solver"].setdefault("max_outer", 4000)
    return spec


def expand_cases(spec):
    """Yield per-case dicts (base spec + angular/geometry defaults + overrides)."""
    ang = spec["angular"]
    geom = spec["geometry"]
    cases = spec.get("cases")
    if cases is None:
        cases = [{}]
    out = []
    for i, c in enumerate(cases):
        c = dict(c)
        c.setdefault("name", c.get("name", f"case{i+1}"))
        c.setdefault("model", ang.get("model", "generic"))
        c.setdefault("I", int(ang.get("I", 30)))
        c.setdefault("M", int(ang.get("M", 24)))
        c.setdefault("subdivide", int(geom.get("subdivide", 1)))
        c["kref"] = c.get("kref", spec.get("kref"))
        out.append(c)
    return out


def arrays(spec):
    """Build (St, Sgg, nuSf, chi) numpy arrays from the spec materials.

    St/nuSf:  [group][material]
    Sgg:      [dest][source][material]   (h5/OpenMOC convention is
             scatter[source*ng+dest], hence the transpose here)
    """
    mats = spec["materials"]
    ng = len(mats[0]["total"])
    St = np.array([[m["total"][g] for m in mats] for g in range(ng)], float)
    Sgg = np.zeros((ng, ng, len(mats)))
    for mi, m in enumerate(mats):
        S = np.asarray(m["scatter"], float)
        for g in range(ng):
            for g2 in range(ng):
                Sgg[g, g2, mi] = S[g2, g]
    nuSf = np.array([[m.get("nu_fission", [0.0] * ng)[g] for m in mats]
                     for g in range(ng)], float)
    chi = np.array(spec["fission_spectrum"], float)
    return St, Sgg, nuSf, chi
