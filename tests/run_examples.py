#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PSN2D regression suite — the paper's example benchmarks.

Purpose
-------
Safety net for code maintenance: every change to psn2d/ must reproduce the
paper's validation examples (checkerboard Fig.3, BWR bundle Table 2 / Fig.8-12,
C5G7 assembly + quarter core) to within tolerance.  keff is bit-for-bit
reproducible across re-runs of the frozen solver, so the baseline is tight.

Each entry is (yaml, case_name, tier):
  fast  — small grids, seconds-to-minutes; run on every edit
  full  — heavy quarter-core / high-subdivide points; run before commits

Usage
-----
  python tests/run_examples.py            # fast tier, compare to baseline
  python tests/run_examples.py --full     # fast + full tier
  python tests/run_examples.py --record   # (re)record baseline from current code
  python tests/run_examples.py --only checkerboard   # filter by yaml basename
  python tests/run_examples.py --tol 2.0  # tolerance in pcm (default 1.0)
  python tests/run_examples.py --json     # machine-readable results

Exit code 0 = all pass, 1 = at least one fail/skip-error, 2 = usage.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psn2d import model
from psn2d import __main__ as M

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
BASELINE = os.path.join(HERE, "baseline_keff.json")

# (yaml basename, case name, tier).  Case names must exist in the yaml's cases
# list (or be the single case if cases: is absent).
CASES = [
    # checkerboard (paper Sec 4.1, Fig.3) — generic polar, 1-group
    ("checkerboard_1g.yaml", "weak_M12", "fast"),
    ("checkerboard_1g.yaml", "weak_M24", "fast"),
    # BWR bundle (paper Sec 4.3, Table 2 / Fig.8-12) — ty3, 2-group
    ("bwr_bundle_2g.yaml", "N2_M8", "fast"),
    ("bwr_bundle_2g.yaml", "N4_M12", "fast"),
    # C5G7 single UO2 assembly (17x17, 7-group)
    ("c5g7_uo2_assembly.yaml", "M8_S2", "fast"),
    # heavy points (quarter core / high-subdivide assembly)
    ("c5g7_uo2_assembly.yaml", "M12_S4", "full"),
    ("c5g7_uo2_assembly.yaml", "M16_S4", "full"),
    ("c5g7_uo2_assembly.yaml", "M16_S6", "full"),
    # C5G7 1/4 core (51x51, 7-group)
    ("c5g7_2d_quarter_core.yaml", "M8_S1", "full"),
    ("c5g7_2d_quarter_core.yaml", "M12_S1", "full"),
    ("c5g7_2d_quarter_core.yaml", "M12_S2", "full"),
    ("c5g7_2d_quarter_core.yaml", "M16_S2", "full"),
]


def _key(yamlb, case):
    return f"{yamlb}::{case}"


_YAML_CACHE = {}


def load_yaml_cache(yamlb):
    path = os.path.join(EXAMPLES, yamlb)
    if yamlb not in _YAML_CACHE:
        _YAML_CACHE[yamlb] = model.load_spec(path)
    return _YAML_CACHE[yamlb]


def run_one(yamlb, case_name, tier):
    spec = load_yaml_cache(yamlb)
    cases = {c["name"]: c for c in model.expand_cases(spec)}
    if case_name not in cases:
        return {"yaml": yamlb, "case": case_name, "tier": tier,
                "status": "NO_CASE", "available": list(cases)}
    case = cases[case_name]
    t0 = time.time()
    try:
        psn = M.build_solver(spec, case)
        k, _, _ = psn.keff(max_outer=int(spec["solver"]["max_outer"]),
                           outer_tol=float(spec["solver"]["keff_tol"]),
                           verbose=False)
    except MemoryError:
        return {"yaml": yamlb, "case": case_name, "tier": tier,
                "status": "OOM", "keff": None}
    except Exception as e:
        return {"yaml": yamlb, "case": case_name, "tier": tier,
                "status": f"ERROR:{type(e).__name__}:{e}", "keff": None}
    dt = time.time() - t0
    return {"yaml": yamlb, "case": case_name, "tier": tier, "status": "OK",
            "keff": float(k), "nodes": psn.nodes, "time_s": round(dt, 1)}


def selected(include_full, only):
    out = []
    for yamlb, case, tier in CASES:
        if only and only not in yamlb:
            continue
        if tier == "full" and not include_full:
            continue
        out.append((yamlb, case, tier))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="include full-tier (heavy) cases")
    p.add_argument("--record", action="store_true",
                   help="(re)record baseline_keff.json from the current code")
    p.add_argument("--only", help="only run cases whose yaml basename contains this string")
    p.add_argument("--tol", type=float, default=1.0, help="tolerance in pcm (default 1.0)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    jobs = selected(args.full, args.only)
    if not jobs:
        print("no cases selected", file=sys.stderr)
        return 2

    baseline = {}
    if not args.record and os.path.exists(BASELINE):
        with open(BASELINE) as f:
            baseline = json.load(f)

    results = []
    for yamlb, case, tier in jobs:
        r = run_one(yamlb, case, tier)
        results.append(r)
        if args.json:
            continue
        if r["status"] != "OK":
            print(f"[{r['status']:>14}] {yamlb}::{case} ({tier})", flush=True)
            continue
        key = _key(yamlb, case)
        line = f"[{r['status']:>14}] {yamlb}::{case} ({tier})  " \
               f"keff={r['keff']:.7f}  {r['time_s']}s"
        if args.record:
            baseline[key] = round(r["keff"], 9)
            line += f"  recorded"
        elif key in baseline:
            dpcm = (r["keff"] - baseline[key]) * 1e5
            ok = abs(dpcm) <= args.tol
            line += f"  base={baseline[key]:.7f}  Δ={dpcm:+.3f} pcm  " \
                    f"{'PASS' if ok else 'FAIL'}"
            r["base"] = baseline[key]; r["dpcm"] = dpcm; r["pass"] = ok
        else:
            line += "  NO_BASELINE"
        print(line, flush=True)

    if args.record:
        with open(BASELINE, "w") as f:
            json.dump(baseline, f, indent=1, ensure_ascii=False)
        print(f"\nrecorded {len(baseline)} baselines -> {BASELINE}")
        return 0

    if args.json:
        for r in results:
            if r["status"] == "OK":
                key = _key(r["yaml"], r["case"])
                if key in baseline:
                    r["base"] = baseline[key]
                    r["dpcm"] = (r["keff"] - baseline[key]) * 1e5
                    r["pass"] = abs(r["dpcm"]) <= args.tol
                else:
                    r["pass"] = False; r["dpcm"] = None
        print(json.dumps(results, indent=1, ensure_ascii=False))

    ok = [r for r in results if r.get("status") == "OK" and r.get("pass")]
    fail = [r for r in results if r.get("status") == "OK" and not r.get("pass")]
    bad = [r for r in results if r.get("status") not in ("OK", "NO_CASE") or
           (r.get("status") == "NO_CASE")]
    print(f"\n=== {len(ok)} passed, {len(fail)} failed, "
          f"{len(bad)} error/skip (tol {args.tol} pcm) ===")
    return 0 if not fail and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
