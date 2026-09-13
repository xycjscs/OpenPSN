# -*- coding: utf-8 -*-
"""Local acceptance: packaged sample points through the upgraded local
OpenPSN, compared against the package's official golden values
(extracted verbatim from validation/*.json, 18 distinct configs).

Gate: |dk| <= 1e-7 per point (the packaged oracle's own
k_difference_to_golden is <= 7e-16, so this is 6 orders looser than the
source of truth).

Usage: /opt/data/venvs/omconda/bin/python tools_ho/accept_local.py
"""
import json
import re
import subprocess
import time

REPO = '/opt/data/workspace/OpenPSN'
PY = '/opt/data/venvs/omconda/bin/python'

# Official goldens from the package validation/*.json (id -> k).
GOLDEN_TY3 = {
    ('mox87', 'M24_A3'): 1.1732714250722445,
    ('mox87', 'M24_A4'): 1.1732879656720250,
    ('mox87', 'M48_A3'): 1.1732952676457347,
    ('mox87', 'M24_A0_orig'): 1.1721653548336128,
    ('uo2', 'M24_A3'): 1.3267607286688914,
    ('uo2', 'M24_A4'): 1.3267700176176846,
    ('uo2', 'M48_A3'): 1.3267741194001486,
    ('uo2', 'M24_A0_orig'): 1.3261464995842203,
}
GOLDEN_GEN = {
    ('mox87', 'M24_A3'): 1.1733041618755806,
    ('mox87', 'M24_A4'): 1.1733207043920697,
    ('mox87', 'M48_A3'): 1.1733279918437869,
    ('mox87', 'I128_M24_A3'): 1.1733256614557841,
    ('mox87', 'M24_A0_orig'): 1.1722675432498990,
    ('uo2', 'M24_A3'): 1.3267922862450716,
    ('uo2', 'M24_A4'): 1.3268015774975108,
    ('uo2', 'M48_A3'): 1.3268056684475726,
    ('uo2', 'I128_M24_A3'): 1.3267951110308969,
    ('uo2', 'M24_A0_orig'): 1.3261374169443916,
}


def parse_case(cname):
    """'I128_M24_A3' -> I=128 M=24 A=3; 'M48_A3' -> M=48 A=3."""
    I = None
    m = re.match(r'^I(\d+)_', cname)
    if m:
        I = int(m.group(1))
        cname = cname[m.end():]
    m2 = re.match(r'^M(\d+)_A(\d+)', cname)
    if not m2:
        raise ValueError(f'unparseable case name: {cname}')
    return I, int(m2.group(1)), int(m2.group(2))


def run_case(fuel, polar, case_name, closure):
    spec_path = f'{REPO}/examples/{fuel}_{polar}_s2.json'
    with open(spec_path) as f:
        spec = json.load(f)
    I, M, A = parse_case(case_name)
    spec['cases'] = [{'name': case_name, 'M': M, 'A': A,
                      **({'I': I} if I else {}),
                      'closure': closure}]
    tmp = f'/tmp/hocase_{fuel}_{polar}_{case_name}.json'
    with open(tmp, 'w') as f:
        json.dump(spec, f)
    t0 = time.time()
    r = subprocess.run(
        [PY, '-m', 'psn2d', 'run', tmp, '--opt', 'off', '--threads', '1',
         '--quiet', '--json'], cwd=REPO, capture_output=True, text=True,
        timeout=3600)
    if r.returncode != 0:
        print(f'  FAILED rc={r.returncode}: {r.stderr[-400:]}', flush=True)
        return None
    out = None
    lines = r.stdout.splitlines()
    for idx, ln in enumerate(lines):
        if ln.strip() == '[':
            try:
                out = json.loads('\n'.join(lines[idx:]))
                break
            except Exception:
                continue
    if out is None:
        print(f'  NO JSON in stdout: {r.stdout[-300:]}', flush=True)
        return None
    out[0]['wall_s'] = round(time.time() - t0, 1)
    return out[0]


def main():
    plan = []
    for fuel in ('mox87', 'uo2'):
        plan += [(fuel, 'ty3', c, 'directional_matrix')
                 for c in ('M24_A3', 'M24_A4', 'M48_A3')]
        plan += [(fuel, 'generic', c, 'directional_matrix')
                 for c in ('M24_A3', 'M24_A4', 'M48_A3', 'I128_M24_A3')]
        plan += [(fuel, 'ty3', 'M24_A0_orig', 'original_azimuth'),
                 (fuel, 'generic', 'M24_A0_orig', 'original_azimuth')]
    n_ok = n_bad = 0
    rows = []
    for (fuel, polar, cname, closure) in plan:
        r = run_case(fuel, polar, cname, closure)
        gold = (GOLDEN_TY3 if polar == 'ty3' else GOLDEN_GEN)[(fuel, cname)]
        if r is None:
            n_bad += 1
            rows.append((fuel, polar, cname, None, None, None))
            print(f'{fuel:6s} {polar:7s} {cname:12s} FAILED (no result)',
                  flush=True)
            continue
        dk = abs(r['keff'] - gold)
        pcm = (r['keff'] - gold) * 1e5
        ok = dk <= 1e-7 and r.get('converged', False)
        n_ok += ok
        n_bad += (not ok)
        rows.append((fuel, polar, cname, r['keff'], dk, pcm))
        print(f"{fuel:6s} {polar:7s} {cname:12s} A={r.get('A')} "
              f"keff={r['keff']:.12f}  golden={gold:.12f}  "
              f"|dk|={dk:.2e}  ({pcm:+.4f} pcm)  "
              f"iter={r['outer_iterations']}  conv={r['converged']}  "
              f"{r['wall_s']}s  {'PASS' if ok else 'FAIL'}", flush=True)
    print(f'\n=== {n_ok} passed, {n_bad} failed (port gate |dk|<=1e-7) ===')
    with open(f'{REPO}/tools_ho/acceptance_local.json', 'w') as f:
        json.dump([dict(fuel=a, polar=b, case=c, keff=d, dk=e, pcm=v)
                   for a, b, c, d, e, v in rows], f, indent=1)


if __name__ == '__main__':
    main()
