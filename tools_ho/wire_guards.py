#!/usr/bin/env python3
"""Wire hard guards into the three sweep drivers (2026-09-14):
_pin_24() + _mem_guard_32g() at run_point entry; coordinator pins too."""
import re, sys

FILES = [
    '/opt/data/workspace/OpenPSN/tools_ho/dir_table3_sweep_gmres.py',
    '/opt/data/workspace/OpenPSN/tools_ho/rect4x4_run.py',
    '/opt/data/workspace/OpenPSN/tools_ho/rect17_run.py',
]

OLD_GUARD = """    spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                           'threads': 1})
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                           (int(VSZ_BACKSTOP_GB * 1e9), hard))
        capped = True
    except (ValueError, OSError):
        capped = False
    try:"""

NEW_GUARD = """    spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                           'threads': 1})
    _pin_24()          # hard: 24-core affinity (inherited by children)
    _mem_guard_32g()   # hard: RLIMIT_AS 32 GiB backstop
    try:"""

REST_OK = """        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        dt = time.time() - t0"""
REST_OK_NEW = "        dt = time.time() - t0"

REST_ME = """    except MemoryError:
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        print("""
REST_ME_NEW = """    except MemoryError:
        print("""

REST_EX = """    except Exception as e:
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        print("""
REST_EX_NEW = """    except Exception as e:
        print("""

for f in FILES:
    t = open(f).read()
    n = 0
    assert t.count(OLD_GUARD) == 1, (f, 'guard')
    t = t.replace(OLD_GUARD, NEW_GUARD); n += 1
    assert t.count(REST_OK) == 1, (f, 'rest_ok')
    t = t.replace(REST_OK, REST_OK_NEW); n += 1
    assert t.count(REST_ME) == 1, (f, 'rest_me')
    t = t.replace(REST_ME, REST_ME_NEW); n += 1
    assert t.count(REST_EX) == 1, (f, 'rest_ex')
    t = t.replace(REST_EX, REST_EX_NEW); n += 1
    n1 = t.count('mem_limit_gb=120')
    t = t.replace('mem_limit_gb=120', 'mem_limit_gb=32')
    open(f, 'w').write(t)
    print(f.split('/')[-1], 'replaced', n, 'memlim', n1)

# coordinator: pin at __main__ entry (children inherit affinity)
for f in FILES:
    t = open(f).read()
    m = re.search(r"if __name__ == '__main__':\n    if len\(sys.argv\)", t)
    assert m, f
    t = t.replace(m.group(0),
                  m.group(0).replace("if len(sys.argv)",
                                     "_pin_24()\n    if len(sys.argv)"),
                  1)
    open(f, 'w').write(t)
    print(f.split('/')[-1], 'coordinator pinned')

print('WIRING DONE')
