#!/usr/bin/env python3
"""Compile eigen_chol.cpp against existing Eigen (and optional METIS).

No package installation, network request, or privilege escalation.

Examples:
  python psn2d/memopt_cxx/build.py
  python psn2d/memopt_cxx/build.py --eigen-include /usr/include/eigen3
  python psn2d/memopt_cxx/build.py --metis-include /path/include --metis-lib /path/lib/libmetis.so
  OPENPSN_EIGEN_INCLUDE=/path/include python psn2d/memopt_cxx/build.py
  OPENPSN_METIS_LIB=/path/lib/libmetis.so python psn2d/memopt_cxx/build.py

Output: psn2d/memopt_cxx/libeigen_chol.so  (the Python side finds it here by
default; override with the OPENPSN_EIGEN_CHOL environment variable).
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eigen-include', type=Path)
    parser.add_argument('--metis-include', type=Path)
    parser.add_argument('--metis-lib', type=Path)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent

    # ---- Eigen headers ----
    candidates = []
    if args.eigen_include:
        candidates.append(Path(args.eigen_include))
    if os.environ.get('OPENPSN_EIGEN_INCLUDE'):
        candidates.append(Path(os.environ['OPENPSN_EIGEN_INCLUDE']))
    candidates.extend([Path('/usr/include/eigen3'), Path('/usr/local/include/eigen3')])
    spec = importlib.util.find_spec('casadi')
    if spec and spec.origin:
        candidates.append(Path(spec.origin).parent / 'include' / 'eigen3')
    include = next((p for p in candidates
                    if (p / 'Eigen' / 'SparseCholesky').is_file()), None)
    if include is None:
        raise SystemExit('Eigen headers not found; provide --eigen-include. '
                         'No packages were installed.')

    # ---- METIS (optional; without it the bridge falls back to AMD) ----
    metis_inc = metis_lib = None
    if args.metis_include:
        metis_inc = Path(args.metis_include)
        metis_lib = args.metis_lib
    if os.environ.get('OPENPSN_METIS_LIB'):
        metis_lib = Path(os.environ['OPENPSN_METIS_LIB'])
    if metis_lib and metis_lib.parent.is_dir() and not metis_inc:
        metis_inc = metis_lib.parent.parent / 'include'
    if metis_inc and not (metis_inc / 'metis.h').is_file():
        print(f'warning: metis.h not found in {metis_inc}; building with AMD only')
        metis_inc = metis_lib = None

    cmd = shlex.split(args.cxx) + ['-std=c++17', '-O3', '-DNDEBUG', '-fPIC',
                                   '-shared', f'-I{include}']
    libs = []
    if metis_inc and metis_lib:
        cmd += [f'-I{metis_inc}', '-DOPENPSN_WITH_METIS',
                f'-L{metis_lib.parent}', f'-Wl,-rpath,{metis_lib.parent}']
        libs.append('-lmetis')
    cmd += [str(here / 'eigen_chol.cpp')]
    # library flags AFTER the source, so undefined symbols are actually
    # resolved at link time; --no-undefined makes any leftover one fatal
    # (a shared lib links "successfully" with unresolved symbols by
    # default and only explodes at dlopen).
    cmd += ['-Wl,--no-undefined'] + libs + ['-o', str(here / 'libeigen_chol.so')]
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f'built: {here / "libeigen_chol.so"}'
          + ('' if metis_lib else '  (AMD ordering only — no METIS linked)'))


if __name__ == '__main__':
    main()
