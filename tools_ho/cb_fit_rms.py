#!/usr/bin/env python3
"""Checkerboard A3 (I=30) M=24-column fit, S=10..20.

Paper claim (Section 4.4, checkerboard paragraph (iii)): over S=10..20 each
M=24 column of the A3 I=30 sweep is fit by
    dS = b + c2/S^2      (two-term)
and by
    dS = b + c1/S        (one-term),
with RMS 0.02/0.08 pcm (weak/strong, two-term) vs 0.16/0.84 pcm
(one-term) --- the quoted RMS use the UNROUNDED sweep values
(keff parsed from the run logs), not the rounded table entries.

Inputs (run logs from the HPC sweep):
    cb_i30_weak.log   / cb_i30_weak_ext.log   (S=1..6 / S=7..20)
    cb_i30_strong.log / cb_i30_strong_ext.log
Five-decimal references (source paper's 2.9a checkerboard):
    weak  1.12974
    strong 0.51673

Usage:  python tools_ho/cb_fit_rms.py
"""
import re
import numpy as np

REF = {"weak": 1.12974, "strong": 0.51673}
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = {
    "weak": [os.path.join(_HERE, f) for f in ("cb_i30_weak.log", "cb_i30_weak_ext.log")],
    "strong": [os.path.join(_HERE, f) for f in ("cb_i30_strong.log", "cb_i30_strong_ext.log")],
}
PAT = re.compile(r"^CB\w* M(\d+)_S(\d+) keff=([\d.]+) .*dref=([+-][\d.]+)pcm")


def collect(tag, tag_prefix):
    rows = {}
    for f in LOGS[tag]:
        try:
            lines = open(f).read().split("\n")
        except FileNotFoundError:
            continue
        for l in lines:
            m = PAT.match(l.strip())
            if not m:
                continue
            M, S, keff = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if M == 24:
                rows[S] = (keff - REF[tag]) * 1e5  # pcm
    return rows


def rms(resid):
    return float(np.sqrt(np.mean(np.asarray(resid) ** 2)))


def fit2(S, y):
    A = np.column_stack([np.ones_like(S), 1.0 / S**2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return rms(y - A @ coef)


def fit1(S, y):
    A = np.column_stack([np.ones_like(S), 1.0 / S])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return rms(y - A @ coef)


def main():
    for tag in ("weak", "strong"):
        rows = collect(tag, tag)
        Ss = sorted(S for S in rows if 10 <= S <= 20)
        if len(Ss) != 11:
            raise SystemExit(
                f"[{tag}] expected S=10..20 (11 points), got {Ss} -> {rows}")
        S = np.array(Ss, dtype=float)
        y = np.array([rows[S] for S in Ss], dtype=float)
        r2, r1 = fit2(S, y), fit1(S, y)
        print(f"[{tag}] n={len(Ss)}  S={Ss[0]:.0f}..{Ss[-1]:.0f}")
        print(f"    b+c2/S^2  RMS = {r2:.4f} pcm")
        print(f"    b+c1/S    RMS = {r1:.4f} pcm   (ratio {r1/r2:.1f}x)")
        for Si, yi in zip(Ss, y):
            print(f"      S={Si:2d}  dk={yi:+8.3f} pcm")


if __name__ == "__main__":
    main()
