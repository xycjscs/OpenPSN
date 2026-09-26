#!/usr/bin/env python3
"""Recompute every Sec. II/IV statistic from all_points_archive.csv.
Stdlib only.  Run: python3 audit_stats.py  (in this directory).
Prints: separability medians & window flags, refinement-direction
counts, Richardson self-consistency, fixed-budget hit fractions.
"""
import csv, collections, itertools, statistics

g = collections.defaultdict(dict)
for r in csv.DictReader(open("all_points_archive.csv")):
    if r["case"] != "checkerboard":
        continue
    S, M = int(r["S"]), int(r["M"])
    if S <= 20 and M <= 192:
        g[(r["closure"], r["material"])][(S, M)] = float(r["dref_pcm"])

for k in sorted(g):
    d = g[k]
    Ms = sorted(set(m for _, m in d)); Ss = sorted(set(s for s, _ in d))
    # 1. full-map rectangle separability
    res = []
    for S1, S2 in itertools.combinations(Ss, 2):
        for M1, M2 in itertools.combinations(Ms, 2):
            try:
                res.append(abs(d[(S1, M1)] - d[(S1, M2)]
                               - d[(S2, M1)] + d[(S2, M2)]))
            except KeyError:
                pass
    # 2. sliding 4-row x 3-col windows, >20 pcm flag
    fl = tot = 0
    for i in range(len(Ss) - 3):
        for j in range(len(Ms) - 2):
            try:
                w = abs(d[(Ss[i], Ms[j])] - d[(Ss[i], Ms[j + 2])]
                        - d[(Ss[i + 3], Ms[j])] + d[(Ss[i + 3], Ms[j + 2])])
                tot += 1; fl += w > 20
            except KeyError:
                pass
    # 3. refinement direction |d(S,2M)|<|d(S,M)|, pairs 48->96, 96->192
    rd = []
    for Ma, Mb in ((48, 96), (96, 192)):
        hit = n = 0
        for S in Ss:
            if (S, Ma) in d and (S, Mb) in d:
                n += 1; hit += abs(d[(S, Mb)]) < abs(d[(S, Ma)])
        rd.append(f"{hit}/{n}")
    # 4. Richardson self-consistency predicting M=192 (rows S>=2)
    hit = n = 0
    for S in Ss:
        if S < 2: continue
        v = {M: d.get((S, M)) for M in (24, 48, 96, 192)}
        if any(x is None for x in v.values()): continue
        bar = abs((4*v[96]-v[48])/3 - (4*v[48]-v[24])/3)
        hit += abs((4*v[96]-v[48])/3 - v[192]) <= bar; n += 1
    # 5. fixed-budget subgrid
    pts = [abs(x) for (S, M), x in d.items() if M in (8, 16, 24, 48)]
    f = lambda t: sum(x < t for x in pts) / len(pts) * 100
    print(f"{k}: rects {len(res)} med {statistics.median(res):.1f} | "
          f"win {tot} flag>20 {fl} | refine {rd} | self-cons {hit}/{n} | "
          f"<20/50pcm {f(20):.0f}/{f(50):.0f}%")
