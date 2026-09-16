import re, sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(REPO, 'examples', 'c5g7', 'study2_rectangular', 'c5g7_core_split4_uo2_mox_quarter_core.yaml')
txt = open(F).read()
m = re.search(r'^  mat_grid:\n(.*?)(?=^  widths_x)', txt, re.M|re.S)
rows, cur = [], None
for ln in m.group(1).splitlines():
    s = ln.strip()
    if s.startswith('- -'):
        cur = [s.split('- -', 1)[1].strip()]
        rows.append(cur)
    elif s.startswith('- '):
        cur.append(s[2:].strip())
print("grid N =", len(rows), "cols:", len(rows[0]))
P = 51
n4 = 0
for r in range(P):
    for c in range(P):
        v = rows[4*r+1][4*c+1]
        if v != '0':
            for rr in (1, 2):
                for cc in (1, 2):
                    assert rows[4*r+rr][4*c+cc] == v
                    n4 += 1
print("fuel blocks:", n4, "(expect 4x1156=4624)")
m2 = re.search(r'^  widths_x:\n(.*?)(?=^  widths_y)', txt, re.M|re.S)
wr, cur2 = [], None
for ln in m2.group(1).splitlines():
    s = ln.strip()
    if s.startswith('- -'):
        cur2 = [s.split('- -', 1)[1].strip()]
        wr.append(cur2)
    elif s.startswith('- '):
        cur2.append(s[2:].strip())
print("widths_x rows:", len(wr), "cols:", len(wr[0]),
      "rowsum:", sum(float(v) for v in wr[0]),
      "row-const:", all(r == wr[0] for r in wr))
for l in txt.split('\n'):
    if 'problem:' in l or 'kref:' in l or 'keff_tol' in l:
        print(l)
sys.path.insert(0, REPO)
from psn2d import model
s = model.load_spec(F)
print('load_spec OK, problem =', s.get('problem'))
