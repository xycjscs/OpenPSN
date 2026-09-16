# -*- coding: utf-8 -*-
"""Generate the C5G7 1/4-core 2x2-fuel-split YAML (2026-09-15 user request:
"core 的 rect4" = quarter core, every pin fuel split 2x2, same convention
as the assembly's rect17_split4).

Source: c5g7_rect_quarter_core.yaml — 153x153 base nodes (51x51 pins,
each pin 3x3 [w,f,w]).
Target: 204x204 base grid (51x51 pins, each pin 4x4).

Transformation per pin (3x3 -> 4x4 base), identical to the assembly
converter (make_rect17_split4.py):
  old sub 0 (water) -> new 0
  old sub 1 (center)-> new {1,2}x{1,2}, same material (or water if the
                       pin center itself is water — the core has 1445
                       water-center pins in the reflector region),
                       width f -> 2*(f/2)
  old sub 2 (water) -> new 3
Physically identical geometry (same pitch, water gap, materials,
boundaries: reflect L/bottom, vacuum R/top); only the fuel square
inside each pin is split 2x2.

S1 = 204x204 base; S2 = subdivide 2 -> 408x408.
kref: 1.1872176 = same-geometry rect OpenMC reference (collision
estimator, per c5g7_rect6_quarter_core.yaml header); 1.18646 =
MCNP5 7-group (per source header).
"""
import os, re, sys

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'examples', 'c5g7', 'study2_rectangular')
N = 153
P = N // 3        # 51 pins per side
N4 = P * 4        # 204


def parse_rows(block):
    rows, cur = [], None
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith('- -'):
            cur = [s.split('- -', 1)[1].strip()]
            rows.append(cur)
        elif s.startswith('- '):
            assert cur is not None, ln
            cur.append(s[2:].strip())
        elif s:
            raise AssertionError('unexpected line: ' + ln)
    return rows


def get_block(txt, key, end_keys):
    pat = r'^  ' + re.escape(key) + r':\n(.*?)(?=^ {0,2}' + \
        '|'.join(re.escape(k.rstrip(':')) for k in end_keys) + r':)'
    m = re.search(pat, txt, re.M | re.S)
    assert m, key
    return m.span(1), m.group(1)


def split_widths(row):
    assert len(row) == N, len(row)
    out = []
    for px in range(P):
        w0, f, w2 = (float(v) for v in row[3 * px:3 * px + 3])
        out += [w0, f / 2.0, f / 2.0, w2]
    return out


def fmt(v):
    return repr(v)


def emit_rows(rows):
    out = []
    for r in rows:
        out.append('  - - ' + fmt(r[0]))
        for v in r[1:]:
            out.append('    - ' + fmt(v))
    return '\n'.join(out) + '\n'


def convert():
    src = EX + '/c5g7_rect_quarter_core.yaml'
    dst = EX + '/c5g7_core_split4_uo2_mox_quarter_core.yaml'
    txt = open(src).read()

    # --- mat_grid ---
    (s, e), block = get_block(txt, 'mat_grid', ['widths_x'])
    old = parse_rows(block)
    assert len(old) == N and all(len(r) == N for r in old)
    # core-specific: off-center must be water; center may be fuel OR
    # water (reflector region)
    n_fuel_center = n_water_center = n_off = 0
    for r in range(P):
        for c in range(P):
            for rr in range(3):
                for cc in range(3):
                    v = old[3 * r + rr][3 * c + cc]
                    if rr == 1 and cc == 1:
                        if v == '0':
                            n_water_center += 1
                        else:
                            n_fuel_center += 1
                    else:
                        assert v == '0', (r, c, rr, cc, v)
                        n_off += 1
    print(f'pin centers: fuel={n_fuel_center} water={n_water_center}')

    new = [['0'] * N4 for _ in range(N4)]
    for r in range(P):
        for c in range(P):
            center = old[3 * r + 1][3 * c + 1]
            for rr in (1, 2):
                for cc in (1, 2):
                    new[4 * r + rr][4 * c + cc] = center
    txt = txt[:s] + emit_rows(new) + txt[e:]

    # --- widths_x (row-constant) ---
    (s, e), block = get_block(txt, 'widths_x', ['widths_y'])
    wx = parse_rows(block)
    assert len(wx) == N and all(len(r) == N for r in wx)
    assert all(r == wx[0] for r in wx), 'widths_x not row-constant'
    new_wx = [split_widths(wx[0])] * N4
    txt = txt[:s] + emit_rows(new_wx) + txt[e:]

    # --- widths_y (each new ROW constant; row pattern expands 153->204)
    (s, e), block = get_block(txt, 'widths_y',
                              ['boundaries:', 'left:', 'right:',
                               'top:', 'bottom:', 'materials_file'])
    wy = parse_rows(block)
    assert len(wy) == N and all(len(r) == N for r in wy)
    assert all(v == r[0] for r in wy for v in r), \
        'widths_y rows not constant'
    wy_new = split_widths([float(r[0]) for r in wy])
    new_wy = [[v] * N4 for v in wy_new]
    txt = txt[:s] + emit_rows(new_wy) + txt[e:]

    # --- problem / description / tolerance ---
    problem = 'c5g7_core_split4_quarter_core'
    txt = re.sub(r'^problem: .*$', 'problem: ' + problem, txt,
                 count=1, flags=re.M)
    desc = ('C5G7 1/4 core, 204x204 base nodes (51x51 pins, each pin '
            'fuel split 2x2 = 4 blocks), PURE XS, reflect L/bottom, '
            'vacuum R/top; geometry identical to c5g7_rect_quarter_core')
    txt = re.sub(r'^description: [^\n]*\n(?:  [^\n]*\n)*?PURE XS',
                 'description: ' + desc, txt, count=1, flags=re.M)
    # description may span two lines; fallback: replace both
    if '153x153 rectangular nodes' in txt.split('\n')[7] if len(txt.split('\n')) > 7 else True:
        lines = txt.split('\n')
        for i, ln in enumerate(lines):
            if ln.startswith('description:'):
                lines[i] = 'description: ' + desc
                # drop continuation line if it exists (indented, part of
                # the same scalar)
                if i + 1 < len(lines) and re.match(r'^  [a-z]',
                                                   lines[i + 1]) and \
                   lines[i + 1].strip() and \
                   ':' not in lines[i + 1]:
                    del lines[i + 1]
                txt = '\n'.join(lines)
                break
    txt = re.sub(r'^  keff_tol: .*$', '  keff_tol: 1.0e-07', txt,
                 flags=re.M)
    m = re.search(r'^  kref: .*$', txt, flags=re.M)
    if m:
        txt = txt[:m.start()] + '  kref: 1.1872176' + txt[m.end():]

    # header comments
    lines = txt.splitlines(keepends=True)
    lines[:8] = [
        '# PSN2D — C5G7 1/4 core, 204x204 base nodes (51x51 pins, each\n',
        '# pin fuel split 2x2 = 4 blocks), PURE XS.  Geometry identical\n',
        '# to c5g7_rect_quarter_core.yaml (153x153, 3x3 pin cells):\n',
        '# same pitch/water/materials/boundaries, only the fuel square\n',
        '# inside each pin is split 2x2.  S1 = 204x204; S2 = 408x408.\n',
        '# kref 1.1872176 = same-geometry rect OpenMC (collision est);\n',
        '# 1.18646 = MCNP5 7-group (C5G7-TD Part I 2D steady).\n',
    ]
    open(dst, 'w').write(''.join(lines))

    # verification
    rowsum_x = sum(float(v) for v in new_wx[0])
    rowsum_y = sum(float(r[0]) for r in new_wy)
    mats = {}
    for r in new:
        for v in r:
            mats[v] = mats.get(v, 0) + 1
    side = 51 * 1.26
    print(f'{os.path.basename(dst)}: 204x204, side_x={rowsum_x:.7f} '
          f'side_y={rowsum_y:.7f} (expect {side:.5f}), '
          f'mats={dict(sorted(mats.items()))}')
    assert abs(rowsum_x - side) < 1e-9 and abs(rowsum_y - side) < 1e-9


if __name__ == '__main__':
    convert()
