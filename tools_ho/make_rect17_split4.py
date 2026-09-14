# -*- coding: utf-8 -*-
"""Generate the 17x17-assembly-with-4-block-pins YAMLs from the rect3x3
YAMLs (2026-09-14 user request: "完全相同的几何只是 pin 中间分成 4 块",
at the FULL assembly level, not the single pin cell).

Transformation, per pin (3x3 base [w,f,w] x [w,f,w] -> 4x4 base):
  old sub-index 0 (water) -> new 0
  old sub-index 1 (fuel)  -> new {1,2} x {1,2}, same material, f -> 2*(f/2)
  old sub-index 2 (water) -> new 3
Everything else (material map, water, pitch, side length 21.42 cm,
all-reflect boundaries, pure XS) is unchanged.  51x51 -> 68x68.

S1 = 68x68 base;  S2 = subdivide 2 -> 136x136.
"""
import os, re, sys

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
N = 51
P = N // 3  # 17 pins per side
N4 = P * 4  # 68

FUEL_SUB = 1  # old (1,1) is the pin center (only non-water sub-cell)


def parse_rows(block):
    """YAML nested-list block ('  - - v' / '    - v') -> list of rows."""
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
    # end keys may be nested (2-space) or top-level (0-space)
    pat = r'^  ' + re.escape(key) + r':\n(.*?)(?=^ {0,2}' + \
        '|'.join(re.escape(k.rstrip(':')) for k in end_keys) + r':)'
    m = re.search(pat, txt, re.M | re.S)
    assert m, key
    return m.span(1), m.group(1)


def split_widths(row):
    assert len(row) == N, len(row)
    out = []
    for px in range(P):
        w0, f, w2 = row[3 * px:3 * px + 3]
        w0, f, w2 = float(w0), float(f), float(w2)
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


def convert(src, dst, problem, desc):
    txt = open(src).read()

    # --- mat_grid ---
    (s, e), block = get_block(txt, 'mat_grid', ['widths_x'])
    old = parse_rows(block)
    assert len(old) == N and all(len(r) == N for r in old)
    # assert only pin centers are non-water
    nz = 0
    for r in range(P):
        for c in range(P):
            for rr in range(3):
                for cc in range(3):
                    v = old[3 * r + rr][3 * c + cc]
                    if rr == 1 and cc == 1:
                        nz += 1 if v != '0' else 0
                    else:
                        assert v == '0', (r, c, rr, cc, v)
    assert nz == P * P, nz

    new = [['0'] * N4 for _ in range(N4)]
    for r in range(P):
        for c in range(P):
            center = old[3 * r + 1][3 * c + 1]
            for rr in (1, 2):
                for cc in (1, 2):
                    new[4 * r + rr][4 * c + cc] = center
    txt = txt[:s] + emit_rows(new) + txt[e:]

    # --- widths_x (rows: x-width per node, identical across rows) ---
    (s, e), block = get_block(txt, 'widths_x', ['widths_y'])
    wx = parse_rows(block)
    assert len(wx) == N and all(len(r) == N for r in wx)
    assert all(r == wx[0] for r in wx), 'widths_x not row-constant'
    new_wx = [split_widths(wx[0])] * N4
    txt = txt[:s] + emit_rows(new_wx) + txt[e:]

    # --- widths_y (rows: y-width per node; structure is the TRANSPOSE
    # of widths_x: row i is constant across columns, identical down
    # the rows) ---
    (s, e), block = get_block(txt, 'widths_y',
                              ['boundaries:', 'left:', 'right:',
                               'top:', 'bottom:', 'materials_file'])
    wy = parse_rows(block)
    assert len(wy) == N and all(len(r) == N for r in wy)
    assert all(v == r[0] for r in wy for v in r), \
        'widths_y rows not constant'
    wy_col = [r[0] for r in wy]          # the 51-pattern, same for all j
    new_wy = [split_widths(wy_col)] * N4
    txt = txt[:s] + emit_rows(new_wy) + txt[e:]

    # --- header / problem / description / tolerance ---
    txt = re.sub(r'^problem: .*$', 'problem: ' + problem, txt,
                 count=1, flags=re.M)
    txt = re.sub(r'^description: [^\n]*\n  PURE material XS, all-reflective',
                 'description: ' + desc, txt, count=1, flags=re.M)
    txt = re.sub(r'^  keff_tol: .*$', '  keff_tol: 1.0e-07', txt,
                 flags=re.M)

    # header comments: replace first 4 lines
    lines = txt.splitlines(keepends=True)
    lines[:4] = ['# PSN2D — C5G7 17x17 assembly, 68x68 base nodes '
                 '(17x17 pins, each pin fuel split 2x2 = 4 blocks), PURE XS\n',
                 f'# S1 = 68x68; S2 = 136x136 (subdivide 2).  Geometry '
                 'identical to the rect3x3 assembly\n',
                 '# (pitch, water, materials, all-reflect); only the fuel '
                 'square inside every\n',
                 '# pin is split 2x2.  kref = rect3x3 OpenMC anchors.\n']
    open(dst, 'w').write(''.join(lines))

    # verification report
    rowsum_x = sum(float(v) for v in new_wx[0])
    rowsum_y = sum(float(v) for v in new_wy[0])
    mats = {}
    for r in new:
        for v in r:
            mats[v] = mats.get(v, 0) + 1
    expect = 17 * 1.26
    print(f'{os.path.basename(dst)}: 68x68, side_x={rowsum_x:.7f} '
          f'side_y={rowsum_y:.7f} (expect {expect:.5f}), '
          f'mats={dict(sorted(mats.items()))}')
    assert abs(rowsum_x - expect) < 1e-9 and abs(rowsum_y - expect) < 1e-9


if __name__ == '__main__':
    convert(EX + '/c5g7_rect_uo2_assembly.yaml',
            EX + '/c5g7_rect17_split4_uo2_assembly.yaml',
            'c5g7_rect17_split4_uo2_assembly',
            'C5G7 17x17 UO2 assembly, 68x68 base nodes (17x17 pins, fuel '
            'split 2x2=4 blocks each), PURE XS, all-reflective')
    convert(EX + '/c5g7_rect_mox_assembly.yaml',
            EX + '/c5g7_rect17_split4_mox_assembly.yaml',
            'c5g7_rect17_split4_mox_assembly',
            'C5G7 17x17 MOX assembly, 68x68 base nodes (17x17 pins, fuel '
            'split 2x2=4 blocks each), PURE XS, all-reflective')
