# -*- coding: utf-8 -*-
"""Demo figure: rect4x4 assembly - FINAL topology (2026-09-14).

S1 = 4x4 NODES: centre 2x2 = 4 FUEL nodes (pins directly touching,
no water between them), outer ring = 12 water nodes.
   widths_x = widths_y = [w, f, f, w]
S2 = 8x8 NODES: subdivide=2 splits every node 2x2 (fuel 2x2 -> 4x4
fuel sub-nodes; edge water strips split into 2 naturally):
   widths = [w/2, w/2, f/2, f/2, f/2, f/2, w/2, w/2]
w = 0.1514375 cm (half gap), f = 0.9571251 cm.
Side = 2*(f+w) = 2.21714 cm.  All four faces reflective.
UO2 / MOX = same layout, fuel material swapped only.
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'rect4x4_demo.png')

try:
    F_TITLE = ImageFont.truetype(
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 27)
    F_SEC = ImageFont.truetype(
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 19)
    F_SMALL = ImageFont.truetype(
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 15)
except Exception:
    F_TITLE = ImageFont.load_default()
    F_SEC = F_SMALL = F_TITLE

COL_BG = (255, 255, 255)
COL_FUEL = (240, 173, 78)
COL_WATER = (173, 216, 230)
COL_GRID = (255, 255, 255)
COL_TEXT = (33, 37, 41)
COL_ACC = (178, 34, 34)

S_CM = 180   # px per cm (true proportions; side = 2.21714cm -> ~399 px)
W_CM = 0.1514375
F_CM = 0.9571251


def draw_grid(d, ox, oy, widths_px, mat):
    """Draw a node grid with true proportions; mat(r,c) -> 'f'/'w'."""
    n = len(widths_px)
    xs = [ox]
    for wpx in widths_px:
        xs.append(xs[-1] + wpx)
    ys = [oy]
    for wpx in widths_px:
        ys.append(ys[-1] + wpx)
    for r in range(n):
        for c in range(n):
            d.rectangle([xs[c], ys[r], xs[c + 1], ys[r + 1]],
                        fill=COL_FUEL if mat(r, c) == 'f' else COL_WATER)
    for nline in range(n + 1):
        d.line([xs[nline], oy, xs[nline], ys[n]], fill=COL_GRID, width=1)
        d.line([ox, ys[nline], xs[n], ys[nline]], fill=COL_GRID, width=1)
    d.rectangle([ox, oy, xs[n], ys[n]], outline=(50, 50, 50), width=2)
    return xs, ys


# ---------------- layout ----------------
WIMG, HIMG = 1360, 860
img = Image.new('RGB', (WIMG, HIMG), COL_BG)
d = ImageDraw.Draw(img)

d.text((60, 22), 'rect4x4 组装体  —  中心 2×2 燃料，外圈一圈水',
       fill=COL_ACC, font=F_TITLE)
d.text((60, 62), 'UO2 / MOX 同布局仅换燃料  |  四边全反射  |  '
      '边长 = 2×(0.9571251+0.1514375) = 2.21714 cm',
       fill=COL_TEXT, font=F_SMALL)

# ---------- left: S1 = 4x4 nodes ----------
LX, LY = 90, 150
d.text((LX, 108), 'S1 = 4×4 节点（真实比例）', fill=COL_TEXT, font=F_SEC)
W1 = [W_CM, F_CM, F_CM, W_CM]
W1px = [w * S_CM for w in W1]
xs, ys = draw_grid(d, LX, LY, W1px,
                   lambda r, c: 'f' if (r in (1, 2) and c in (1, 2)) else 'w')
# fuel 2x2 outline
d.rectangle([xs[1], ys[1], xs[3], ys[3]], outline=(140, 70, 0), width=2)
d.text((xs[1] + 10, (ys[1] + ys[3]) // 2 - 14), '燃料 2×2',
       fill=(140, 70, 0), font=F_SMALL)
# water ring bracket (left side)
d.line([LX - 10, ys[0], LX - 10, ys[1]], fill=(200, 120, 0), width=2)
d.text((LX - 60, (ys[0] + ys[1]) // 2 - 8), '水',
       fill=(200, 120, 0), font=F_SMALL)
# fuel bracket
d.line([LX - 22, ys[1], LX - 22, ys[3]], fill=(140, 70, 0), width=2)
d.text((LX - 70, (ys[1] + ys[3]) // 2 - 8), '燃料',
       fill=(140, 70, 0), font=F_SMALL)
# width labels below
d.text((LX, ys[4] + 10), '0.1514  |  0.9571  |  0.9571  |  0.1514',
       fill=(90, 90, 90), font=F_SMALL)

# ---------- middle: S2 = 8x8 nodes ----------
MX, MY = 520, 150
d.text((MX, 108), 'S2 = 8×8 节点（subdivide=2，真实比例）',
       fill=COL_TEXT, font=F_SEC)
W2 = [W_CM / 2] * 2 + [F_CM / 2] * 4 + [W_CM / 2] * 2
W2px = [w * S_CM for w in W2]
xs2, ys2 = draw_grid(d, MX, MY, W2px,
                     lambda r, c: 'f'
                     if (2 <= r <= 5 and 2 <= c <= 5) else 'w')
# fuel 4x4 sub-node outline
d.rectangle([xs2[2], ys2[2], xs2[6], ys2[6]], outline=(140, 70, 0), width=2)
d.text((xs2[2] + 8, (ys2[2] + ys2[6]) // 2 - 14), '燃料 4×4 子节点',
       fill=(140, 70, 0), font=F_SMALL)
d.text((MX, ys2[8] + 10),
       'w/2 ×2  |  f/2 ×4  |  w/2 ×2（边缘水条自然一分为二）',
       fill=(90, 90, 90), font=F_SMALL)

# ---------- right: notes ----------
RX = 1010
d.text((RX, 108), '拓扑要点', fill=COL_TEXT, font=F_SEC)
notes = ('S1：中心 2×2 燃料节点直接相邻，\n'
         '无水夹层；外圈 12 个水节点。\n\n'
         'S2：subdivide=2 均匀 2×2 分解——\n'
         '燃料 2×2 → 4×4 子节点（16 个），\n'
         '水节点 → 2 段，边缘水条顺其自然\n'
         '一分为二，网格自动对齐。\n\n'
         '水节点 = 0.1514375 cm（半间隙，\n'
         '与 rect3x3 约定一致）\n'
         '燃料节点 = 0.9571251 cm\n\n'
         '待算 4 点：UO2/MOX × S1/S2 × M24\n'
         'OpenMC 参考沿用 rect3x3 锚点：\n'
         'UO2 1.3347260 / MOX 1.1851053\n'
         '（2×2 pin 簇 keff 预期偏离，主验证 =\n'
         'S1 vs S2 交叉一致性）')
d.text((RX, 150), notes, fill=COL_TEXT, font=F_SMALL, spacing=7)

# ---------- bottom: width sequences (S1 & S2) ----------
TY = 700
d.text((60, TY - 30), '节点宽度序列（x = y，cm）', fill=COL_TEXT, font=F_SEC)
d.text((60, TY + 6), 'S1 (4 节点):', fill=COL_TEXT, font=F_SMALL)
cx = 180
for i, wpx in enumerate(W1px):
    d.rectangle([cx, TY, cx + wpx, TY + 34],
                fill=COL_FUEL if i in (1, 2) else COL_WATER,
                outline=(80, 80, 80), width=1)
    d.text((cx + 2, TY + 8), f'{W1[i]:.4f}', fill=(20, 20, 20), font=F_SMALL)
    cx += wpx

d.text((60, TY + 56), 'S2 (8 节点):', fill=COL_TEXT, font=F_SMALL)
cx = 180
for i, wpx in enumerate(W2px):
    d.rectangle([cx, TY + 50, cx + wpx, TY + 84],
                fill=COL_FUEL if 2 <= i <= 5 else COL_WATER,
                outline=(80, 80, 80), width=1)
    d.text((cx + 2, TY + 60), f'{W2[i]:.4f}', fill=(20, 20, 20), font=F_SMALL)
    cx += wpx

img.save(OUT)
print(OUT)
