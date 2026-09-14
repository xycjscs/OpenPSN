# -*- coding: utf-8 -*-
"""Demo figure: rect4x4 assembly - FINAL topology v2 (2026-09-14).

User decision: "rect4 应该是完全相同的几何只是 pin 中间分成 4 块" —
rect4x4 = ONE rect3x3 PIN CELL (pitch 1.26 cm, all-reflective) with
the centred fuel square split into 2x2 = 4 touching blocks.

  side = f + 2w = 0.9571251 + 2*0.1514375 = 1.26 cm
  f = 0.9571251 (rect3x3 fuel square side), w = 0.1514375 (water ring,
  = half of the 0.3029 inter-cell gap, same as rect3x3)
  water area fraction = 1 - (f/side)^2 = 42.3%  (identical to rect3x3)

S1 = 4x4 nodes:  [w, f/2, f/2, w]  (centre 2x2 = 4 fuel blocks)
S2 = 8x8 nodes:  subdivide=2 -> fuel 2x2 -> 4x4 sub-nodes (f/4),
                 each water node -> 2 (w/2)
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

S_CM = 320   # px per cm (true proportions; side 1.26 cm -> ~403 px)
W_CM = 0.1514375          # water ring thickness (rect3x3 w)
F_CM = 0.9571251          # rect3x3 fuel square side
W2 = W_CM / 2             # 0.0757187 (S2 water sub-node)
F2 = F_CM / 4             # 0.2392813 (S2 fuel sub-node)


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

d.text((60, 22), 'rect4x4 组装体  —  rect3x3 pin 单元，燃料方块 2×2 分块',
       fill=COL_ACC, font=F_TITLE)
d.text((60, 62), '几何与 rect3x3 完全相同（边长 1.26 cm = 0.9571+2×0.1514）'
      '  |  仅燃料内部划分为 4 块  |  四边全反射',
       fill=COL_TEXT, font=F_SMALL)

# ---------- left: S1 = 4x4 nodes ----------
LX, LY = 90, 150
d.text((LX, 108), 'S1 = 4×4 节点（真实比例）', fill=COL_TEXT, font=F_SEC)
W1 = [W_CM, F_CM / 2, F_CM / 2, W_CM]
W1px = [w * S_CM for w in W1]
xs, ys = draw_grid(d, LX, LY, W1px,
                   lambda r, c: 'f' if (r in (1, 2) and c in (1, 2)) else 'w')
# fuel 2x2 outline
d.rectangle([xs[1], ys[1], xs[3], ys[3]], outline=(140, 70, 0), width=2)
d.text((xs[1] + 10, (ys[1] + ys[3]) // 2 - 14), '燃料 2×2 分块',
       fill=(140, 70, 0), font=F_SMALL)
# water ring bracket (left side)
d.line([LX - 10, ys[0], LX - 10, ys[1]], fill=(200, 120, 0), width=2)
d.text((LX - 34, (ys[0] + ys[1]) // 2 - 8), '水',
       fill=(200, 120, 0), font=F_SMALL)
# fuel bracket
d.line([LX - 22, ys[1], LX - 22, ys[3]], fill=(140, 70, 0), width=2)
d.text((LX - 46, (ys[1] + ys[3]) // 2 - 8), '燃料',
       fill=(140, 70, 0), font=F_SMALL)
# width labels below
d.text((LX, ys[4] + 10), '0.1514  |  0.4786  |  0.4786  |  0.1514  (cm)',
       fill=(90, 90, 90), font=F_SMALL)

# ---------- middle: S2 = 8x8 nodes ----------
MX, MY = 540, 150
d.text((MX, 108), 'S2 = 8×8 节点（subdivide=2，真实比例）',
       fill=COL_TEXT, font=F_SEC)
W2px = [W2 * S_CM, W2 * S_CM] + [F2 * S_CM] * 4 + [W2 * S_CM] * 2
xs2, ys2 = draw_grid(d, MX, MY, W2px,
                     lambda r, c: 'f'
                     if (2 <= r <= 5 and 2 <= c <= 5) else 'w')
# fuel 4x4 sub-node outline
d.rectangle([xs2[2], ys2[2], xs2[6], ys2[6]], outline=(140, 70, 0), width=2)
d.text((xs2[2] + 8, (ys2[2] + ys2[6]) // 2 - 14), '燃料 4×4 子节点',
       fill=(140, 70, 0), font=F_SMALL)
d.text((MX, ys2[8] + 10),
       'w/2 ×2  |  f/4 ×4  |  w/2 ×2（水条各一分为二）',
       fill=(90, 90, 90), font=F_SMALL)

# ---------- right: notes ----------
RX = 1010
d.text((RX, 108), '拓扑要点', fill=COL_TEXT, font=F_SEC)
notes = ('= rect3x3 的单个 pin 单元：\n'
         '边长 1.26 cm = 0.9571 + 2×0.1514\n'
         '（f + 2w，与 rect3x3 节距一致）\n\n'
         '燃料：中心 0.9571 cm 方块\n'
         '分成 2×2 = 4 块（各 0.4786，\n'
         '直接相邻、无夹层水）\n\n'
         '水环：四边 0.1514 cm（相邻单元\n'
         '间完整间隙 0.3029，同 rect3x3）\n'
         '水面积占比 42.3%（同 rect3x3）\n\n'
         'S1：4×4 节点（燃料 2×2）\n'
         'S2：8×8 节点（燃料 4×4 子节点）\n\n'
         '待算 4 点：UO2/MOX × S1/S2 × M24\n'
         '参考锚点：UO2 1.3347260 /\n'
         'MOX 1.1851053（OpenMC, rect3x3）\n'
         '同几何预期 keff 贴近锚点')
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
W2seq = [W2, W2, F2, F2, F2, F2, W2, W2]
for i, wpx in enumerate(W2px):
    d.rectangle([cx, TY + 50, cx + wpx, TY + 84],
                fill=COL_FUEL if 2 <= i <= 5 else COL_WATER,
                outline=(80, 80, 80), width=1)
    d.text((cx + 2, TY + 60), f'{W2seq[i]:.4f}', fill=(20, 20, 20), font=F_SMALL)
    cx += wpx

img.save(OUT)
print(OUT)
