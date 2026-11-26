r"""Regenerate the Study IV S x M matrices of the closure of record.

Every panel is a spatial subdivision S by azimuthal sector M matrix of the eigenvalue
deviation, read back from the tracked per-point data rather than from a scratch journal::

    python -m scripts.study4_sm_figures                      # all five figure sets
    python -m scripts.study4_sm_figures --figures checker,checker-wide

``checker`` and ``checker-wide`` are the strong and weak absorber checkerboards of
Section 4, ``assembly-2x2`` and ``assembly-wall`` the four C5G7 assemblies (square on top,
rectangular below; UO$_2$ left, MOX right), and ``overview`` stacks all of them next to the
published quarter-core matrix.

Each panel has its own scale. Use the printed pcm values to compare panels. Blank cells
represent scheduled points without a measurement, and hatching represents unscheduled cells.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.colors import LogNorm, SymLogNorm                         # noqa: E402
from matplotlib.gridspec import GridSpec                                  # noqa: E402
from matplotlib.patches import Circle, Rectangle                          # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLIES = {"square_uo2": r"Square UO$_2$", "square_mox": "Square MOX",
              "rect_uo2": r"Rectangular UO$_2$", "rect_mox": "Rectangular MOX"}
ABSORBERS = {"strong": "Strong absorber", "weak": "Weak absorber"}
CORE_PANEL = "core"


def panel_of(family, case, degree):
    """Which figure panel a planned or measured point belongs to."""
    if family in ("square", "rect"):
        return case
    if family == "sm6_checker":
        return f"wide:{case}"
    return f"checker:{case}:{degree}"


def load_plan(path):
    """Planned cells per panel, and an index that classifies a measured cell."""
    planned, index = {}, {}
    for point in json.loads(path.read_text()):
        panel = panel_of(point["family"], point.get("case"), int(point.get("A", 1)))
        cell = (int(point.get("subdivide", point.get("S"))), int(point["M"]))
        planned.setdefault(panel, set()).add(cell)
        index[(point.get("case"), int(point.get("A", 1)), cell)] = panel
    return planned, index


def quarter_core(path):
    """The published quarter-core matrix: header ``S,M2,M4,...``, one row per S."""
    header, *rows = path.read_text().splitlines()
    sectors = [int(cell.lstrip("M")) for cell in header.split(",")[1:]]
    matrix = {}
    for line in rows:
        cells = line.split(",")
        matrix.update({(int(cells[0]), m): float(value)
                       for m, value in zip(sectors, cells[1:]) if value.strip()})
    return matrix


def read_points(plan_path, checker_path, points_path, core_path, closure):
    """Measured cells by panel for one closure, as {panel: {(S, M): pcm}}."""
    planned, index = load_plan(plan_path)
    groups = {panel: {} for panel in planned}
    seconds = {panel: {} for panel in planned}
    with checker_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["closure"] != closure:
                continue
            cell = (int(row["subdivide"]), int(row["M"]))
            # A handful of legacy rows carry no polynomial degree; they are not in the plan.
            degree = int(row["A"]) if row["A"] else -1
            panel = index.get((row["problem"].split("-")[-1], degree, cell))
            if panel:                                   # a point this campaign measured
                groups[panel][cell] = float(row["pcm"])
    with points_path.open() as handle:
        for row in csv.DictReader(handle):
            if (row.get("closure") or "sector_average") != closure:
                continue
            panel, cell = row["case"], (int(row["S"]), int(row["M"]))
            groups.setdefault(panel, {})[cell] = float(row["pcm"])
            seconds.setdefault(panel, {})[cell] = float(row["seconds"])
    groups[CORE_PANEL] = quarter_core(core_path)
    planned[CORE_PANEL] = {(s, m) for s in range(1, 6) for m in (2, 4, 8, 12, 16, 24)}
    # "checker" and "sm6_checker" are two sweep grids of one problem, not two problems: the
    # driver labels a point by the list it came from.  The wide panel therefore shows the
    # whole A=1 matrix, the 5 x 4 published grid included, so that no measured point of it
    # reads as a gap.
    for case in ABSORBERS:
        narrow = panel_of("checker", case, 1)
        wide = panel_of("sm6_checker", case, 1)
        planned[wide] |= planned[narrow]
        groups[wide] = {**groups[narrow], **groups[wide]}
    return planned, groups, seconds


def label_of(panel):
    if panel in ASSEMBLIES:
        return f"{ASSEMBLIES[panel]} ($A{{=}}1$, $I{{=}}8$)"
    if panel == CORE_PANEL:
        return "C5G7 quarter core ($A{=}1$, TY3)"
    kind, case = panel.split(":")[:2]
    figure = ABSORBERS[case]
    if kind == "wide":
        return f"{figure} checkerboard, $A{{=}}1$, $S{{=}}1{{-}}20$"
    return f"{figure} checkerboard, $A={panel.rpartition(':')[2]}$, $I{{=}}30$"


def short_label(panel):
    """Panel title for the overview, where the band already names the problem family."""
    if panel in ASSEMBLIES:
        return ASSEMBLIES[panel]
    if panel == CORE_PANEL:
        return "Quarter core (TY3)"
    kind, case = panel.split(":")[:2]
    if kind == "wide":
        return f"{ABSORBERS[case]}, $S{{=}}1{{-}}20$"
    return f"$A={panel.rpartition(':')[2]}$, $I{{=}}30$"


def grid_of(points, planned):
    """Dense matrix and its axes; an unscheduled cell still keeps its row and column."""
    S = sorted({s for s, _ in planned} | {s for s, _ in points})
    M = sorted({m for _, m in planned} | {m for _, m in points})
    values = np.full((len(S), len(M)), np.nan)
    for (i, s) in enumerate(S):
        for (j, m) in enumerate(M):
            values[i, j] = points.get((s, m), np.nan)
    return values, S, M


def best_cell(points):
    """The point of a panel with the smallest |delta k| -- the one the ring marks."""
    return min(points, key=lambda cell: abs(points[cell])) if points else None


def draw(axis, panel, points, planned, field="pcm", fontsize=7.5, missing_text="",
         title=None, mark=None, colorbar="label", ylabel=True):
    """One matrix on its own scale, ringed at the accuracy-optimal point.

    ``mark`` lets the wall-time panel ring the same point as its error panel, so the two
    figures tell the reader what the winning point costs rather than what the cheapest one
    costs.  ``colorbar`` is "label", "bare" or None, and ``ylabel`` is off for every panel
    but the first of a band, whose colour bar would otherwise sit on the neighbour's label.
    """
    values, S, M = grid_of(points, planned)
    finite = values[np.isfinite(values)]
    if field == "seconds":
        low, high = (max(finite.min(), .01), finite.max()) if finite.size else (1, 10)
        options = dict(cmap="magma", norm=LogNorm(low, max(high, low * 1.1)))
    else:
        limit = max(np.abs(finite).max(), 1) if finite.size else 1
        options = dict(cmap="RdBu_r", norm=SymLogNorm(1, vmin=-limit, vmax=limit))
    image = axis.imshow(values, origin="lower", aspect="auto", **options)
    for y, s in enumerate(S):
        for x, m in enumerate(M):
            if (s, m) not in planned:
                axis.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="none",
                                         edgecolor="#9a9a9a", hatch="///", linewidth=0))
            elif missing_text and not np.isfinite(values[y, x]):
                axis.text(x, y, missing_text, ha="center", va="center",
                          fontsize=max(fontsize - 1, 5), color="#666666")
    axis.set(title=title or f"{label_of(panel)}   {len(points)}/{len(planned)} points",
             xlabel="Azimuthal sectors M", ylabel="Spatial subdivision S" if ylabel else "",
             xticks=range(len(M)), yticks=range(len(S)))
    axis.set_xticklabels(M)
    axis.set_yticklabels([S[i] if len(S) <= 8 or S[i] in (1, 5, 10, 15, 20) else ""
                          for i in range(len(S))])
    if len(S) <= 8:
        for (y, x), value in np.ndenumerate(values):
            if not np.isfinite(value):
                continue
            red, green, blue, _ = image.cmap(image.norm(value))
            axis.text(x, y, f"{value:.0f}" if field == "seconds" else f"{value:+.1f}",
                      ha="center", va="center", fontsize=fontsize,
                      color="black" if .2126 * red + .7152 * green + .0722 * blue > .55 else "white")
    ring = mark or best_cell(points)
    if ring and ring in points:
        # Circle() takes (column, row), which is the transpose of a (S, M) cell.
        axis.add_patch(Circle((M.index(ring[1]), S.index(ring[0])), .46, fill=False,
                              color="#1a9641", linewidth=2))
    if colorbar:
        axis.figure.colorbar(image, ax=axis, pad=.02,
                             label={"label": "Wall time (s)" if field == "seconds"
                                    else r"$\Delta k$ (pcm, this panel only)"}.get(colorbar))
    return image


def sheet(panels, planned, values_of, out, name, columns=2, panel_size=(4.75, 3.5),
          fontsize=7.5, missing_text="", field="pcm", marks=None):
    rows = -(-len(panels) // columns)
    figure, axes = plt.subplots(rows, columns,
                                figsize=(panel_size[0] * columns, panel_size[1] * rows),
                                constrained_layout=True)
    flat = np.atleast_1d(axes).flat
    for axis, panel in zip(flat, panels):
        draw(axis, panel, values_of(panel), planned[panel], field, fontsize, missing_text,
             mark=(marks or {}).get(panel))
    for axis in list(flat)[len(panels):]:
        axis.remove()
    figure.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
    figure.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)
    return f"{name}.png"


def overview(bands, planned, pcm_of, out):
    """Every matrix of the campaign in one figure, the published quarter core heading it."""
    figure = plt.figure(figsize=(15.5, 4.0 * len(bands)))
    grid = GridSpec(len(bands), 12, figure=figure, hspace=0.42, wspace=0.5)
    for row, (panels, geometry, band) in enumerate(bands):
        for column, ((start, width), panel) in enumerate(zip(geometry, panels)):
            points = pcm_of(panel)
            draw(figure.add_subplot(grid[row, start:start + width]), panel, points,
                 planned[panel], fontsize=6.5, missing_text="n/a", colorbar="bare",
                 ylabel=column == 0,
                 title=f"{short_label(panel)}   {len(points)}/{len(planned[panel])} pts",
                 mark=best_cell(points))
        figure.text(0.008, 0.955 - row * (0.91 / len(bands)), band, rotation=90, fontsize=9,
                    ha="center", va="top", color="#555555")
    figure.suptitle(r"OpenPSN sector_diffusion: every $S\times M$ matrix of the re-sweep "
                    r"campaign ($\Delta k$ in pcm, each panel on its own scale)", fontsize=13)
    figure.savefig(out / "study4_sm_overview.png", dpi=150, bbox_inches="tight")
    figure.savefig(out / "study4_sm_overview.pdf", bbox_inches="tight")
    plt.close(figure)
    return "study4_sm_overview.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", default="sector_diffusion",
                        help="which of the two finite-sector closures is plotted")
    parser.add_argument("--figures", default="checker,checker-wide,assembly-2x2,"
                                             "assembly-wall,overview")
    parser.add_argument("--checker", type=Path,
                        default=ROOT / "paper/data/study4-checkerboard.csv")
    parser.add_argument("--points", type=Path,
                        default=ROOT / "paper/data/study4_assembly_points.csv")
    parser.add_argument("--plan", type=Path, default=ROOT / "paper/data/uniform/plan.json")
    parser.add_argument("--core", type=Path,
                        default=ROOT / "examples/c5g7/study4_sector_core/sm_error_pcm.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    planned, groups, seconds = read_points(args.plan, args.checker, args.points, args.core,
                                           args.closure)
    checker = [panel_of("checker", case, degree) for case in ("strong", "weak")
               for degree in (0, 1, 2)]
    wide = [panel_of("sm6_checker", case, 1) for case in ("strong", "weak")]
    three, four = [(0, 4), (4, 4), (8, 4)], [(0, 3), (3, 3), (6, 3), (9, 3)]
    bands = (([CORE_PANEL] + wide, three, "Quarter core and the wide $A{=}1$ checkerboard"),
             (checker[0:3], three, "Strong-absorber checkerboard, $I{=}30$"),
             (checker[3:6], three, "Weak-absorber checkerboard, $I{=}30$"),
             (list(ASSEMBLIES), four, "C5G7 assemblies, $A{=}1$, $I{=}8$"))
    chosen = set(args.figures.split(","))
    sheets = {"checker": ("study4_checker_sm", checker, 3, "pcm", (6.25, 3.75)),
              "checker-wide": ("study4_sm6_sm", wide, 2, "pcm", (5.75, 4.25)),
              "assembly-2x2": ("study4_assembly_grid2x2", list(ASSEMBLIES), 2, "pcm", (6.25, 5.75)),
              "assembly-wall": ("study4_assembly_wall", list(ASSEMBLIES), 2, "seconds",
                                (6.25, 5.75))}
    written = []
    for name, panels, columns, field, size in (sheets[k] for k in sheets if k in chosen):
        # The four-assembly sheets have room for larger values and explicit missing cells.
        roomy = name.startswith("study4_assembly")
        written.append(sheet(panels, planned, (seconds.get if field == "seconds" else groups.get),
                             args.out, name, columns=columns, panel_size=size,
                             fontsize=10 if roomy else 7.5, missing_text="n/a" if roomy else "",
                             field=field,
                             marks={panel: best_cell(groups[panel]) for panel in panels}))
    if "overview" in chosen:
        written.append(overview(bands, planned, groups.get, args.out))
    print(f"{args.closure}: " + ", ".join(f"{panel} {len(groups[panel])}/{len(planned[panel])}"
                                          for panel in [*wide, *checker, *ASSEMBLIES, CORE_PANEL]))
    print("wrote " + ", ".join(written) + " into " + str(args.out))


if __name__ == "__main__":
    main()
