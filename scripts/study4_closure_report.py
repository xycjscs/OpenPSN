r"""Rebuild the closure tables of Study IV from the per-point results.

The two finite-sector closures are measured on the same meshes at the same angular
settings, so both live in the same per-point files and are separated by their
``closure`` column.  This script turns them into the three things the manuscripts
include::

    python -m scripts.study4_closure_report

* ``paper/tables/study4-checker-sector.tex`` -- the checkerboard matrix of the
  direction-wise ``sector_diffusion`` closure, which is the closure of record;
* ``paper/tables/study4-checker-average.tex`` -- the same matrix of the
  cell-average ``sector_average`` closure, kept as the comparison;
* ``paper/tables/study4-closure-ratio.tex`` -- the point-by-point ratio of the two,
  $|\Delta k|_{\rm diffusion}/|\Delta k|_{\rm average}$, with the aggregate counts
  quoted in its caption;
* ``paper/data/study4_closure_pairs.csv`` -- one row per paired point, the source of
  every number in the caption and in the text.

Two grids are distinguished.  The *published* checkerboard grid is every
``sector_average`` point of the quadratic coefficient subspace (the bilinear
representation has no counterpart in the other closure, whose azimuthal coefficient
is a Gauss direction rather than a polynomial).  Every published point must have a
partner of the new closure; the ``--allow-missing`` flag is what an unfinished
re-sweep needs, and it says so loudly.  The wide $S{=}1\ldots20$ matrices of the
assembly data file are reported in the printed summary only, because their
re-sweep is longer.
"""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = (("strong", "strong absorber"), ("weak", "weak absorber"))
LABELS = {"strong": "strong absorber", "weak": "weak absorber",
          "square_uo2": r"square UO$_2$ assembly", "square_mox": "square MOX assembly",
          "rect_uo2": r"rectangular UO$_2$ assembly", "rect_mox": "rectangular MOX assembly"}
DEGREES = (0, 1, 2)
ROWS = ((2, 4), (4, 8), (8, 12), (12, 24), (12, 48), (12, 96))
PAIR_FIELDS = ("case", "A", "S", "M", "I", "pcm_sector_average", "pcm_sector_diffusion",
               "abs_ratio", "seconds_sector_average", "seconds_sector_diffusion",
               "second_ratio")


def checker_points(path, closure):
    """Checkerboard points of one closure, keyed by (case, A, S, M).

    ``slope_space`` marks the coefficient subspace of the cell-average closure; only
    the quadratic one is comparable, because the direction-wise closure replaces the
    azimuthal polynomial by Gauss directions and has no bilinear variant.  The
    ``strict_phase_space`` rows carry no azimuthal degree at all.
    """
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("closure") == closure]
    rows = [row for row in rows if row["A"] != "" and row.get("slope_space") != "bilinear"]
    return {(row["problem"].split("-")[-1], int(row["A"]), int(row["subdivide"]),
             int(row["M"])): row for row in rows}


def assembly_points(path, closure):
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("case") and row.get("closure", "sector_average") == closure]
    return {(row["case"], int(row["A"]), int(row["S"]), int(row["M"])): row for row in rows}


def seconds(row):
    for field in ("time_s", "seconds"):
        if row.get(field):
            return float(row[field])
    raise KeyError("point records no wall time")


def value(row):
    return float(row["pcm"])


def cell(index, key):
    row = index.get(key)
    return "---" if row is None else f"${value(row):+.2f}$"


def ratio_table(rows):
    """Median, extremes and sign structure of a list of paired points."""
    ratios = [float(row["abs_ratio"]) for row in rows]
    flips = [row for row in rows
             if float(row["pcm_sector_average"]) * float(row["pcm_sector_diffusion"]) < 0.0]
    ordered = sorted(ratios)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2)
    # One phrase per (case, A, M) group of sign changes, so the caption can point at the
    # column they sit in instead of listing fifteen cells.
    groups = {}
    for row in flips:
        groups.setdefault((row["case"], int(row["A"]), int(row["M"])), []).append(int(row["S"]))
    text = "; ".join(f"$A$={a}, $M$={m}, $S=$" + ",".join(str(s) for s in sorted(sub))
                     + f" of the {LABELS[case]}" for (case, a, m), sub in sorted(groups.items()))
    coarse = bool(flips) and all(int(row["M"]) == 4 and int(row["A"]) == 1 for row in flips)
    # The largest ratio is one point two cells away from a single-element mesh, where both
    # errors are far from converged anyway; quoting it alone would overstate how often the
    # direction-wise tensor loses, so the caption names it and then the rest.
    worst_row = max(rows, key=lambda row: float(row["abs_ratio"])) if rows else None
    rest = [float(row["abs_ratio"]) for row in rows if row is not worst_row]
    worst_text = ""
    if worst_row is not None and rest:
        worst_text = (f"the worst is {float(worst_row['abs_ratio']):.2f} at the "
                      f"{LABELS[worst_row['case']]} with $A$={worst_row['A']}, "
                      f"$S$={worst_row['S']}, $M$={worst_row['M']}, where the two errors are "
                      f"{abs(float(worst_row['pcm_sector_average'])):.0f} and "
                      f"{abs(float(worst_row['pcm_sector_diffusion'])):.0f} pcm; the other "
                      f"{len(rest)} pairs are all at most {max(rest):.2f}")
    return dict(n=len(rows), smaller=sum(1 for ratio in ratios if ratio < 1.0),
                flips=len(flips), median=median, worst=max(ratios) if ratios else 0.0,
                best=min(ratios) if ratios else 0.0, worst_text=worst_text,
                flip_text=text or "none of them",
                flip_class=("all of them in the $M{{=}}4$ column of the "
                          "$A{{=}}1$ scan, where a sector holds two Gauss directions "
                          if coarse else ""))


def seconds_of(row, field):
    return float(row[field]) if row[field] else 0.0


def matrix(index, label, note, path, missing_ok, expected):
    """One $S,M$ row per refinement pair, one column per absorber and azimuth degree."""
    header = (" & ".join(f"$A$={a}" for _ in CASES for a in DEGREES))
    body, missing = [], []
    for s, m in ROWS:
        line = [str(s), str(m)]
        for case, _ in CASES:
            for a in DEGREES:
                key = (case, a, s, m)
                line.append(cell(index, key))
                if key not in index and key in expected:
                    missing.append(key)
        body.append(" & ".join(line) + r"\\")
    counts = {case: sum(1 for key in index if key[0] == case) for case, _ in CASES}
    assert not missing or missing_ok, f"{label}: points not measured {missing}"
    columns = "rr" + "r" * (len(CASES) * len(DEGREES))
    path.write_text(rf"""\begin{{table}}[htbp]
\centering
\caption{{{note}}}
\label{{tab:study4-checker-{label}}}
\centering\small
\begin{{tabular}}{{{columns}}}
\hline
& & \multicolumn{{{len(DEGREES)}}}{{c}}{{strong absorber}} &
  \multicolumn{{{len(DEGREES)}}}{{c}}{{weak absorber}}\\
\cline{{3-{2 + len(DEGREES)}}}\cline{{{3 + len(DEGREES)}-{2 + 2 * len(DEGREES)}}}
$S$ & $M$ & {header}\\
\hline
{chr(10).join(body)}
\hline
\end{{tabular}}
\end{{table}}
""")
    print(f"study4-checker-{label}.tex: {sum(counts.values())} points "
          f"({counts['strong']} strong, {counts['weak']} weak)"
          f"{', ' + str(len(missing)) + ' cells blank' if missing else ', every published cell filled'}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, default=ROOT / "paper/data/study4-checkerboard.csv")
    parser.add_argument("--assembly", type=Path,
                        default=ROOT / "paper/data/study4_assembly_points.csv")
    parser.add_argument("--latex", type=Path, default=ROOT / "paper/tables")
    parser.add_argument("--pairs", type=Path, default=ROOT / "paper/data/study4_closure_pairs.csv")
    parser.add_argument("--allow-missing", action="store_true",
                        help="publish a matrix whose re-sweep is still running")
    args = parser.parse_args()

    published = checker_points(args.checker, "sector_average")
    diffusion = checker_points(args.checker, "sector_diffusion")
    expected = set(published)
    args.latex.mkdir(parents=True, exist_ok=True)

    matrix(published, "average",
           r"Strong and weak absorber checkerboards, $10^{5}(\keff-k_\mathrm{ref})$ in "
           r"pcm, against the published $0.51673$ and $1.12974$, $I{=}30$, measured with "
           r"the cell-average closure \texttt{sector\_average}: $A{=}0$ keeps the angular "
           r"average over each sector, $A{=}1$ and $A{=}2$ add local azimuthal modes "
           r"inside the sector. Kept as the comparison for "
           r"Table~\ref{tab:study4-checker-sector}, whose points cost the same to "
           r"within a factor of $1.5$ at the same $A$ and $M$ "
           r"(Table~\ref{tab:study4-closure-cost}). The bilinear variant of $A{=}1$ is "
           r"measured on this closure only and is quoted in the text.",
           args.latex / "study4-checker-average.tex", args.allow_missing, expected)
    matrix(diffusion, "sector",
           r"The direction-wise closure \texttt{sector\_diffusion} on the same meshes, "
           r"same $I$, same $M$, and the same number of angular unknowns per sector "
           r"($K=A+1$): the diffusion tensor of each sector is used per retained "
           r"direction instead of being averaged over the sector first. This is the "
           r"closure of record; every point of the cell-average comparison is paired "
           r"with the point here that shares its $A$, $S$ and $M$.",
           args.latex / "study4-checker-sector.tex", args.allow_missing, expected)

    paired = []
    for key, old in sorted(published.items()):
        new = diffusion.get(key)
        if new is None:
            continue
        case, a, s, m = key
        old_pcm, new_pcm = value(old), value(new)
        paired.append(dict(case=case, A=a, S=s, M=m, I=old["I"],
                           pcm_sector_average=repr(old_pcm),
                           pcm_sector_diffusion=repr(new_pcm),
                           abs_ratio=repr(abs(new_pcm) / abs(old_pcm)),
                           seconds_sector_average=repr(seconds(old)),
                           seconds_sector_diffusion=repr(seconds(new)),
                           second_ratio=repr(seconds(new) / seconds(old))))
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    wide = assembly_points(args.assembly, "sector_average")
    wide_new = assembly_points(args.assembly, "sector_diffusion")
    pairs = {(row["case"], int(row["A"]), int(row["S"]), int(row["M"])): row
             for row in paired}
    for (case, a, s, m), old in sorted(wide.items()):
        new = wide_new.get((case, a, s, m))
        if new is None or (case, a, s, m) in pairs:
            continue                    # the 40 shared points are already paired
        pairs[(case, a, s, m)] = dict(
            case=case, A=a, S=s, M=m, I=new["I"], pcm_sector_average=repr(value(old)),
            pcm_sector_diffusion=repr(value(new)),
            abs_ratio=repr(abs(value(new)) / abs(value(old))),
            seconds_sector_average=repr(seconds(old)),
            seconds_sector_diffusion=repr(seconds(new)),
            second_ratio=repr(seconds(new) / seconds(old)))
    all_paired = sorted(pairs.values(), key=lambda row: (row["case"], row["A"], row["S"],
                                                         row["M"]))
    stats = ratio_table(all_paired)
    counts = {
        "StudyFourPairs": stats["n"],
        "StudyFourSmaller": stats["smaller"],
        "StudyFourMedianRatio": f"{stats['median']:.3f}",
        "StudyFourSignChanges": stats["flips"],
    }
    (args.latex / "study4-pair-counts.tex").write_text(
        "% Generated by scripts.study4_closure_report.\n" +
        "".join(f"\\newcommand{{\\{name}}}{{{count}}}\n" for name, count in counts.items()))
    with args.pairs.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in PAIR_FIELDS}
                         for row in all_paired)

    by_case = {}
    for row in all_paired:
        by_case.setdefault(row["case"], []).append(row)
    stored = {(case, a, s, m): True for case, a, s, m in list(published) + list(wide)}
    for case in LABELS:
        rows = by_case.get(case, [])
        planned = sum(1 for key in stored if key[0] == case)
        if not rows:
            print(f"{LABELS[case]:32s} no paired point yet, of {planned} stored")
            continue
        stat = ratio_table(rows)
        print(f"{LABELS[case]:32s} {len(rows):3d}/{planned:3d} paired, "
              f"{stat['smaller']:3d} smaller, median {stat['median']:.3f}, "
              f"best {stat['best']:.3f}, worst {stat['worst']:.3f}, "
              f"{stat['flips']} sign change(s)")

    checker_pairs = {(row["case"], int(row["A"]), int(row["S"]), int(row["M"])): row
                     for row in paired}
    with (args.latex / "study4-closure-ratio.tex").open("w") as handle:
        body = []
        for s, m in ROWS:
            line = [str(s), str(m)]
            for case, _ in CASES:
                for a in DEGREES:
                    row = checker_pairs.get((case, a, s, m))
                    line.append("---" if row is None else f"{float(row['abs_ratio']):.3f}")
            body.append(" & ".join(line) + r"\\")
        handle.write(rf"""\begin{{table}}[htbp]
\centering
\caption{{$\lvert\Delta k\rvert$ of the direction-wise closure divided by that of the
cell-average closure at the same $A$, $S$, $M$ and $I$; below $1$ the direction-wise
tensor is the more accurate of the two at equal angular unknowns. Over all
{stats['n']} paired checkerboard and assembly points the direction-wise closure is
smaller at {stats['smaller']} of them, the median ratio is {stats['median']:.3f}
(best {stats['best']:.3f}), {stats['worst_text']}, and the sign of the difference
changes between the two closures at {stats['flips']} of them, {stats['flip_class']}
({stats['flip_text']}). The time these points take is
not compared here, because the two campaigns ran under different thread budgets; the
matched pairs of Table~\ref{{tab:study4-closure-cost}} are.}}
\label{{tab:study4-closure-ratio}}
\centering\small
\begin{{tabular}}{{rr{'r' * (len(CASES) * len(DEGREES))}}}
\hline
& & \multicolumn{{{len(DEGREES)}}}{{c}}{{strong absorber}} &
  \multicolumn{{{len(DEGREES)}}}{{c}}{{weak absorber}}\\
\cline{{3-{2 + len(DEGREES)}}}\cline{{{3 + len(DEGREES)}-{2 + 2 * len(DEGREES)}}}
$S$ & $M$ & {" & ".join(f"$A$={a}" for _ in CASES for a in DEGREES)}\\
\hline
{chr(10).join(body)}
\hline
\end{{tabular}}
\end{{table}}
""")
    print(f"{stats['n']} paired points: {stats['smaller']} smaller with the "
          f"direction-wise closure, median ratio {stats['median']:.3f}, "
          f"{stats['flips']} sign changes"
          f"{' (' + stats['flip_text'] + ')' if stats['flips'] else ''}")
    print(f"wrote {args.latex}/study4-checker-sector.tex, study4-checker-average.tex, "
          f"study4-closure-ratio.tex and {args.pairs}")


if __name__ == "__main__":
    main()
