r"""Rebuild the Study IV assembly summary table from the per-point results.

The table is shared by the OpenPSN paper (Section 4, Study IV) and by the
standalone finite-sector manuscript, so it is generated from
``paper/data/study4_assembly_points.csv`` rather than typed into either
document::

    python -m scripts.study4_assembly_report

Every case must have completed its planned grid, the best point of each case is
re-derived from the per-point rows, and the pin-wise field errors are taken from
the summary file that was produced together with the fission maps.

Both finite-sector closures live in the same per-point file and are separated by its
``closure`` column, so ``--closure`` selects which of the two is tabulated; the
stored rows of the published campaign are ``sector_average``.  The summary file
carries the same column.

    python -m scripts.study4_assembly_report --closure sector_average
    python -m scripts.study4_assembly_report --closure sector_diffusion \
        --table paper/tables/study4-assembly-sector.tex --label tab:study4-assembly-sector

``--allow-partial`` permits coverage shorter than the plan. The default keeps the
all-or-nothing gate. Coverage and the packed-factor count are stated in the
direction-wise table.
"""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = (("strong", "Strong absorber"), ("weak", "Weak absorber"),
         ("square_uo2", r"Square UO$_2$"), ("square_mox", "Square MOX"),
         ("rect_uo2", r"Rectangular UO$_2$"), ("rect_mox", "Rectangular MOX"))
CAPTION = (r"Coverage, best measured point, fuel-pin error and wall time of the six "
           r"$S\times M$ matrices, measured with the \texttt{{{closure}}} closure at $A=1$. "
           r"The sum column adds the individual point times; several cases ran "
           r"concurrently on the same host.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=ROOT / "paper/data/study4_assembly_points.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "paper/data/study4_assembly_summary.csv")
    parser.add_argument("--table", type=Path, default=ROOT / "paper/tables/study4-assembly.tex")
    parser.add_argument("--label", default="tab:study4-assembly",
                        help="LaTeX label of the table; a second closure goes under its own")
    parser.add_argument("--closure", default="sector_average",
                        help="which closure of the two in the per-point file is tabulated")
    parser.add_argument("--allow-partial", action="store_true",
                        help="tabulate a closure whose grid is not complete, printing the coverage")
    args = parser.parse_args()

    with args.points.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("case")
                and row.get("closure", "sector_average") == args.closure]
    with args.summary.open() as handle:
        summary = {(row.get("closure", "sector_average"), row["case"]): row
                   for row in csv.DictReader(handle)}

    body, counts, partial = [], [], []
    show_coverage = args.closure != "sector_average"
    for key, label in CASES:
        points = [row for row in rows if row["case"] == key]
        assert points, f"no points recorded for {key}"
        counts.append(len(points))
        best = min(points, key=lambda row: abs(float(row["pcm"])))
        info = summary[(args.closure, key)]
        assert int(info["complete"]) == len(points), (key, info["complete"])
        if len(points) != int(info["planned"]):
            assert args.allow_partial, \
                f"{key}: {len(points)} of {info['planned']} planned points measured"
            partial.append(f"{label} {len(points)}/{info['planned']}")
        assert (int(info["S"]), int(info["M"])) == (int(best["S"]), int(best["M"])), key
        assert abs(float(info["pcm"]) - float(best["pcm"])) < 1e-9, key
        assert abs(float(info["sum_s"]) - sum(float(row["seconds"]) for row in points)) < 0.05, key
        pin = [(f"{float(info[column]):.3f}\\%" if info[column] else "---")
               for column in ("fuel_rms_percent", "fuel_max_percent")]
        coverage = f"{len(points)}/{info['planned']} & " if show_coverage else ""
        body.append(f"{label} & {coverage}$({best['S']},{best['M']})$ & {float(best['pcm']):+.2f} & "
                    + " & ".join(pin) + f" & {float(info['sum_s']):.1f}~s" + r"\\")

    args.table.parent.mkdir(parents=True, exist_ok=True)
    closure = args.closure.replace("_", "\\_")      # the caption is LaTeX text
    columns = "l r r r r r r" if show_coverage else "l r r r r r"
    header = "Coverage & " if show_coverage else ""
    total = "Total time" if show_coverage else f"Sum, {min(counts)}--{max(counts)} points"
    caption = CAPTION.format(closure=closure)
    packed = sum(row["backend"] == "chol-packed-rs" for row in rows) if show_coverage else 0
    if packed:
        caption += (f" Packed PCG was used at {packed} points. "
                    "Run-level backend and thread settings are in the per-point file.")
    args.table.write_text(rf"""\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{{args.label}}}
\centering\small
\begin{{tabular}}{{{columns}}}
\hline
Case & {header}Best $(S,M)$ & $\Delta k$ (pcm) & Pin RMS & Pin max & {total}\\
\hline
{chr(10).join(body)}
\hline
\end{{tabular}}
\end{{table}}
""")
    print(f"{len(rows)} {args.closure} assembly points over {len(CASES)} cases, "
          + ("incomplete grids: " + "; ".join(partial) if partial
             else "every planned grid complete"))
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
