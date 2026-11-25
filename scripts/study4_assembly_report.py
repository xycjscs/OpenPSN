"""Rebuild the Study IV assembly summary table from the per-point results.

The table is shared by the OpenPSN paper (Section 4, Study IV) and by the
standalone finite-sector manuscript, so it is generated from
``paper/data/study4_assembly_points.csv`` rather than typed into either
document::

    python -m scripts.study4_assembly_report

Every case must have completed its planned grid, the best point of each case is
re-derived from the per-point rows, and the pin-wise field errors are taken from
the summary file that was produced together with the fission maps.
"""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = (("strong", "Strong absorber"), ("weak", "Weak absorber"),
         ("square_uo2", r"Square UO$_2$"), ("square_mox", "Square MOX"),
         ("rect_uo2", r"Rectangular UO$_2$"), ("rect_mox", "Rectangular MOX"))
CAPTION = (r"Coverage, best measured point, fuel-pin error and wall time of the six "
           r"$S\times M$ matrices. The sum column adds the individual point times; several "
           r"cases ran concurrently on the same host.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=ROOT / "paper/data/study4_assembly_points.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "paper/data/study4_assembly_summary.csv")
    parser.add_argument("--table", type=Path, default=ROOT / "paper/tables/study4-assembly.tex")
    args = parser.parse_args()

    with args.points.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("case")]
    with args.summary.open() as handle:
        summary = {row["case"]: row for row in csv.DictReader(handle)}

    body, counts = [], []
    for key, label in CASES:
        points = [row for row in rows if row["case"] == key]
        assert points, f"no points recorded for {key}"
        counts.append(len(points))
        best = min(points, key=lambda row: abs(float(row["pcm"])))
        info = summary[key]
        assert int(info["complete"]) == len(points) == int(info["planned"]), (key, info["complete"])
        assert (int(info["S"]), int(info["M"])) == (int(best["S"]), int(best["M"])), key
        assert abs(float(info["pcm"]) - float(best["pcm"])) < 1e-9, key
        assert abs(float(info["sum_s"]) - sum(float(row["seconds"]) for row in points)) < 0.05, key
        pin = [(f"{float(info[column]):.3f}\\%" if info[column] else "---")
               for column in ("fuel_rms_percent", "fuel_max_percent")]
        body.append(f"{label} & $({best['S']},{best['M']})$ & {float(best['pcm']):+.2f} & "
                    + " & ".join(pin) + f" & {float(info['sum_s']):.1f}~s" + r"\\")

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(rf"""\begin{{table}}[htbp]
\centering
\caption{{{CAPTION}}}
\label{{tab:study4-assembly}}
\centering\small
\begin{{tabular}}{{l r r r r r}}
\hline
Case & Best $(S,M)$ & $\Delta k$ (pcm) & Pin RMS & Pin max & Sum, {min(counts)}--{max(counts)} points\\
\hline
{chr(10).join(body)}
\hline
\end{{tabular}}
\end{{table}}
""")
    print(f"{len(rows)} assembly points over {len(CASES)} cases, every planned grid complete")
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
