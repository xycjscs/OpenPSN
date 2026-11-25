"""Rebuild the Study IV checkerboard table and its data file from the run logs.

The square-checkerboard claims of the paper are the sign structure and the
refinement trend, so this script re-derives both from the archived logs rather
than trusting the table::

    python -m scripts.study4_checkerboard_report            # writes paper/tables + data
    python -m scripts.study4_checkerboard_report --logs DIR

Checks performed on every sector-closure point: the closure, azimuth degree,
quadrature size, kref and the two residuals are recorded; the eigenvalue error
is negative; and along each axis, at fixed values of the other, the error moves
monotonically toward zero.  The published legacy values of the same
two problems are already tabulated in ``docs/fig3ref.json`` and reproduced by
``tests/run_examples.py``.
"""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KREF = {"strong": 0.51673, "weak": 1.12974}
SCAN = {  # table column -> (log file, expected azimuth degree, slope space)
    ("strong", "A0"): ("strict_average_strong.log", 0, None),
    ("weak", "A0"): ("strict_average_weak.log", 0, None),
    ("strong", "A1"): ("sector_linear_strong.log", 1, None),
    ("weak", "A1"): ("sector_linear_weak.log", 1, None),
    ("strong", "A2"): ("sector_quadratic_strong.log", 2, None),
    ("strong", "bilin"): ("timing_bilinear_full_strong.log", 1, "bilinear"),
    ("weak", "bilin"): ("timing_bilinear_full_weak.log", 1, "bilinear"),
    ("strong", "valley"): ("strong_strict_sector_valley.log", None, None),
}
ROWS = ((2, 4), (4, 8), (8, 12), (12, 24), (12, 48), (12, 96))
COLUMNS = ("strong", "A0"), ("strong", "A1"), ("strong", "bilin"), ("strong", "A2"), \
          ("weak", "A0"), ("weak", "A1")
CAPTION = (r"Strong and weak absorber checkerboards, $10^{5}(\keff-k_\mathrm{ref})$ in pcm, "
           r"against the published $0.51673$ and $1.12974$. $I{=}30$. $A{=}0$ keeps the angular "
           r"average over each sector, $A{=}1$ and $A{=}2$ add local azimuthal modes inside the "
           r"sector, and ``bilin.'' represents the $A{=}1$ coefficient in the continuous "
           r"bilinear subspace of the same mesh. The bilinear representation was scanned over the "
           r"same $S{=}2,\ldots,12$ $\times$ $M{=}4,8,12,24$ grid for both absorbers; its "
           r"values here are the strong-absorber column, and the weak absorber reaches "
           r"$-8.94\pcm$ at $S{=}12$, $M{=}24$.")


def read_records(path):
    """Return every converged result record in a run log."""
    records = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("converged"):
            records.append(item)
    if not records:
        raise SystemExit(f"no converged result record in {path}")
    return records


def load_scans(logs):
    scans, table = {}, {}
    for key, (filename, degree, slope) in SCAN.items():
        records = read_records(logs / filename)
        points = {}
        for record in records:
            assert abs(record["kref"] - KREF[key[0]]) < 1e-12, (filename, record["kref"])
            assert record["I"] == 30, (filename, record["I"])
            assert record.get("A") == degree, (filename, record.get("A"), degree)
            assert (record.get("slope_space") or "quadratic") == (slope or "quadratic"), \
                (filename, record.get("slope_space"))
            assert record["source_residual"] <= 5e-9 and record["keff_residual"] <= 5e-9, record
            pcm = 1e5 * (record["keff"] - record["kref"])
            assert abs(pcm - record["pcm"]) < 1e-6, record
            points[(record["subdivide"], record["M"])] = (record, pcm)
        scans[key] = points
        if key[1] != "valley":  # the valley control run is allowed to change sign
            positive = [point for point, pcm in points.values() if pcm >= 0]
            assert not positive, f"{filename}: non-negative error at {positive}"
        table[key] = points
    return scans, table


def check_monotone(scans):
    """Along each axis at fixed values of the other, |error| must not grow."""
    for key, points in scans.items():
        for (s1, m1), (_, first) in points.items():
            for (s2, m2), (_, second) in points.items():
                refined = (s1 == s2 and m1 < m2) or (m1 == m2 and s1 < s2)
                if refined:
                    assert abs(second) <= abs(first) + 1e-9, \
                        f"{key}: ({s1},{m1}) {first:+.2f} -> ({s2},{m2}) {second:+.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=ROOT / "examples/psn_repro/logs")
    parser.add_argument("--table", type=Path, default=ROOT / "paper/tables/study4-checker.tex")
    parser.add_argument("--data", type=Path, default=ROOT / "paper/data/study4-checkerboard.csv")
    args = parser.parse_args()

    scans, _ = load_scans(args.logs)
    check_monotone({key: points for key, points in scans.items() if key[1] != "valley"})

    columns = ["problem", "source", "closure", "A", "slope_space", "subdivide", "M", "I",
               "nodes", "keff", "kref", "pcm", "source_residual", "keff_residual",
               "sweeps_executed", "backend", "time_s"]
    rows = []
    for (case, kind), points in scans.items():
        filename = SCAN[(case, kind)][0]
        for (subdivision, M), (record, pcm) in sorted(points.items()):
            rows.append({
                "problem": f"checkerboard-{case}", "source": f"examples/psn_repro/logs/{filename}",
                "closure": record["closure"], "A": record.get("A", ""),
                "slope_space": record.get("slope_space") or "quadratic",
                "subdivide": subdivision, "M": M, "I": record["I"], "nodes": record["nodes"],
                "keff": f"{record['keff']:.15g}", "kref": record["kref"], "pcm": f"{pcm:.6g}",
                "source_residual": f"{record['source_residual']:.3g}",
                "keff_residual": f"{record['keff_residual']:.3g}",
                "sweeps_executed": record.get("sweeps_executed", ""),
                "backend": record.get("backend", ""),
                "time_s": record["time_s"]})
    rows.sort(key=lambda row: (row["problem"], row["closure"], row["A"], row["subdivide"], row["M"]))
    args.data.parent.mkdir(parents=True, exist_ok=True)
    with args.data.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(case, kind):
        return {point: pcm for point, (_, pcm) in scans.get((case, kind), {}).items()}

    body = []
    for subdivision, M in ROWS:
        line = [str(subdivision), str(M)]
        for case, kind in COLUMNS:
            value = cell(case, kind).get((subdivision, M))
            line.append("---" if value is None else f"${value:+.2f}$")
        body.append(" & ".join(line) + r"\\")

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(rf"""\begin{{table}}[htbp]
\centering
\caption{{{CAPTION}}}
\label{{tab:study4-checker}}
\centering\small
\begin{{tabular}}{{rrrrrrrr}}
\hline
& & \multicolumn{{4}}{{c}}{{strong absorber}} & \multicolumn{{2}}{{c}}{{weak absorber}}\\
\cline{{3-6}}\cline{{7-8}}
$S$ & $M$ & $A{{=}}0$ & $A{{=}}1$ & bilin. & $A{{=}}2$ & $A{{=}}0$ & $A{{=}}1$\phantom{{, bil.}}\\
\hline
{chr(10).join(body)}
\hline
\end{{tabular}}
\end{{table}}
""")
    sector = sum(len(points) for key, points in scans.items() if key[1] != "valley")
    print(f"{sector} sector points, all one-signed and monotone under refinement")
    print(f"wrote {args.data} ({len(rows)} rows)")
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
