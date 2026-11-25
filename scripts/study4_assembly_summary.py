r"""Derive the assembly summary rows of a re-swept closure.

``paper/data/study4_assembly_summary.csv`` is the roll-up that
``scripts/study4_assembly_report.py`` checks its per-point file against: how much of the
planned grid a case covers, where its best point sits, and what the wall clock added up
to.  The rows of the published ``sector_average`` campaign came from the archived sweep
harness, which is not in this repository, so they are carried over untouched and only the
closures named by ``--closure`` are derived --- keyed by closure, idempotent, exactly like
the re-sweep merge that fills the per-point file::

    python -m scripts.study4_assembly_summary --closure sector_diffusion

``planned`` of a closure measured by ``scripts/study4_uniform_grid.py`` is read from that
campaign's own plan rather than from the rows that happen to exist, which is what keeps a
grid the host could not finish visible as ``26/36`` instead of quietly complete.  The
pin-wise fuel errors stay empty for a derived row: those numbers come from the fission
maps of Study III, not from an eigenvalue sweep, and the assembly table prints ``---``
where they are missing.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["closure", "case", "complete", "planned", "S", "M", "pcm", "keff",
          "seconds_at_best", "sum_s", "median_s", "max_s", "factor_sum_s",
          "fuel_rms_percent", "fuel_max_percent"]
CASES = ("strong", "weak", "square_uo2", "square_mox", "rect_uo2", "rect_mox")


def summary_row(closure, case, points, planned):
    """One case of one closure, rolled up from its per-point rows."""
    best = min(points, key=lambda row: abs(float(row["pcm"])))
    seconds = [float(row["seconds"]) for row in points]
    factors = [float(row["factor_seconds"]) for row in points if row.get("factor_seconds")]
    return dict(closure=closure, case=case, complete=len(points), planned=planned,
                S=best["S"], M=best["M"], pcm=best["pcm"], keff=best["keff"],
                seconds_at_best=best["seconds"], sum_s=f"{sum(seconds):.10g}",
                median_s=f"{median(seconds):.10g}", max_s=f"{max(seconds):.10g}",
                factor_sum_s=f"{sum(factors):.10g}" if factors else "",
                fuel_rms_percent="", fuel_max_percent="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path,
                        default=ROOT / "paper/data/study4_assembly_points.csv")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "paper/data/study4_assembly_summary.csv")
    parser.add_argument("--plan", type=Path, default=ROOT / "paper/data/uniform/plan.json",
                        help="point plan of the campaign, the source of `planned`")
    parser.add_argument("--closure", action="append", default=[],
                        help="closure whose rows are derived (repeatable)")
    args = parser.parse_args()

    with args.points.open() as handle:
        points = [row for row in csv.DictReader(handle) if row.get("case")]
    with args.summary.open() as handle:
        reader = csv.DictReader(handle)
        stored, fields = list(reader), list(reader.fieldnames)

    planned_by = {}
    if args.plan.exists():
        for point in json.loads(args.plan.read_text()):
            planned_by[point["case"]] = planned_by.get(point["case"], 0) + 1

    # A row measured by this script is replaced, anything else is kept as it stands: the
    # published campaign's own roll-up is the only record of what it planned.
    derived = {}
    for closure in args.closure or ["sector_diffusion"]:
        for case in CASES:
            rows = [row for row in points if row["case"] == case
                    and row.get("closure", "sector_average") == closure]
            if not rows:
                print(f"no {closure} points for {case}, row left as stored")
                continue
            planned = planned_by.get(case, len(rows))
            derived[(closure, case)] = summary_row(closure, case, rows, planned)

    rows = [dict(row, closure=row.get("closure", "sector_average")) for row in stored]
    rows = [derived.pop((row["closure"], row["case"]), None) or row for row in rows]
    rows += [derived[key] for key in sorted(derived, key=lambda key: CASES.index(key[1]))]

    columns = ["closure"] + [field for field in fields if field != "closure"]
    if sorted(columns) != sorted(FIELDS):
        raise SystemExit(f"{args.summary}: expected columns {FIELDS}, found {columns}")
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)

    print(f"{len(rows)} summary rows in {args.summary}")
    for key in sorted(derived, key=lambda key: CASES.index(key[1])):
        row = derived[key]
        print(f"  {key[0]} {key[1]:11s} {row['complete']}/{row['planned']} points, "
              f"best ({row['S']},{row['M']}) {float(row['pcm']):+.2f} pcm, "
              f"sum {float(row['sum_s']):.1f} s")


if __name__ == "__main__":
    main()
