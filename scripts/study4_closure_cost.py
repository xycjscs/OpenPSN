"""Measure what the direction-wise closure costs against the cell-average one.

Both closures of the finite-sector scheme keep the same number of angular unknowns
per sector at the same azimuthal degree --- ``K = A + 1`` --- so a run of one and a
run of the other at the same ``S``, ``M``, ``A``, ``I``, thread budget and factor
kernel differ only in the closure.  This script runs such pairs back to back through
``scripts/sector_diffusion_benchmark.py`` and writes the record the cost section
quotes::

    python -m scripts.study4_closure_cost                       # default pairs
    python -m scripts.study4_closure_cost --point strong:12,24,1 --threads 8

Points are skipped when too little memory is free, because the host also carries the
rest of the measurement campaign; the load average and the free memory at the start
of each run are recorded with it, so the table states its own conditions.  Each pair
appears twice in the log, cell-average run first, then direction-wise.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = ("strong:8,16,1", "strong:12,24,1", "strong:12,24,0", "weak:12,24,1",
                  "weak:8,16,1", "weak:12,24,0", "strong:4,8,2", "weak:4,8,2",
                  "square_uo2:4,8,1", "square_mox:4,8,1")
FIELDS = ("model", "S", "M", "A", "I", "closure", "keff", "pcm", "seconds", "sweeps",
          "factor_seconds", "factor_bytes", "backend", "threads", "mem_limit_gb",
          "load_average", "available_gb", "node")
LABEL = {"strong": "Strong absorber", "weak": "Weak absorber",
         "square_uo2": r"Square UO$_2$", "square_mox": "Square MOX",
         "rect_uo2": r"Rectangular UO$_2$", "rect_mox": "Rectangular MOX"}


def available_gb():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20


def parse(text):
    """``model:S,M,A`` -- one paired point of the cost table."""
    model, _, grid = text.partition(":")
    s, m, a = grid.split(",")
    return dict(model=model, S=int(s), M=int(m), A=int(a))


def benchmark(args, point, closure):
    """One run of the repository's own producer, returned as its JSON record."""
    cmd = [sys.executable, "-u", "-m", "scripts.sector_diffusion_benchmark",
           point["model"], "--sm", f"{point['S']},{point['M']}", "--closure", closure,
           "--azimuth-degree", str(point["A"]), "--opt", args.backend,
           "--threads", str(args.threads), "--mem-limit-gb", str(args.memory)]
    if closure == "sector_average":
        cmd += ["--slope-space", args.slope_space]
    started = perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    record = next((json.loads(line) for line in reversed(proc.stdout.splitlines())
                   if line.startswith("{")), None)
    if record is None:
        raise RuntimeError(f"{point['model']} S={point['S']} M={point['M']} "
                           f"{closure}: " + (proc.stderr.strip().splitlines()
                                             or ["no output"])[-1])
    record["seconds"] = record.get("time_s", perf_counter() - started)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", action="append", type=parse, dest="points",
                        help="model:S,M,A -- repeatable; defaults to the published pairs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", type=float, default=16.0)
    parser.add_argument("--backend", default="sector-cholmod",
                        help="factor kernel, identical for both closures of a pair")
    parser.add_argument("--slope-space", default="quadratic",
                        choices=("quadratic", "bilinear", "quadratic-linear"))
    parser.add_argument("--csv", type=Path, default=ROOT / "paper/data/study4_closure_cost.csv")
    parser.add_argument("--table", type=Path,
                        default=ROOT / "paper/tables/study4-closure-cost.tex")
    args = parser.parse_args()

    rows, skipped = [], []
    for point in args.points or [parse(text) for text in DEFAULT_POINTS]:
        if available_gb() < args.memory + 1.0:
            skipped.append(f"{point['model']} S={point['S']} M={point['M']}")
            print(f"skipped {point['model']} S={point['S']} M={point['M']}: only "
                  f"{available_gb():.1f} GiB free")
            continue
        pair = {}
        for closure in ("sector_average", "sector_diffusion"):
            record = benchmark(args, point, closure)
            load = round(os.getloadavg()[0], 2)
            row = dict(model=point["model"], S=point["S"], M=point["M"], A=point["A"],
                       I=record.get("I", ""), closure=closure, keff=repr(record["keff"]),
                       pcm=repr(record["pcm"]), seconds=repr(record["seconds"]),
                       sweeps=record.get("sweeps_executed", ""),
                       factor_seconds=repr(record.get("factor_time_s", record.get("factor_seconds", ""))),
                       factor_bytes=record.get("factor_bytes", ""), backend=args.backend,
                       threads=args.threads, mem_limit_gb=args.memory,
                       load_average=load, available_gb=round(available_gb(), 1),
                       node=hex(sum(1 << cpu for cpu in os.sched_getaffinity(0))))
            rows.append(row)
            pair[closure] = row
            print(f"{point['model']} S={point['S']} M={point['M']} A={point['A']} "
                  f"{closure:17s} {record['pcm']:+9.2f} pcm {record['seconds']:8.2f} s "
                  f"load {load}")
        ratio = float(pair["sector_diffusion"]["seconds"]) / float(pair["sector_average"]["seconds"])
        print(f"{point['model']} S={point['S']} M={point['M']} A={point['A']} "
              f"direction-wise / cell-average = {ratio:.2f}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    index = {(row["model"], row["S"], row["M"], row["A"], row["closure"]): row
             for row in rows}
    body, ratios = [], []
    for key in sorted({(row["model"], row["S"], row["M"], row["A"]) for row in rows}):
        model, s, m, a = key
        old, new = index.get((*key, "sector_average")), index.get((*key, "sector_diffusion"))
        if old is None or new is None:
            continue
        ratio = float(new["seconds"]) / float(old["seconds"])
        ratios.append(ratio)
        body.append(f"{LABEL[model]} & $({s},{m})$ & ${a}$ & "
                    f"{float(old['seconds']):.2f}~s / {float(old['pcm']):+.2f} & "
                    f"{float(new['seconds']):.2f}~s / {float(new['pcm']):+.2f} & "
                    f"{ratio:.2f}" + r"\\")
    assert ratios, "no complete pair was measured, nothing to tabulate"
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(rf"""\begin{{table}}[htbp]
\centering
\caption{{Cost of the two closures at equal angular unknowns.  Each pair was run
through \texttt{{scripts/sector\_diffusion\_benchmark.py}} on one host, one closure
immediately after the other, at the same $M$, $A$, $I$, thread budget, factor kernel
and memory cap ({args.threads} threads, {args.memory:.0f}~GiB,
\texttt{{{args.backend}}}); the load average at the start of each run is in the data
file, and the host was also carrying the rest of the campaign, so these are workload
records.  $s$ is the end-to-end time, pcm the difference from the same reference.
Median ratio {sorted(ratios)[len(ratios) // 2]:.2f}, range {min(ratios):.2f}--{max(ratios):.2f},
over {len(ratios)} pairs.}}
\label{{tab:study4-closure-cost}}
\centering\small
\begin{{tabular}}{{l r r l l r}}
\hline
Case & $(S,M)$ & $A$ & \texttt{{sector\_average}} & \texttt{{sector\_diffusion}} & ratio \\
\hline
{chr(10).join(body)}
\hline
\end{{tabular}}
\end{{table}}
""")
    print(f"wrote {args.csv} and {args.table} over {len(ratios)} pairs"
          + (f", skipped {len(skipped)} for memory" if skipped else ""))


if __name__ == "__main__":
    main()
