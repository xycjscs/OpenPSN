r"""Re-measure every published grid point with the direction-wise sector closure.

The published checkerboard and assembly matrices were swept with the cell-average
closure.  To put both closures on the same meshes the re-sweep mirrors those files
point for point, so each new ``sector_diffusion`` measurement has a stored counterpart
of the same case, ``S``, ``M``, polar order and azimuthal degree.  Three subcommands::

    python -m scripts.study4_uniform_grid plan            # write the point plan
    python -m scripts.study4_uniform_grid run             # run it, resumable
    python -m scripts.study4_uniform_grid merge           # fold it into paper/data

``plan`` reads the two per-point CSVs and adds the weak-absorber $A{=}2$ column the
archived campaign left out; ``run`` is one serial, memory-aware queue that goes
through the repository's own producers --- ``scripts/sector_diffusion_benchmark.py``
for the checkerboards and the quarter core, ``scripts/study3_bilinear.py
--closure sector_diffusion`` for the four C5G7 assemblies, whose checkpoints also
carry the per-pin fission fields --- and appends one JSON record per converged point
to ``paper/data/uniform/points.jsonl``; ``merge`` writes those records back into
``paper/data/study4-checkerboard.csv`` and ``paper/data/study4_assembly_points.csv``
beside the stored rows, keyed by closure as well as by case and grid, so a published
point is never overwritten and the merge is idempotent.

A point whose factor pool exceeds the cap fails fast with ``MemoryError``, is requeued
with a larger tier estimate and retried when the window widens.  While another
campaign shares the host, ``--yield-to`` keeps this queue below a fixed memory cap,
lets ``--free-gib`` stay free, and stands down for ``--yield-seconds`` every
``--yield-every`` points; pass ``--yield-to none`` on a free machine.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep, strftime, time


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "paper/data/uniform"
SOURCE = "paper/data/uniform/points.jsonl"
CASE_OF = {"checkerboard-strong": "strong", "checkerboard-weak": "weak"}
CHECKER_FIELDS = ["problem", "source", "closure", "A", "slope_space", "subdivide", "M", "I",
                  "nodes", "keff", "kref", "pcm", "source_residual", "keff_residual",
                  "sweeps_executed", "backend", "time_s"]
ASSEMBLY_FIELDS = ["case", "S", "M", "I", "A", "keff", "reference", "pcm", "seconds",
                   "factor_seconds", "factor_bytes", "sweeps", "source_residual",
                   "keff_residual", "backend", "threads", "tolerance", "field_seconds",
                   "closure"]
# Factor-pool tier in GiB by family and spatial subdivision, measured on the host the
# campaign runs on: the I=30 checkerboards stay below 0.15 GiB, the square assemblies
# need 3.4 GiB at S=4 M=12, and the rectangular ones are an order of magnitude heavier.
TIER = {"checker": lambda s, m: 2,
        "sm6_checker": lambda s, m: 2,
        "square": lambda s, m: {1: 2, 2: 3, 3: 5, 4: 8, 5: 12, 6: 16}[s],
        "rect": lambda s, m: {1: 4, 2: 7, 3: 12, 4: 18, 5: 26, 6: 34}[s]}


def converged(data):
    """A finished point: the benchmark says so, the assembly sweep only implies it.

    ``scripts/sector_diffusion_benchmark.py`` records ``converged``; the Study III
    sweep prints its row only after the eigenvalue has passed both acceptance gates, so
    for it a row carrying an eigenvalue *is* a converged point.  The two producers also
    name the radial subdivision differently: ``subdivide`` and ``S``.
    """
    if "case" not in data:
        return False
    measured = "keff" in data and "pcm" in data and "keff_residual" in data
    return bool(data.get("converged") or measured)


def grid_of(data):
    return (data["case"], int(data["A"]), int(data.get("subdivide", data.get("S"))),
            int(data["M"]))


def available_gib():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20


def plan_path(args):
    return args.plan or args.state / "plan.json"


def do_plan(args):
    """One point per published point, plus the missing weak-absorber A=2 column."""
    points, seen = [], set()

    def add(family, case, s, m, a, i):
        if (case, a, s, m) in seen:
            return
        seen.add((case, a, s, m))
        points.append({"family": family, "case": case, "S": s, "M": m, "A": a, "I": i,
                       "tier": TIER[family](s, m)})

    with (ROOT / "paper/data/study4-checkerboard.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row.get("closure") != "sector_average" or row["A"] == "":
                continue
            add("checker", CASE_OF[row["problem"]], int(row["subdivide"]), int(row["M"]),
                int(row["A"]), int(row["I"]))
    for m in (4, 8, 12, 24, 48, 96):
        for s in (2, 4, 8, 12):
            add("checker", "weak", s, m, 2, 30)
    with (ROOT / "paper/data/study4_assembly_points.csv").open() as handle:
        for row in csv.DictReader(handle):
            if not row.get("case"):
                continue
            case = row["case"]
            family = "sm6_checker" if case in ("strong", "weak") else (
                "square" if case.startswith("square") else "rect")
            add(family, case, int(row["S"]), int(row["M"]), int(row["A"]), int(row["I"]))

    points.sort(key=lambda point: (point["tier"], point["family"], point["S"], point["M"],
                                   point["A"]))
    path = plan_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(points, indent=1))
    counts = {}
    for point in points:
        counts[point["family"]] = counts.get(point["family"], 0) + 1
    print(f"{len(points)} points {counts}, tiers {min(p['tier'] for p in points)}"
          f"..{max(p['tier'] for p in points)} GiB, wrote {path}")


def records(path):
    if not path.exists():
        return {}
    out = {}
    with path.open() as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if converged(data):
                data.setdefault("subdivide", data.get("S"))
                data.setdefault("S", data["subdivide"])
                out[grid_of(data)] = data
    return out


def checkpoint(point, fields):
    """The row a previous sweep wrote for this assembly point, if any.

    The Study III sweep is checkpointed itself and prints nothing when its checkpoint
    exists, so re-dispatching such a point looks like a silent failure; reading the
    checkpoint's own row turns that into the result it is.
    """
    if point["family"] not in ("square", "rect"):
        return None
    path = fields / f"{point['case']}_sector_diffusion" / f"s{point['S']}_m{point['M']}.npz"
    if not path.exists():
        return None
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as saved:
            return json.loads(str(saved["info"].item()))
    except (ImportError, OSError, ValueError, KeyError):
        return None


def say(text, log):
    line = f"{strftime('%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with log.open("a") as handle:
        handle.write(line + "\n")


def busy(pattern):
    """True while a process other than this one matches ``pattern``."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or entry.name == str(os.getpid()):
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd:
            return True
    return False


def do_run(args):
    points = json.loads(plan_path(args).read_text())
    done = records(args.points)
    pending = [point for point in points if grid_of_record(point) not in done]
    started, finished = time(), 0
    args.points.parent.mkdir(parents=True, exist_ok=True)
    say(f"queue start: {len(pending)} of {len(points)} points pending", args.log)
    while pending and time() - started < args.max_hours * 3600:
        competing = args.yield_to != "none" and busy(args.yield_to)
        limit = (args.heavy_gib if competing else max(available_gib() - args.free_gib, 0.0))
        index = next((i for i, point in enumerate(pending) if point["tier"] <= limit), None)
        if index is None:
            say(f"waiting: {len(pending)} pending, lightest tier "
                f"{min(point['tier'] for point in pending)} GiB, cap {limit:.1f} GiB, "
                f"{available_gib():.1f} GiB free", args.log)
            sleep(args.poll_seconds)
            continue
        point = pending.pop(index)
        cap = min(point["tier"] + 2.0, limit)
        while (available_gib() < cap + args.free_gib
               and checkpoint(point, args.fields) is None):
            say(f"memory wait: need {cap + args.free_gib:.1f} GiB for {point['case']} "
                f"S={point['S']} M={point['M']}, {available_gib():.1f} GiB free", args.log)
            sleep(args.poll_seconds)
        record = run_point(args, point, cap)
        if converged(record):
            say(f"ok {point['family']:12s} {point['case']:10s} S={point['S']} "
                f"M={point['M']} A={point['A']} {record['pcm']:+9.2f} pcm  "
                f"{record.get('time_s', record.get('seconds', 0.0)):8.1f} s  fac "
                f"{record.get('factor_bytes', 0) / 2 ** 30:6.2f} GiB  left {len(pending)}",
                args.log)
        else:
            heavy = "exceeds cap" in record.get("error", "")
            if heavy:
                # "projected sector diffusion factor pool 34.51 GiB exceeds cap 32.00 GiB"
                need = float(record["error"].split("pool")[1].split("GiB")[0]) + 4.0
                point["tier"] = min(max(need, point["tier"] + 8.0), args.heavy_gib)
            say(f"{'requeued' if heavy else 'failed'} {point['case']} S={point['S']} "
                f"M={point['M']} A={point['A']}: {record.get('error', '')[:120]} "
                f"(tier now {point['tier']})", args.log)
            pending.append(point)
        finished += 1
        if competing and finished % args.yield_every == 0 and pending:
            say(f"standing down {args.yield_seconds} s so the other campaign gets a "
                f"window ({len(pending)} points still pending)", args.log)
            sleep(args.yield_seconds)
        else:
            sleep(15 if competing else 5)
    say(f"queue exit: {len(pending)} pending", args.log)


def grid_of_record(point):
    return (point["case"], point["A"], point["S"], point["M"])


def run_point(args, point, cap):
    if point["family"] in ("square", "rect"):
        # The assembly rows also carry pin-wise fission fields, which only the Study III
        # sweep writes, so they come from there and the best-point maps of the new
        # closure need no second transport run.
        cmd = [sys.executable, "-u", "-m", "scripts.study3_bilinear", point["case"],
               "--output", str(args.fields), "--s", str(point["S"]), "--m", str(point["M"]),
               "--closure", "sector_diffusion", "--threads", str(args.threads),
               "--memory", f"{cap:.1f}", "--backend", args.backend]
    else:
        cmd = [sys.executable, "-u", "-m", "scripts.sector_diffusion_benchmark",
               point["case"], "--sm", f"{point['S']},{point['M']}",
               "--closure", "sector_diffusion", "--azimuth-degree", str(point["A"]),
               "--opt", args.backend, "--threads", str(args.threads),
               "--mem-limit-gb", f"{cap:.1f}"]
    started = time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    record = checkpoint(point, args.fields) or next(
        (json.loads(line) for line in reversed(proc.stdout.splitlines())
         if line.startswith("{")), None)
    if record is None:
        error = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        record = {"converged": False, "error": error[:200],
                  "seconds": round(time() - started, 2)}
    record["case"] = point["case"]
    record.setdefault("A", point["A"])
    record.setdefault("subdivide", point["S"])
    record.setdefault("M", point["M"])
    if converged(record):
        record["converged"] = True       # one test for every later reader
    record.update(family=point["family"], tier_gib=point["tier"], mem_cap_gib=round(cap, 1),
                  wall_s=round(time() - started, 1))
    with args.points.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def num(value, default=-1):
    """Grid key as an int; a stored row may leave A blank (the strict rows do)."""
    return default if str(value) == "" else int(value)


def checker_row(data):
    return dict(problem=f"checkerboard-{data['case']}", source=SOURCE,
                closure="sector_diffusion", A=data["A"], slope_space="",
                subdivide=data["subdivide"], M=data["M"], I=data["I"], nodes=data["nodes"],
                keff=repr(data["keff"]), kref=repr(data["kref"]), pcm=repr(data["pcm"]),
                source_residual=repr(data["source_residual"]),
                keff_residual=repr(data["keff_residual"]),
                sweeps_executed=data["sweeps_executed"], backend=data["backend"],
                time_s=repr(data["time_s"]))


def assembly_row(data):
    return dict(case=data["case"], S=data.get("S", data["subdivide"]), M=data["M"],
                I=data.get("I", 30), A=data["A"], keff=repr(data["keff"]),
                reference=repr(data.get("reference", data.get("kref"))), pcm=repr(data["pcm"]),
                seconds=repr(data.get("seconds", data.get("time_s"))),
                factor_seconds=repr(data["factor_seconds"]) if "factor_seconds" in data else "",
                factor_bytes=repr(data.get("factor_bytes", "")),
                sweeps=data.get("sweeps", data.get("sweeps_executed")),
                source_residual=repr(data["source_residual"]),
                keff_residual=repr(data["keff_residual"]), backend=data["backend"],
                threads=data.get("threads", ""), tolerance=repr(data.get("tolerance", 1e-9)),
                field_seconds=data.get("field_seconds", ""), closure="sector_diffusion")


def rewrite(path, fields, key, new_rows, label, dry):
    """Replace or append the re-sweep rows, keeping every stored row untouched."""
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:                      # the stored campaign has no closure column
        row.setdefault("closure", "sector_average")
    index = {key(row): i for i, row in enumerate(rows)}
    replaced = 0
    for row in new_rows:
        where = index.get(key(row))
        if where is None:
            rows.append(row)
            index[key(row)] = len(rows) - 1
        else:
            rows[where] = row
            replaced += 1
    print(f"{label}: {len(new_rows)} re-sweep rows, {replaced} already present, "
          f"{len(rows)} total")
    if dry:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def do_merge(args):
    points = records(args.points)
    print(f"{len(points)} converged re-sweep points")
    rewrite(ROOT / "paper/data/study4-checkerboard.csv", CHECKER_FIELDS,
            lambda row: (row["problem"], row["closure"], num(row["A"]),
                         int(row["subdivide"]), int(row["M"])),
            [checker_row(data) for data in points.values()
             if data["family"] in ("checker", "sm6_checker")],
            "study4-checkerboard.csv", args.dry_run)
    rewrite(ROOT / "paper/data/study4_assembly_points.csv", ASSEMBLY_FIELDS,
            lambda row: (row["case"], row["closure"], num(row["A"]), int(row["S"]),
                         int(row["M"])),
            [assembly_row(data) for data in points.values()
             if data["family"] in ("square", "rect") or data["case"] in ("strong", "weak")],
            "study4_assembly_points.csv", args.dry_run)
    if args.dry_run:
        print("(dry run, nothing written)")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", type=Path, default=STATE,
                        help="where the plan, the point records and the checkpoints live")
    common.add_argument("--plan", type=Path, help="point plan (default <state>/plan.json)")
    common.add_argument("--points", type=Path, help="point records (default <state>/points.jsonl)")
    common.add_argument("--log", type=Path, help="driver log (default <state>/driver.log)")
    common.add_argument("--fields", type=Path,
                        help="checkpoint directory of the assembly sweep (default <state>/fields)")
    common.add_argument("--backend", default="sector-cholmod")
    common.add_argument("--threads", type=int, default=8)
    common.add_argument("--free-gib", type=float, default=12.0,
                        help="memory to leave free for whatever else runs on the host")
    common.add_argument("--heavy-gib", type=float, default=16.0,
                        help="cap while another campaign is running")
    common.add_argument("--yield-to", default="run_point.py",
                        help="process pattern of the other campaign, or none")
    common.add_argument("--yield-every", type=int, default=20)
    common.add_argument("--yield-seconds", type=float, default=420)
    common.add_argument("--poll-seconds", type=float, default=300)
    common.add_argument("--max-hours", type=float, default=20)
    common.add_argument("--dry-run", action="store_true")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("plan", "write the point plan from the published grids"),
                            ("run", "run the plan, resumable, one point at a time"),
                            ("merge", "fold the converged points into paper/data")):
        sub.add_parser(name, parents=[common], help=help_text)
    args = parser.parse_args()
    args.points = args.points or args.state / "points.jsonl"
    args.log = args.log or args.state / "driver.log"
    args.fields = args.fields or args.state / "fields"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    {"plan": do_plan, "run": do_run, "merge": do_merge}[args.command](args)


if __name__ == "__main__":
    main()
