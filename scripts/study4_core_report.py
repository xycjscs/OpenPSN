"""Assemble the rectangular quarter-core sector-diffusion S/M tables and figure.

Every table cell is read back from its own run log, so the deliverable is
regenerated from the recorded eigenvalue, residual and wall time rather than
from hand-copied numbers::

    python -m scripts.study4_core_report --logs <dir-with-S*_M*.log> --output <dir>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Rectangle


S_VALUES = (1, 2, 3, 4, 5)
M_VALUES = (2, 4, 8, 12, 16, 24)
KREF = 1.187217554586068
SOURCE_RESIDUAL = 5e-9
EIGEN_RESIDUAL = 1e-9


def read_record(path):
    """Return the last converged result record in a run log."""
    for line in reversed(path.read_text(errors="replace").splitlines()):
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("converged"):
            return item
    raise ValueError(f"no converged result in {path}")


def timing_basis(logs):
    """Which points reused a factorisation stored by an earlier run.

    ``logs/timing_basis.csv`` records how each archived run was launched; every
    point marked ``reused`` is checked below against the factor time the run
    itself recorded, so the file cannot quietly overstate the savings.
    """
    path = logs / "timing_basis.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {(int(row["subdivide"]), int(row["M"])): row["factors"]
                for row in csv.DictReader(handle)}


def load(logs, expected_nodes, allow_missing=False, basis=None):
    basis = basis or {}
    """Read every point of the grid; a missing log is an error unless allowed."""
    records, missing = {}, []
    for s in S_VALUES:
        for m in M_VALUES:
            if not (logs / f"S{s}_M{m}.log").exists():
                if allow_missing:
                    missing.append((s, m))
                    continue
                raise SystemExit(f"missing log for S={s}, M={m}: {logs / f'S{s}_M{m}.log'}")
            data = read_record(logs / f"S{s}_M{m}.log")
            assert data["closure"] == "sector_diffusion" and data["model"] == "ty3", data["name"]
            assert (data["subdivide"], data["M"], data["A"], data["threads"]) == (s, m, 1, 60)
            assert data["nodes"] == expected_nodes[s]
            assert data["kref"] == KREF
            assert data["source_residual"] <= SOURCE_RESIDUAL
            assert data["keff_residual"] <= EIGEN_RESIDUAL
            assert abs(1e5*(data["keff"] - KREF) - data["pcm"]) < 1e-8
            reused = bool(data.get("factor_cache_used")) or basis.get((s, m)) == "reused"
            records[s, m] = dict(data, factors_reused=reused)
    for (s, m), record in records.items():
        if not record["factors_reused"]:
            continue
        # Factor cost grows with both axes, so a reused factorisation has to undercut
        # every fresh one measured at the same S with no more sectors.
        fresh = [r["factor_time_s"] for (s2, m2), r in records.items()
                 if s2 == s and m2 <= m and not r["factors_reused"] and r.get("factor_time_s")]
        assert record.get("factor_time_s") and (not fresh
                   or record["factor_time_s"] < min(fresh)), (s, m, record.get("factor_time_s"))
    if missing:
        print(f"INCOMPLETE: no converged run yet for {missing}")
    return records, missing


def fmt(value, key):
    """CSV cell text: blank for a point with no converged run yet."""
    return "" if value != value else value  # NaN is the only value unequal to itself


def write(filename, columns, rows, output):
    with (output / filename).open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        writer.writerows(rows)


def latex_table(label, caption, rows, units):
    """One S x M matrix as a float, best cell in bold."""
    body = "\n".join(f"{s} & " + " & ".join(cells) + " \\\\" for s, cells in rows)
    columns = " & ".join(str(m) for m in M_VALUES)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{rrrrrrr}}
\\hline
 & \\multicolumn{{6}}{{c}}{{{units}}}\\\\
\\cline{{2-7}}
$S$ & {columns}\\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="directory of S{S}_M{M}.log run logs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latex", type=Path, help="also write the paper tables into this directory")
    parser.add_argument("--mirror-md", type=Path, action="append", default=[],
                        help="markdown files whose SM_MATRIX block to refresh from the tables")
    parser.add_argument("--allow-missing", action="store_true",
                        help="report the points measured so far, leaving the rest as '---'")
    parser.add_argument("--fission-summary", type=Path,
                        help="best_fission_summary.json for the pin-wise errors of the best cell")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records, missing = load(args.logs, {s: (153*s)**2 for s in S_VALUES}, args.allow_missing,
                            timing_basis(args.logs))
    incomplete = ""
    if missing:
        cells = ", ".join(f"$(S,M)=({s},{m})$" for s, m in missing)
        incomplete = (f" \\textbf{{Incomplete:}} still running when this table was written --- "
                      f"{cells}; those cells are left blank.")
    def cell(s, m, key, default=float("nan")):
        record = records.get((s, m))
        return default if record is None else record[key]

    head = ["S", *(f"M{m}" for m in M_VALUES)]
    for key, filename in (("keff", "sm_keff.csv"), ("pcm", "sm_error_pcm.csv"),
                          ("time_s", "sm_wall_seconds.csv")):
        write(filename, head, [[s, *(fmt(cell(s, m, key), key) for m in M_VALUES)]
                               for s in S_VALUES], args.output)
    columns = list(dict.fromkeys(key for record in records.values() for key in record))
    write("points.csv", columns,
          [[record.get(key, "") for key in columns] for record in records.values()], args.output)

    best = min(records.values(), key=lambda row: abs(row["pcm"]))
    best_cell = (best["subdivide"], best["M"])
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), layout="constrained")
    figure.suptitle("C5G7 rectangular quarter-core | TY3 sector diffusion | 60-thread cap", fontsize=15)
    panels = ((axes[0], "pcm", "Signed eigenvalue deviation", "Blues_r",
               "Δk (pcm, symlog scale)", "Azimuth sectors M"),
              (axes[1], "time_s", "Observed wall time", "YlOrBr",
               "Seconds (log scale)", "Azimuth sectors M (* cached factors)"))
    for axis, key, title, cmap, units, xlabel in panels:
        matrix = np.array([[cell(s, m, key) for m in M_VALUES] for s in S_VALUES])
        low, high = float(np.nanmin(matrix)), float(np.nanmax(matrix))
        norm = LogNorm(vmin=low, vmax=high) if key == "time_s" \
            else SymLogNorm(linthresh=5, vmin=low, vmax=high)
        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set(title=title, xlabel=xlabel, ylabel="Spatial subdivision S",
                 xticks=range(len(M_VALUES)), xticklabels=M_VALUES,
                 yticks=range(len(S_VALUES)), yticklabels=S_VALUES)
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            if not np.isfinite(value):
                axis.text(j, i, "---", ha="center", va="center", fontsize=9, color="#94a3b8")
                continue
            rgba = image.cmap(image.norm(value))
            light = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
            label = f"{value:+.2f}" if key == "pcm" else f"{value:.1f}"
            if key == "time_s" and records[S_VALUES[i], M_VALUES[j]]["factors_reused"]:
                label += "*"
            axis.text(j, i, label, ha="center", va="center", fontsize=9,
                      color="white" if light < 0.43 else "#152735")
        if key == "pcm":
            axis.add_patch(Rectangle((M_VALUES.index(best["M"]) - 0.47, S_VALUES.index(best["subdivide"]) - 0.47),
                                     0.94, 0.94, fill=False, edgecolor="#b45309", linewidth=1.5))
        figure.colorbar(image, ax=axis, label=units, shrink=0.84)
    figure.savefig(args.output / "sector_diffusion_sm.png", dpi=200)
    figure.savefig(args.output / "sector_diffusion_sm.pdf")
    print(f"best S={best['subdivide']} M={best['M']} {best['pcm']:+.2f} pcm")
    for s in S_VALUES:
        print(s, [round(cell(s, m, "pcm"), 2) if np.isfinite(cell(s, m, "pcm")) else None
                  for m in M_VALUES],
              [round(cell(s, m, "time_s"), 1) if np.isfinite(cell(s, m, "time_s")) else None
               for m in M_VALUES])

    if args.latex:
        args.latex.mkdir(parents=True, exist_ok=True)
        def cells(key, plus):
            rows = []
            for s in S_VALUES:
                row = []
                for m in M_VALUES:
                    if (s, m) not in records:
                        row.append("---")
                        continue
                    value = records[s, m][key]
                    text = f"{value:{'+.2f' if plus else '.1f'}}"
                    if key == "time_s" and records[s, m]["factors_reused"]:
                        text += "*"
                    if key == "pcm" and (s, m) == best_cell:
                        text = rf"\textbf{{{text}}}"
                    row.append(text)
                rows.append((s, row))
            return rows

        pcm = (r"Signed eigenvalue difference $10^{5}(\keff-k_\mathrm{ref})$ in pcm on the "
               r"explicit quarter core, against the same-geometry OpenMC reference "
               f"${KREF:.12f}$ ($3.41\\,\\mathrm{{pcm}}$ standard error). Bold marks the "
               r"smallest measured absolute difference.")
        seconds = (r"End-to-end wall time in seconds for the same points, including assembly, "
                   r"factor construction, outer iterations and the verification sweep. "
                   + (r"An asterisk marks a point that reused a stored factorisation. "
                      if any(record["factors_reused"] for record in records.values()) else "")
                   + r"Points sharing a time window ran concurrently, so the column reports "
                     r"load rather than throughput.")
        tables = {"study4-core-pcm.tex": latex_table("tab:study4-core-pcm", pcm + incomplete,
                                                     cells("pcm", True), r"Azimuth sectors $M$"),
                  "study4-core-time.tex": latex_table("tab:study4-core-time", seconds + incomplete,
                                                      cells("time_s", False),
                                                      r"Azimuth sectors $M$")}
        pin = ""
        if args.fission_summary:
            summary = json.loads(args.fission_summary.read_text())
            with (args.output / "best_fission.csv").open() as handle:
                pins = list(csv.DictReader(handle))
            assert len(pins) == 51**2 and sum(int(row["fuel"]) for row in pins) == 1056
            assert abs(sum(float(row["sector_diffusion"]) for row in pins) - 1) < 1e-9, \
                "the shipped pin table is not the normalised one of the best cell"
            assert (summary["subdivide"], summary["M"]) == best_cell, "pin table is not the best cell"
            pin = (f", with a relative pin-wise fission-rate difference of RMS "
                   f"{summary['fuel_rms_percent']:.3f}\\% and maximum absolute value "
                   f"{summary['fuel_max_abs_percent']:.3f}\\%")
        cached = " (stored factors)" if best["factors_reused"] else ""
        tables["study4-core-best.tex"] = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{The cell of Table~\\ref{{tab:study4-core-pcm}} with the smallest absolute
eigenvalue difference{pin}. The node count is $(153S)^2$.}}
\\label{{tab:study4-core-best}}
\\begin{{tabular}}{{lr}}
\\hline
spatial subdivision $S$ & {best['subdivide']}\\\\
azimuth sectors $M$ & {best['M']}\\\\
nodes & {best['nodes']}\\\\
$\\keff$ & {best['keff']:.12f}\\\\
$\\Delta k$ & ${best['pcm']:+.2f}\\pcm$\\\\
wall time & {best['time_s']:.1f}~s{cached}\\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        for name, text in tables.items():
            (args.latex / name).write_text(text)
            print(f"wrote {args.latex / name}")

        def markdown(key, plus):
            labels = ["S \\ M", *(str(m) for m in M_VALUES)]
            lines = ["| " + " | ".join(labels) + " |", "|" + "---:|" * len(labels)]
            for s in S_VALUES:
                row = [str(s)]
                for m in M_VALUES:
                    if (s, m) not in records:
                        row.append("---")
                        continue
                    text = f"{records[s, m][key]:{'+.2f' if plus else '.1f'}}"
                    if key == "time_s" and records[s, m]["factors_reused"]:
                        text += "\\*"
                    if key == "pcm" and (s, m) == best_cell:
                        text = f"**{text}**"
                    row.append(text)
                lines.append("| " + " | ".join(row) + " |")
            return "\n".join(lines)

        (args.output / "sm_matrix.md").write_text(
            f"### Δk 矩阵（pcm，k_ref={KREF:.9f}）\n\n{markdown('pcm', True)}\n\n"
            f"### 墙钟时间矩阵（秒，`*` 复用分解因子）\n\n{markdown('time_s', False)}\n"
            + ("\n> 未完成（写表时仍在运行）：" + "、".join(f"S={s}, M={m}" for s, m in missing) + "\n"
               if missing else ""))
        print(f"wrote {args.output / 'sm_matrix.md'}")

        if args.mirror_md:
            note = ("\n两张矩阵由本脚本从 `logs/S{S}_M{M}.log` 逐格读回；`logs/timing_basis.csv` "
                    "记录每格的因子来源（六格取自此前的同配置算例，其原始运行名保留在各日志末行的 "
                    "`name` 字段中），`*` 只标给复用了已存分解因子的格子。\n")
            block = ("<!-- SM_MATRIX_BEGIN -->\n\n" + (args.output / "sm_matrix.md").read_text()
                     + note + "\n<!-- SM_MATRIX_END -->\n")
            for path in args.mirror_md:
                text = path.read_text()
                if "SM_MATRIX_BEGIN" not in text:
                    raise SystemExit(f"no SM_MATRIX_BEGIN marker in {path}")
                head, _, rest = text.partition("<!-- SM_MATRIX_BEGIN -->")
                rest = rest.split("<!-- SM_MATRIX_END -->", 1)[-1]
                path.write_text(head + block + rest.lstrip("\n"))
                print(f"refreshed the matrix block of {path}")


if __name__ == "__main__":
    main()
