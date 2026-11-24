"""Export Study III S/M tables, timing summaries and paper-style pin maps."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

from psn2d.model import arrays, load_spec
from scripts.study3_bilinear import CASES, EXAMPLES, M_VALUES, point

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle


LABELS = {
    "strong": "Strong absorber", "weak": "Weak absorber",
    "square_uo2": "Square UO₂", "square_mox": "Square MOX",
    "rect_uo2": "Rectangular UO₂", "rect_mox": "Rectangular MOX",
}
REFERENCES = {
    "square_uo2": "omc_hom_asm_fission_true.npz",
    "square_mox": "omc_hom_mox_asm_fission_true.npz",
    "rect_uo2": "omc_rect_asm_power.npy",
    "rect_mox": "omc_mox_rect_asm_power.npy",
}
FUEL_LIBRARY = {"uo2": "UO2", "mox43": "MOX-4.3%", "mox7": "MOX-7%", "mox87": "MOX-8.7%"}


def read_checkers(args, rows):
    for case in ("strong", "weak"):
        src = getattr(args, case)
        for line in src.read_text().splitlines():
            result = json.loads(line)
            if not result["converged"]:
                raise ValueError(f"unconverged checkerboard result in {src}")
            s, m = result["subdivide"], result["M"]
            if (s not in range(1, 21) or m not in M_VALUES or result["I"] != 30
                    or result["A"] != 1 or result["slope_space"] != "bilinear"):
                raise ValueError(f"unexpected checkerboard setting in {src}")
            if (s, m) in rows[case]:
                raise ValueError(f"duplicate S={s}, M={m} in {src}")
            rows[case][s, m] = dict(case=case, S=s, M=m, I=30, A=1,
                                   keff=result["keff"], reference=result["kref"], pcm=result["pcm"],
                                   seconds=result["time_s"], factor_seconds=result["factor_time_s"],
                                   factor_bytes=result["factor_bytes"], sweeps=result["sweeps_executed"],
                                   source_residual=result["source_residual"], keff_residual=result["keff_residual"],
                                   threads=12, tolerance=1e-9, backend=result["backend"])


def cases_data(args):
    rows = {key: {} for key in LABELS}
    read_checkers(args, rows)
    for case in CASES:
        for s in range(1, 7):
            for m in M_VALUES:
                path = point(args.points / case, s, m)
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        row = json.loads(saved["info"].item())
                    if (row["case"], row["S"], row["M"]) != (case, s, m):
                        raise ValueError(f"wrong point metadata in {path}")
                    # The first direct-factor sweeps predate these metadata fields.
                    if row.get("backend") == "sector-cholmod":
                        threads = {("rect_uo2", 2): 16, ("rect_uo2", 3): 24}
                        row.setdefault("threads", threads.get((case, s), 12))
                        row.setdefault("tolerance", 1e-9)
                    rows[case][s, m] = row
    for case, group in rows.items():
        for row in group.values():
            tol = row.get("tolerance", 1e-9)
            if (not np.isfinite([row["keff"], row["pcm"], row["seconds"],
                                row["factor_seconds"], row["keff_residual"],
                                row["source_residual"]]).all()
                    or row["keff_residual"] > tol or row["source_residual"] > 5 * tol):
                raise ValueError(f"point failed residual gates: {case} S={row['S']} M={row['M']}")
    return rows


def nrows(case):
    return 20 if case in ("strong", "weak") else 6


def matrix(group, case, field):
    return np.array([[group.get((s, m), {}).get(field, np.nan) for m in M_VALUES]
                     for s in range(1, nrows(case) + 1)])


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save(fig, output, name):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmaps(groups, best, output, name, field="pcm"):
    fig, axes = plt.subplots(len(groups) // 2, 2, figsize=(9.5, 3.5*len(groups)//2),
                             constrained_layout=True)
    for ax, (case, group) in zip(axes.flat, groups.items()):
        values = matrix(group, case, field)
        finite = values[np.isfinite(values)]
        if field == "seconds":
            lo, hi = (max(finite.min(), .01), finite.max()) if finite.size else (1, 10)
            options = dict(cmap="magma", norm=LogNorm(lo, max(hi, lo * 1.1)))
        else:
            lim = max(np.abs(finite).max(), 1) if finite.size else 1
            options = dict(cmap="RdBu_r", norm=SymLogNorm(1, vmin=-lim, vmax=lim))
        plane = ax.imshow(values, origin="lower", aspect="auto", **options)
        ax.set(title=f"{LABELS[case]} ({len(group)}/{nrows(case)*len(M_VALUES)})", xlabel="M", ylabel="S")
        ax.set_xticks(range(len(M_VALUES)), M_VALUES)
        rows = np.arange(0, nrows(case)) if nrows(case) <= 6 else np.array([0, 4, 9, 14, 19])
        ax.set_yticks(rows, rows + 1)
        if nrows(case) <= 6:
            for (y, x), val in np.ndenumerate(values):
                if not np.isfinite(val):
                    continue
                red, green, blue, _ = plane.cmap(plane.norm(val))
                luminance = .2126 * red + .7152 * green + .0722 * blue
                label = f"{val:+.1f}" if field == "pcm" else f"{val:.0f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                        color="black" if luminance > .55 else "white")
        if case in best:
            row = best[case]
            ax.add_patch(Circle((M_VALUES.index(row["M"]), row["S"] - 1), .46,
                                fill=False, color="#19a34a", linewidth=2))
        fig.colorbar(plane, ax=ax, label="Wall time (s)" if field == "seconds" else "Δk (pcm)")
    save(fig, output, name)


def pin_maps(case, row, points, refs):
    relpath, _ = CASES[case]
    spec = load_spec(EXAMPLES / relpath)
    geometry = spec["geometry"]
    rect = case.startswith("rect")
    material = np.asarray(geometry["mat_grid"])[1::3, 1::3] if rect else np.asarray(geometry["grid"])
    names = np.array([mat["name"].lower() for mat in spec["materials"]])[material]
    with np.load(point(points / case, row["S"], row["M"]), allow_pickle=False) as saved:
        if "pin_source" in saved:
            model = saved["pin_source"]
        elif not rect:
            model = np.einsum("gyx,gyx->yx", saved["pin_flux"], arrays(spec)[2][:, material])
        else:
            raise ValueError(f"rectangular best point for {case} lacks its area-weighted pin tally")
    if rect:
        with h5py.File(EXAMPLES / "c5g7-mgxs.h5") as library:
            nu = {}
            for key, label in {**FUEL_LIBRARY, "fc": "Fission Chamber"}.items():
                group = library["material"][label]
                nu[key] = group["nu-fission"][:].sum() / group["fission"][:].sum()
        # The paper converts the pin neutron-production map using material-average nu.
        model /= np.array([nu.get(key, 1) for key in names.flat]).reshape(17, 17)
        fuel = np.isin(names, tuple(FUEL_LIBRARY))
        reference = np.load(refs / REFERENCES[case], allow_pickle=False)
    else:
        fuel = np.isin(material, (0, 1, 2, 3))
        with np.load(refs / REFERENCES[case], allow_pickle=False) as saved:
            reference = saved["fission"]
    if (model.shape != (17, 17) or reference.shape != model.shape or fuel.sum() != 264
            or not np.isfinite(model).all() or not np.isfinite(reference).all()):
        raise ValueError(f"invalid pin map or 17×17 material mask for {case}")
    # Reproduce the paper figure script's minimum subtraction and unit pin sum.
    model, reference = [(a-a.min()) / (a-a.min()).sum() for a in (model, reference)]
    if not np.all(reference[fuel] > 0):
        raise ValueError(f"nonpositive OpenMC reference on fuel pins in {case}")
    error = np.full_like(model, np.nan)
    error[fuel] = (model[fuel] / reference[fuel] - 1) * 100
    return model, reference, error, fuel


def field_report(best, points, refs, output):
    records = []
    selected = [(case, best[case]) for case in CASES if case in best]
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.0*len(selected)),
                             constrained_layout=True, squeeze=False)
    for (case, row), panels in zip(selected, axes):
        model, ref, err, fuel = pin_maps(case, row, points, refs)
        rms, maximum = np.sqrt(np.mean(err[fuel]**2)), np.abs(err[fuel]).max()
        records.append((case, row["S"], row["M"], row["pcm"], rms, maximum))
        write_csv(output / (case + "_best_pin_fission.csv"),
                  ["X", "Y", "OpenPSN", "OpenMC", "relative_error_percent", "fuel"],
                  ((x+1, y+1, model[y, x], ref[y, x], err[y, x], int(fuel[y, x]))
                   for y, x in np.ndindex(model.shape)))
        bound = max(np.abs(err[fuel]).max(), .1)
        scale = (min(model.min(), ref.min()), max(model.max(), ref.max()))
        panels[0].set_ylabel(LABELS[case] + ("\n$\\Sigma_f\\phi$" if case.startswith("rect") else "\n$\\nu\\Sigma_f\\phi$"))
        titles = (f"OpenPSN S={row['S']} M={row['M']}", "OpenMC (same geometry)",
                  f"Fuel-pin error  RMS {rms:.3f}%  max {maximum:.3f}%")
        for ax, values, title, limits, cmap in zip(panels, (model, ref, err), titles,
                                                   (scale, scale, (-bound, bound)),
                                                   ("viridis", "viridis", "RdBu_r")):
            image = ax.imshow(values, origin="lower", vmin=limits[0], vmax=limits[1], cmap=cmap)
            ax.set(xticks=[], yticks=[], title=title)
            fig.colorbar(image, ax=ax, fraction=.05)
    save(fig, output, "studyIII_best_fission")
    return {r[0]: r[4:] for r in records}


def checker_fields(best, points, output):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for case, ax in zip(("strong", "weak"), axes):
        r = best[case]
        with np.load(points / (case + "_best_field.npz"), allow_pickle=False) as saved:
            if (saved["S"].item(), saved["M"].item()) != (r["S"], r["M"]):
                raise ValueError(f"checker field is from the wrong point for {case}")
            field = saved["fission"]
        if not np.isclose(field.sum(), 1):
            raise ValueError(f"checker fission map is not normalized for {case}")
        np.savetxt(output / (case + "_best_fission.csv"), field, delimiter=",")
        im = ax.imshow(field, extent=(0, 2, 0, 2), origin="lower", cmap="viridis")
        ax.set(title=f"{LABELS[case]}  S={r['S']}, M={r['M']}", xlabel="x (cm)", ylabel="y (cm)")
        fig.colorbar(im, ax=ax, label="Fission source fraction per cell")
    save(fig, output, "studyIII_checkerboard_best_fission")


def manuscript(summary, output):
    complete = all(row["complete"] == row["planned"] for row in summary) and len(summary) == len(LABELS)
    titles = {k: v.replace("UO₂", "UO$_2$") for k, v in LABELS.items()}
    lines = [r"\subsection{Study III --- finite-sector diffusion with bilinear azimuth coefficients}",
             r"\label{sec:study3-bilinear}",
             r"The even-parity diffusion operator is integrated over each finite angular cell.",
             r"The azimuthal mean is represented in a biquadratic spatial space and the linear",
             r"azimuth coefficient in its continuous bilinear subspace. The angular diffusion",
             r"coupling remains in the same weak form.",
             r"The checkerboards use $I=30$ and $S=1,\ldots,20$. The C5G7 assemblies use",
             r"$I=8$ finite polar subdivisions and $S=1,\ldots,6$. Each study uses",
             r"$M=2,4,8,12,16,24$. The square model uses the pin-homogenised cross sections",
             r"from Study I and the rectangular model uses the explicit $3\times3$ tiling and",
             r"material cross sections from Study II. The geometry-matched OpenMC eigenvalues",
             r"are 1.3336285 and 1.1853993 for the square UO$_2$ and MOX assemblies,",
             r"and 1.3347260 and 1.1851053 for their rectangular counterparts.", "",
             r"Each result passed a separate transport sweep to confirm the eigenvalue",
             r"and source residual. The best sampled setting in each matrix minimises",
             r"$|10^5(k-k_{\mathrm{ref}})|$. Table~\ref{tab:study3-bilinear-best} gives",
             r"the complete timing summary by case and the fuel-pin shape error at those settings.", ""]
    if not complete:
        lines.append(r"\textbf{Partial coverage.} Empty matrix entries carry no calculation result.")
    for image, label, caption in (
        ("studyIII_checkerboard_sm", "checker", "Strong and weak absorber checkerboards. The green marker indicates the smallest measured absolute eigenvalue difference in each panel."),
        ("studyIII_c5g7_sm", "c5g7", "Signed eigenvalue differences for the pin-homogenised square and explicit rectangular C5G7 assemblies, in pcm relative to each same-geometry OpenMC reference."),
        ("studyIII_best_fission", "fission", r"Best measured eigenvalue points. From left to right, OpenPSN, OpenMC and fuel-pin relative error. The square rows use $\nu\Sigma_f\phi$ and the rectangular rows use $\Sigma_f\phi$, each normalized to unit pin sum as in the corresponding paper figures."),
        ("studyIII_wall_time", "timing", "Measured wall-clock time per S/M point on the study host. The color scales are logarithmic. The underlying timing table records the solver backend and thread budget for each point."),
    ):
        lines += [r"\begin{figure}[htbp]", r"\centering",
                  r"\includegraphics[width=.86\linewidth]{" + image + ".pdf}",
                  r"\caption{" + caption + "}", r"\label{fig:study3-bilinear-" + label + "}", r"\end{figure}", ""]
    lines += [r"\begin{table}[htbp]", r"\centering",
              r"\caption{Coverage, best measured point, fuel-pin error and timing. The sum column adds individual wall times. Several cases were run concurrently.}",
              r"\begin{tabular}{lrrrrrrr}", r"\toprule",
              r"Case & Points & $S$ & $M$ & $\Delta k$ (pcm) & RMS (\%) & Sum (s) & Median (s) \\", r"\midrule"]
    for row in summary:
        rms = "--" if row["fuel_rms_percent"] is None else f"{row['fuel_rms_percent']:.3f}"
        lines.append(f"{titles[row['case']]} & {row['complete']}/{row['planned']} & {row['S']} & {row['M']} & "
                     f"{row['pcm']:+.2f} & {rms} & {row['sum_s']:.1f} & {row['median_s']:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\label{tab:study3-bilinear-best}", r"\end{table}"]
    (output / "studyIII_bilinear.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("points", "strong", "weak", "refs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = cases_data(args)
    best = {case: min(group.values(), key=lambda row: abs(row["pcm"]))
            for case, group in rows.items() if group}
    fields = ["case", "S", "M", "I", "A", "keff", "reference", "pcm", "seconds", "factor_seconds", "factor_bytes",
              "sweeps", "source_residual", "keff_residual", "backend", "threads", "tolerance", "field_seconds"]
    write_csv(args.output / "studyIII_points.csv", fields,
              ([r.get(f, "") for f in fields] for group in rows.values() for _, r in sorted(group.items())))
    for case, group in rows.items():
        for field in ("pcm", "seconds"):
            values = matrix(group, case, field)
            write_csv(args.output / f"{case}_{field}.csv", ["S", *M_VALUES],
                      ([s, *["" if np.isnan(x) else x for x in row]] for s, row in enumerate(values, 1)))
    heatmaps({key: rows[key] for key in ("strong", "weak")}, best, args.output, "studyIII_checkerboard_sm")
    heatmaps({key: rows[key] for key in CASES}, best, args.output, "studyIII_c5g7_sm")
    heatmaps(rows, best, args.output, "studyIII_wall_time", "seconds")
    rms = field_report(best, args.points, args.refs, args.output)
    if all((args.points / (key + "_best_field.npz")).is_file() for key in ("strong", "weak")):
        checker_fields(best, args.points, args.output)
    summary = []
    for case, group in rows.items():
        if not group:
            continue
        r = best[case]
        times = [p["seconds"] for p in group.values()]
        summary.append(dict(case=case, complete=len(group), planned=nrows(case) * len(M_VALUES),
                            S=r["S"], M=r["M"], pcm=r["pcm"], keff=r["keff"],
                            seconds_at_best=r["seconds"], sum_s=sum(times),
                            median_s=float(np.median(times)), max_s=max(times),
                            factor_sum_s=sum(p["factor_seconds"] for p in group.values()),
                            fuel_rms_percent=rms.get(case, (None, None))[0], fuel_max_percent=rms.get(case, (None, None))[1]))
    (args.output / "studyIII_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary:
        write_csv(args.output / "studyIII_summary.csv", list(summary[0]),
                  (row.values() for row in summary))
    manuscript(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
