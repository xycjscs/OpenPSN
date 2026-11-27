"""Export complete paired Study IV C5G7 matrices from the tracked data.

Use --fields to refresh the assembly RMS from the original checkpoints,
and scripts.study4_core_pin_rms to refresh the quarter-core provenance CSV.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.study3_bilinear_report import CASES, LABELS, M_VALUES, write_csv
from scripts.study4_core_report import S_VALUES as CORE_S, load, timing_basis
from scripts.study4_pin_rms import S_VALUES as ASSEMBLY_S, assembly_points, calculate, read_matrix
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/c5g7"
DATA = EXAMPLES / "study4_c5g7_matrices"
METRICS = (("keff", "k-effective", ".9f"), ("delta_k_pcm", "Signed Δk (pcm)", "+.2f"),
           ("pin_rms_percent", "Fuel-pin RMS (%)", ".5f"))


def collect(data, fields, core_provenance):
    cases = {case: (LABELS[case], ASSEMBLY_S, assembly_points(case),
                    calculate(fields, case) if fields else
                    read_matrix(data / f"{case}_pin_rms_percent.csv")) for case in CASES}
    logs = EXAMPLES / "study4_sector_core/logs"
    core_records, _ = load(logs, {s: (153*s)**2 for s in CORE_S}, basis=timing_basis(logs))
    with core_provenance.open() as handle:
        estimates = list(csv.DictReader(handle))
    pin_rows = {(int(row["S"]), int(row["M"])): row for row in estimates}
    if len(estimates) != len(pin_rows) or pin_rows.keys() != core_records.keys():
        raise ValueError("expected one recovered pin RMS for each quarter-core cell")
    for key, row in pin_rows.items():
        if float(row["archived_keff"]) != core_records[key]["keff"]:
            raise ValueError(f"quarter-core pin provenance does not match {key}")
    core_rms = np.array([[float(pin_rows[s, m]["fuel_rms_percent"]) for m in M_VALUES] for s in CORE_S])
    cases["rect_core"] = ("Rectangular quarter core", CORE_S, core_records, core_rms)

    for case, (label, spatial, points, rms) in cases.items():
        grid = [[(s, m) for m in M_VALUES] for s in spatial]
        keff, pcm = (np.array([[float(points[key][field]) for key in row] for row in grid])
                     for field in ("keff", "pcm"))
        refs = {float(row.get("reference", row.get("kref"))) for row in points.values()}
        if len(refs) != 1 or rms.shape != keff.shape or not np.isfinite(rms).all() or (rms < 0).any():
            raise ValueError(f"invalid or incomplete matrix for {case}")
        reference, = refs
        np.testing.assert_allclose(pcm, 1e5 * (keff - reference), atol=1e-8, rtol=0)
        yield case, label, spatial, reference, (keff, pcm, rms)


def pdf_page(pdf, label, spatial, reference, arrays, max_pcm, max_rms):
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), layout="constrained")
    fig.suptitle(f"Study IV · {label}\nSame-geometry OpenMC k-reference = {reference:.12f}")
    for ax, (key, title, digits), values in zip(axes, METRICS, arrays):
        if key == "keff":
            ax.axis("off")
            table = ax.table(cellText=[[format(value, digits) for value in row] for row in values],
                             colLabels=[f"M={m}" for m in M_VALUES],
                             rowLabels=[f"S={s}" for s in spatial], loc="center", cellLoc="right")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax.set_title(title)
            continue
        if key == "delta_k_pcm":
            image = ax.imshow(values, cmap="RdBu_r", aspect="auto",
                              norm=SymLogNorm(1, vmin=-max_pcm, vmax=max_pcm))
        else:
            image = ax.imshow(values, cmap="viridis_r", aspect="auto", vmin=0, vmax=max_rms)
        ax.set(title=title, xlabel="Azimuthal sectors M", ylabel="Spatial subdivision S",
               xticks=range(len(M_VALUES)), xticklabels=M_VALUES,
               yticks=range(len(spatial)), yticklabels=spatial)
        fig.colorbar(image, ax=ax, pad=.02)
        for (i, j), value in np.ndenumerate(values):
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = .2126*red + .7152*green + .0722*blue
            ax.text(j, i, format(value, digits), color="black" if luminance > .50 else "white",
                    ha="center", va="center")
        i, j = np.unravel_index(np.abs(values).argmin(), values.shape)
        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, linewidth=1.8,
                               edgecolor="#ce3c65", fill=False))
    pdf.savefig(fig)
    plt.close(fig)


def markdown_section(label, spatial, reference, arrays):
    lines = [f"## {label}", "", f"Same-geometry OpenMC reference `k_ref={reference:.12f}`.", ""]
    header = ["S / M", *(str(m) for m in M_VALUES)]
    for (_, title, digits), values in zip(METRICS, arrays):
        lines.extend([f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---:|" * len(header)])
        for s, row in zip(spatial, values):
            lines.append("| " + " | ".join([str(s), *(format(value, digits) for value in row)]) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--data", type=Path, default=DATA, help="tracked assembly RMS matrix directory")
    parser.add_argument("--core-provenance", type=Path, default=DATA / "core_pin_provenance.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = list(collect(args.data, args.fields, args.core_provenance))
    max_pcm = max(np.abs(arrays[1]).max() for *_, arrays in cases)
    max_rms = max(arrays[2].max() for *_, arrays in cases)
    args.output.mkdir(parents=True, exist_ok=True)
    markdown = ["# Study IV C5G7 S–M matrices", "",
                r"| Case | Min \|Δk\| cell (S, M) | Signed Δk (pcm) | Min RMS cell (S, M) | RMS (%) |",
                "|:---|---:|---:|---:|---:|"]
    for _, label, spatial, _, (_, pcm, rms) in cases:
        i, j = np.unravel_index(np.abs(pcm).argmin(), pcm.shape)
        u, v = np.unravel_index(rms.argmin(), rms.shape)
        markdown.append(f"| {label} | ({spatial[i]}, {M_VALUES[j]}) | {pcm[i, j]:+.2f} | "
                        f"({spatial[u]}, {M_VALUES[v]}) | {rms[u, v]:.5f} |")
    markdown.append("")
    with plt.rc_context({"pdf.fonttype": 42}), PdfPages(args.output / "study4_c5g7_matrices.pdf") as pdf:
        for case, label, spatial, reference, arrays in cases:
            pdf_page(pdf, label, spatial, reference, arrays, max_pcm, max_rms)
            markdown.append(markdown_section(label, spatial, reference, arrays))
            for (key, _, _), matrix in zip(METRICS, arrays):
                write_csv(args.output / f"{case}_{key}.csv", ["S", *(f"M{m}" for m in M_VALUES)],
                          ([s, *row] for s, row in zip(spatial, matrix)))
    (args.output / "sm_matrices.md").write_text("\n".join(markdown))
    print(f"Exported {sum(len(spatial) * len(M_VALUES) for _, _, spatial, *_ in cases)} cells over five cases")


if __name__ == "__main__":
    main()
