"""Build the Study IV square UO₂ fuel-pin RMS matrix and its figure.

With --fields, compute the matrix from the 36 assembly checkpoints.
Without it, redraw the figure and table from the tracked matrix CSV.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.study3_bilinear_report import M_VALUES, pin_maps, save, write_csv
from scripts.study4_core_report import EIGEN_RESIDUAL, SOURCE_RESIDUAL, latex_table
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
CASE = "square_uo2"
CLOSURE = "sector_diffusion"
S_VALUES = tuple(range(1, 7))
CSV_HEADER = ["S", *(f"M{m}" for m in M_VALUES)]
NAME = "study4_square_uo2_pin_rms"
LABEL = "tab:study4-square-uo2-pin-rms"
CAPTION = (r"RMS relative error (\%) in normalized neutron production over the 264 fuel "
           r"pins of the square UO$_2$ assembly, against the same-geometry OpenMC reference. "
           r"The calculation uses direction-wise sector diffusion, $A=1$ and $I=8$. "
           r"Bold marks the minimum of the unrounded values.")


def assembly_points(case):
    with (ROOT / "paper/data/study4_assembly_points.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if (row["case"], row["closure"], row["A"], row["I"]) == (case, CLOSURE, "1", "8")]
    points = {(int(row["S"]), int(row["M"])): row for row in rows}
    expected = {(s, m) for s in S_VALUES for m in M_VALUES}
    if points.keys() != expected or len(rows) != len(expected):
        raise ValueError(f"expected one {case} record for each S–M cell")
    return points


def calculate(fields, case=CASE):
    points = assembly_points(case)
    rms = np.empty((len(S_VALUES), len(M_VALUES)))
    for i, s in enumerate(S_VALUES):
        for j, m in enumerate(M_VALUES):
            row = points[s, m]
            if not (0 <= float(row["source_residual"]) <= SOURCE_RESIDUAL
                    and 0 <= float(row["keff_residual"]) <= EIGEN_RESIDUAL):
                raise ValueError(f"{case} S={s}, M={m} failed the verification criteria")
            _, _, error, fuel = pin_maps(case, row, fields, ROOT / "tools_ho/data", closure=CLOSURE)
            rms[i, j] = np.sqrt(np.mean(error[fuel]**2))
    return rms


def read_matrix(path):
    with path.open() as handle:
        header, *rows = csv.reader(handle)
    if header != CSV_HEADER or [int(row[0]) for row in rows] != list(S_VALUES):
        raise ValueError(f"unexpected S–M axes in {path}")
    return np.array([[float(value) for value in row[1:]] for row in rows])


def plot(rms, best, output):
    fig, ax = plt.subplots(figsize=(7.3, 4.7), layout="constrained")
    image = ax.imshow(rms, cmap="viridis_r", vmin=0, aspect="auto")
    ax.set(title=r"Square UO$_2$ | Study IV fuel-pin RMS ($A=1$, $I=8$)",
           xlabel="Azimuthal sectors M", ylabel="Spatial subdivision S",
           xticks=range(len(M_VALUES)), xticklabels=M_VALUES,
           yticks=range(len(S_VALUES)), yticklabels=S_VALUES)
    for (i, j), value in np.ndenumerate(rms):
        red, green, blue, _ = image.cmap(image.norm(value))
        luminance = .2126 * red + .7152 * green + .0722 * blue
        ax.text(j, i, f"{value:.5f}", ha="center", va="center",
                color="black" if luminance > .50 else "white")
    i, j = best
    ax.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False,
                           edgecolor="#b81935", linewidth=2))
    fig.colorbar(image, ax=ax, label="Fuel-pin RMS relative error (%)")
    save(fig, output, NAME)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, help="original assembly checkpoints")
    parser.add_argument("--matrix", type=Path, default=ROOT / f"paper/data/{NAME}.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latex", type=Path, help="output file for the paper table")
    args = parser.parse_args()
    rms = calculate(args.fields) if args.fields else read_matrix(args.matrix)
    if rms.shape != (len(S_VALUES), len(M_VALUES)) or not np.isfinite(rms).all() or (rms < 0).any():
        raise ValueError("expected a complete nonnegative RMS matrix")
    if args.fields:
        write_csv(args.matrix, CSV_HEADER, ([s, *row] for s, row in zip(S_VALUES, rms)))
    best = np.unravel_index(rms.argmin(), rms.shape)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(rms, best, args.output)
    if args.latex:
        rows = [(s, [rf"\textbf{{{value:.5f}}}" if (i, j) == best else f"{value:.5f}"
                     for j, value in enumerate(row)])
                for i, (s, row) in enumerate(zip(S_VALUES, rms))]
        args.latex.write_text(latex_table(LABEL, CAPTION, rows, r"Fuel-pin RMS (\%)"))
    print(f"36 square UO₂ points, minimum {rms[best]:.8f}% at "
          f"S={S_VALUES[best[0]]}, M={M_VALUES[best[1]]}")


if __name__ == "__main__":
    main()
