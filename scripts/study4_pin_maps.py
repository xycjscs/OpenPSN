"""Render the finest Study IV pin maps on one relative-error scale.

The published pin CSVs suffice to redraw both figures. Supply the original
assembly checkpoints with --fields to regenerate the assembly CSVs as well.
"""

import argparse
import csv
import json
from math import ceil
from pathlib import Path

import numpy as np

from scripts.study3_bilinear_report import CASES, LABELS, pin_maps, save, write_csv
from scripts.study4_core_report import M_VALUES, S_VALUES
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/c5g7"
ASSEMBLY_GRID = (6, 24)
CORE_GRID = (max(S_VALUES), max(M_VALUES))


def read_pins(path, size, model_column):
    model, ref, error = [np.full((size, size), np.nan) for _ in range(3)]
    with path.open() as handle:
        reader = csv.DictReader(handle)
        x_column, y_column = reader.fieldnames[:2]
        for row in reader:
            x, y = int(row[x_column]) - 1, int(row[y_column]) - 1
            if not (0 <= x < size and 0 <= y < size) or np.isfinite(model[y, x]):
                raise ValueError(f"invalid or duplicate pin coordinate in {path}")
            model[y, x], ref[y, x] = float(row[model_column]), float(row["OpenMC"])
            if int(row["fuel"]):
                error[y, x] = float(row["relative_error_percent"])
    fuel = np.isfinite(error)
    if (not np.isfinite(model).all() or not np.isfinite(ref).all()
            or fuel.sum() != (264 if size == 17 else 1056) or (ref[fuel] <= 0).any()
            or not np.allclose(error[fuel], 100 * (model[fuel] / ref[fuel] - 1), atol=1e-12)):
        raise ValueError(f"incomplete or inconsistent pin fields in {path}")
    return model, ref, error, fuel


def write_pins(path, model, ref, error, fuel):
    write_csv(path, ["X", "Y", "OpenPSN", "OpenMC", "relative_error_percent", "fuel"],
              ((x + 1, y + 1, model[y, x], ref[y, x], error[y, x], int(fuel[y, x]))
               for y, x in np.ndindex(model.shape)))


def plot(rows, grid, bound, size, output, name):
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.1 * len(rows)),
                             layout="constrained", squeeze=False)
    s, m = grid
    fig.suptitle(f"Study IV pin fields | S={s}, M={m}")
    error_colors = plt.get_cmap("RdBu_r").with_extremes(bad="#d9d9d9")
    for (label, pcm, data), panels in zip(rows, axes):
        model, ref, error, fuel = data
        error_values = error[fuel]
        rms = np.sqrt(np.mean(error_values**2))
        maximum = np.abs(error_values).max()
        titles = (f"OpenPSN\nΔk = {pcm:+.2f} pcm", "OpenMC\nSame geometry",
                  f"Fuel-pin relative error\nRMS {rms:.3f}% · max {maximum:.3f}%")
        scale = (min(model.min(), ref.min()), max(model.max(), ref.max()))
        panels[0].set_ylabel(label)
        for axis, values, title in zip(panels, (model, ref, error), titles):
            is_error = axis is panels[2]
            limits = (-bound, bound) if is_error else scale
            image = axis.imshow(values[:size, :size], origin="lower", interpolation="nearest",
                                cmap=error_colors if is_error else "viridis",
                                vmin=limits[0], vmax=limits[1])
            axis.set(xticks=[], yticks=[])
            axis.set_title(title, fontsize=10)
            if not is_error:
                fig.colorbar(image, ax=axis, fraction=.05)
    fig.colorbar(image, ax=axes[:, 2].tolist(), ticks=np.linspace(-bound, bound, 5),
                 label="Relative error (%)", fraction=.05)
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fields", type=Path, help="original assembly sweep checkpoints")
    parser.add_argument("--assembly-data", type=Path, default=EXAMPLES / "study4_sector_assembly")
    parser.add_argument("--core-data", type=Path, default=EXAMPLES / "study4_sector_core")
    args = parser.parse_args()

    with (ROOT / "paper/data/study4_assembly_points.csv").open() as handle:
        measured = [row for row in csv.DictReader(handle) if row["closure"] == "sector_diffusion"
                    and (int(row["S"]), int(row["M"])) == ASSEMBLY_GRID and row["case"] in CASES]
    by_case = {row["case"]: row for row in measured}
    if len(measured) != len(CASES) or set(by_case) != CASES.keys():
        raise ValueError("a unique finest-grid record is required for each Study IV assembly")
    core = json.loads((args.core_data / "best_fission_summary.json").read_text())
    if (core["subdivide"], core["M"]) != CORE_GRID or core["closure"] != "sector_diffusion":
        raise ValueError("the quarter-core pin tally must come from the finest grid")
    core_pins = read_pins(args.core_data / "best_fission.csv", 51, "sector_diffusion")

    assemblies = []
    if args.fields:
        args.assembly_data.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        row = by_case[case]
        path = args.assembly_data / f"{case}_finest_pin_fission.csv"
        if args.fields:
            data = pin_maps(case, row, args.fields, ROOT / "tools_ho/data", closure="sector_diffusion")
            write_pins(path, *data)
        else:
            data = read_pins(path, 17, "OpenPSN")
        quantity = r"$\Sigma_f\phi$" if case.startswith("rect") else r"$\nu\Sigma_f\phi$"
        assemblies.append((f"{LABELS[case]}\n{quantity}", float(row["pcm"]), data))

    core_rows = [("Rectangular quarter core", float(core["pcm"]), core_pins)]
    maximum = max(np.abs(error[fuel]).max() for _, _, (_, _, error, fuel) in assemblies + core_rows)
    bound = max(.05, ceil(20 * maximum) / 20)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(assemblies, ASSEMBLY_GRID, bound, 17, args.output, "study4_assembly_fission")
    plot(core_rows, CORE_GRID, bound, 34, args.output, "study4_core_fission")
    print(f"Study IV assembly {ASSEMBLY_GRID} and quarter-core {CORE_GRID} maps share ±{bound:.2f}%")


if __name__ == "__main__":
    main()
