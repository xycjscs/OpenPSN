"""Recover the Study IV quarter-core pin RMS at each certified S–M point.

The input manifest has columns S,M,checkpoint,run_log. Paths are relative to
the manifest directory. A separate run log is needed for source-only snapshots
without a solver record embedded in the checkpoint.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from psn2d.model import arrays, load_spec
from scripts.sector_diffusion_benchmark import INPUTS
from scripts.study3_bilinear_report import EXAMPLES, FUEL_LIBRARY
from scripts.study4_core_report import (EIGEN_RESIDUAL, SOURCE_RESIDUAL, KREF, S_VALUES, load,
                                        read_record, timing_basis)


PINS = 51
ROOT = EXAMPLES.parents[1]
CORE = EXAMPLES / "study4_sector_core"
# Independent, residual-accepted solves of the same cell must agree to 0.001 pcm.
KEFF_MATCH = 1e-8


def pin_sum(field, stride):
    return field.reshape(PINS, stride, PINS, stride).sum(axis=(1, 3))


def run_record(saved, run_log):
    embedded = json.loads(saved["info"].item()) if "info" in saved else None
    if run_log is None:
        if embedded is None:
            raise ValueError("no certified solver record for the checkpoint")
        return embedded
    record = read_record(run_log)
    if embedded and record["keff"] != embedded["keff"]:
        raise ValueError(f"checkpoint and run log disagree: {run_log}")
    return record


def checkpoint_metrics(point, archived, checkpoint, run_log, material, area,
                       scattering, nu_fission, chi, xs, reference, fuel):
    s, m = point
    with np.load(checkpoint, allow_pickle=False) as saved:
        record = run_record(saved, run_log)
        if (int(saved["s"]) != s
                or (record["subdivide"], record["M"], record["A"], record["I"]) != (s, m, 1, 8)
                or record["model"] != "ty3" or record["closure"] != "sector_diffusion"
                or record["kref"] != KREF or not record["converged"]
                or not 0 <= record["source_residual"] <= SOURCE_RESIDUAL
                or not 0 <= record["keff_residual"] <= EIGEN_RESIDUAL
                or not abs(record["keff"] - archived["keff"]) <= KEFF_MATCH):
            raise ValueError(f"uncertified or different discretization in {checkpoint}")
        q = saved["q"]
        direct = saved["flux"] if "flux" in saved else None
    if q.shape != (7, len(material), 9) or not np.isfinite(q).all():
        raise ValueError(f"invalid saved source in {checkpoint}")

    source = q[:, :, 0]
    flux = np.empty_like(source)
    rebuild_norm_sq = 0.0
    weight_sq = np.empty(xs.shape[1])
    for i in range(xs.shape[1]):
        select = material == i
        coupling = scattering[:, :, i] + np.outer(chi, nu_fission[:, i]) / record["keff"]
        weight_sq[i] = np.linalg.norm(np.linalg.solve(coupling.T, xs[:, i]))**2
        if select.any():
            flux[:, select] = np.linalg.solve(coupling, source[:, select])
            rebuild_norm_sq += np.linalg.norm(coupling @ flux[:, select] - source[:, select])**2
    rebuild = np.sqrt(rebuild_norm_sq) / np.linalg.norm(source)
    if rebuild >= 5e-14 or not np.isfinite(flux).all():
        raise ValueError(f"failed local source inversion in {checkpoint}")
    direct_difference = ""
    if direct is not None:
        if direct.shape != flux.shape:
            raise ValueError(f"invalid verified flux shape in {checkpoint}")
        direct_difference = np.linalg.norm(flux - direct) / np.linalg.norm(direct)
        if not direct_difference < 1e-6:
            raise ValueError(f"recovered and verified flux disagree in {checkpoint}")
        flux = direct

    production = nu_fission[:, material] * area
    recovered_keff = np.sum(production * flux) / production.sum()
    if not abs(recovered_keff - record["keff"]) < 1e-7:
        raise ValueError(f"pin flux does not reproduce the eigenvalue in {checkpoint}")
    mean = pin_sum(area * np.sum(flux * xs[:, material], axis=0), 3 * s)
    if not (mean[fuel] > 0).all():
        raise ValueError(f"nonpositive fuel fission rates in {checkpoint}")
    error = 100 * (mean[fuel] / mean.sum() / reference[fuel] - 1)

    pin_norm = np.sqrt(pin_sum(area**2 * weight_sq[material], 3 * s))
    source_error = np.linalg.norm(q) * record["source_residual"]
    total_error = np.linalg.norm(pin_norm) * source_error
    normalized = mean[fuel] / mean.sum()
    shift = (pin_norm[fuel] * source_error + normalized * total_error)
    if total_error >= mean.sum():
        raise ValueError(f"source residual gives no finite map bound in {checkpoint}")
    relative_shift = 100 * np.max(shift / ((mean.sum() - total_error) * normalized))

    return dict(S=s, M=m, checkpoint=checkpoint.name, field_run=record["name"],
                archived_keff=archived["keff"], field_keff=record["keff"],
                field_keff_residual=record["keff_residual"], field_source_residual=record["source_residual"],
                map_flux="direct verified sweep" if direct is not None else "source inversion",
                recovered_keff=recovered_keff, source_rebuild_relative_residual=rebuild,
                recovered_vs_direct_relative_difference=direct_difference,
                map_shift_bound_percent=relative_shift,
                fuel_rms_percent=np.sqrt(np.mean(error**2)), fuel_max_abs_percent=np.abs(error).max())


def calculate(manifest, logs=CORE / "logs"):
    records, _ = load(logs, {s: (3 * PINS * s)**2 for s in S_VALUES}, basis=timing_basis(logs))
    with manifest.open() as handle:
        entries = list(csv.DictReader(handle))
    sources = {(int(row["S"]), int(row["M"])): row for row in entries}
    if sources.keys() != records.keys() or len(entries) != len(sources):
        raise ValueError("the manifest must cover each quarter-core cell exactly once")

    spec = load_spec(INPUTS["rect"])
    geom = spec["geometry"]
    base = np.asarray(geom["mat_grid"])
    _, scattering, nu_fission, chi = arrays(spec)
    names = {**FUEL_LIBRARY, "fc": "Fission Chamber"}
    xs = np.zeros_like(nu_fission)
    with h5py.File(EXAMPLES / "c5g7-mgxs.h5") as library:
        for i, material in enumerate(spec["materials"]):
            if material["name"] in names:
                table = library["material"][names[material["name"]]]
                np.testing.assert_array_equal(table["nu-fission"][:], nu_fission[:, i])
                xs[:, i] = table["fission"][:]
    fuel = np.isin(base[1::3, 1::3], [i for i, mat in enumerate(spec["materials"])
                                     if mat["name"] in FUEL_LIBRARY])
    reference = np.load(ROOT / "tools_ho/data/omc_rect_core_power.npy", allow_pickle=False)
    if reference.shape != (PINS, PINS) or fuel.sum() != 1056 or not (reference[fuel] > 0).all():
        raise ValueError("invalid same-geometry quarter-core OpenMC tally")
    reference = reference / reference.sum()

    results = []
    for s in S_VALUES:
        material = base.repeat(s, axis=0).repeat(s, axis=1).ravel()
        hx = (np.asarray(geom["widths_x"])[0] / s).repeat(s)
        hy = (np.asarray(geom["widths_y"])[:, 0] / s).repeat(s)
        area = np.outer(hy, hx).ravel()
        for point, archived in records.items():
            if point[0] != s:
                continue
            row = sources[point]
            checkpoint = manifest.parent / row["checkpoint"]
            run_log = manifest.parent / row["run_log"] if row["run_log"] else None
            result = checkpoint_metrics(point, archived, checkpoint, run_log, material, area,
                                        scattering, nu_fission, chi, xs, reference, fuel)
            results.append(result)
            print(f"Core S={s} M={point[1]} RMS {result['fuel_rms_percent']:.7f}%", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = calculate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
