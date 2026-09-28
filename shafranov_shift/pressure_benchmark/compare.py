"""Compare VMEX axes with the traced vacuum axis and the first-order theory.

    python compare.py RUNS_DIR [RUN ...]

For every run directory with a row.json: vacuum-axis bias (VMEX axis at beta = 0 minus the
traced coil axis), and for beta > 0 the displacement relative to the matching beta = 0 run of
the same case, radius and numerical settings, projected on the theory shape:
ratio = <dX, xi> / <xi, xi> over phi, with the residual orthogonal to xi.
"""

import json
import sys
from pathlib import Path

import numpy as np

MU0 = 4e-7 * np.pi


def key(row):
    return (row["case"], row["radius_m"], tuple(row["ns"]), tuple(row["ftol"]), row["mpol"], row["ntor"],
            row["nzeta"], row["delt"])


def load(root, names=None):
    rows = {}
    for path in sorted(Path(root).glob("*/row.json")):
        if names and path.parent.name not in names:
            continue
        row = json.loads(path.read_text())
        row["name"] = path.parent.name
        rows[row["name"]] = row
    return rows


def analyse(root, names=None):
    root = Path(root)
    rows = load(root, names)
    theory = {}
    out = []
    for name, row in rows.items():
        case = row["case"]
        if case not in theory:
            theory[case] = np.load(root / "theory" / f"theory_{case}.npz")
        th = theory[case]
        X = np.array([row["axis_R"], row["axis_Z"]])
        traced = th["traced_axis"]
        bias = X - traced
        entry = dict(name=name, case=case, radius_m=row["radius_m"], beta=row["beta_axis_target"],
                     converged=row["converged"], iterations=row["iterations"], restart=row["restart"],
                     ns=row["ns"], mpol=row["mpol"],
                     axis_minus_traced_rms_m=float(np.sqrt(np.mean(np.sum(bias**2, 0)))),
                     axis_minus_traced_R0_m=float(bias[0, 0]))
        if row["beta_axis_target"] > 0:
            xi = row["p0_Pa"] * th["traced"]
            # Same case, radius, settings and protocol (cold, or continued from a wout).
            ref = [r for r in rows.values() if r["beta_axis_target"] == 0 and key(r) == key(row)
                   and (r["restart"] is None) == (row["restart"] is None)]
            entry["theory_rms_m"] = float(np.sqrt(np.mean(np.sum(xi**2, 0))))
            entry["theory_R0_m"] = float(xi[0, 0])
            entry["ratio_vs_traced"] = float(np.sum(bias * xi) / np.sum(xi * xi))
            if ref:
                V = np.array([ref[0]["axis_R"], ref[0]["axis_Z"]])
                d = X - V
                ratio = float(np.sum(d * xi) / np.sum(xi * xi))
                entry.update(vacuum_run=ref[0]["name"], ratio=ratio,
                             residual_rms_over_theory_rms=float(np.sqrt(np.mean(np.sum((d - ratio * xi) ** 2, 0)))
                                                                / entry["theory_rms_m"]),
                             dR0_m=float(d[0, 0]))
        out.append(entry)
    return out


if __name__ == "__main__":
    for e in analyse(sys.argv[1], sys.argv[2:] or None):
        print(json.dumps(e))
