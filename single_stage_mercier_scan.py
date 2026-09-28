"""Near-axis stage of the single-stage design with a hard Mercier hinge: the quasisymmetry it costs.

python single_stage_mercier_scan.py   -> single_stage_mercier_scan.json (about 5 minutes)

Each row re-solves stage 1 of drivers/optimize_single_stage_nearaxis_finite_beta.py (the equilibrium
alone, from the seed, 400 evaluations) with the driver's weights, except for the changes listed in the
row. The driver itself keeps a magnetic-well margin and a weak Mercier term (see its MERCIER_WEIGHT).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
D = ROOT / "drivers"
sys.path.insert(0, str(D))
os.chdir(D)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

SOURCE = (D / "optimize_single_stage_nearaxis_finite_beta.py").read_text()
SOURCE = SOURCE[:SOURCE.index('""" Optimization, one segment per run """')]
FOUR = "rc=[1.0, 0.155, 0.0102, 0.0, 0.0, 0.0, 0.0], zs=[0.0, 0.154, 0.0111, 0.0, 0.0, 0.0, 0.0]"
TWO = "rc=[1.0, 0.155, 0.0102, 0.0, 0.0], zs=[0.0, 0.154, 0.0111, 0.0, 0.0]"
# (label, radius [m], Mercier weight, well weight, elongation cap, extra axis harmonics)
ROWS = [("driver: weak Mercier term, well margin", 0.10, 0.01, 100.0, 6.0, 4),
        ("no Mercier term, no well margin", 0.10, 0.0, 0.0, 6.0, 4),
        ("hard Mercier hinge", 0.10, 100.0, 0.0, 6.0, 4),
        ("hard Mercier hinge", 0.10, 100.0, 0.0, 6.0, 2),
        ("hard Mercier hinge, elongation cap 8", 0.10, 100.0, 0.0, 8.0, 4),
        ("hard Mercier hinge, elongation cap 8", 0.12, 100.0, 0.0, 8.0, 4),
        ("hard Mercier hinge", 0.15, 100.0, 0.0, 6.0, 4),
        ("hard Mercier hinge", 0.15, 100.0, 0.0, 6.0, 2),
        ("hard Mercier hinge", 0.20, 100.0, 0.0, 6.0, 2)]


def solve(radius, mercier, well, elongation, harmonics):
    src = (SOURCE.replace("PLASMA_RADIUS = 0.10", f"PLASMA_RADIUS = {radius}")
           .replace("MAX_ELONGATION = 6.0", f"MAX_ELONGATION = {elongation}")
           .replace(FOUR, FOUR if harmonics == 4 else TWO)
           .replace('OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_stage_finite_beta"',
                    f'OUTPUT_DIR = Path("{ROOT}/runs/mercier_scan")'))
    src = re.sub(r"MERCIER_WEIGHT = [0-9.]+", f"MERCIER_WEIGHT = {mercier}", src)
    src = re.sub(r"WELL_MARGIN = ([0-9.]+); WELL_WEIGHT = [0-9.]+", rf"WELL_MARGIN = \1; WELL_WEIGHT = {well}", src)
    g = {"__name__": "scan", "__file__": str(D / "scan.py")}
    exec(compile(src, "single_stage", "exec"), g)
    full, index = g["initial_dofs"], g["SHAPE_INDEX"]
    function = lambda x: g["physics_residuals"](full.at[index].set(x))
    residuals, jacobian = jax.jit(function), jax.jit(jax.jacfwd(function))
    seed, n = np.asarray(g["seed_shape"]), g["N_MODES"]
    width = np.r_[np.full(2 * n, g["AXIS_SHAPE_BOUND"]), 0.5, 3.0, 0.5]  # As in the driver's stage 1
    result = least_squares(lambda x: np.asarray(residuals(jnp.asarray(x))), seed,
                           jac=lambda x: np.asarray(jacobian(jnp.asarray(x))), bounds=(seed - width, seed + width),
                           x_scale="jac", ftol=1e-10, gtol=1e-10, xtol=1e-12, max_nfev=400)
    near = g["make_near_axis"](jnp.asarray(result.x), 151)
    s = near.solution
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)[None]
    X = np.asarray(radius * s.X1c[:, None] * np.cos(theta) + radius**2 * (
        s.X20[:, None] + s.X2s[:, None] * np.sin(2 * theta) + s.X2c[:, None] * np.cos(2 * theta)))
    Y = np.asarray(radius * (s.Y1s[:, None] * np.sin(theta) + s.Y1c[:, None] * np.cos(theta)) + radius**2 * (
        s.Y20[:, None] + s.Y2s[:, None] * np.sin(2 * theta) + s.Y2c[:, None] * np.cos(2 * theta)))
    return dict(beta=float(g["beta_of"](jnp.asarray(result.x))), iota=float(near.iota),
                B20_variation_R0sq_over_B0=float(s.second_order.B20_variation * g["R0"]**2),
                DMerc_times_r2=float(s.DMerc_times_r2), DWell_times_r2=float(s.DWell_times_r2),
                r_singularity_over_a=float(s.r_singularity / radius), max_elongation=float(jnp.max(near.elongation)),
                boundary_reach_from_axis_m=float(np.hypot(X, Y).max()), cost=float(result.cost))


rows = []
for label, radius, mercier, well, elongation, harmonics in ROWS:
    row = dict(label=label, a_m=radius, mercier_weight=mercier, well_weight=well, max_elongation_cap=elongation,
               extra_axis_harmonics=harmonics, **solve(radius, mercier, well, elongation, harmonics))
    print(json.dumps(row), flush=True)
    rows.append(row)
(ROOT / "single_stage_mercier_scan.json").write_text(json.dumps(rows, indent=1) + "\n")
