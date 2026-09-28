"""Exact B2c elimination vs the stored eight-mode axis refinement (seed: database 57409)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    COLORS,
    DOUBLE,
    configuration,
    draw_surface,
    field_period_angle,
    plt,
    qsc,
    save,
)

NPHI = 121
RADIUS = 0.075

stock = qsc.optimize_B2c(configuration("database_low_b20_57409", nphi=NPHI))
refined = configuration("b20_optimized_good", nphi=NPHI)
d_stock, d_ref = stock.diagnostics, qsc.b20_diagnostics(refined)
check = {
    n: float(qsc.b20_diagnostics(configuration("b20_optimized_good", nphi=n)).weighted_l2)
    for n in (NPHI, 2 * NPHI + 1)
}

figure = plt.figure(figsize=(DOUBLE, 4.4))
grid = figure.add_gridspec(
    2,
    2,
    width_ratios=(1.0, 1.15),
    hspace=0.6,
    wspace=0.3,
    left=0.0,
    right=0.98,
    top=0.95,
    bottom=0.1,
)
ax3d = figure.add_subplot(grid[:, 0], projection="3d")
draw_surface(ax3d, refined, RADIUS, color=COLORS[2], view=(60, 38))
ax3d.set_title(
    rf"(a) refined QH, $r={RADIUS}$ m, $|\iota_0|={abs(float(refined.iota)):.3f}$", pad=-10
)
for row, (label, sol, d, color, exponent) in enumerate(
    (
        (r"(b) ID 57409 + exact $B_{2c}$", stock.solution, d_stock, COLORS[0], -2),
        (r"(c) eight-mode axis refinement", refined, d_ref, COLORS[2], -10),
    )
):
    ax = figure.add_subplot(grid[row, 1])
    scale = 10.0**exponent
    ax.plot(
        np.asarray(field_period_angle(sol)),
        np.asarray(d.anomaly / sol.inputs.B0) / scale,
        color=color,
    )
    norm = float(d.weighted_l2 / sol.inputs.B0)
    ax.set_title(label + rf", $\|\cdot\|_2={norm / scale:.2f}\times10^{{{exponent}}}$ m$^{{-2}}$")
    ax.set_ylabel(
        rf"$(B_{{20}}-\langle B_{{20}}\rangle)/B_0$" "\n" rf"[$10^{{{exponent}}}$ m$^{{-2}}$]"
    )
ax.set_xlabel(r"$N_\mathrm{fp}\varphi/2\pi$")
save(
    figure,
    "B20_optimization",
    {
        "seed": "https://stellarator.physics.wisc.edu/app/plot/57409",
        "nphi": NPHI,
        "surface_radius_m": RADIUS,
        "stock_exact_B2c": float(stock.B2c_optimal),
        "refined_B2c": float(refined.inputs.B2c),
        "stock_weighted_l2": float(d_stock.weighted_l2),
        "refined_weighted_l2": float(d_ref.weighted_l2),
        "improvement_factor": float(d_stock.weighted_l2 / d_ref.weighted_l2),
        "refined_r_singularity_m": float(refined.r_singularity),
        "refined_weighted_l2_vs_nphi": check,
    },
)
