"""Plasma/external split of the on-axis field for a finite-p2, I2=0 stellarator (ID 52521)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyqsc_jax.plotting import field_split_frenet_components

from common import COLORS, DOUBLE, configuration, field_period_angle, plt, qsc, save

FORMAL_RADIUS = 0.15
sol = configuration("plasma_stellarator", nphi=121)
result = qsc.plasma_hessian_on_axis(sol, formal_radius=FORMAL_RADIUS, angular_resolution=128)
B0 = float(sol.inputs.B0)
comp = np.asarray(field_split_frenet_components(result, sol)) / B0
plasma = np.asarray(result.field.field.field)
fraction = np.linalg.norm(plasma, axis=1) / np.linalg.norm(np.asarray(sol.B_axis), axis=1)
x = np.asarray(field_period_angle(sol))

figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.25), constrained_layout=True)
axes[0].plot(x, 100 * fraction, color=COLORS[1])
axes[0].set(ylabel=r"$|\mathbf{B}_\mathrm{pl}|/|\mathbf{B}|$ [%]", title="(a) plasma fraction")
for i, (label, ls) in enumerate((("total", "-"), ("plasma", "-"), ("external", "--"))):
    axes[1].plot(x, 1e4 * (comp[i, 0] - (0 if i == 1 else 1)), ls, color=COLORS[i], label=label)
axes[1].set(ylabel=r"$\Delta B_t/B_0$ [$10^{-4}$]", title="(b) tangential")
axes[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.72), fontsize=6.3)
for j, (name, color) in enumerate(((r"n", COLORS[3]), (r"b", COLORS[4]))):
    axes[2].plot(x, 1e3 * comp[1, j + 1], color=color, label=rf"plasma $B_{name}$")
    axes[2].plot(x, 1e3 * comp[2, j + 1], "--", color=color, label=rf"ext. $B_{name}$")
axes[2].set(ylabel=r"$B/B_0$ [$10^{-3}$]", title="(c) transverse (total $=0$)", ylim=(-2.6, 3.4))
axes[2].legend(ncol=2, fontsize=6, loc="upper center", columnspacing=0.8)
for ax in axes:
    ax.set_xlabel(r"$N_\mathrm{fp}\varphi/2\pi$")
save(
    figure,
    "plasma_external_jet",
    {
        "source": "https://stellarator.physics.wisc.edu/app/plot/52521",
        "formal_radius_m": FORMAL_RADIUS,
        "I2": float(sol.inputs.I2),
        "p2": float(sol.inputs.p2),
        "plasma_fraction_min_mean_max": [
            float(fraction.min()),
            float(fraction.mean()),
            float(fraction.max()),
        ],
        "max_total_transverse_over_B0": float(np.abs(comp[0, 1:]).max()),
        "r_singularity_m": float(sol.r_singularity),
        "field_remainder": float(result.field.field.estimated_field_remainder),
        "hessian_remainder": float(result.estimated_hessian_remainder),
    },
)
