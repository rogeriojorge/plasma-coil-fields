"""VMEX radial profiles from near-axis boundaries and an implicit magnetic-well gradient."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax

from common import COLORS, DOUBLE, configuration, plt, qsc, save

CASES = (("qa", "vacuum QA"), ("plasma_stellarator", r"finite $p_2$, $I_2=0$ (ID 52521)"))
SETTINGS = dict(
    r=0.02,
    qs_surfaces=(0.2, 0.4, 0.6, 0.8, 1.0),
    ntheta=8,
    mpol=3,
    ntor=2,
    ns_array=(7,),
    ftol=1.0e-7,
    max_iterations=1200,
    adjoint_tol=1.0e-8,
    multigrid=False,
)

records = []
for name, label in CASES:
    sol = configuration(name, nphi=31)
    problem = qsc.to_vmex_problem(sol, **SETTINGS)
    records.append((label, sol, problem, problem.solve().quantities))
problem = records[-1][2]
well, grad = jax.value_and_grad(lambda p: qsc.vmex_radial_quantities(problem, p).magnetic_well)(
    problem.parameters
)

figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), constrained_layout=True)
for (label, sol, _, q), color in zip(records, COLORS):
    axes[0].plot(np.asarray(q.s), np.asarray(q.iota), "-o", ms=3, color=color, label=label)
    axes[0].plot([0], [float(sol.iota)], "x", ms=6, color=color)
    axes[1].semilogy(
        np.asarray(q.qs_surfaces), np.asarray(q.quasisymmetry), "-o", ms=3, color=color
    )
axes[0].set(xlabel="$s$", ylabel=r"$\iota(s)$", title=r"(a) $\iota$ (x: near-axis $\iota_0$)")
axes[0].legend(fontsize=6.3)
axes[1].set(xlabel="$s$", ylabel="QS residual", title="(b) quasisymmetry error")
wells = [100 * float(q.magnetic_well) for *_, q in records]
axes[2].bar(["vacuum\nQA", "finite\n$p_2$"], wells, color=COLORS[:2], width=0.55)
axes[2].axhline(0, color="0.2", lw=0.8)
axes[2].set(ylabel=r"magnetic well $W$ [%]", title="(c) magnetic well")
axes[2].text(
    0.5,
    0.5,
    "implicit gradient\n"
    rf"$\partial W/\partial p_\mathrm{{scale}}={float(grad.pres_scale):.1e}$",
    transform=axes[2].transAxes,
    ha="center",
    fontsize=6.5,
)
save(
    figure,
    "vmex_radial_profiles",
    {
        "settings": SETTINGS,
        "cases": {
            label: {
                "iota": np.asarray(q.iota).tolist(),
                "s": np.asarray(q.s).tolist(),
                "quasisymmetry": np.asarray(q.quasisymmetry).tolist(),
                "magnetic_well": float(q.magnetic_well),
                "near_axis_iota": float(sol.iota),
            }
            for label, sol, _, q in records
        },
        "finite_beta_well_gradient_pres_scale": float(grad.pres_scale),
        "finite_beta_well_gradient_rbc_norm": float(np.linalg.norm(np.asarray(grad.rbc))),
    },
)
