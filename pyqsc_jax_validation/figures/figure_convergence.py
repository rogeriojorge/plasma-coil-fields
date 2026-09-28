"""Grid convergence of iota and the B20 residual for database QA 139524."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SINGLE, configuration, plt, qsc, save

RESOLUTIONS = (11, 15, 21, 31, 45, 61, 91, 121)
REFERENCE = 301

sols = [configuration("database_qa_139524", nphi=n) for n in RESOLUTIONS + (REFERENCE,)]
iota = np.array([float(s.iota) for s in sols])
b20 = np.array([float(qsc.b20_diagnostics(s).weighted_l2 / s.inputs.B0) for s in sols])
floor = 1e-16
e_iota = np.maximum(np.abs(iota[:-1] - iota[-1]) / abs(iota[-1]), floor)
e_b20 = np.maximum(np.abs(b20[:-1] - b20[-1]) / b20[-1], floor)

figure, ax = plt.subplots(figsize=(SINGLE, 2.5), constrained_layout=True)
ax.semilogy(RESOLUTIONS, e_iota, "-o", label=r"$\iota_0$")
ax.semilogy(RESOLUTIONS, e_b20, "-s", label=r"$\|B_{20}-\langle B_{20}\rangle\|/B_0$")
ax.axhline(np.finfo(float).eps, color="0.5", ls=":", lw=0.9)
ax.text(
    RESOLUTIONS[-1],
    2.2e-16,
    "machine precision",
    ha="right",
    va="bottom",
    fontsize=6.5,
    color="0.4",
)
ax.set(
    xlabel=r"grid points per field period $N_\varphi$",
    ylabel=f"relative difference to $N_\\varphi={REFERENCE}$",
)
ax.legend()
save(
    figure,
    "convergence",
    {
        "configuration": "database_qa_139524",
        "source": "https://stellarator.physics.wisc.edu/app/plot/139524",
        "order": "r3",
        "resolutions": RESOLUTIONS,
        "reference": REFERENCE,
        "iota": iota.tolist(),
        "B20_weighted_l2_over_B0": b20.tolist(),
    },
)
