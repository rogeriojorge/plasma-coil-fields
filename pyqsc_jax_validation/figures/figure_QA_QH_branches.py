"""iota and elongation along etabar for a QA (database 139524) and a QH (database 3) axis."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    COLORS,
    CONFIGURATIONS,
    DOUBLE,
    plt,
    qsc,
    save,
)

BRANCHES = {
    "QA (ID 139524)": ("database_qa_139524", np.linspace(-1.15, -0.4, 16)),
    "QH (ID 3)": ("database_example_3", np.linspace(0.8, 1.8, 16)),
}
NPHI = 61

figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), constrained_layout=True)
metadata = {"nphi": NPHI, "order": "first", "branches": {}}
for (label, (name, etabars)), color in zip(BRANCHES.items(), COLORS):
    c = CONFIGURATIONS[name]
    axis = qsc.Axis(rc=c["rc"], zs=c["zs"], nfp=c["nfp"])
    sols = [qsc.solve(axis=axis, etabar=e, nphi=NPHI) for e in etabars]
    iota = np.array([float(s.iota) for s in sols])
    elong = np.array([float(s.elongation.max()) for s in sols])
    axes[0].plot(etabars, np.abs(iota), "-o", ms=3, color=color, label=label)
    axes[1].semilogy(etabars, elong, "-o", ms=3, color=color, label=label)
    axes[0].axvline(c["etabar"], color=color, ls=":", lw=0.9)
    axes[1].axvline(c["etabar"], color=color, ls=":", lw=0.9)
    metadata["branches"][label] = {
        "source": c["source"],
        "etabar": etabars.tolist(),
        "iota": iota.tolist(),
        "max_elongation": elong.tolist(),
    }
axes[0].set(xlabel=r"$\bar\eta\,R_0$", ylabel=r"$|\iota_0|$", title="(a) rotational transform")
axes[1].set(xlabel=r"$\bar\eta\,R_0$", ylabel="max elongation", title="(b) maximum elongation")
axes[0].legend(loc="center left")
save(figure, "QA_QH_branches", metadata)
