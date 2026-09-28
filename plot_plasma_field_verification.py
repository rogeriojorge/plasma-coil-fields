"""Figure of the plasma-field verification: volume Biot-Savart and fixed-Cartesian derivatives against a.

python plot_plasma_field_verification.py -> docs/figures/00_plasma_field_verification.png
Reads volume_table.json and cartesian_table.json only.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parent
plt.style.use(REPO / "paper.mplstyle")
volume = json.loads((REPO / "volume_table.json").read_text())["errors"]
cartesian = json.loads((REPO / "cartesian_table.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
for name, rows in volume.items():
    a = np.array(sorted(map(float, rows)))
    axes[0].loglog(a, [rows[str(x)] for x in a], "o-", label=name)
a = np.array([0.01, 0.08])
axes[0].loglog(a, volume["pressure-only"]["0.01"] * (a / 0.01) ** 2, ":", color="0.5", label=r"$\propto a^2$")
axes[0].set_xlabel("flux radius a [m]")
axes[0].set_ylabel("relative error of on-axis field")
axes[0].set_title("(a) matched field vs volume Biot–Savart")
axes[0].legend(loc="upper left")
for i, (case, marker) in enumerate((("qa-pressure", "o"), ("qa-current", "s"), ("circular", "^"))):
    rows = cartesian[case]
    a = np.array(sorted(map(float, rows)))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
    axes[1].loglog(a, [rows[str(x)]["gradient"] for x in a], marker + "-", color=color, label=f"{case}: gradient")
    axes[1].loglog(a, [rows[str(x)]["hessian"] for x in a], marker + "--", color=color, label=f"{case}: Hessian")
axes[1].set_xlabel("source radius a [m]")
axes[1].set_ylabel("relative difference")
axes[1].set_title("(b) gradient, Hessian vs fixed-Cartesian")
axes[1].set_ylim(1e-4, 3)
axes[1].legend(fontsize=6.2, loc="upper left", ncols=2)
for ax, ticks in zip(axes, ([0.01, 0.02, 0.04, 0.08], [0.01, 0.02, 0.04])):
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
fig.savefig(REPO / "docs" / "figures" / "00_plasma_field_verification.png")
