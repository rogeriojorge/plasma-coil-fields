"""B20 flattening: optimizer comparison (benchmarks/benchmark_b20_optimizers.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import COLORS, DOUBLE, latest_report, plt, save

report = latest_report("b20-optimizers")
rows = report["results"]
labels = [
    "exact $B_{2c}$\nonly",
    "L-BFGS-B",
    "least\nsquares",
    "diff. evol.\n(120 evals)",
    "staged\n8-mode",
]
values = np.array([r["weighted_l2"] for r in rows])

figure, axes = plt.subplots(
    1, 2, figsize=(DOUBLE, 2.5), constrained_layout=True, gridspec_kw={"width_ratios": (1.35, 1)}
)
bars = axes[0].bar(
    labels, values, color=["0.6", COLORS[0], COLORS[1], COLORS[4], COLORS[2]], width=0.65
)
axes[0].set_yscale("log")
axes[0].set_ylim(values.min() / 5, values.max() * 8)
axes[0].bar_label(bars, labels=[f"{v:.1e}" for v in values], padding=2, fontsize=6.5)
axes[0].set(
    ylabel=r"$\|B_{20}-\langle B_{20}\rangle\|_2/B_0$ [m$^{-2}$]",
    title="(a) residual after optimization",
)
axes[0].tick_params(axis="x", labelsize=6.8)
for r, color, off in zip(
    rows[1:4], (COLORS[0], COLORS[1], COLORS[4]), ((7, 6), (7, -8), (7, 0))
):
    axes[1].loglog(r["evaluations"], r["weighted_l2"], "o", ms=7, color=color)
    axes[1].annotate(
        f"{r['method'].split(' (')[0]}\n{r['seconds']:.2f} s",
        (r["evaluations"], r["weighted_l2"]),
        xytext=off,
        textcoords="offset points",
        fontsize=6.5,
        va="center",
    )
axes[1].axhline(values[0], color="0.6", ls=":", lw=0.9)
axes[1].set_xlim(15, 1000)
axes[1].set_ylim(3e-5, 3)
axes[1].set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
axes[1].minorticks_off()
axes[1].set(
    xlabel="objective evaluations",
    ylabel=r"residual / $B_0$ [m$^{-2}$]",
    title="(b) cost (modes 1-3, bounded)",
)
save(
    figure,
    "optimizer_comparison",
    {"report": report["_file"], "improvement_exact_to_staged": float(values[0] / values[-1])},
)
