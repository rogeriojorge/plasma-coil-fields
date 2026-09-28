"""Cold vs warm first-order solve, JVP and VMAP timings (from benchmarks/benchmark_core.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import COLORS, DOUBLE, latest_report, plt, save

report = latest_report("core")
rows = report["results"]
nphi = np.array([r["nphi"] for r in rows])
batch = report["case"]["vmap_batch"]

figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), constrained_layout=True)
for key, label, color, marker in (
    ("solve", "solve", COLORS[0], "o"),
    ("jvp", "solve + JVP", COLORS[1], "s"),
    ("vmap", f"vmap, {batch} cases", COLORS[2], "^"),
):
    axes[0].loglog(
        nphi,
        [1e3 * r[f"{key}_warm_median_seconds"] for r in rows],
        "-" + marker,
        color=color,
        label=label,
    )
    axes[0].loglog(
        nphi, [1e3 * r[f"{key}_cold_seconds"] for r in rows], ":" + marker, color=color, mfc="none"
    )
axes[0].set(
    xlabel=r"$N_\varphi$",
    ylabel="wall time [ms]",
    title="(a) warm (solid) vs compile + first call (dotted)",
)
axes[0].set_xticks(nphi, [str(n) for n in nphi])
axes[0].minorticks_off()
axes[0].legend(loc="center right")
per_case = [1e6 * r["vmap_warm_median_seconds"] / batch for r in rows]
single = [1e6 * r["solve_warm_median_seconds"] for r in rows]
axes[1].loglog(nphi, single, "-o", color=COLORS[0], label="single solve")
axes[1].loglog(nphi, per_case, "-^", color=COLORS[2], label="per case in vmap batch")
axes[1].set(xlabel=r"$N_\varphi$", ylabel=r"time per solve [$\mu$s]", title="(b) batching")
axes[1].set_xticks(nphi, [str(n) for n in nphi])
axes[1].minorticks_off()
axes[1].legend()
env = report["environment"]
figure.suptitle(
    f"{report['hardware']}, JAX {env['jax']} ({env['jax_backend']}, float64), "
    f"medians of {report['case']['repeats']}",
    fontsize=7,
    color="0.35",
)
save(figure, "core_performance", {"report": report["_file"]})
