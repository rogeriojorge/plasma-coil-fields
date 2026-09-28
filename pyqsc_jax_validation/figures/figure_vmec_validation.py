"""VMEC export: on-axis iota convergence (frozen VMEC 9.0 run) and export timing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

from common import (
    COLORS,
    DOUBLE,
    REPORTS,
    latest_report,
    plt,
    save,
)

# VMEC (version 9.0) is not rerun here: the radius scan is the frozen 2026-07-30 report.
frozen = json.loads((REPORTS / "2026-07-30-b20-vmec-apple-m4.json").read_text())
scan = frozen["vmec_radius_convergence"]
r = np.array([row["radius"] for row in scan["results"]])
err = np.array([row["relative_iota_error"] for row in scan["results"]])
timing = latest_report("vmec-export")

figure, axes = plt.subplots(
    1, 2, figsize=(DOUBLE, 2.4), constrained_layout=True, gridspec_kw={"width_ratios": (1.3, 1)}
)
axes[0].loglog(r, err, "-o", label="VMEC vs near-axis")
axes[0].loglog(r, err[0] * (r / r[0]) ** 2, "--", color="0.5", label=r"$\propto r^2$")
axes[0].set(
    xlabel="boundary radius $r$ [m]",
    ylabel=r"$|\iota_\mathrm{VMEC}(0)-\iota_0|/\iota_0$",
    title="(a) on-axis rotational transform",
)
axes[0].legend()
axes[0].set_xticks(r, [f"{v:g}" for v in r])
axes[0].minorticks_off()
cold, warm = (1e3 * timing[k] for k in ("cold_compile_and_execute_seconds", "warm_median_seconds"))
bars = axes[1].bar(
    ["compile +\nfirst call", "warm"], [cold, warm], color=["0.6", COLORS[0]], width=0.55
)
axes[1].set_yscale("log")
axes[1].set_ylim(warm / 3, cold * 6)
axes[1].bar_label(bars, fmt="%.1f ms", padding=2, fontsize=6.8)
c = timing["case"]
axes[1].set(ylabel="export time [ms]", title=f"(b) export, mpol={c['mpol']}, ntor={c['ntor']}")
axes[1].text(
    0.97,
    0.95,
    f"max $|\\Delta R|/R_0$={timing['maximum_R_reconstruction_error_m']:.1e}\n"
    f"max $|\\Delta Z|/R_0$={timing['maximum_Z_reconstruction_error_m']:.1e}",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=6.5,
)
save(
    figure,
    "vmec_validation",
    {
        "frozen_vmec_report": "2026-07-30-b20-vmec-apple-m4.json",
        "vmec_version": scan["vmec_version"],
        "near_axis_iota": scan["near_axis_iota"],
        "export_report": timing["_file"],
    },
)
