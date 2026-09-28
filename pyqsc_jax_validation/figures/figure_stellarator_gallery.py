"""Gallery of four screened near-axis designs (database entries and one refinement)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    COLORS,
    CONFIGURATIONS,
    DOUBLE,
    configuration,
    draw_surface,
    plt,
    qsc,
    save,
)

CASES = (  # name, title, radius [m], view
    ("database_qa_139524", "QA, ID 139524", 0.03, (60, 55)),
    ("database_example_3", "QH, ID 3", 0.075, (60, 35)),
    ("b20_optimized_good", r"QH, flattened $B_{20}$", 0.075, (60, 28)),
    (
        "database_large_singularity_107579",
        r"QH, ID 107579 (large $r_\mathrm{sing}$)",
        0.15,
        (60, 40),
    ),
)

figure = plt.figure(figsize=(DOUBLE, 5.4))
metadata = {"nphi": 121, "cases": {}}
for panel, (name, title, radius, view) in enumerate(CASES):
    sol = configuration(name, nphi=121)
    ax = figure.add_subplot(2, 2, panel + 1, projection="3d")
    draw_surface(ax, sol, radius, color=COLORS[panel], view=view)
    rs = float(sol.r_singularity)
    ax.set_title(
        f"({'abcd'[panel]}) {title}\n"
        rf"$|\iota_0|={abs(float(sol.iota)):.3f}$, $r={radius}$ m, "
        rf"$r_\mathrm{{sing}}={rs:.3f}$ m",
        pad=-4,
    )
    metadata["cases"][name] = {
        "source": CONFIGURATIONS[name]["source"],
        "radius_m": radius,
        "iota": float(sol.iota),
        "r_singularity_m": rs,
        "B20_weighted_l2": float(qsc.b20_diagnostics(sol).weighted_l2),
    }
figure.subplots_adjust(left=0, right=1, bottom=0, top=0.93, wspace=0, hspace=0.12)
save(figure, "stellarator_gallery", metadata)
