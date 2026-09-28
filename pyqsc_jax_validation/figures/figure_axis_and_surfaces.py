"""QA and QH near-axis surfaces at r = 0.05 m (pyQSC r2 sections 5.1 and 5.5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import COLORS, DOUBLE, draw_surface, plt, qsc, save

CASES = {
    # Landreman & Sengupta, J. Plasma Phys. 85, 815850601 (2019), sections 5.1 and 5.5
    "QA": dict(rc=[1.0, 0.155, 0.0102], zs=[0.0, 0.154, 0.0111], nfp=2, etabar=0.64, B2c=-0.00322),
    "QH": dict(
        rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-5],
        zs=[0.0, 0.1581, 0.01820, 0.001548, 7.772e-5],
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
    ),
}
RADIUS = 0.05

figure = plt.figure(figsize=(DOUBLE, 2.9))
metadata = {"radius_m": RADIUS, "cases": {}}
for panel, (name, p) in enumerate(CASES.items()):
    solution = qsc.solve(
        axis=qsc.Axis(rc=p["rc"], zs=p["zs"], nfp=p["nfp"]),
        etabar=p["etabar"],
        B2c=p["B2c"],
        nphi=61,
        order="r2",
    )
    ax = figure.add_subplot(1, 2, panel + 1, projection="3d")
    draw_surface(ax, solution, RADIUS, color=COLORS[panel], view=(45, 40))
    ax.set_title(
        rf"({'ab'[panel]}) {name}, $n_\mathrm{{fp}}={p['nfp']}$, "
        rf"$\iota_0={float(solution.iota):.3f}$",
        pad=-6,
    )
    metadata["cases"][name] = {
        **p,
        "iota": float(solution.iota),
        "r_singularity": float(solution.r_singularity),
    }
figure.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0)
save(figure, "axis_and_surfaces", metadata)
