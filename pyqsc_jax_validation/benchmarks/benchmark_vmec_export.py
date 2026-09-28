"""Cold and warm VMEC boundary-export timings and reconstruction errors."""

import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax

from common import qsc, write_report

CASE = {"radius": 0.03, "nphi": 61, "ntheta": 40, "mpol": 12, "ntor": 14}
REPEATS = 9

solution = qsc.solve(
    axis=qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3),
    etabar=-0.9,
    nphi=CASE["nphi"],
    order="r2",
)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "input.benchmark"

    def export():
        return qsc.to_vmec(
            solution,
            path,
            r=CASE["radius"],
            ntheta=CASE["ntheta"],
            mpol=CASE["mpol"],
            ntor=CASE["ntor"],
        )

    jax.clear_caches()
    cold = export()
    warm = [export().conversion_seconds for _ in range(REPEATS)]
boundary = cold.boundary
write_report(
    "vmec-export",
    {
        "case": CASE,
        "cold_compile_and_execute_seconds": cold.conversion_seconds,
        "warm_median_seconds": statistics.median(warm),
        "warm_min_seconds": min(warm),
        "maximum_toroidal_angle_residual": float(boundary.maximum_toroidal_angle_residual),
        "maximum_R_reconstruction_error_m": float(boundary.maximum_R_reconstruction_error),
        "maximum_Z_reconstruction_error_m": float(boundary.maximum_Z_reconstruction_error),
    },
)
