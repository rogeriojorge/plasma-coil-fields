"""VMEX forward-solve and implicit magnetic-well gradient timings (needs vmex)."""

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax

from common import configuration, qsc, write_report

CASE = dict(
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

problem = qsc.to_vmex_problem(configuration("plasma_stellarator", nphi=31), **CASE)
jax.clear_caches()
start = perf_counter()
equilibrium = problem.solve()
jax.block_until_ready(equilibrium.quantities)
forward = perf_counter() - start
start = perf_counter()
well, gradient = jax.value_and_grad(lambda p: qsc.vmex_radial_quantities(problem, p).magnetic_well)(
    problem.parameters
)
jax.block_until_ready((well, gradient))
gradient_seconds = perf_counter() - start
write_report(
    "vmex-interface",
    {
        "case": {"configuration": "plasma_stellarator (database 52521)", "nphi": 31, **CASE},
        "forward_seconds_including_compile": forward,
        "value_and_gradient_seconds_including_compile": gradient_seconds,
        "magnetic_well": float(well),
        "gradient_pres_scale": float(gradient.pres_scale),
        "gradient_rbc_norm": float(np.linalg.norm(np.asarray(gradient.rbc))),
    },
)
