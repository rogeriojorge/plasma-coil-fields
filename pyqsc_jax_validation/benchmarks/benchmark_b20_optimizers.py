"""Compare SciPy optimizers on the B20 flattening problem seeded by database entry 57409.

Axis Fourier modes 1-3 (R and Z) vary within a bounded box; B2c is eliminated
exactly at every evaluation. The residual is the arclength-weighted, projected
B20/B0 anomaly. The former pyQSC_JAX multistart Levenberg-Marquardt row
(`search_axis`) is retired with that research API.
"""

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp

from common import CONFIGURATIONS, configuration, qsc, write_report

NPHI = 121
SEED = CONFIGURATIONS["database_low_b20_57409"]
AXIS = qsc.Axis(rc=SEED["rc"], zs=SEED["zs"], nfp=SEED["nfp"])
NMODES = len(SEED["rc"])
INITIAL = jnp.asarray(SEED["rc"][1:] + SEED["zs"][1:])
HALF_WIDTH = jnp.asarray((0.14, 0.07, 0.03, 0.14, 0.07, 0.03))
BOUNDS = (np.asarray(INITIAL - HALF_WIDTH), np.asarray(INITIAL + HALF_WIDTH))


def residual(variables):
    rc = jnp.concatenate((jnp.ones(1), variables[: NMODES - 1]))
    zs = jnp.concatenate((jnp.zeros(1), variables[NMODES - 1 :]))
    solution = qsc.solve(
        axis=qsc.Axis(rc=rc, zs=zs, nfp=SEED["nfp"]),
        etabar=SEED["etabar"],
        p2=SEED["p2"],
        nphi=NPHI,
        order="r2",
    )
    result = qsc.optimize_B2c(solution)
    w = result.solution.geometry.d_l_d_phi
    return jnp.sqrt(w / jnp.sum(w)) * result.diagnostics.anomaly / result.solution.inputs.B0


f = jax.jit(residual)
jac = jax.jit(jax.jacrev(residual))
vg = jax.jit(jax.value_and_grad(lambda v: 0.5 * jnp.sum(residual(v) ** 2)))
start = perf_counter()
jax.block_until_ready((f(INITIAL), jac(INITIAL), vg(INITIAL)))
compile_seconds = perf_counter() - start


def l2(v):
    return float(np.linalg.norm(np.asarray(f(jnp.asarray(v)))))


rows = [{"method": "exact B2c only", "seconds": 0.0, "evaluations": 1, "weighted_l2": l2(INITIAL)}]


def record(method, run):
    start = perf_counter()
    result = run()
    rows.append(
        {
            "method": method,
            "seconds": perf_counter() - start,
            "evaluations": int(result.nfev),
            "weighted_l2": l2(result.x),
        }
    )
    print(rows[-1])


record(
    "L-BFGS-B",
    lambda: minimize(
        lambda v: tuple(np.asarray(a, dtype=float) for a in vg(jnp.asarray(v))),
        np.asarray(INITIAL),
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(*BOUNDS)),
        options={"maxiter": 60, "ftol": 1e-18, "gtol": 1e-12},
    ),
)
record(
    "least squares (TRF)",
    lambda: least_squares(
        lambda v: np.asarray(f(jnp.asarray(v))),
        np.asarray(INITIAL),
        jac=lambda v: np.asarray(jac(jnp.asarray(v))),
        bounds=BOUNDS,
        max_nfev=60,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        x_scale="jac",
    ),
)
record(
    "differential evolution (low budget)",
    lambda: differential_evolution(
        lambda v: l2(v) ** 2,
        list(zip(*BOUNDS)),
        seed=7,
        popsize=4,
        maxiter=4,
        polish=False,
        updating="immediate",
    ),
)

refined = configuration("b20_optimized_good", nphi=NPHI, order="r2")
rows.append(
    {
        "method": "staged 8-mode refinement (stored result)",
        "seconds": None,
        "evaluations": None,
        "weighted_l2": float(qsc.b20_diagnostics(refined).weighted_l2 / refined.inputs.B0),
    }
)
write_report(
    "b20-optimizers",
    {
        "case": {
            "seed_database_id": 57409,
            "nphi": NPHI,
            "modes": [1, 2, 3],
            "half_width": np.asarray(HALF_WIDTH).tolist(),
        },
        "compile_seconds": compile_seconds,
        "timing_note": "optimizer timings exclude the shared JIT compilation (compile_seconds)",
        "results": rows,
    },
)
