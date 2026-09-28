"""Synchronized first-order compile, execution, JVP and VMAP timings."""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp

from common import qsc, write_report

AXIS = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
ETABAR = -0.9
RESOLUTIONS = (15, 31, 61, 121)
REPEATS = 15
BATCH = 64


def timed(function, *arguments):
    start = time.perf_counter()
    jax.block_until_ready(function(*arguments))
    return time.perf_counter() - start


def measure(nphi):
    def iota(etabar):
        return qsc.solve(axis=AXIS, etabar=etabar, nphi=nphi).iota

    row = {"nphi": nphi}
    kernels = {
        "solve": (jax.jit(iota), jnp.asarray(ETABAR)),
        "jvp": (jax.jit(lambda e: jax.jvp(iota, (e,), (jnp.ones_like(e),))), jnp.asarray(ETABAR)),
        "vmap": (jax.jit(jax.vmap(iota)), jnp.linspace(-1.0, -0.8, BATCH)),
    }
    jax.clear_caches()
    for name, (function, argument) in kernels.items():
        row[f"{name}_cold_seconds"] = timed(function, argument)
        samples = [timed(function, argument) for _ in range(REPEATS)]
        row[f"{name}_warm_median_seconds"] = statistics.median(samples)
        row[f"{name}_warm_min_seconds"] = min(samples)
    print(row)
    return row


write_report(
    "core",
    {
        "case": {
            "rc": [1.0, 0.045],
            "zs": [0.0, -0.045],
            "nfp": 3,
            "etabar": ETABAR,
            "order": "first",
            "repeats": REPEATS,
            "vmap_batch": BATCH,
        },
        "results": [measure(n) for n in RESOLUTIONS],
    },
)
