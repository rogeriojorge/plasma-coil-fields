"""Shared style, reference configurations and output helpers for the pyQSC_JAX figures.

The configuration values below were bundled in pyQSC_JAX's former
``configurations`` module (branch ``preserve/pr2-before-refactor-2026-09-27``)
and are inlined here with their original sources. Database entries are from the
QUASR/Curvo stellarator database, https://stellarator.physics.wisc.edu/app/plot/<id>.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyqsc_jax as qsc

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
FIGURES = HERE / "figures"
REPORTS = HERE / "benchmarks" / "reports"
SINGLE = 3.4  # journal single-column width [in]
DOUBLE = 6.6  # journal double-column width [in]
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")

plt.style.use(STUDY / "paper.mplstyle")
plt.rcParams["axes.formatter.use_mathtext"] = True

DB = "https://stellarator.physics.wisc.edu/app/plot/"
CONFIGURATIONS = {
    # Axis and etabar of Landreman & Sengupta, J. Plasma Phys. 85, 815850601 (2019),
    # section 5.1; B0 and B2c as bundled in pyQSC_JAX for pyQSC parity.
    "qa": dict(
        rc=(1.0, 0.155, 0.0102),
        zs=(0.0, 0.154, 0.0111),
        nfp=2,
        etabar=0.64,
        B0=1.006541121335688,
        B2c=0.9427285320639192,
        order="r2",
        source=None,
    ),
    "database_example_3": dict(
        rc=(1.0, -0.53677857, -0.046455786, -0.0070183445),
        zs=(0.0, -0.5888703, -0.04447083, -0.009581006),
        nfp=4,
        etabar=1.4014399,
        B2c=-0.7512066,
        p2=-74635.375,
        order="r3",
        source=DB + "3",
    ),
    "database_qa_139524": dict(
        rc=(1.0, -0.06883207, 0.0017516185, 0.023231717),
        zs=(0.0, -0.28447896, 0.074662544, 0.07483574),
        nfp=1,
        etabar=-0.7771866,
        B2c=-1.8120022,
        p2=-232743.75,
        order="r3",
        source=DB + "139524",
    ),
    "database_low_b20_57409": dict(
        rc=(1.0, -0.51677144, -0.009499784, -0.005914526),
        zs=(0.0, -0.5420635, -0.012225689, -0.0059485724),
        nfp=4,
        etabar=-1.3295174,
        B2c=-0.7577404,
        p2=-23501.281,
        order="r3",
        source=DB + "57409",
    ),
    # Eight-mode refinement of database entry 57409 (exact B2c elimination plus
    # staged bounded least squares on the axis Fourier modes; pyQSC_JAX, 2026-07).
    "b20_optimized_good": dict(
        rc=(
            1.0,
            -0.5039436500066075,
            -0.043965867334349464,
            -0.00654919263283669,
            -3.047898633100702e-06,
            2.7519909478191202e-05,
            6.071324626161564e-06,
            8.46692978641554e-07,
            5.99743474381913e-08,
        ),
        zs=(
            0.0,
            -0.5050020451312105,
            -0.045010140721391825,
            -0.006585874304053245,
            -1.5550078876295003e-05,
            2.6653739413147386e-05,
            5.978859527401134e-06,
            8.327402553082346e-07,
            5.9109366390399247e-08,
        ),
        nfp=4,
        etabar=-1.3295174,
        B2c=-1.132420959333329,
        p2=-23501.281,
        order="r3",
        source=DB + "57409",
    ),
    "database_large_singularity_107579": dict(
        rc=(1.0, -0.54465365, 0.005908036, 0.0054288576),
        zs=(0.0, 0.540987, -0.009328544, -0.003327578),
        nfp=3,
        etabar=-0.95917624,
        B2c=0.10977107,
        p2=-19211.062,
        order="r3",
        source=DB + "107579",
    ),
    "plasma_stellarator": dict(
        rc=(1.0, -0.5415884, 0.029195854, 0.0048646266),
        zs=(0.0, -0.57113713, 0.029922731, 0.0041398546),
        nfp=4,
        etabar=1.1396117,
        B2c=-0.050057083,
        p2=-28248.188,
        order="r3",
        source=DB + "52521",
    ),
}


def configuration(name: str, *, nphi: int = 61, **overrides):
    """Solve one inlined reference configuration with the current pyQSC_JAX API."""

    parameters = {k: v for k, v in CONFIGURATIONS[name].items() if k != "source"}
    parameters.update(overrides)
    axis = qsc.Axis(rc=parameters.pop("rc"), zs=parameters.pop("zs"), nfp=parameters.pop("nfp"))
    return qsc.solve(axis=axis, nphi=nphi, **parameters)


def environment() -> dict:
    """Package versions and source commits (no machine paths)."""

    def commit(module):
        try:
            return subprocess.check_output(
                ("git", "-C", str(Path(module.__file__).parent), "rev-parse", "HEAD"),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "pyqsc_jax": qsc.__version__,
        "pyqsc_jax_commit": commit(qsc),
    }
    try:
        import vmex

        info["vmex"] = getattr(vmex, "__version__", None)
        info["vmex_commit"] = commit(vmex)
    except ImportError:
        pass
    return info


def save(figure, name: str, metadata: dict | None = None) -> None:
    """Write ``figures/<name>.png``/``.pdf`` and a small provenance JSON."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    payload = {"environment": environment(), **(metadata or {})}
    (FIGURES / f"{name}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    plt.close(figure)
    print("saved", name)


def field_period_angle(solution):
    return solution.varphi * solution.inputs.axis.nfp / (2 * jax.numpy.pi)


def draw_surface(ax, solution, radius, *, color=COLORS[0], ntheta=40, view=(28, 35), zoom=1.25):
    """Shaded 3-D near-axis surface plus the magnetic axis, with equal aspect."""

    import numpy as np
    from pyqsc_jax.plotting import surface_coordinates

    x, y, z = (np.asarray(c) for c in surface_coordinates(solution, radius=radius, ntheta=ntheta))
    x, y, z = (np.concatenate((c, c[:1]), axis=0) for c in (x, y, z))
    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        shade=True,
        linewidth=0,
        antialiased=True,
        rcount=x.shape[0],
        ccount=x.shape[1],
        alpha=0.95,
        rasterized=True,
    )
    phi = np.linspace(0, 2 * np.pi, 400)
    axis = solution.inputs.axis
    n = np.arange(len(axis.rc))
    R = np.cos(np.outer(phi, n * axis.nfp)) @ np.asarray(axis.rc)
    Z = np.sin(np.outer(phi, n * axis.nfp)) @ np.asarray(axis.zs)
    ax.plot(R * np.cos(phi), R * np.sin(phi), Z, color="0.1", lw=0.9)
    for setter, c in ((ax.set_xlim, x), (ax.set_ylim, y), (ax.set_zlim, z)):
        setter(c.min(), c.max())
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)), zoom=zoom)
    ax.view_init(*view)
    ax.set_axis_off()


def write_report(name: str, payload: dict) -> Path:
    """Write ``benchmarks/reports/<date>-<name>.json`` with environment and hardware."""

    import datetime
    import os

    hardware = None
    try:
        hardware = subprocess.check_output(
            ("sysctl", "-n", "machdep.cpu.brand_string"), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        hardware = platform.processor() or None
    date = datetime.date.today().isoformat()
    path = REPORTS / f"{date}-{name}.json"
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "date": date,
        "hardware": hardware,
        "cpu_count": os.cpu_count(),
        "environment": environment(),
        **payload,
    }
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    print("saved", path.name)
    return path


def latest_report(name: str) -> dict:
    """Load the newest ``benchmarks/reports/*-<name>.json``."""

    paths = sorted(REPORTS.glob(f"????-??-??-{name}.json"))
    if not paths:
        raise FileNotFoundError(f"run benchmarks/benchmark_{name}.py first")
    return json.loads(paths[-1].read_text()) | {"_file": paths[-1].name}
