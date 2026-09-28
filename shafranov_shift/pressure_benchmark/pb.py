"""Fixed-coil pressure response of the magnetic axis: first-order theory against VMEX.

Two subcommands, one heavy job each.

    python pb.py theory CASE --output DIR
        Trace the closed vacuum axis of the coils, then compute the first-order axis
        displacement per unit p2 three ways: the exact periodic linear response of the traced
        closed field line to the near-axis plasma field (primary), the near-axis Frenet operator
        with the actual coil gradient, and the ideal near-axis operator.

    python pb.py solve CASE --radius A --beta B --output DIR [--restart WOUT] [--ns ...]
        One VMEX free-boundary equilibrium in the fixed coils with p = p0 (1 - s), zero current,
        PHIEDGE = pi B0 A^2 and on-axis beta B = 2 mu0 p0 / B0^2. The magnetic axis is written
        on the same phi grid as the theory.

The pressure p = p0 (1 - r^2/A^2) has p2 = -p0 / A^2. The near-axis on-axis plasma field is
proportional to p2 A^2 = -p0, so the first-order displacement is xi = p0 * X with X from
``theory``: it depends on the on-axis beta = 2 mu0 p0 / B0^2 only, not on A.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from drivers_dir import drivers_dir  # noqa: E402

sys.path.insert(0, drivers_dir())
import nearaxis_finite_beta_helpers as helpers  # noqa: E402
from essos.coils import Coils  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402
from essos.objective_functions import frenet_axis_response, pressure_axis_response  # noqa: E402
from pyqsc_jax.near_axis import near_axis  # noqa: E402
from pyqsc_jax.vmec import to_vmec  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from axis_operator_check import (  # noqa: E402
    closed_axis,
    cylindrical_rhs,
    lab_displacement,
    periodic_linear_response,
)

MU0 = 4e-7 * np.pi
NPHI = 101  # axis points per field period (theory and VMEX axis samples)

# Vacuum near-axis axes of the three coil sets. The QA coils were fitted to this vacuum axis;
# the QH and hybrid coils were fitted to the external field of a finite-pressure (and, for the
# hybrid, finite-current) near-axis equilibrium on the same axis, so their traced vacuum axes
# differ from it. The theory is therefore evaluated on the traced axis.
CASES = {
    "qa": dict(rc=[1.0902382491800673], zs=None, nfp=2, etabar=None, B2c=None, B0=1.0,
               reference=HERE.parent / "reference" / "vacuum_fitted_reference.json"),
    "qh": dict(rc=[1.0, 0.17, 0.01804, 0.001409, 5.877e-05], zs=[0.0, 0.1581, 0.0182, 0.001548, 7.772e-05],
               nfp=4, etabar=1.569, B2c=0.1348, B0=1.0, coils=HERE / "reference" / "qh_coils.json",
               design_radius=0.035),
    "hybrid": dict(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3, etabar=0.9, B2c=0.0, B0=1.0,
                   coils=HERE / "reference" / "hybrid_coils.json", design_radius=0.04),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(name, segments=480):
    case = dict(CASES[name])
    if "reference" in case:
        spec = json.loads(Path(case["reference"]).read_text())
        case.update(rc=spec["rc"], zs=spec["zs"], nfp=spec["nfp"], etabar=spec["etabar"],
                    B2c=spec["B2c"], B0=spec["B0"], coils=Path(case["reference"]).parent / spec["coil_file"],
                    design_radius=spec["design_radius_m"])
    coils = Coils.from_json(str(case["coils"]))
    coils.curves.n_segments = segments
    solution = near_axis(rc=jnp.asarray(case["rc"]), zs=jnp.asarray(case["zs"]), nfp=case["nfp"],
                         etabar=case["etabar"], nphi=NPHI, order="r3", B0=case["B0"], B2c=case["B2c"],
                         p2=0.0, I2=0.0).solution
    case["coil_sha256"] = digest(case["coils"])
    return case, BiotSavart(coils), solution


def traced_axis(field, solution, nfp):
    period = 2 * np.pi / nfp
    phis = np.asarray(solution.phi)
    rhs = cylindrical_rhs(field, lambda x: jnp.zeros(3))
    X0, closure = closed_axis(rhs, [float(solution.R0[0]), float(solution.Z0[0])], period, phis)
    f0 = jax.jit(rhs)
    dense = solve_ivp(lambda p, y: np.asarray(f0(p, jnp.asarray(y))), (0, period), X0[:, 0],
                      method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True).sol
    return rhs, X0, closure, dense, period, phis


def theory(args):
    case, field, solution = load_case(args.case)
    nfp, B0 = case["nfp"], case["B0"]
    rhs, X0, closure, dense, period, phis = traced_axis(field, solution, nfp)
    # The on-axis plasma field is proportional to p2 radius^2 = -p0: evaluate p0 = 1 Pa.
    radius = case["design_radius"]
    response = pressure_axis_response(solution, radius, -1.0 / radius**2)
    normal = np.asarray(solution.geometry.normal_cartesian)
    binormal = np.asarray(solution.geometry.binormal_cartesian)
    Bp = response["source_n"][:, None] * normal + response["source_b"][:, None] * binormal
    n = len(phis)
    wave = nfp * np.fft.fftfreq(n, d=1 / n)
    coeff = np.fft.fft(Bp, axis=0) / n

    def forcing(phi, X):
        dB = np.real(np.exp(1j * wave * phi) @ coeff)
        R = X[0]
        x = np.array([R * np.cos(phi), R * np.sin(phi), X[1]])
        B = np.asarray(field.B(jnp.asarray(x)))
        c, s = np.cos(phi), np.sin(phi)
        BR, Bphi = B[0] * c + B[1] * s, -B[0] * s + B[1] * c
        dBR, dBphi = dB[0] * c + dB[1] * s, -dB[0] * s + dB[1] * c
        return np.array([R * (dBR / Bphi - BR * dBphi / Bphi**2),
                         R * (dB[2] / Bphi - B[2] * dBphi / Bphi**2)])

    traced, monodromy = periodic_linear_response(rhs, dense, forcing, period, phis)
    ideal = np.stack((response["delta_R"], response["delta_Z"]))
    curve = np.asarray(solution.geometry.position_cartesian)
    grad = np.asarray(jax.jit(jax.vmap(field.dB_by_dX))(jnp.asarray(curve)))
    Bc = np.asarray(jax.jit(jax.vmap(field.B))(jnp.asarray(curve)))
    Bt = np.sum(Bc * np.asarray(solution.geometry.tangent_cartesian), 1)
    u, v = frenet_axis_response(solution, Bp, gradient=grad, tangent_field=Bt)
    frenet_actual = lab_displacement(solution, u, v)
    iota = float(np.arccos(np.clip(np.trace(monodromy) / 2, -1, 1)) / (2 * np.pi) * nfp)
    rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))
    report = dict(
        case=args.case, coil_sha256=case["coil_sha256"], nfp=nfp, B0=B0,
        traced_axis_closure_m=closure,
        traced_axis_offset_from_near_axis_max_m=float(np.max(np.hypot(X0[0] - np.asarray(solution.R0),
                                                                       X0[1] - np.asarray(solution.Z0)))),
        traced_axis_R0_m=float(X0[0, 0]), monodromy_iota_lab_abs=iota,
        near_axis_iota=float(solution.iota),
        displacement_per_unit_p0_m_per_Pa=dict(
            traced_delta_R0=float(traced[0, 0]), traced_rms=float(np.sqrt(np.mean(np.sum(traced**2, 0)))),
            ideal_delta_R0=float(ideal[0, 0]), frenet_actual_delta_R0=float(frenet_actual[0, 0])),
        traced_vs_ideal_rms_rel=rel(ideal, traced),
        traced_vs_frenet_actual_rms_rel=rel(frenet_actual, traced),
        note="xi = p0 * X with p0 = -p2 a^2 the axis pressure; X per Pa; traced is the primary theory",
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"theory_{args.case}.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(out / f"theory_{args.case}.npz", phi=phis, traced_axis=X0, traced=traced, ideal=ideal,
             frenet_actual=frenet_actual)
    print(json.dumps(report, indent=2))


def solve(args):
    import vmex as vj
    from vmex.core.plotting import surface_rz

    case, field, solution = load_case(args.case)
    B0 = case["B0"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ns = tuple(args.ns)
    ftol = tuple(args.ftol) if len(args.ftol) == len(ns) else tuple(args.ftol) * len(ns)
    niter = tuple(args.niter) if len(args.niter) == len(ns) else tuple(args.niter) * len(ns)
    export = to_vmec(solution, out / "input.seed", r=args.radius, mpol=args.mpol, ntor=args.ntor,
                     ntheta=64, parameters=dict(ns_array=ns, ftol_array=ftol, niter_array=niter,
                                                delt=args.delt))
    base = vj.VmecInput.from_file(export.path)
    base = dataclasses.replace(base, phiedge=float(np.sign(export.phiedge)) * np.pi * B0 * args.radius**2)
    p0 = args.beta * B0**2 / (2 * MU0)
    p2 = -p0 / args.radius**2
    # The deck family is p = alpha * (-p2_star a^2) (1 - s); beta = 0 is alpha = 0.
    inp = helpers.pressure_family_input(base, args.radius, float(args.beta > 0),
                                        p2 if args.beta > 0 else -1.0, args.nzeta)
    inp.to_indata(out / "input.runtime")
    start = perf_counter()
    with (out / "vmex.log").open("w") as log:
        def emit(*values, **kwargs):
            print(*values, **{**kwargs, "file": log, "flush": True})
        result = vj.solve_free_boundary_multigrid(
            inp, external_field=field, restart_from=args.restart, verbose=True, emit=emit,
            raise_on_max_iterations=False)
    wout = vj.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                              fsql=float(result.fsql), niter=int(result.iterations),
                              converged=bool(result.converged), vacuum_output=result.vacuum)
    vj.write_wout(out / "wout.nc", wout)
    phis = np.asarray(solution.phi)
    R, Z = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phis)
    row = dict(
        case=args.case, radius_m=args.radius, beta_axis_target=args.beta, p0_Pa=p0, p2_Pa_per_m2=p2,
        ns=list(ns), ftol=list(ftol), niter=list(niter), mpol=args.mpol, ntor=args.ntor, nzeta=args.nzeta,
        delt=args.delt, restart=None if args.restart is None else Path(args.restart).parent.name,
        converged=bool(result.converged and result.vacuum is not None), iterations=int(result.iterations),
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        seconds=perf_counter() - start, betatotal=float(wout.betatotal), beta_axis=float(wout.betaxis),
        iota_axis_raw=float(np.asarray(wout.iotaf)[0]), coil_sha256=case["coil_sha256"],
        wout_sha256=digest(out / "wout.nc"), axis_R=np.asarray(R[0]).tolist(), axis_Z=np.asarray(Z[0]).tolist())
    (out / "row.json").write_text(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if not k.startswith("axis_")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("theory")
    t.add_argument("case", choices=sorted(CASES))
    t.add_argument("--output", required=True)
    s = sub.add_parser("solve")
    s.add_argument("case", choices=sorted(CASES))
    s.add_argument("--radius", type=float, required=True, help="boundary flux radius a_b (m)")
    s.add_argument("--beta", type=float, required=True, help="on-axis beta 2 mu0 p0 / B0^2")
    s.add_argument("--output", required=True)
    s.add_argument("--restart", help="wout.nc to warm-start from")
    s.add_argument("--ns", type=int, nargs="+", default=[17, 33, 65])
    s.add_argument("--ftol", type=float, nargs="+", default=[1e-8, 1e-9, 1e-10])
    s.add_argument("--niter", type=int, nargs="+", default=[2000, 4000, 8000])
    s.add_argument("--mpol", type=int, default=8)
    s.add_argument("--ntor", type=int, default=8)
    s.add_argument("--nzeta", type=int, default=32)
    s.add_argument("--delt", type=float, default=0.5)
    args = parser.parse_args()
    theory(args) if args.command == "theory" else solve(args)


if __name__ == "__main__":
    main()
