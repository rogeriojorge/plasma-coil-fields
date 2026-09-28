"""Coils for a fixed quasi-helically symmetric near-axis equilibrium at finite pressure.

The equilibrium is the four-field-period QH axis of Landreman & Sengupta (2019, section 5.4)
with zero on-axis current and a prescribed pressure. Only the coils are optimized. Their
field, gradient and Hessian on the magnetic axis are fitted to the EXTERNAL target, the total
near-axis field minus the analytic plasma field of pyQSC_JAX. With I2 = 0 the plasma field is
local and its gradient is the order-a^2 zero-current gradient.

The coils are then benchmarked with VMEX free-boundary equilibria, one per entry of
VMEX_RADIUS_FRACTIONS. Fractions above one ask whether the coils hold closed surfaces beyond
the design radius. The pressure profile keeps its local coefficient p2 there, so a larger
boundary also has a larger central pressure.
"""
import hashlib
import json
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from essos.coils import Coils
from essos.fields import BiotSavart
from pyqsc_jax.near_axis import near_axis
import nearaxis_finite_beta_helpers as helpers


""" Input parameters """
RC = [1.0, 0.17, 0.01804, 0.001409, 5.877e-5]   # Axis R cosine coefficients [m]
ZS = [0.0, 0.1581, 0.01820, 0.001548, 7.772e-5]  # Axis Z sine coefficients [m]
NFP = 4; ETABAR = 1.569; B2C = 0.1348             # Landreman & Sengupta (2019) section 5.4
B0 = 1.0                              # Field on the axis [T]
I2 = 0.0                              # On-axis current: zero, as for a QH stellarator
P2 = -1.76e6                          # Pressure p = p0 (1 - s), p0 = -p2 a^2 [Pa/m^2]
# Boundary flux radius a [m], PHIEDGE = pi B0 a^2. This QH is strongly shaped: the fitted coils match the
# quadratic target, so B.n/|B| on the boundary grows as a^3, and a = 0.035 keeps it near 1 %.
PLASMA_RADIUS = 0.035
NPHI = 61                             # Axis points per field period in the fit
NPHI_DIAGNOSTIC = 151                 # ... and in the diagnostics

# Coils: N_COILS per half period, rotated and reflected by stellarator symmetry
N_COILS = 4; FOURIER_ORDER = 10; N_SEGMENTS = 120; N_SEGMENTS_BENCHMARK = 480
COIL_MINOR_RADIUS = 0.3               # Initial circular coils about the axis [m]
HESSIAN_WEIGHT = 0.01
# Coil limits, evaluated on 16 points per Fourier order (helpers.coil_limit_residuals). The first pass
# applied the length and curvature hinges at the N_SEGMENTS quadrature points only and had no distance
# terms: its coils passed within 4-7 mm of each other and exceeded the curvature limit between samples.
# With 0.08 m, 0.12 m and 30 m^-2 the limits bound the fit (B.n/|B| stalled at 4 % for the QH and 12 % for
# the hybrid after 700-850 evaluations; runs qh3 and hybrid3 in examples_results.json).
COIL_COIL_DISTANCE = 0.05             # Minimum distance between any two coils [m]
COIL_PLASMA_DISTANCE = 0.10           # Minimum distance from any coil to the r = a surface [m]
MSC_TARGET = 50.0                     # Arclength-mean squared curvature [1/m^2]
ARCLENGTH_VARIATION_TARGET = 0.2      # var(|gamma'|) / mean(|gamma'|)^2
# Limit weight per phase: (last evaluation, weight). Weight 100 from the start stalls the fit, because
# least_squares scales every variable by its Jacobian column (x_scale="jac") and the limit rows then
# dominate (runs qh4 and hybrid4: B.n/|B| 1 % and 6 %, rising or flat). So the fit runs with soft limits
# and a last phase enforces them, as the coil_limits stage of the single-stage driver does.
LIMIT_WEIGHTS = ((1600, 1.0), (2000, 100.0))   # The last entry ends at MAX_FUNCTION_EVALUATIONS
# Points per Fourier order for the distance terms. At 8 (80 points, 4.4 cm apart on a 3.5 m coil) the
# point-pair minimum overestimates the curve distance by ~1 cm: the qh5 and hybrid5 coils met the 0.05 m
# limit there and sit at 0.040-0.042 m at 480 points. 24 samples them 1.5 cm apart.
DISTANCE_POINTS_PER_ORDER = 8
WARM_START = None                     # optimized_dofs.npz of an earlier run to start from (its hash is in CONFIG)
LENGTH_TARGET = 5.0; CURVATURE_TARGET = 12.0
MAX_FUNCTION_EVALUATIONS = 2000
OPTIMIZE = True                       # False: reuse OUTPUT_DIR/optimized_dofs.npz
EVALUATIONS_PER_RUN = None            # e.g. 200: optimize in shorter runs, each resuming from the checkpoint

# VMEX free-boundary benchmark
RUN_VMEX = True
VMEX_RADIUS_FRACTIONS = (1.0,)        # Boundary radius / PLASMA_RADIUS; e.g. (1.0, 1.5, 2.0) for a scan
FLUX_LEVELS = (0.0625, 0.25, 0.5625, 1.0)
VMEX = dict(mpol=8, ntor=8, nzeta=32, ntheta_boundary=64, delt=0.5, ns=(17, 33, 65),
            ftol=(1e-8, 1e-9, 1e-10), niter=(2000, 4000, 8000), flux_levels=FLUX_LEVELS)
SHOW_PLOTS = True
OUTPUT_DIR = Path(__file__).resolve().parent / "output_coils_qh_finite_beta"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not SHOW_PLOTS:
    plt.switch_backend("Agg")


""" Near-axis equilibrium """
def make_near_axis(nphi):
    return near_axis(rc=jnp.asarray(RC), zs=jnp.asarray(ZS), etabar=ETABAR, nfp=NFP, nphi=nphi, order="r3",
                     B0=B0, I2=I2, p2=P2, B2c=B2C)

near = make_near_axis(NPHI)
solution = near.solution
p0 = -P2 * PLASMA_RADIUS**2
print(f"QH equilibrium: nfp = {NFP}, iota = {float(near.iota):.4f}, r_singularity = {float(solution.r_singularity):.3f} m, "
      f"a = {PLASMA_RADIUS} m, central pressure {p0:.0f} Pa, <beta> = mu0 p0 / B0^2 = {4e-7 * np.pi * p0 / B0**2:.2e}")
targets = helpers.coil_targets(solution, PLASMA_RADIUS)


""" Initial coils """
R0 = float(near.R0[0])
current = 2 * np.pi * R0 * B0 / (4e-7 * np.pi * 2 * NFP * N_COILS)
curves = helpers.axis_centered_curves(solution, N_COILS, FOURIER_ORDER, COIL_MINOR_RADIUS, N_SEGMENTS)
field_initial = BiotSavart(Coils(curves=curves, currents=jnp.full(N_COILS, current)))


""" Coil optimization against the fixed external-field target """
boundary = helpers.boundary_points(solution, PLASMA_RADIUS, ntheta=32)
limits = dict(length=LENGTH_TARGET, curvature=CURVATURE_TARGET, msc=MSC_TARGET, arclength_variation=ARCLENGTH_VARIATION_TARGET,
              coil_coil=COIL_COIL_DISTANCE, coil_plasma=COIL_PLASMA_DISTANCE, surface=boundary, n_base=N_COILS,
              distance_points=DISTANCE_POINTS_PER_ORDER * FOURIER_ORDER)
checkpoint = OUTPUT_DIR / "optimized_dofs.npz"
initial_dofs, unravel = ravel_pytree(field_initial)
# Everything that defines the fit; a checkpoint made for anything else is refused.
CONFIG = json.dumps(dict(rc=RC, zs=ZS, nfp=NFP, etabar=ETABAR, B2c=B2C, B0=B0, I2=I2, p2=P2, a=PLASMA_RADIUS,
                         nphi=NPHI, n_coils=N_COILS, order=FOURIER_ORDER, n_segments=N_SEGMENTS,
                         coil_minor_radius=COIL_MINOR_RADIUS, hessian_weight=HESSIAN_WEIGHT, length=LENGTH_TARGET,
                         curvature=CURVATURE_TARGET, max_nfev=MAX_FUNCTION_EVALUATIONS, coil_coil=COIL_COIL_DISTANCE,
                         coil_plasma=COIL_PLASMA_DISTANCE, msc=MSC_TARGET, arclength_variation=ARCLENGTH_VARIATION_TARGET,
                         limit_weights=LIMIT_WEIGHTS, points_per_order=helpers.PENALTY_POINTS_PER_ORDER,
                         # Only when changed, so that runs made before these two options still match:
                         **({} if DISTANCE_POINTS_PER_ORDER == 8 else dict(distance_points_per_order=DISTANCE_POINTS_PER_ORDER)),
                         **({} if WARM_START is None else dict(warm_start=hashlib.sha256(Path(WARM_START).read_bytes()).hexdigest()))),
                    sort_keys=True)
start, cost_history, segments = initial_dofs, [], []
if WARM_START:
    with np.load(WARM_START) as saved:
        start = jnp.asarray(saved["optimized"])
if checkpoint.exists():
    with np.load(checkpoint) as saved:
        if str(saved["config"]) != CONFIG:
            raise ValueError(f"{checkpoint} was made for a different fit. Delete it or restore the inputs.")
        start, cost_history = jnp.asarray(saved["optimized"]), list(saved["cost_history"])
        segments = json.loads(str(saved["segments"]))
    converged = {g.get("limit_weight") for g in segments if g["status"] != 0}  # Phases that converged early
    finished = len(cost_history) >= MAX_FUNCTION_EVALUATIONS or LIMIT_WEIGHTS[-1][1] in converged
    if OPTIMIZE and finished or not OPTIMIZE:
        OPTIMIZE = False
        print(f"Reusing {checkpoint} ({len(cost_history)} evaluations)")
    else:
        print(f"Continuing from evaluation {len(cost_history)} of {MAX_FUNCTION_EVALUATIONS}")
if OPTIMIZE:
    converged = {g.get("limit_weight") for g in segments if g["status"] != 0}
    phase_end, weight = next(p for p in LIMIT_WEIGHTS if p[0] > len(cost_history) and p[1] not in converged)
    limits["weight"] = weight
    budget = phase_end - len(cost_history)
    field_optimized, record, history = helpers.fit_coils(
        unravel(start), solution, targets, hessian_weight=HESSIAN_WEIGHT, length_target=LENGTH_TARGET,
        curvature_target=CURVATURE_TARGET, max_nfev=min(budget, EVALUATIONS_PER_RUN or budget), limits=limits)
    cost_history, segments = cost_history + history, segments + [dict(record, limit_weight=weight)]
    print(f"Coil fit: {record['nfev']} evaluations in {record['seconds']:.1f} s: {record['message']}")
    np.savez(checkpoint, initial=np.asarray(initial_dofs), optimized=np.asarray(ravel_pytree(field_optimized)[0]),
             cost_history=cost_history, segments=json.dumps(segments), config=CONFIG)
else:
    field_optimized = unravel(jnp.asarray(start))
optimization = dict(segments[-1] if segments else {}, nfev=len(cost_history), segments=segments,
                    seconds=sum(g["seconds"] for g in segments))


""" Diagnostics at higher axis and coil resolution """
diagnostic = make_near_axis(NPHI_DIAGNOSTIC).solution
states = {name: helpers.coil_state(field, diagnostic, PLASMA_RADIUS, N_SEGMENTS_BENCHMARK)
          for name, field in (("initial", field_initial), ("optimized", field_optimized))}
print("\n" + "#" * 78)
boundary_fine = helpers.boundary_points(diagnostic, PLASMA_RADIUS)
for name, state in states.items():
    helpers.print_state(name.capitalize(), state)
    state["quality"] = helpers.coil_quality(state["field"], boundary_fine, 480)
    print("   at 480 points: " + ", ".join(f"{k} {v:.4g}" for k, v in state["quality"].items()))
    state["field"].coils.to_json(str(OUTPUT_DIR / f"coils_{name}.json"))
summary = dict(case="qh_finite_beta", inputs=dict(rc=RC, zs=ZS, nfp=NFP, etabar=ETABAR, B2c=B2C, B0=B0, I2=I2, p2=P2,
                                                   a=PLASMA_RADIUS), optimization=optimization,
               states={n: dict(axis_match=s["match"], boundary=s["normal"], coil_quality=s["quality"])
                       for n, s in states.items()}, vmex={})
# The boundary B.n/|B| after every segment: a diagnostic of the fit, not part of the objective.
with open(OUTPUT_DIR / "trajectory.jsonl", "a") as trajectory:
    trajectory.write(json.dumps(dict(evaluations=len(cost_history), cost=cost_history[-1] if cost_history else None,
                                     normal=states["optimized"]["normal"], quality=states["optimized"]["quality"],
                                     axis=states["optimized"]["match"])) + "\n")


""" VMEX free-boundary equilibria in the optimized coils """
equilibria = {}
if RUN_VMEX:
    for fraction in VMEX_RADIUS_FRACTIONS:
        radius, name = fraction * PLASMA_RADIUS, f"direct_{fraction:g}"
        print(f"\n{'=' * 78}\nVMEX free boundary at a_b = {fraction:g} a = {radius:.4g} m")
        wout, report = helpers.solve_free_boundary(diagnostic, states["optimized"]["field"], radius,
                                                   OUTPUT_DIR / "vmex", name, VMEX)
        summary["vmex"][f"{fraction:g}"] = report
        equilibria[fraction] = wout
        if report["converged"]:
            near_axis_report = report["near_axis"]
            print(f"   converged in {report['iterations']} iterations: beta {report['betatotal']:.2e}, "
                  f"iota lab {report['signed']['vmex_iota_axis_lab']:+.4f} (near axis {report['signed']['near_axis_iota_lab']:+.4f}), "
                  f"axis shift / a_b {near_axis_report['axis_shift_over_benchmark_radius']:.3e}, "
                  f"LCFS shape / a_b {near_axis_report['surfaces'][-1]['shape_rms_over_flux_radius']:.3e}")
        helpers.save_json(OUTPUT_DIR / "summary.json", summary)
helpers.save_json(OUTPUT_DIR / "summary.json", summary)


""" Plots """
label = f"QH, a = {PLASMA_RADIUS} m, p$_2$ = {P2:g} Pa/m$^2$"
helpers.plot_optimization(cost_history, {n: s["match"] for n, s in states.items()}, OUTPUT_DIR / "optimization.png", label)
helpers.plot_axis_profiles(diagnostic, states["optimized"]["targets"], states["optimized"]["field"],
                           OUTPUT_DIR / "axis_fields.png", label)
helpers.plot_coils_and_normal_error(states, {n: s["surface"] for n, s in states.items()},
                                    OUTPUT_DIR / "coils_and_normal_field.png", label)
for fraction, wout in equilibria.items():
    if wout is not None and summary["vmex"][f"{fraction:g}"]["converged"]:
        helpers.plot_cross_sections({"optimized": states["optimized"]}, {"optimized": {"direct": wout}},
                                    fraction * PLASMA_RADIUS, FLUX_LEVELS,
                                    OUTPUT_DIR / f"cross_sections_{fraction:g}.png", label)
print(f"\nResults saved in {OUTPUT_DIR}")
if SHOW_PLOTS:
    plt.show()
