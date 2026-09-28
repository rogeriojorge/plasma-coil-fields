"""Collect the QH, hybrid and single-stage example runs into examples_results.json and draw their figures.

python collect_examples.py   (reads runs/, writes examples_results.json and figures/examples/)
Native WOUT files stay out of git; their sha256 is recorded.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "drivers"))
import nearaxis_finite_beta_helpers as helpers  # noqa: E402
from essos.coils import Coils  # noqa: E402

RUNS, OUT = ROOT / "runs", ROOT / "figures" / "examples"
OUT.mkdir(parents=True, exist_ok=True)
plt.style.use(ROOT / "paper.mplstyle")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ITER = re.compile(r"^\s*(\d+)\s+([0-9.]+E[-+]\d+)\s+([0-9.]+E[-+]\d+)\s+([0-9.]+E[-+]\d+)\s", re.M)


def coil_metrics(path, n=240):
    coils = Coils.from_json(str(path))
    curves = coils.curves.copy()
    curves.n_segments = n
    fine = Coils(curves=curves, currents=coils.dofs_currents_raw, currents_scale=coils.currents_scale)
    return dict(points_per_coil=n, max_length_m=float(np.max(fine.length)), max_curvature_per_m=float(np.max(fine.curvature)))


def log_residuals(log):
    rows = [tuple(map(float, m.groups()[1:])) for m in ITER.finditer(Path(log).read_text())]
    last = rows[-1] if rows else None
    return dict(final_fsq=last)


def vmex_case(run, attempt):
    s = json.loads((RUNS / run / "summary.json").read_text())
    rep = next(iter(s["vmex"].values()))
    wout = RUNS / run / "vmex" / f"wout_{rep['route']}.nc"
    row = dict(run=run, attempt=attempt, a_b_m=rep["radius_m"], converged=rep["converged"],
               **log_residuals(RUNS / run / "run.log"))
    if wout.exists():
        row["wout_sha256"] = sha(wout)
    if rep["converged"]:
        na = rep["near_axis"]
        row.update(iterations=rep["iterations"], betatotal=rep["betatotal"], iota_vmex_lab=rep["signed"]["vmex_iota_axis_lab"],
                   iota_near_axis_lab=rep["signed"]["near_axis_iota_lab"],
                   axis_offset_max_over_ab=na["axis_shift_over_benchmark_radius"],
                   lcfs_shape_rms_over_ab=na["surfaces"][-1]["shape_rms_over_flux_radius"],
                   shape_rms_over_flux_radius={f"{r['s']:.4g}": r["shape_rms_over_flux_radius"] for r in na["surfaces"]},
                   tangential_jump_max_over_B=rep["interface"]["tangential_jump_max_over_B"],
                   tangential_jump_rms_over_B=rep["interface"]["tangential_jump_rms_over_B"])
    return row


# (run directory, a_b / a, settings) of every free-boundary attempt, in the order they were run
ATTEMPTS = {
    "qh": [("qh_ab1", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
           ("qh_ab1.25", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
           ("qh_ab1.5", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
           ("qh_ab1.5_f9", "NS 17/33/65, FTOL 1e-8/1e-9/1e-9, DELT 0.5"),
           ("qh_ab1.5_d25", "NS 17/33, FTOL 1e-8/1e-9, DELT 0.25"),
           ("qh_ab2", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
           ("qh_ab2_d25", "NS 17/33, FTOL 1e-8/1e-9, DELT 0.25")],
    "hybrid": [("hybrid_ab1", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
               ("hybrid_ab1.25", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
               ("hybrid_ab1.5", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
               ("hybrid_ab1.5_f9", "NS 17/33/65, FTOL 1e-8/1e-9/1e-9, NITER 16000 at NS 65, DELT 0.5"),
               ("hybrid_ab1.5_ns33", "NS 17/33, FTOL 1e-8/1e-9, DELT 0.25"),
               ("hybrid_ab2", "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"),
               ("hybrid_ab2_d25", "NS 17/33/65, FTOL 1e-8/1e-9/1e-9, DELT 0.25"),
               ("hybrid_ab2_ns33", "NS 17/33, FTOL 1e-8/1e-9, DELT 0.25")],
}

results = {}
for case, attempts in ATTEMPTS.items():
    s = json.loads((RUNS / case / "summary.json").read_text())
    opt = s["optimization"]
    m = s["states"]["optimized"]["axis_match"]
    results[case] = dict(
        inputs=s["inputs"],
        optimization=dict(evaluations=opt["nfev"], cost=opt["cost"], message=opt["message"],
                          checkpoint_sha256=sha(RUNS / case / "optimized_dofs.npz")),
        axis_match=dict(field_rms_over_B0=m["field_rms_over_B0"], gradient_rms_R0_over_B0=m["gradient_rms_R0_over_B0"],
                        hessian_rms_R0sq_over_B0=m["hessian_rms_R0sq_over_B0"],
                        target_hessian_rms_R0sq_over_B0=m["target_hessian_rms_R0sq_over_B0"]),
        boundary_normal_field=dict(a_m=s["states"]["optimized"]["boundary"]["radius_m"],
                                   max_over_B=s["states"]["optimized"]["boundary"]["normal_error_max"],
                                   rms_over_B=s["states"]["optimized"]["boundary"]["normal_error_rms"]),
        coils=coil_metrics(RUNS / case / "coils_optimized.json"),
        free_boundary=[vmex_case(run, note) for run, note in attempts])

# Single stage: the design as optimized, and the free-boundary ladder on the fixed coils
single = json.loads((RUNS / "single" / "summary.json").read_text())
values = single["terms"]["after coil_limits"]["values"]
ladder = []
for name, note in (("single/vmex_free_boundary", "NS 17, FTOL 1e-8, DELT 0.5 (previous run, as archived)"),
                   ("single_fb/ns33_d025", "NS 33, FTOL 1e-9, DELT 0.25, from NS 17"),
                   ("single_fb/ns33_f8_d025", "NS 33, FTOL 1e-8, DELT 0.25, from NS 17"),
                   ("single_fb/ns65_f8_a", "NS 65, FTOL 1e-8, DELT 0.25, from NS 33, 1500 iterations"),
                   ("single_fb/ns65_f8_b", "NS 65, FTOL 1e-8, DELT 0.25, continued"),
                   ("single_fb/ns65_f9_a", "NS 65, FTOL 1e-9, DELT 0.25, from the NS 65 FTOL 1e-8 state")):
    d = RUNS / name
    if (d / "report.json").exists():
        r = json.loads((d / "report.json").read_text())
    else:
        r = dict(single["vmex"]["benchmark"])
    row = dict(attempt=note, ns=r["ns"], ftol=r["ftol"], iterations=r["iterations"], converged=r["converged"],
               fsq=[r["fsqr"], r["fsqz"], r["fsql"]], betatotal=r["betatotal"], iota_axis_vmex=r["iota_axis"],
               iota_edge_vmex=r["iota_edge"], wout_sha256=sha(d / r["wout"]))
    if "near_axis" in r:
        row.update(axis_offset_max_over_a=r["near_axis"]["axis_shift_over_benchmark_radius"],
                   lcfs_shape_rms_over_a=r["near_axis"]["surfaces"][-1]["shape_rms_over_flux_radius"],
                   tangential_jump_max_over_B=r["interface"]["tangential_jump_max_over_B"],
                   iota_near_axis_lab=r["signed"]["near_axis_iota_lab"], iota_vmex_lab=r["signed"]["vmex_iota_axis_lab"])
    ladder.append(row)
sm = single["states"]["optimized"]
results["single_stage"] = dict(
    config_hash=single["config_hash"], near_axis_beta=values["beta"], iota_near_axis=values["iota"],
    quasisymmetry_B20_variation_R0sq_over_B0=values["B20_variation_R0sq_over_B0"],
    DMerc_times_r2=values["DMerc_times_r2"], r_singularity_over_a=values["r_singularity_over_a"],
    min_coil_coil_distance_m=values["min_coil_coil_distance"], min_coil_plasma_distance_m=values["min_coil_plasma_distance"],
    requirements=single["requirements"], coils=coil_metrics(RUNS / "single" / "coils_optimized.json"),
    boundary_normal_field=dict(a_m=sm["boundary"]["radius_m"], max_over_B=sm["boundary"]["normal_error_max"],
                               rms_over_B=sm["boundary"]["normal_error_rms"]),
    axis_match_T=sm["axis_match"], free_boundary=ladder)
previous = ROOT / "examples_results.json"
if previous.exists():  # keep the hand-recorded package versions
    results = {"versions": json.loads(previous.read_text()).get("versions"), **results}
previous.write_text(json.dumps(results, indent=1) + "\n")

# Figure: boundary-radius scan (converged solves only)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), layout="constrained")
for case, marker in (("qh", "o"), ("hybrid", "s")):
    a = results[case]["inputs"]["a"]
    rows = [r for r in results[case]["free_boundary"] if r["converged"]]
    x = np.array([r["a_b_m"] / a for r in rows])
    label = dict(qh="QH (nfp 4)", hybrid="hybrid (nfp 2, I$_2$ = 0.4)")[case]
    axes[0].plot(x, [100 * r["axis_offset_max_over_ab"] for r in rows], marker, ls="-", label=label)
    axes[0].plot(x, [100 * r["lcfs_shape_rms_over_ab"] for r in rows], marker, ls="--", mfc="none",
                 color=axes[0].lines[-1].get_color())
    axes[1].plot(x, [abs(r["iota_vmex_lab"] / r["iota_near_axis_lab"]) for r in rows], marker, ls="-", label=label)
axes[0].set(xlabel=r"$a_b / a$", ylabel=r"% of $a_b$", xlim=(0.9, 1.6))
axes[0].text(0.03, 0.97, "solid: axis offset (max)\ndashed: LCFS shape (RMS)", transform=axes[0].transAxes, va="top", fontsize=7)
axes[0].set_ylim(0, 32)
axes[1].set(xlabel=r"$a_b / a$", ylabel=r"$\iota_\mathrm{VMEX}/\iota_\mathrm{near\,axis}$ on axis", xlim=(0.9, 1.6))
axes[1].axhline(1, color="0.5", lw=0.6)
axes[1].set_ylim(0.93, 1.03)
axes[1].legend(loc="upper right", title="a$_b$ = 2a: no converged solve", title_fontsize=6.5)
fig.savefig(OUT / "qh_hybrid_boundary_scan.png", dpi=300)

# Figure: single-stage residual ladder and transform against resolution
fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
conv = [r for r in ladder if r["converged"]]
ax.plot([r["ns"] for r in conv], [r["iota_axis_vmex"] for r in conv], "o-", label="VMEX, converged (FTOL 1e-8)")
bad = [r for r in ladder if not r["converged"]]
ax.plot([r["ns"] for r in bad], [r["iota_axis_vmex"] for r in bad], "x", color="C1", label="VMEX, not converged")
ax.axhline(abs(values["iota"]), color="0.3", ls="--", lw=1, label="near-axis design")
ax.set(xscale="log", xticks=[17, 33, 65], xticklabels=["17", "33", "65"], xlabel="NS", ylabel=r"$\iota$ on axis")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(0.36, 0.70)
ax.legend(fontsize=6.5, loc="center right")
fig.savefig(OUT / "single_stage_iota_ladder.png", dpi=300)
print(json.dumps({k: [(r.get("run"), r["converged"]) for r in v["free_boundary"]] for k, v in results.items() if k != "single_stage"}))
