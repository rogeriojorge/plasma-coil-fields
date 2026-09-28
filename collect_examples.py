"""Collect the QH, hybrid and single-stage example runs into examples_results.json and draw their figures.

python collect_examples.py   (reads runs/, writes examples_results.json and figures/examples/)
Native WOUT files stay out of git; their sha256 is recorded.
"""
import hashlib
import json
import re
import shutil
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

RUNS, OUT = ROOT / "runs", ROOT / "figures" / "examples"
OUT.mkdir(parents=True, exist_ok=True)
plt.style.use(ROOT / "paper.mplstyle")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ITER = re.compile(r"^\s*(\d+)\s+([0-9.]+E[-+]\d+)\s+([0-9.]+E[-+]\d+)\s+([0-9.]+E[-+]\d+)\s", re.M)


def trajectory(run):
    """B.n/|B| on r = a, axis match and coil quality after every optimization segment of a run."""
    rows, seen = [], set()
    for line in (RUNS / run / "trajectory.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["evaluations"] in seen:
            continue
        seen.add(d["evaluations"])
        rows.append(dict(evaluations=d["evaluations"], normal_max_over_B=d["normal"]["normal_error_max"],
                         normal_rms_over_B=d["normal"]["normal_error_rms"],
                         field_rms_over_B0=d["axis"]["field_rms_over_B0"],
                         gradient_rms_R0_over_B0=d["axis"]["gradient_rms_R0_over_B0"],
                         hessian_rms_R0sq_over_B0=d["axis"]["hessian_rms_R0sq_over_B0"], coils=d["quality"]))
    return rows


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


LADDER = "NS 17/33/65, FTOL 1e-8/1e-9/1e-10, DELT 0.5"
# Second pass. Fit: FIT[case][0] (2000 evaluations: soft coil limits to 1600, weight 100 after), then
# FIT[case][1] (400 evaluations at weight 100 with the distance terms at 24 points per order, from the
# first). Free-boundary attempts on the final coils, in the order they were run.
FIT = {"qh": ("qh5", "qh6"), "hybrid": ("hybrid5", "hybrid6")}
ATTEMPTS = {
    "qh": [("qh6_ab1", LADDER), ("qh6_ab1.25", LADDER), ("qh6_ab1.5", LADDER), ("qh6_ab2", LADDER)],
    "hybrid": [("hybrid6_ab1", LADDER), ("hybrid6_ab1.25", LADDER), ("hybrid6_ab1.5", LADDER),
               ("hybrid6_ab1.5_ns33", "NS 17/33, FTOL 1e-8/1e-9, DELT 0.25"), ("hybrid6_ab2", LADDER)],
}
# Runs of the second pass that were stopped and replaced; their last recorded state is kept.
ABANDONED = {
    "qh": [("qh2", "limits 0.08/0.12 m, MSC 30, weight 1: replaced to make the limits near-hard"),
           ("qh3", "limits 0.08/0.12 m, MSC 30, weight 100 from the start: fit stalled"),
           ("qh4", "limits 0.05/0.10 m, MSC 50, weight 100 from the start: B.n rising as the fit continued")],
    "hybrid": [("hybrid2", "limits 0.08/0.12 m, MSC 30, weight 1: replaced to make the limits near-hard"),
               ("hybrid3", "limits 0.08/0.12 m, MSC 30, weight 100 from the start: fit stalled at B.n 12 %"),
               ("hybrid4", "limits 0.05/0.10 m, MSC 50, weight 100 from the start: stalled at B.n 6 %")],
}

previous_path = ROOT / "examples_results.json"
previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
# The first-pass results are kept verbatim under "first_pass".
first_pass = previous.get("first_pass") or {k: previous[k] for k in ("qh", "hybrid", "single_stage") if k in previous}
results = {"versions": previous.get("versions")}
for case, attempts in ATTEMPTS.items():
    fit, final = FIT[case]
    s = json.loads((RUNS / final / "summary.json").read_text())
    opt = s["optimization"]
    m = s["states"]["optimized"]["axis_match"]
    abandoned = []
    for run, note in ABANDONED[case]:
        if (RUNS / run / "trajectory.jsonl").exists():
            abandoned.append(dict(run=run, note=note, last=trajectory(run)[-1]))
    results[case] = dict(
        inputs=s["inputs"],
        optimization=dict(evaluations=dict(fit=json.loads((RUNS / fit / "summary.json").read_text())["optimization"]["nfev"],
                                           limits=opt["nfev"]),
                          cost=opt["cost"], message=opt["message"],
                          checkpoint_sha256={fit: sha(RUNS / fit / "optimized_dofs.npz"),
                                             final: sha(RUNS / final / "optimized_dofs.npz")}),
        axis_match=dict(field_rms_over_B0=m["field_rms_over_B0"], gradient_rms_R0_over_B0=m["gradient_rms_R0_over_B0"],
                        hessian_rms_R0sq_over_B0=m["hessian_rms_R0sq_over_B0"],
                        target_hessian_rms_R0sq_over_B0=m["target_hessian_rms_R0sq_over_B0"]),
        boundary_normal_field=dict(a_m=s["states"]["optimized"]["boundary"]["radius_m"],
                                   max_over_B=s["states"]["optimized"]["boundary"]["normal_error_max"],
                                   rms_over_B=s["states"]["optimized"]["boundary"]["normal_error_rms"]),
        coils=s["states"]["optimized"]["coil_quality"],
        trajectory=trajectory(fit) + trajectory(final),
        abandoned_runs=abandoned,
        free_boundary=[vmex_case(run, note) for run, note in attempts if (RUNS / run / "summary.json").exists()])

# Single stage: the design as optimized, and the free-boundary ladder on the fixed coils
SINGLE = "single4"
single = json.loads((RUNS / SINGLE / "summary.json").read_text())
values = single["terms"]["after coil_limits"]["values"]
ladder = []
for name, note in (("ns17_f10", "NS 17, FTOL 1e-10, DELT 0.5, 3000 iterations from scratch"),
                   ("ns17_f10_b", "NS 17, FTOL 1e-10, DELT 0.5, continued"),
                   ("ns33_f10", "NS 33, FTOL 1e-10, DELT 0.5, from NS 17"),
                   ("ns33_f10_b", "NS 33, FTOL 1e-10, DELT 0.5, continued (residual rises)"),
                   ("ns33_f9", "NS 33, FTOL 1e-9, DELT 0.5, from NS 17"),
                   ("ns65_f9", "NS 65, FTOL 1e-9, DELT 0.5, from NS 33"),
                   ("ns65_f10", "NS 65, FTOL 1e-10, DELT 0.5, from the NS 65 FTOL 1e-9 state")):
    d = RUNS / f"{SINGLE}_fb" / name
    if not (d / "report.json").exists():
        continue
    r = json.loads((d / "report.json").read_text())
    row = dict(run=name, attempt=note, ns=r["ns"], ftol=r["ftol"], iterations=r["iterations"], converged=r["converged"],
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
    run=SINGLE, config_hash=single["config_hash"], near_axis_beta=values["beta"], iota_near_axis=values["iota"],
    quasisymmetry_B20_variation_R0sq_over_B0=values["B20_variation_R0sq_over_B0"],
    DMerc_times_r2=values["DMerc_times_r2"], DWell_times_r2=values["DWell_times_r2"],
    r_singularity_over_a=values["r_singularity_over_a"], max_elongation=values["max_elongation"],
    requirements=single["requirements"], coils=sm["coil_quality"],
    boundary_normal_field=dict(a_m=sm["boundary"]["radius_m"], max_over_B=sm["boundary"]["normal_error_max"],
                               rms_over_B=sm["boundary"]["normal_error_rms"]),
    axis_match_T=sm["axis_match"], free_boundary=ladder,
    mercier_scan=json.loads((ROOT / "single_stage_mercier_scan.json").read_text()),
    abandoned_runs=[dict(run="single2_attempt_d020_r050", note="a = 0.15 m, hard Mercier hinge, coil-plasma 0.20 m from "
                         "0.5 m circles: coil stage ran to 52 m coils; stopped in stage 2"),
                    dict(run="single3_attempt_a015", note="a = 0.15 m, hard Mercier hinge, coil-plasma 0.15 m from "
                         "0.72 m circles: finished, but the r = a surface folds (normals), B.n not evaluable, "
                         "DMerc r^2 -0.0015 and B20 variation 0.13 after the coil-limit stage")])
results["first_pass"] = first_pass
previous_path.write_text(json.dumps(results, indent=1) + "\n")

# Driver figures of the final runs, under stable names
for source, target in (("qh6/axis_fields.png", "qh_axis_fields.png"), ("qh6/coils_and_normal_field.png", "qh_coils_and_normal_field.png"),
                       ("qh5/optimization.png", "qh_optimization.png"),
                       ("hybrid6/axis_fields.png", "hybrid_axis_fields.png"),
                       ("hybrid6/coils_and_normal_field.png", "hybrid_coils_and_normal_field.png"),
                       ("hybrid5/optimization.png", "hybrid_optimization.png"),
                       (f"{SINGLE}/coils_and_normal_field.png", "single_stage_coils_and_normal_field.png"),
                       (f"{SINGLE}/optimization.png", "single_stage_optimization.png")):
    shutil.copyfile(RUNS / source, OUT / target)
for case in ATTEMPTS:
    for row in results[case]["free_boundary"]:
        label = row["run"].split("_ab")[1]
        fraction = label.split("_")[0]
        figure = RUNS / row["run"] / f"cross_sections_{float(fraction):g}.png"
        target = OUT / f"{case}_cross_sections_ab{fraction}.png"
        if row["converged"] and figure.exists():
            shutil.copyfile(figure, target)
        elif target.exists() and not any(r["converged"] and r["run"].split("_ab")[1].split("_")[0] == fraction
                                         for r in results[case]["free_boundary"]):
            target.unlink()  # A first-pass figure of a radius that no longer converges

# Figure: boundary-radius scan (converged solves only)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), layout="constrained")
unconverged = []
for case, marker in (("qh", "o"), ("hybrid", "s")):
    a = results[case]["inputs"]["a"]
    rows = [r for r in results[case]["free_boundary"] if r["converged"]]
    x = np.array([r["a_b_m"] / a for r in rows])
    label = dict(qh="QH (nfp 4)", hybrid="hybrid (nfp 3, I$_2$ = 0.4 T/m)")[case]
    axes[0].plot(x, [100 * r["axis_offset_max_over_ab"] for r in rows], marker, ls="-", label=label)
    axes[0].plot(x, [100 * r["lcfs_shape_rms_over_ab"] for r in rows], marker, ls="--", mfc="none",
                 color=axes[0].lines[-1].get_color())
    axes[1].plot(x, [abs(r["iota_vmex_lab"] / r["iota_near_axis_lab"]) for r in rows], marker, ls="-", label=label)
    tried = {round(r["a_b_m"] / a, 2) for r in results[case]["free_boundary"]}
    unconverged += [f"{case} {f:g}a" for f in sorted(tried - {round(v, 2) for v in x})]
for ax in axes:
    ax.set(xlabel=r"$a_b / a$", xlim=(0.9, 2.1), xticks=[1, 1.25, 1.5, 1.75, 2])
axes[0].set(ylabel=r"% of $a_b$")
axes[0].set_ylim(bottom=0)
axes[0].text(0.42, 0.97, "solid: axis offset (max)\ndashed: LCFS shape (RMS)", transform=axes[0].transAxes, va="top", fontsize=7)
axes[1].set(ylabel=r"$\iota_\mathrm{VMEX}/\iota_\mathrm{near\,axis}$ on axis")
axes[1].axhline(1, color="0.5", lw=0.6)
axes[1].legend(loc="center right", title=("not converged: " + ", ".join(unconverged)) if unconverged else None,
               title_fontsize=6.5)
fig.savefig(OUT / "qh_hybrid_boundary_scan.png", dpi=300)

# Figure: single-stage transform against radial resolution
fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
for converged, style, text in ((True, "o", "VMEX, converged"), (False, "x", "VMEX, not converged")):
    rows = [r for r in ladder if r["converged"] == converged]
    ax.plot([r["ns"] for r in rows], [r["iota_axis_vmex"] for r in rows], style, ls="none", label=text)
for ftol, color in ((1e-10, "C0"), (1e-9, "C2")):
    rows = [r for r in ladder if r["converged"] and r["ftol"] == ftol]
    for r in rows:
        ax.annotate(f"FTOL {ftol:.0e}".replace("e-", "e−"), (r["ns"], r["iota_axis_vmex"]), textcoords="offset points",
                    xytext=(4, 4), fontsize=6)
ax.axhline(abs(values["iota"]), color="0.3", ls="--", lw=1, label="near-axis design")
ax.set(xscale="log", xticks=[17, 33, 65], xticklabels=["17", "33", "65"], xlabel="NS", ylabel=r"$\iota$ on axis")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.margins(x=0.15, y=0.25)
ax.legend(fontsize=6.5, loc="upper right")
fig.savefig(OUT / "single_stage_iota_ladder.png", dpi=300)
print(json.dumps({k: [(r.get("run"), r["converged"]) for r in results[k]["free_boundary"]] for k in ("qh", "hybrid", "single_stage")}))
