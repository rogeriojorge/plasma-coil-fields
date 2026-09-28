"""Largest-boundary free-boundary equilibrium of every design against the near-axis surfaces.

python max_radius.py DESIGN ...   # rebuild the near-axis solution (no optimization, no VMEX) and compare
python max_radius.py --plot       # figures and the summary from the stored JSON only

For each design, the WOUT named in DESIGNS is the free-boundary VMEX equilibrium at the largest
boundary radius a_b that converged in the scan recorded in max_radius_scan.json. The near-axis
solution is rebuilt from the design's checkpoint by the same driver and overrides that made it.
Results go to max_radius/DESIGN.json (metrics and contours in four toroidal planes); --plot writes
figures/max_radius/*.png and docs/figures/13_max_radius_*.png and max_radius_results.json.

WOUT files are not tracked: they are looked up under runs/ and their sha256 is recorded.
p2 is fixed, so the central pressure grows as a_b^2 along the scan.
"""
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
OUT = REPO / "max_radius"
LEVELS = (0.0625, 0.25, 0.5625, 1.0)
PLANES = (0.0, 0.25, 0.5, 0.75)  # Toroidal planes, in field periods
NTHETA = 129

QA = "optimize_coils_and_nearaxis_finite_beta.py"
QH = "optimize_coils_nearaxis_qh_finite_beta.py"
HY = "optimize_coils_nearaxis_hybrid_finite_current.py"
SECOND_PASS = ["MAX_FUNCTION_EVALUATIONS = 400", "LIMIT_WEIGHTS = ((400, 100.0),)", "DISTANCE_POINTS_PER_ORDER = 24"]
# checkpoint: run directory whose optimized_dofs.npz defines the coils; wout: the maximum-radius equilibrium.
DESIGNS = {
    "qa": dict(label="QA finite beta (a = 30 mm)", driver=QA, checkpoint="qa", args=[],
               wout="qa_ab1.5/vmex_fitted_optimized/wout_direct.nc", fraction=1.5),
    "nohess": dict(label="Hessian omitted", driver=QA, checkpoint="nohess", args=["HESSIAN_WEIGHT = 0.0"],
                   wout=None, fraction=None),
    "control": dict(label="total-field control", driver=QA, checkpoint="control",
                    args=["SUBTRACT_PLASMA_FIELD = False"], wout=None, fraction=None),
    "vacuum": dict(label="vacuum QA", driver=QA, checkpoint="vacuum",
                   args=['CASE = "qa_vacuum"', "TRACE_FIELD_LINES = False"], wout=None, fraction=None),
    "axisym": dict(label="axisymmetric", driver=QA, checkpoint="axisym", args=['CASE = "axisymmetric"'],
                   wout=None, fraction=None),
    "qh": dict(label="QH (nfp 4)", driver=QH, checkpoint="qh6",
               args=SECOND_PASS + ['WARM_START = "../runs/qh5/optimized_dofs.npz"'], wout=None, fraction=None),
    "hybrid": dict(label="hybrid (nfp 3)", driver=HY, checkpoint="hybrid6",
                   args=SECOND_PASS + ['WARM_START = "../runs/hybrid5/optimized_dofs.npz"'], wout=None, fraction=None),
    "single": dict(label="single stage (well margin)", driver="single", checkpoint="single4", args=[],
                   wout=None, fraction=None),
}


# Every free-boundary attempt of the scan: design -> [(run directory under runs/, a_b / a)]. Single-stage
# entries are single_stage_free_boundary.py level directories (runs/single4_fb/NAME/report.json).
SCAN = {
    "qa": [("qa", 1.0), ("qa_ab1.25", 1.25), ("qa_ab1.5", 1.5), ("qa_ab1.625", 1.625), ("qa_ab1.75", 1.75),
           ("qa_ab2.0", 2.0), ("qa_ab2.5", 2.5), ("qa_ab3.0", 3.0)],
    "nohess": [("nohess", 1.0), ("nohess_ab1.125", 1.125), ("nohess_ab1.25", 1.25)],
    "control": [("control", 1.0), ("control_ab1.125", 1.125), ("control_ab1.25", 1.25)],
    "vacuum": [("vacuum", 0.7), ("V_ftol", 0.7), ("vacuum_ab0.85", 0.85), ("V_ab10", 1.0), ("vacuum_ab1.0", 1.0)],
    "axisym": [("axisym", 1.0)] + [(f"axisym_ab{f}", float(f)) for f in ("1.25", "1.5", "1.75", "2.0", "2.25", "2.5", "2.75", "2.875", "3.0")],
    "qh": [("qh6_ab1", 1.0), ("qh6_ab1.25", 1.25), ("qh6_ab1.5", 1.5), ("qh6_ab2", 2.0)]
          + [(f"qh6_ab{f}", float(f)) for f in ("2.25", "2.5", "2.75", "3.0", "3.25", "3.375", "3.5")],
    "hybrid": [("hybrid6_ab1", 1.0), ("hybrid6_ab1.25", 1.25), ("hybrid6_ab1.375", 1.375), ("hybrid6_ab1.5", 1.5),
               ("hybrid6_ab1.5_ns33", 1.5), ("hybrid6_ab2", 2.0)],
    "single": [(f"single4_fb/{n}", 1.0) for n in ("ns17_f10", "ns17_f10_b", "ns33_f10", "ns33_f10_b", "ns33_f9",
                                                  "ns65_f10", "ns65_f9")]
              + [(f"single4_fb/ab{f:g}_{n}", f) for f in (1.125, 1.25) for n in ("ns17_f10", "ns17_f10_b")],
}
# The tolerance a solve must reach to count. FTOL 1e-10 at NS 65 everywhere, except the single-stage design,
# whose NS 65 solve stalls at 5e-10 already at a_b = a (README Sec. 7); looser solves stay listed as attempts.
REQUIRED_FTOL = dict.fromkeys(DESIGNS, 1e-10) | {"single": 1e-9}


def shape(report):
    """Axis offset and LCFS shape discrepancy of a converged attempt, relative to a_b (as the driver reported)."""
    near = report.get("near_axis") if report.get("converged") else None
    if not near:
        return {}
    return dict(axis_offset_over_a_b=near["axis_shift_over_benchmark_radius"],
                lcfs_shape_over_a_b=near["surfaces"][-1]["shape_rms_over_flux_radius"],
                iota_lab_vmex=report.get("signed", {}).get("vmex_iota_axis_lab"))


def attempt(run, fraction):
    """Outcome of one free-boundary attempt, read from the run's own summary."""
    import re
    directory = RUNS / run
    report = directory / "report.json"
    if report.exists():  # single_stage_free_boundary.py level
        r = json.loads(report.read_text())
        return dict(run=run, fraction=fraction, ns=r["ns"], ftol=r["ftol"], converged=r["converged"],
                    fsqr=r["fsqr"], iterations=r["iterations"], wout=f"{run}/{r['wout']}", **shape(r))
    summary = directory / "summary.json"
    if not summary.exists():
        return dict(run=run, fraction=fraction, converged=False, note="no summary: killed at the 600 s cap")
    vmex = json.loads(summary.read_text())["vmex"]
    entry = vmex["fitted"]["optimized"]["direct"] if "fitted" in vmex else vmex[f"{fraction:g}"]
    wout = next(directory.glob(f"vmex*/wout_{entry['route']}.nc"), None)
    deck = next(directory.glob(f"vmex*/input.{entry['route']}"), None)
    ns = ftol = None
    if deck is not None:
        text = deck.read_text()
        ns = int(re.search(r"NS_ARRAY\s*=\s*([^\n]*)", text, re.I).group(1).split(",")[-1])
        ftol = float(re.search(r"FTOL_ARRAY\s*=\s*([^\n]*)", text, re.I).group(1).split(",")[-1])
    return dict(run=run, fraction=fraction, ns=ns, ftol=ftol, converged=bool(entry.get("converged")),
                fsqr=entry.get("fsqr"), iterations=entry.get("iterations"),
                wout=str(wout.relative_to(RUNS)) if wout is not None else None, **shape(entry))


def collect():
    """max_radius_scan.json: every attempt and the largest converged a_b of each design."""
    designs = {}
    for name, runs in SCAN.items():
        attempts = sorted((attempt(run, f) for run, f in runs), key=lambda x: (x["fraction"], -(x.get("ns") or 0)))
        good = [x for x in attempts if x["converged"] and x.get("ns") == 65 and x["ftol"] <= REQUIRED_FTOL[name] * 1.01]
        best = max(good, key=lambda x: (x["fraction"], x.get("ns") or 0, -(x.get("ftol") or 1)))
        failed = sorted({x["fraction"] for x in attempts if x["fraction"] > best["fraction"]}
                        - {x["fraction"] for x in good})
        designs[name] = dict(max_converged=dict(best, setting=f"NS {best['ns']}, FTOL {best['ftol']:.0e}"),
                             first_failed=failed[0] if failed else None, attempts=attempts)
    (REPO / "max_radius_scan.json").write_text(json.dumps(dict(
        note="Free-boundary VMEX attempts per design against a_b / a. Converged = FTOL reached with the vacuum "
             "region active. p2 is fixed, so the central pressure grows as (a_b/a)^2.", designs=designs), indent=1) + "\n")


def load_scan():
    """The chosen maximum-radius WOUT of each design, from max_radius_scan.json."""
    scan = json.loads((REPO / "max_radius_scan.json").read_text())
    for name, entry in scan["designs"].items():
        DESIGNS[name].update(wout=entry["max_converged"]["wout"], fraction=entry["max_converged"]["fraction"])
    return scan


def near_axis(name):
    """Rebuild the design's near-axis solution with its own driver (reusing its checkpoint)."""
    design = DESIGNS[name]
    work = RUNS / "max_radius_near_axis" / name
    work.mkdir(parents=True, exist_ok=True)
    if design["driver"] == "single":
        sys.argv = ["single_stage_free_boundary.py"]
        os.environ["SINGLE_RUN"] = design["checkpoint"]
        drivers = REPO / "drivers"
        sys.path.insert(0, str(drivers))
        os.chdir(drivers)
        import matplotlib
        matplotlib.use("Agg")
        src = (drivers / "optimize_single_stage_nearaxis_finite_beta.py").read_text()
        src = src.replace("RUN_VMEX = True", "RUN_VMEX = False", 1).replace(
            'OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_stage_finite_beta"',
            f'OUTPUT_DIR = Path("{RUNS / design["checkpoint"]}")', 1)
        g = {"__file__": str(drivers / "x.py"), "__name__": "__main__"}
        exec(compile(src, "single", "exec"), g)
        return g["states"]["optimized"]["solution"], float(g["PLASMA_RADIUS"]), g["helpers"]
    shutil.copy(RUNS / design["checkpoint"] / "optimized_dofs.npz", work / "optimized_dofs.npz")
    os.environ["EXAMPLE"] = design["driver"]
    sys.argv = ["run.py", str(work), "OPTIMIZE = False", "RUN_VMEX = False", *design["args"]]
    ns = runpy.run_path(str(REPO / "run.py"))["NAMESPACE"]
    solution = ns["diagnostic"] if design["driver"] in (QH, HY) else ns["states"]["optimized"]["solution"]
    return solution, float(ns["PLASMA_RADIUS"]), ns["helpers"]


def compare(name):
    from vmex.core.plotting import surface_rz
    import vmex as vj
    solution, a, helpers = near_axis(name)
    design = DESIGNS[name]
    path = RUNS / design["wout"]
    wout = vj.read_wout(path)
    a_b = design["fraction"] * a
    metrics = helpers.compare_to_near_axis(wout, solution, a_b, LEVELS)
    theta = np.arange(64) * 2 * np.pi / 64
    RM, ZM = surface_rz(wout, s_index=int(wout.ns) // 2, theta=theta, phi=np.zeros(1))
    RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=np.zeros(1))
    vmex_sign = helpers.poloidal_orientation(RM[:, 0], ZM[:, 0], float(RA[0, 0]), float(ZA[0, 0]))
    near_lab, _ = helpers.near_axis_lab_iota(solution, a_b)
    iota = np.asarray(wout.iotaf)
    phi_grid = np.asarray(solution.phi)
    period = 2 * np.pi / int(solution.inputs.axis.nfp)
    theta = np.arange(NTHETA) * 2 * np.pi / (NTHETA - 1)
    planes = []
    for fraction in PLANES:
        k = int(np.argmin(np.abs(phi_grid - fraction * period)))
        R0, Z0 = float(solution.R0[k]), float(solution.Z0[k])
        RA, ZA = surface_rz(wout, s_index=0, theta=np.zeros(1), phi=phi_grid[k:k + 1])
        plane = dict(phi_over_period=float(phi_grid[k] / period), near_axis_axis=[R0, Z0],
                     vmex_axis=[float(RA[0, 0]), float(ZA[0, 0])], near_axis=[], vmex=[])
        for index, s in helpers.flux_indices(wout, LEVELS):
            RV, ZV = surface_rz(wout, s_index=index, theta=theta, phi=phi_grid[k:k + 1])
            plane["vmex"].append(dict(s=s, R=np.round(RV[:, 0], 7).tolist(), Z=np.round(ZV[:, 0], 7).tolist()))
            surface = helpers.flux_surface(solution, a_b * np.sqrt(s), NTHETA - 1)
            R, Z = np.r_[surface["R"][:, k], surface["R"][0, k]], np.r_[surface["Z"][:, k], surface["Z"][0, k]]
            plane["near_axis"].append(dict(s=s, R=np.round(R, 7).tolist(), Z=np.round(Z, 7).tolist()))
        planes.append(plane)
    result = dict(design=name, label=design["label"], a_m=a, fraction=design["fraction"], a_b_m=a_b,
                  wout=design["wout"], wout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  ns=int(wout.ns), betatotal=float(wout.betatotal),
                  iota_lab=dict(vmex_axis=float(vmex_sign * iota[0]), vmex_edge=float(vmex_sign * iota[-1]),
                                near_axis=float(near_lab)),
                  comparison=metrics, planes=planes)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "planes"}, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(REPO / "paper.mplstyle")
    scan = load_scan()
    results = {n: json.loads((OUT / f"{n}.json").read_text()) for n in DESIGNS if (OUT / f"{n}.json").exists()}
    (REPO / "figures" / "max_radius").mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for name, r in results.items():
        fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.05), layout="constrained")
        a_b = r["a_b_m"]
        for ax, plane in zip(axes, r["planes"]):
            R0, Z0 = plane["near_axis_axis"]
            for c in plane["near_axis"]:
                ax.plot((np.array(c["R"]) - R0) / a_b, (np.array(c["Z"]) - Z0) / a_b, color="0.1", lw=1.0)
            for c in plane["vmex"]:
                ax.plot((np.array(c["R"]) - R0) / a_b, (np.array(c["Z"]) - Z0) / a_b, "--", color=colors[1], lw=1.0)
            ax.plot(0, 0, "+", color="0.1", ms=6, mew=1.2)
            ax.plot((plane["vmex_axis"][0] - R0) / a_b, (plane["vmex_axis"][1] - Z0) / a_b, "x", color=colors[1],
                    ms=5, mew=1.2)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(rf"$\phi$ = {plane['phi_over_period']:.2f} period")
            ax.set_xlabel(r"$(R - R_0)/a_b$")
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        axes[0].set_ylabel(r"$(Z - Z_0)/a_b$")
        handles = [plt.Line2D([], [], color="0.1", lw=1.0, label="near axis (+ axis)"),
                   plt.Line2D([], [], color=colors[1], ls="--", lw=1.0, label="VMEX free boundary (x axis)")]
        fig.legend(handles=handles, loc="outside lower center", ncols=2)
        fig.savefig(REPO / "figures" / "max_radius" / f"{name}_cross_sections.png")
        plt.close(fig)
    # Discrepancy against flux radius, all designs.
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained", sharey=True)
    for i, (name, r) in enumerate(results.items()):
        rows = r["comparison"]["surfaces"]
        rho = [np.sqrt(x["s"]) * r["fraction"] for x in rows]
        color = (colors + ["0.45", "0.1"])[i]
        marker = "osD^vPXh"[i]
        axes[0].semilogy(rho, [x["rms_over_flux_radius"] for x in rows], marker + "-", color=color, label=r["label"])
        axes[1].semilogy(rho, [x["shape_rms_over_flux_radius"] for x in rows], marker + "-", color=color)
    axes[0].set_title("raw (with axis offset)")
    axes[1].set_title("axis-registered shape")
    for ax in axes:
        ax.set_xlabel(r"flux radius $r/a$")
    axes[0].set_ylabel("RMS distance / flux radius")
    fig.legend(loc="outside right center")
    fig.savefig(REPO / "figures" / "max_radius" / "discrepancy_vs_radius.png")
    plt.close(fig)
    summary = dict(note="Maximum converged boundary radius a_b per design and the comparison with the near-axis "
                        "surfaces at s = " + ", ".join(f"{s:g}" for s in LEVELS) + ". p2 is fixed, so the central "
                        "pressure grows as a_b^2. Source: max_radius/*.json (this file is written by max_radius.py --plot).",
                   designs={})
    for name, r in results.items():
        c = r["comparison"]
        summary["designs"][name] = dict(
            label=r["label"], a_m=r["a_m"], a_b_over_a=r["fraction"], a_b_m=r["a_b_m"],
            setting=scan["designs"][name]["max_converged"]["setting"],
            first_failed=scan["designs"][name].get("first_failed"),
            axis_offset_max_over_a_b=c["axis_shift_over_benchmark_radius"],
            raw_rms_over_flux_radius={f"{x['s']:.4f}": x["rms_over_flux_radius"] for x in c["surfaces"]},
            shape_rms_over_flux_radius={f"{x['s']:.4f}": x["shape_rms_over_flux_radius"] for x in c["surfaces"]},
            iota_lab=r["iota_lab"], betatotal=r["betatotal"], wout_sha256=r["wout_sha256"])
    (REPO / "max_radius_results.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--collect"]:
        collect()
    elif sys.argv[1:] == ["--plot"]:
        plot()
    else:
        load_scan()
        compare(sys.argv[1])
