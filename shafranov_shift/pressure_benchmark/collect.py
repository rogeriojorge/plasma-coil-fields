"""Collect the pressure-response benchmark into results JSON and figures.

    python collect.py [--runs RUN_DIR]

With --runs, the row.json of every VMEX run and the theory files are first copied from RUN_DIR
(outside Git: it also holds the WOUT files) into ../results/pressure_benchmark/runs/. The
analysis then uses only the copies in the repository.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "pressure_benchmark"
RUNS = OUT / "runs"
sys.path.insert(0, str(HERE))
from compare import analyse  # noqa: E402

ADOPTED_FTOL = 1e-15


def pick(entries, **match):
    return [e for e in entries if all(e.get(k) == v for k, v in match.items())]


def one(entries, name):
    found = [e for e in entries if e["name"] == name]
    return found[0] if found else None


def ratio_with(entries, rows, name, vacuum):
    """Projected response ratio of run ``name`` against an explicit vacuum run."""
    th = np.load(RUNS / "theory" / "theory_qh.npz")
    r, v = rows[name], rows[vacuum]
    xi = r["p0_Pa"] * th["traced"]
    d = np.array([r["axis_R"], r["axis_Z"]]) - np.array([v["axis_R"], v["axis_Z"]])
    return float(np.sum(d * xi) / np.sum(xi * xi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    if args.runs:
        (RUNS / "theory").mkdir(parents=True, exist_ok=True)
        for path in sorted(args.runs.glob("*/row.json")):
            target = RUNS / path.parent.name
            target.mkdir(exist_ok=True)
            shutil.copy(path, target / "row.json")
            for extra in ("input.runtime",):
                if (path.parent / extra).exists():
                    shutil.copy(path.parent / extra, target / extra)
        for path in (args.runs / "theory").glob("theory_*"):
            shutil.copy(path, RUNS / "theory" / path.name)

    entries = [e for e in analyse(RUNS) if e["case"] == "qh"]
    rows = {p.parent.name: json.loads(p.read_text()) for p in RUNS.glob("*/row.json")}
    theory = json.loads((RUNS / "theory" / "theory_qh.json").read_text())

    def ftol_of(e):
        return rows[e["name"]]["ftol"][-1]

    for e in entries:
        e["ftol_final"] = ftol_of(e)
        e["fsqr"] = rows[e["name"]]["fsqr"]
        e["seconds"] = rows[e["name"]]["seconds"]

    # 1. Vacuum-axis bias against the traced coil axis, a_b = 35 mm, by final force tolerance.
    vac35 = [dict(run=e["name"], ftol=e["ftol_final"], mpol=e["mpol"], ns=e["ns"][-1],
                  restart=e["restart"], iterations=e["iterations"],
                  bias_rms_m=e["axis_minus_traced_rms_m"], bias_rms_over_ab=e["axis_minus_traced_rms_m"] / 0.035)
             for e in pick(entries, radius_m=0.035, beta=0.0)]
    vac_radius = [dict(run=e["name"], radius_m=e["radius_m"], ftol=e["ftol_final"], mpol=e["mpol"],
                       bias_rms_m=e["axis_minus_traced_rms_m"],
                       bias_rms_over_ab=e["axis_minus_traced_rms_m"] / e["radius_m"])
                  for e in entries if e["beta"] == 0 and e["mpol"] == 8]

    # 2. Pressure ladder at a_b = 35 mm, NS 65, MPOL 8, continued to FTOL 1e-15.
    ladder_names = ["qh_a35_b0.675e-3_cont15", "qh_a35_b1.35e-3_cont15", "qh_a35_b2.7e-3_cont15",
                    "qh_a35_b5.4e-3_cont15", "qh_a35_b1.08e-2_cont15"]
    vac_refs = ["qh_a35_b0_f14", "qh_a35_b0_cont14", "qh_a35_b0_cont15", "qh_a35_b0_cont16"]
    ladder = []
    for name in ladder_names:
        e = one(entries, name)
        if e is None:
            continue
        by_ref = {v: ratio_with(entries, rows, name, v) for v in vac_refs if v in rows}
        ladder.append(dict(run=name, beta_axis=e["beta"], theory_rms_m=e["theory_rms_m"],
                           theory_rms_over_ab=e["theory_rms_m"] / 0.035, ratio=e["ratio"],
                           ratio_by_vacuum_reference=by_ref,
                           vacuum_reference_half_spread=float(np.ptp(list(by_ref.values())) / 2),
                           ratio_vs_traced_axis=e["ratio_vs_traced"],
                           shape_residual_rms_over_theory_rms=e["residual_rms_over_theory_rms"],
                           iterations=e["iterations"], fsqr=e["fsqr"]))
    beta = np.array([r["beta_axis"] for r in ladder])
    amp = np.array([r["ratio"] * r["beta_axis"] for r in ladder])  # response amplitude in theory units
    # amplitude(beta) = k1 beta + k2 beta^2: k1 is the first-order VMEX/theory ratio
    design = np.stack((beta, beta**2), 1)
    (k1, k2), *_ = np.linalg.lstsq(design, amp, rcond=None)
    resid = amp - design @ np.array([k1, k2])
    cov = np.linalg.inv(design.T @ design) * (np.sum(resid**2) / max(len(beta) - 2, 1))
    linear_fit = dict(model="ratio*beta = k1*beta + k2*beta^2 over the ladder", k1=float(k1),
                      k1_fit_sigma=float(np.sqrt(cov[0, 0])), k2=float(k2),
                      quadratic_fraction_at_max_beta=float(k2 * beta.max() / k1))
    # Headline fit: displacement from the TRACED coil axis (no vacuum VMEX run needed), with a free
    # offset that absorbs the constant part of the vacuum-axis bias: c0 + k1 beta + k2 beta^2.
    amp_t = np.array([r["ratio_vs_traced_axis"] * r["beta_axis"] for r in ladder])
    D = np.stack((np.ones_like(beta), beta, beta**2), 1)
    coef = np.linalg.lstsq(D, amp_t, rcond=None)[0]
    res_t = amp_t - D @ coef
    cov_t = np.linalg.inv(D.T @ D) * (res_t @ res_t) / max(len(beta) - 3, 1)
    drop_one = [float(np.linalg.lstsq(D[np.arange(len(beta)) != i], amp_t[np.arange(len(beta)) != i],
                                      rcond=None)[0][1]) for i in range(len(beta))]
    offset_fit = dict(model="ratio_vs_traced*beta = c0 + k1*beta + k2*beta^2", k1=float(coef[1]),
                      k1_fit_sigma=float(np.sqrt(cov_t[1, 1])), k2=float(coef[2]), c0=float(coef[0]),
                      offset_rms_m=float(coef[0] / 2.7e-3 * [r for r in ladder if r["beta_axis"] == 2.7e-3][0][
                          "theory_rms_m"]),
                      k1_drop_one=drop_one, k1_drop_one_half_spread=float(np.ptp(drop_one) / 2),
                      quadratic_fraction_at_max_beta=float(coef[2] * beta.max() / coef[1]))

    # 3. Restart dependence at beta = 2.7e-3, a_b = 35 mm, by final force tolerance.
    restart = {}
    for label, names in {
        "cold ladder then continued": ["qh_a35_b2.7e-3", "qh_a35_b2.7e-3_cont14", "qh_a35_b2.7e-3_cont15"],
        "cold, one run": ["qh_a35_b2.7e-3_f14"],
        "warm from the vacuum": ["qh_a35_b2.7e-3_warmvac", "qh_a35_b2.7e-3_warmvac_cont15",
                                 "qh_a35_b2.7e-3_warmvac_cont16"],
        "warm from beta = 5.4e-3": ["qh_a35_b2.7e-3_warmabove", "qh_a35_b2.7e-3_warmabove_cont15",
                                    "qh_a35_b2.7e-3_warmabove_cont16"],
    }.items():
        restart[label] = [dict(run=n, ftol=ftol_of(one(entries, n)), ratio=one(entries, n).get("ratio"),
                               ratio_vs_traced_axis=one(entries, n)["ratio_vs_traced"],
                               iterations=one(entries, n)["iterations"])
                          for n in names if one(entries, n) is not None]
    final = [r[-1]["ratio"] for r in restart.values() if r and r[-1]["ftol"] <= ADOPTED_FTOL]
    restart_half_spread = float(np.ptp(final) / 2)

    # 4. Mode resolution at a_b = 35 mm, beta = 2.7e-3.
    resolution = {}
    for label, pair in {"MPOL 8 NTOR 8 NZETA 32": ("qh_a35_b2.7e-3_cont15", "qh_a35_b0_cont15"),
                        "MPOL 12 NTOR 12 NZETA 64": ("qh_a35_b2.7e-3_m12_cont15", "qh_a35_b0_m12_cont15"),
                        "NS 129 (MPOL 8)": ("qh_a35_b2.7e-3_ns129_cont15", "qh_a35_b0_ns129_cont15")}.items():
        if all(p in rows for p in pair):
            resolution[label] = dict(run=pair[0], vacuum=pair[1], ratio=ratio_with(entries, rows, *pair))
    base_ratio = resolution["MPOL 8 NTOR 8 NZETA 32"]["ratio"]
    resolution_shift = float(np.sqrt(sum((v["ratio"] - base_ratio) ** 2 for v in resolution.values())))

    # 5. Boundary radius at FTOL 1e-15.
    radius = []
    for a, name, vac in ((0.015, "qh_a15_b2.7e-3_cont15", "qh_a15_b0_cont15"),
                         (0.025, "qh_a25_b1.378e-3_cont15", "qh_a25_b0_cont15"),
                         (0.035, "qh_a35_b2.7e-3_cont15", "qh_a35_b0_cont15"),
                         (0.045, "qh_a45_b4.463e-3_cont15", "qh_a45_b0_cont15")):
        if name in rows and vac in rows:
            e = one(entries, name)
            radius.append(dict(radius_m=a, run=name, beta_axis=e["beta"], ratio=ratio_with(entries, rows, name, vac),
                               vacuum_bias_rms_m=one(entries, vac)["axis_minus_traced_rms_m"],
                               theory_rms_m=e["theory_rms_m"],
                               bias_over_signal=one(entries, vac)["axis_minus_traced_rms_m"] / e["theory_rms_m"],
                               shape_residual=e["residual_rms_over_theory_rms"]))
    # a_b = 15 mm (a_b/R0 = 0.013) stalls: the force residual floors at 2e-14 (vacuum) and 7e-14
    # (pressure) over 54000 iterations, and the ratio drifts. Reported, not used.
    if "qh_a15_b2.7e-3_cont14" in rows and "qh_a15_b0_cont14" in rows:
        e = one(entries, "qh_a15_b2.7e-3_cont14")
        stalled = dict(radius_m=0.015, run=e["name"], converged=False, fsqr=rows[e["name"]]["fsqr"],
                       vacuum_fsqr=rows["qh_a15_b0_cont14"]["fsqr"],
                       iterations_total=int(sum(rows[f"qh_a15_b2.7e-3_cont14_hop{h}"]["iterations"] for h in range(1, 7)
                                                if f"qh_a15_b2.7e-3_cont14_hop{h}" in rows)),
                       ratio_last=ratio_with(entries, rows, e["name"], "qh_a15_b0_cont14"),
                       ratio_by_hop=[one(entries, f"qh_a15_b2.7e-3_cont14_hop{h}")["ratio_vs_traced"] for h in range(1, 7)
                                     if one(entries, f"qh_a15_b2.7e-3_cont14_hop{h}")],
                       vacuum_bias_rms_m=one(entries, "qh_a15_b0_cont14")["axis_minus_traced_rms_m"])
    else:
        stalled = None
    a_arr = np.array([r["radius_m"] for r in radius]) / theory["traced_axis_R0_m"]
    r_arr = np.array([r["ratio"] for r in radius])
    extrap = {}
    for label, power in (("linear in a/R0", 1), ("quadratic in a/R0", 2)):
        A = np.stack((np.ones_like(a_arr), a_arr**power), 1)
        coef, *_ = np.linalg.lstsq(A, r_arr, rcond=None)
        extrap[label] = dict(intercept=float(coef[0]), slope=float(coef[1]),
                             max_abs_residual=float(np.max(np.abs(A @ coef - r_arr))))

    budget = dict(
        restart_half_spread=restart_half_spread,
        vacuum_reference_half_spread_at_2p7e_3_not_in_headline=[r for r in ladder if r["beta_axis"] == 2.7e-3][0][
            "vacuum_reference_half_spread"] if any(r["beta_axis"] == 2.7e-3 for r in ladder) else None,
        mode_resolution_shift=resolution_shift,
        theory_operator_traced_vs_ideal_rms=theory["traced_vs_ideal_rms_rel"],
        fit_sigma=offset_fit["k1_fit_sigma"],
        fit_drop_one_half_spread=offset_fit["k1_drop_one_half_spread"],
    )
    parts = [v for k, v in budget.items() if v is not None and not k.endswith("not_in_headline")]
    budget["combined_quadrature"] = float(np.sqrt(np.sum(np.square(parts))))

    results = dict(
        description="Fixed-coil pressure response of the magnetic axis: VMEX free boundary vs first-order theory",
        configuration=dict(case="QH, nfp 4, second-pass coils (examples/second-pass)", coil_sha256=theory["coil_sha256"],
                           traced_axis_R0_m=theory["traced_axis_R0_m"], traced_iota_lab=theory["monodromy_iota_lab_abs"],
                           pressure="p = p0 (1 - s), zero current, PHIEDGE = pi B0 a_b^2, B0 = 1 T"),
        solver=dict(vmex_commit="b68807d8d51ec69d166cd336b1691454fd1c4f2f",
                    settings="NS 17/33/65 cold to FTOL 1e-10, then restart on NS 65 to 1e-14 and 1e-15 (1e-16 "
                             "for the warm checks); MPOL 8, NTOR 8, NZETA 32, DELT 0.5; direct Biot-Savart coils"),
        theory=theory,
        vacuum_bias_by_ftol_ab35=vac35, vacuum_bias_by_radius=vac_radius,
        ladder_ab35=ladder, linear_fit_vacuum_subtracted=linear_fit, headline_fit=offset_fit, restart_dependence_ab35_beta2p7e_3=restart,
        mode_resolution=resolution, radius_dependence=radius, radius_extrapolation=extrap,
        radius_15mm_not_converged=stalled,
        uncertainty_budget_ab35=budget,
        all_runs=[{k: v for k, v in e.items()} for e in entries],
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(dict(linear_fit=linear_fit, headline=offset_fit, budget=budget, radius=radius, extrap=extrap,
                          restart={k: v[-1] for k, v in restart.items() if v}, resolution=resolution), indent=1))
    figures(results, entries, rows)


def figures(results, entries, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(HERE.parent.parent / "paper.mplstyle")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    th = np.load(RUNS / "theory" / "theory_qh.npz")
    Xrms = results["theory"]["displacement_per_unit_p0_m_per_Pa"]["traced_rms"]
    R0 = results["theory"]["traced_axis_R0_m"]
    mu0 = 4e-7 * np.pi
    ab = 0.035
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))

    # (a) signal and vacuum-axis bias against beta
    ax = axes[0, 0]
    b = np.logspace(-3.4, -1.85, 50)
    ax.loglog(b, Xrms * b / (2 * mu0) / ab, color="0.2", lw=1.2, label="first-order theory")
    for ftol, fill, label in ((1e-10, "none", "VMEX, FTOL 1e-10"), (1e-15, colors[0], "VMEX, FTOL 1e-15")):
        pts = sorted((e["beta"], e["ratio"] * e["theory_rms_m"] / ab) for e in entries
                     if e["radius_m"] == ab and e["beta"] > 0 and e["mpol"] == 8 and e["ns"][-1] == 65
                     and e["ftol_final"] == ftol and e.get("ratio") and "warm" not in e["name"])
        if pts:
            pts = np.array(pts)
            ax.loglog(pts[:, 0], pts[:, 1], "o", mfc=fill, color=colors[0], label=label)
    for ftol, ls in ((1e-10, ":"), (1e-15, "--")):
        vals = [v["bias_rms_over_ab"] for v in results["vacuum_bias_by_ftol_ab35"]
                if v["ftol"] == ftol and v["mpol"] == 8 and v["ns"] == 65]
        if vals:
            ax.axhline(np.mean(vals), color=colors[1], ls=ls, lw=1.0, label=f"vacuum-axis bias, {ftol:.0e}")
    ax.set_xlabel(r"on-axis $\beta$")
    ax.set_ylabel(r"axis displacement (rms) / $a_b$")
    ax.set_title(r"(a) QH, $a_b$ = 35 mm", loc="left")
    ax.set_ylim(3e-4, 0.3)
    ax.legend(loc="upper left", fontsize=5.8)

    # (b) displacement profile at beta = 5.4e-3
    ax = axes[0, 1]
    name, vac = "qh_a35_b5.4e-3_cont15", "qh_a35_b0_cont15"
    r, v = rows[name], rows[vac]
    phi = th["phi"] * 4 / (2 * np.pi)
    xi = r["p0_Pa"] * th["traced"]
    d = np.array([r["axis_R"], r["axis_Z"]]) - np.array([v["axis_R"], v["axis_Z"]])
    for k, (lab, c) in enumerate((("R", colors[0]), ("Z", colors[2]))):
        ax.plot(phi, xi[k] / ab, color=c, label=rf"$\delta {lab}$ theory")
        ax.plot(phi[::4], d[k][::4] / ab, "o", color=c, ms=3.0, mfc="none", label=rf"$\delta {lab}$ VMEX")
    ax.set_xlabel(r"$n_{fp}\phi / 2\pi$")
    ax.set_ylabel(r"displacement / $a_b$")
    ax.set_title(r"(b) $\beta$ = 5.4e-3, $a_b$ = 35 mm", loc="left")
    ax.set_ylim(-0.02, 0.05)
    ax.legend(ncol=2, loc="upper center", fontsize=6.0)

    # (c) vacuum-axis bias against force tolerance
    ax = axes[0, 2]
    for k, (label, sel) in enumerate((("NS 65, MPOL 8", lambda v: v["ns"] == 65 and v["mpol"] == 8),
                                      ("NS 65, MPOL 12", lambda v: v["ns"] == 65 and v["mpol"] == 12),
                                      ("NS 129, MPOL 8", lambda v: v["ns"] == 129))):
        pts = {}
        for v in results["vacuum_bias_by_ftol_ab35"]:
            if sel(v) and "hop" not in v["run"]:
                pts.setdefault(v["ftol"], []).append(v["bias_rms_over_ab"])
        if pts:
            f = sorted(pts, reverse=True)
            ax.loglog(f, [np.mean(pts[x]) for x in f], "o-", color=colors[k], ms=3.2, label=label)
    st = results.get("radius_15mm_not_converged")
    if st:
        ax.loglog([1e-14], [st["vacuum_bias_rms_m"] / 0.015], "x", color=colors[3], ms=5,
                  label=r"$a_b$ = 15 mm (stalls)")
    ax.set_xlabel("final force tolerance FTOL")
    ax.set_ylabel(r"vacuum-axis bias (rms) / $a_b$")
    ax.set_title("(c) vacuum axis vs traced", loc="left")
    ax.invert_xaxis()
    ax.set_ylim(None, 0.02)
    ax.legend(loc="upper right", fontsize=6.0)

    # (d) restart dependence against force tolerance
    ax = axes[1, 0]
    for k, (label, series) in enumerate(results["restart_dependence_ab35_beta2p7e_3"].items()):
        pts = np.array([(x["ftol"], x["ratio"]) for x in series if x["ratio"] is not None])
        if len(pts) > 1:
            ax.semilogx(pts[:, 0], pts[:, 1], "o-", color=colors[k], label=label, ms=3.2)
    ax.axhline(1.0, color="0.2", lw=1.0)
    ax.set_xlabel("final force tolerance FTOL")
    ax.set_ylabel("VMEX / theory")
    ax.set_title(r"(d) restart policy, $\beta$ = 2.7e-3", loc="left")
    ax.invert_xaxis()
    ax.legend(loc="lower right", fontsize=6.2)

    # (e) linearity in beta
    ax = axes[1, 1]
    lad = results["ladder_ab35"]
    hf = results["headline_fit"]
    bl = np.array([x["beta_axis"] for x in lad])
    ax.plot(bl * 1e3, [x["ratio"] for x in lad], "o", color=colors[0], label="vacuum run subtracted")
    ax.plot(bl * 1e3, [x["ratio_vs_traced_axis"] - hf["c0"] / x["beta_axis"] for x in lad], "s", mfc="none",
            color=colors[1], label="traced axis, fitted offset")
    bb = np.linspace(0, bl.max(), 50)
    ax.plot(bb * 1e3, hf["k1"] + hf["k2"] * bb, color=colors[1], lw=1.0)
    ax.axhline(1.0, color="0.2", lw=1.0)
    ax.set_xlabel(r"on-axis $\beta$ (10$^{-3}$)")
    ax.set_ylabel("VMEX / theory")
    ax.set_title(r"(e) linearity, $a_b$ = 35 mm", loc="left")
    ax.set_ylim(0.97, 1.08)
    ax.legend(loc="lower right", fontsize=6.2)

    # (f) boundary radius
    ax = axes[1, 2]
    rad = results["radius_dependence"]
    x = np.array([q["radius_m"] for q in rad]) / R0
    ax.plot(x, [q["ratio"] for q in rad], "o", color=colors[0], label="VMEX, FTOL 1e-15")
    xx = np.linspace(0, x.max() * 1.05, 50)
    for k, (label, fit) in enumerate(results["radius_extrapolation"].items()):
        power = 1 if label.startswith("linear") else 2
        ax.plot(xx, fit["intercept"] + fit["slope"] * xx**power, ls="--" if power == 1 else ":",
                color=colors[1 + k], lw=1.0, label=f"{label}: {fit['intercept']:.3f}")
    ax.axhline(1.0, color="0.2", lw=1.0)
    ax.set_xlabel(r"$a_b / R_0$")
    ax.set_ylabel("VMEX / theory")
    ax.set_title("(f) boundary radius", loc="left")
    ax.set_xlim(0, None)
    ax.legend(loc="upper left", fontsize=6.2)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"qh_fixed_coil_pressure_response.{ext}")


if __name__ == "__main__":
    main()
