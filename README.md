# plasma-coil-fields

[![License](https://img.shields.io/github/license/rogeriojorge/plasma-coil-fields)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![ESSOS](https://img.shields.io/badge/ESSOS-PR%20%2370-informational)](https://github.com/uwplasma/ESSOS/pull/70)
[![pyQSC_JAX](https://img.shields.io/badge/pyQSC__JAX-PR%20%232-informational)](https://github.com/uwplasma/pyQSC_JAX/pull/2)

Plasma self-fields and pressure response in stellarator coil design.

At finite pressure, the coils of a stellarator do not have to produce the
equilibrium magnetic field. They have to produce the equilibrium field
**minus the field of the plasma's own currents**. Near a quasisymmetric
magnetic axis that plasma field, together with its gradient and Hessian, has
a closed form. Coils can then be designed against the external field
directly, without a plasma boundary or a virtual-casing integral. This
repository holds the study built on that idea: finite-beta coil designs
checked with free-boundary equilibria, ablations, the fixed-coil pressure
displacement of the magnetic axis, and the verification of each piece.

![Finite-pressure QA coils fitted to the external field: normal-field error on the plasma boundary before and after optimization](docs/figures/01_qa_coils_and_normal_field.png)

*Figure 1. Coils for a finite-pressure quasi-axisymmetric stellarator (nfp = 2,
a = 30 mm, p₂ = −6×10⁵ Pa/m², no net current), optimized together with the
near-axis equilibrium against the external field. Color: the normal field of
the coils minus the external target on the plasma boundary. The maximum drops
from 28.6% to 0.62% of |B| (RMS from 15.3% to 0.059%).*

**Contents:**
[1 Plasma field and coil target](#1-the-plasma-field-and-the-coil-target) ·
[2 Finite-pressure design](#2-a-finite-pressure-design-checked-at-free-boundary) ·
[3 Ablations](#3-ablations-what-each-ingredient-buys) ·
[4 Vacuum limit](#4-the-vacuum-limit) ·
[5 Pressure response](#5-fixed-coil-pressure-response-of-the-magnetic-axis) ·
[6 Fractional bootstrap](#6-fractional-bootstrap-current-near-the-axis) ·
[7 QH, hybrid, single stage](#7-qh-hybrid-and-single-stage-examples) ·
[Status](#status) · [Reproduce](#reproducing-the-results) · [Layout](#repository-layout)

## 1. The plasma field and the coil target

On the magnetic axis the coils must supply

```text
B_coils = B_total − B_plasma,   ∇B_coils = ∇B_total − ∇B_plasma,   ∇∇B_coils = ∇∇B_total − ∇∇B_plasma
```

`B_plasma` is the free-space field of the Pfirsch–Schlüter, diamagnetic and
net currents inside a flux radius a. pyQSC_JAX evaluates it by matched
asymptotics from the near-axis solution. The result is a
divergence- and curl-free external target: 3 field, 5 gradient and
7 Hessian components per axis point. ESSOS fits coils to it with
`near_axis_coil_residuals`.

The Hessian matters most. The plasma part of the field is O(a²), but its
Hessian is order one: 13.9 of 14.6 T/m² here. Coils fitted to the total
field reproduce the wrong plasma shaping.

![Plasma field on the axis and the residual coil mismatch](docs/figures/02_qa_axis_fields.png)

*Figure 2. Left: plasma field on the axis in the Frenet frame, 2.6 mT RMS
against B₀ = 1 T. Right: the optimized coils minus the external target,
0.12 mT RMS, 21 times below the plasma field they leave out.*

## 2. A finite-pressure design, checked at free boundary

The coils of Fig. 1 were given to VMEX as a fixed external field. VMEX
finds, with no knowledge of the near-axis target, the free-boundary
equilibrium those coils hold at the same pressure and flux.

![Near-axis and free-boundary flux surfaces in four toroidal planes](docs/figures/03_qa_cross_sections.png)

*Figure 3. Flux surfaces s = 1/16, 1/4, 9/16, 1: near-axis design (black) and
free-boundary VMEX equilibria in the optimized coils, with the coil field
evaluated directly (dashed) or through a MAKEGRID file (dotted).*

| Quantity (a = 30 mm) | Value |
|---|---|
| volume-averaged β | 6.75×10⁻⁴ |
| transform, VMEX / near-axis | −0.2099 / −0.2125 |
| boundary shape discrepancy (axis-registered, RMS) | 2.2% of a |
| raw axis offset | 5.0% of a (1.5 mm) |
| max tangential jump at the plasma–vacuum interface | 0.96% of B |
| direct coils vs MAKEGRID route | ≤ 6.5×10⁻⁴ of a |

Resolution ladder (NS 33/65/129, modes 8/10, 32/64 toroidal points,
FTOL 10⁻⁹…3×10⁻¹¹, 240/480 coil points, three field tables): the shape
discrepancy is 2.16–2.32% of a, the axis offset 4.2–5.3%, and the transform
0.2055–0.2132.

## 3. Ablations: what each ingredient buys

Same coil count, order and pressure. Only the target changes.

| Target | Boundary shape | Axis offset | Interface jump | Transform (VMEX) |
|---|---|---|---|---|
| external field, gradient and Hessian (Fig. 3) | **2.2% a** | **5.0% a** | **0.96%** | −0.210 |
| Hessian omitted | 4.8% a | 6.1% a | 4.9% | −0.209 |
| total field (no plasma subtraction) | 8.5% a | 59% a | 2.7% | −0.286 |

| Hessian omitted | Total-field target |
|---|---|
| ![Cross sections without the Hessian target](docs/figures/05_no_hessian_cross_sections.png) | ![Cross sections of the total-field control](docs/figures/04_control_cross_sections.png) |

*Figure 4. Left: without the Hessian residual the coils reproduce the ellipse
but not the triangular shaping. Right: fitting the total field instead of the
external field moves the equilibrium off the design by more than half a
minor radius.*

## 4. The vacuum limit

With zero pressure the coil field is the total field, so field-line tracing
is an exact check.

![Poincaré sections of the vacuum coils against the near-axis surfaces](docs/figures/06_vacuum_poincare.png)

*Figure 5. Field lines of the vacuum coils against the near-axis surfaces. The
surfaces are confined out to 0.91 a. The traced transform is −0.2076, against
−0.2125 near-axis. VMEX and VMEC2000 both give −0.188 to −0.197 on the same
deck; that 5–10% gap is common to VMEC-type solvers and not yet explained.*

## 5. Fixed-coil pressure response of the magnetic axis

Adding pressure at fixed coils moves the axis. To first order, the closed
field line of B_coils + α B_plasma is displaced by the periodic solution of
the linearized field-line equation. The forcing is the on-axis plasma field
of Sec. 1. ESSOS's `pressure_axis_response` solves it for a current-free QS
axis. It returns the displacement in fixed cylindrical planes and the
first variation of axis length, which is positive on this branch.

![Normal and binormal axis displacement per unit pressure amplitude](docs/figures/10_linear_response.png)

*Figure 6. First-order axis displacement per unit α for the QA axis (a = 30 mm, p₂ = −6×10⁵ Pa/m², axis β 1.4×10⁻³),
in the Frenet frame. The closed form and an independent monodromy solve
agree to 2.4×10⁻⁷, and the length slope matches its analytic formula to
2.4×10⁻⁷.*

Every piece of this prediction is verified independently of any MHD solver:

![Source and closed-axis operator verification](docs/figures/07_source_and_operator_verification.png)

*Figure 7. (a) Biot–Savart of the near-axis current distribution converges
to the closed-form on-axis field as nφ⁻²; after Richardson extrapolation
the difference is 6.8×10⁻⁴. (b) A known vertical field added to the fitted
coils: the exact linear response on the traced axis agrees with nonlinear
tracing to 1.8×10⁻⁸. The near-axis Frenet operator agrees to 0.22%, the size
of the 82 µm offset between the ideal and traced axes.*

Free-boundary equilibria do **not** resolve this response at a = 17.8 mm and
β ~ 10⁻⁴. The predicted shift at the linearity-screen amplitude α_max is
0.52 mm:

![Fixed-coil pressure response from VMEX and VMEC2000](docs/figures/08_fixed_coil_pressure_response.png)

*Figure 8. (a) Axis shift against pressure. The untuned theory and the coils
traced with a frozen first-order current (linear to 2% up to 4 α_max) are
compared with free-boundary VMEC2000 (NS 65, 129) and VMEX (cold start).
(b) Response ratio at α_max against radial resolution and restart policy.
(c) Offset of the free-boundary vacuum axis from the directly traced coil
axis. Open circles: VMEC2000 with capped iteration counts. Diamond:
fixed-boundary solve. Dashed line: the predicted shift.*

| δR/δR_theory at α_max | NS 33 | NS 65 | NS 129 | NS 257 |
|---|---|---|---|---|
| VMEC2000 free, cold | 0.042 | 0.068 | 0.071 | 0.115 |
| VMEX free, warm continuation | – | −0.029 | −0.005 | – |
| VMEX free, cold | – | 0.020 | 0.023 | – |
| **vacuum-axis offset from traced coil axis (µm), VMEC2000 / VMEX** | 536 / – | 476 / 688 | 479 / 658 | 475 / – |

The sign depends on the restart policy. VMEC2000's response is non-analytic
in α and grows with its iteration count (1171 → 5165 from α_max to
12 α_max). With the iteration count capped at 1k/4k/16k/64k, the vacuum axis
wanders by 200 µm and the ratio takes the values 0.10, 0.40, 0.12 and 0.34,
all at F_sq ≤ 3×10⁻¹².

Every free-boundary vacuum axis is 0.47–0.69 mm from the traced coil axis,
as large as the signal itself. Fixed-boundary is within 33 µm; MPOL 16/NTOR 12
and a four-times finer MAKEGRID change nothing. The axis is a soft mode that
descent solvers stop before relaxing, so these equilibria cannot give the
derivative at this β. Nothing here contradicts the first-order theory, and
no full-equilibrium calculation here confirms it.

![Field of the equilibria's own current and the local force floor](docs/figures/09_equilibrium_current_diagnostic.png)

*Figure 9. (a) Vertical plasma field on the vacuum axis per unit α. The
near-axis current gives +9.3×10⁻⁴ T; the solvers' own currents (Ampère's
law on the WOUT field, differenced against their vacuum) give −0.1 to
−1.4×10⁻⁴ T. (b) The angle-resolved m = 1 radial-force residual of the vacuum
solutions (5–15 Pa) exceeds the Pfirsch–Schlüter force they must resolve
(3.3 Pa). Surface-averaged force balance holds to ≤1.3%.*

Details, all numbers and the reproduction commands:
[`shafranov_shift/results/verification/README.md`](shafranov_shift/results/verification/README.md).
A drop-in manuscript section is in
[`pressure_scan_status_revised.tex`](shafranov_shift/results/verification/pressure_scan_status_revised.tex).

## 6. Fractional bootstrap current near the axis

This is a separate, reduced problem. A collisionless bootstrap current
j∥ ∝ λ½ r^½ near a QS axis produces a non-integer transverse field
Ψ ∝ ρ^{5/2} f(γ). The mean transform coefficient is independent of the
first-order ellipse:

```text
mean_θ P = 4/25   and   ι½ = (2/5) ℓ λ½
```

The surface correction does depend on the ellipse, because P(θ) does.

![Fractional source and transform check](docs/figures/11_fractional_bootstrap.png)

*Figure 10. Left: the angular source P(θ, φ = 0) for a circular section, a
fixed ellipse and the nonplanar QS reference; all three average to exactly
4/25. Right: direct 64-period integration of the non-averaged field-line
Hamiltonian recovers ι-coefficient 0.4 with error ∝ ε_b (9.4×10⁻⁵ and
1.9×10⁻⁵ at ε_b = 5×10⁻⁴). This verifies the reduced transverse problem. It is
not a full MHD equilibrium or a kinetic closure.*

## 7. QH, hybrid and single-stage examples

Three further designs, run with the drivers in `drivers/` and VMEX main
`204c8a9c` (0.11.4). All numbers are in [`examples_results.json`](examples_results.json),
written by `collect_examples.py` from the run directories (WOUT files are
not tracked; their sha256 is recorded there). The first-pass results are kept
unchanged under `first_pass` in the same file. Coil metrics are evaluated at
480 points per coil. Free-boundary equilibria use the direct coil field.

**Second pass: coil limits.** The first pass hinged length and curvature only at the
Biot-Savart quadrature points and had no distance terms. A kink grew between the samples of a
single-stage coil (curvature 9.8 m⁻¹ at 60 points, 7778 m⁻¹ at 480), and the QH and hybrid coils
passed 4–7 mm from each other. Now length, curvature, mean-squared curvature (MSC) and
arclength variation are evaluated on 16 points per Fourier order. Coil–coil and coil–plasma
distance terms were added. The QH and hybrid fits run 1600 evaluations with soft limits
(weight 1), then 400 at weight 100, then 400 more at weight 100 with the distance terms at 24
points per order. The earlier attempts with near-hard limits from the start stalled and are
recorded under `abandoned_runs`.

**QH boundary error (the remedy).** In the first pass, B·n/|B| on r = a rose from 1.3% to
4.2% while the axis match improved. That fit had no coil–plasma distance term: the coils ended
8 mm from each other and 83 mm from the plasma, where the field content beyond the fitted
quadratic jet is large. The remedy stays within the method: stronger coil regularization
(coil–plasma ≥ 0.10 m, coil–coil ≥ 0.05 m, MSC ≤ 50 m⁻², curvature evaluated at 16 points per
order). The objective is still the axis jet only, and B·n is only a diagnostic
(`trajectory` in the JSON, one row per segment). With the regularization, max B·n/|B| drops to
0.77% by 900 evaluations and stays within 0.77–0.79% to 2000. It is 0.68% after the limit phase.

| | QH (nfp 4, a = 35 mm, p₂ = −1.76×10⁶ Pa/m²) | Hybrid (nfp 3, a = 40 mm, I₂ = 0.4 T/m, p₂ = −6×10⁵ Pa/m²) |
|---|---|---|
| coils − target on axis: field / B₀, gradient R₀/B₀, Hessian R₀²/B₀ | 3.4×10⁻⁴ / 4.3×10⁻³ / 0.28 (target 39) | 7.2×10⁻⁴ / 2.3×10⁻³ / 0.16 (target 27) |
| B·n/\|B\| on r = a, max / RMS (first pass) | 0.68% / 0.16% (4.19% / 0.55%) | 1.29% / 0.21% (1.05% / 0.16%) |
| coil length, curvature max (limit), MSC max | 3.63 m, 12.08 m⁻¹ (12), 49.2 m⁻² | 3.49 m, 12.13 m⁻¹ (12), 49.2 m⁻² |
| min coil–coil / coil–plasma (limits 0.05 / 0.10 m; first pass) | 0.050 / 0.131 m (0.004 / 0.083) | 0.050 / 0.100 m (0.007 / 0.092) |
| ι lab, VMEX / near axis (a_b = a) | 1.1396 / 1.1441 (first pass 1.118) | −0.708 / −0.713 |
| free boundary converged (NS 65, FTOL 10⁻¹⁰) at a_b = | a, 1.25a, 1.5a, 2a | a, 1.25a |
| axis offset max / a_b, at a, 1.25a, 1.5a, 2a | 0.7%, 1.9%, 3.3%, 5.3% | 6.1%, 29% |
| LCFS shape RMS / a_b, same | 0.57%, 0.89%, 1.5%, 6.3% (first pass 5.3, 9.2, 13, –) | 3.3%, 9.2% |
| tangential interface jump max / \|B\|, same | 0.7%, 0.9%, 1.0%, 1.6% | 0.9%, 1.3% |

The QH free-boundary agreement is an order of magnitude better than in the first pass, and the
full ladder converges out to 2a. The hybrid matches the first pass. At 1.5a it stalls at
1×10⁻⁸ (NS 33, FTOL 10⁻⁹, DELT 0.25) and at 3×10⁻⁶ (NS 65). At 2a it stalls at 2×10⁻⁷. The coils
satisfy their limits at 480 points (curvature 1% over through the smooth hinge), but they are
wavy (`figures/examples/*_coils_and_normal_field.png`), because a curvature limit of 12 m⁻¹ on 3.5 m coils still allows that.

![Axis offset, boundary shape and transform against boundary radius](figures/examples/qh_hybrid_boundary_scan.png)

*Figure 11. Converged free-boundary equilibria of the QH and hybrid coils against the boundary radius.*

**Single stage (nfp 2, a = 0.1 m, near-axis ⟨β⟩ = μ₀p₀/B₀² = 3.00%).** Axis harmonics up to n = 6
are free. Mercier stability, D_Merc r² ≥ 0, cannot be met together with quasisymmetry in this
family at 3% β (`single_stage_mercier_scan.py`/`.json`: a hard Mercier hinge raises the B₂₀
variation from 4×10⁻⁴ to 0.47 at a = 0.1 m and breaks the r_singularity, elongation and ι
limits). At a = 0.15 m it is compatible, but the r = a surface then reaches 0.56 m from the axis and
folds (run abandoned). The design therefore keeps a magnetic-well margin D_Well r² ≥ 1 as a
near-hard hinge, plus a weak Mercier term:

- ⟨β⟩ 3.00%, ι 0.474, B₂₀ variation 0.0165 B₀/R₀² (first pass 0.016), r_singularity 1.51a, elongation 6.00;
- D_Well r² = 1.17; **D_Merc r² = −0.97** (first pass −2.35): still Mercier-unstable near the axis;
- coils at 480 points: length 4.91 m (limit 5), curvature 6.00 m⁻¹ (limit 6), MSC 9.6 m⁻² (limit 10), coil–plasma
  0.197 m (limit 0.2), coil–coil **0.083 m** (limit 0.1; the driver's distance term uses the 60 Biot–Savart points);
- B·n/|B| on r = a: max 21.7%, RMS 3.0% (first pass 7.1% / 1.3%). The coils are loopy (`single_stage_coils_and_normal_field.png`);
- free boundary: NS 17 converged at FTOL 10⁻¹⁰. At NS 33 and 65, FTOL 10⁻¹⁰ stalls (5×10⁻¹⁰–1.5×10⁻⁹, with ι
  drifting at NS 33), so they converged at FTOL 10⁻⁹. DELT 0.25 was not needed. VMEX betatotal 2.91%;
- on-axis ι: 0.500 / 0.475 / 0.457 at NS 17 / 33 / 65, against 0.474 near axis (first pass 0.556–0.617 against 0.417).
  It is **not resolution-converged**: it falls by about 0.02 per NS doubling;
- axis offset 2.6 / 1.6 / 1.3% of a (first pass 19%); LCFS shape 18–20% of a; tangential jump 13%.

![Single-stage transform against radial resolution](figures/examples/single_stage_iota_ladder.png)

*Figure 12. On-axis transform of the single-stage free-boundary equilibria against NS.*

## Status

| Result | State |
|---|---|
| Plasma field, gradient, Hessian on axis (pyQSC_JAX) | verified: volume Biot–Savart, fixed-Cartesian differences, 7 cases, 4 orientations (`validation/`) |
| External-field coil design, QA a = 30 mm | done: resolution ladder, direct and MAKEGRID routes, three ablations |
| Vacuum limit | done; the 5–10% VMEC-type transform gap is unexplained |
| First-order pressure response (theory, source, operator) | verified independently of MHD solvers |
| Fixed-coil pressure derivative from free-boundary equilibria | **unresolved**: solver path-dependent; vacuum-axis bias ≥ signal |
| QH (a = 35 mm) and stellarator–tokamak hybrid | second pass: coils within length, curvature and distance limits at 480 points; QH free boundary converged to 2a with LCFS shape 0.6% at a; hybrid converged at a and 1.25a only ([Sec. 7](#7-qh-hybrid-and-single-stage-examples)) |
| Single-stage 3% β design (a = 0.1 m) | second pass: no coil kink, well margin met, **Mercier unstable** (D_Merc r² −0.97, incompatible with QS here); coil–coil 0.083 m < 0.1; B·n max 22%; ι 0.50→0.46 with NS vs 0.47 near axis |
| Matched design and end-to-end timing comparison | not started; needs a resolved pressure derivative |

## Reproducing the results

The solvers live in their own packages. This repository has the study
drivers, compact outputs (hashes of every native WOUT) and figures.

| Package | Branch / version | Provides |
|---|---|---|
| [ESSOS](https://github.com/uwplasma/ESSOS) | `plasma-coil` ([PR #70](https://github.com/uwplasma/ESSOS/pull/70)) | coils, Biot–Savart, `near_axis_coil_targets`, `near_axis_coil_residuals`, `pressure_axis_response` |
| [pyQSC_JAX](https://github.com/uwplasma/pyQSC_JAX) | `refactor/pyqsc-jax-complete` ([PR #2](https://github.com/uwplasma/pyQSC_JAX/pull/2)) | near-axis equilibria and `pyqsc_jax.plasma` |
| [VMEX](https://github.com/uwplasma/VMEX) | v0.11.2 (`926892ab`); Sec. 7: 0.11.4 (`204c8a9c`) | free-boundary equilibria |
| VMEC2000 (optional) | STELLOPT `ee175502` | independent free-boundary check |

```bash
git clone -b plasma-coil https://github.com/uwplasma/ESSOS.git
git clone -b refactor/pyqsc-jax-complete https://github.com/uwplasma/pyQSC_JAX.git
git clone https://github.com/uwplasma/VMEX.git && git -C VMEX checkout 926892ab
git clone https://github.com/rogeriojorge/plasma-coil-fields.git && cd plasma-coil-fields
export PYTHONPATH=$PWD/../ESSOS:$PWD/../pyQSC_JAX/src:$PWD/../VMEX JAX_ENABLE_X64=1
```

```bash
pytest                                                        # plasma-field references and study checks (PYQSC_RUN_VOLUME_SWEEP=1 adds the full sweep)
python shafranov_shift/axis_operator_check.py --output runs/axis_operator_check.json   # Fig. 7b, ~4 min
python shafranov_shift/scan.py --reference shafranov_shift/reference/vacuum_fitted_reference.json \
  --output runs/qa18 --radius 0.018 --segments 480 --mpol 10 --ntor 10 --run-vmex --pressure-scan
python shafranov_shift/make_verification_figures.py           # Figs. 7-9 from the tracked JSON
cd independent_checks && python independent_checks.py && python fractional_bootstrap.py   # Figs. 6, 10
```

The Sec. 2–4 designs come from `drivers/optimize_coils_and_nearaxis_finite_beta.py`
through `run.py` (overrides), `jobs.sh` and the `*.jobs` lists. The recorded
runs are in `results.json` and the method is described in
[`archive_report.md`](archive_report.md). The full native archive (44 runs,
201 MB) is to be deposited with the paper.

## Repository layout

| Path | Contents |
|---|---|
| `docs/figures/` | the README and paper figures (compressed PNG) |
| `drivers/` | full optimization drivers (QA, QH, hybrid, single stage) and `nearaxis_finite_beta_helpers.py` (VMEX/MAKEGRID benchmarks, diagnostics, plots) |
| `run.py`, `jobs.sh`, `*.jobs`, `segments*.sh`, `adopt.py` | run orchestration for the archived designs |
| `collect_examples.py`, `examples_results.json`, `figures/examples/` | Sec. 7 results and figures |
| `make_tables.py`, `make_figures.py`, `replot.py`, `results.json`, `tables.tex` | tables and figures of Secs. 2–4 |
| `figures/` | full-resolution design figures, all cases |
| `investigation/` | vacuum-transform investigation and single-stage records |
| `shafranov_shift/` | pressure-response scan, reference coils, verification campaign (`results/verification/`) |
| `validation/` | heavy pyQSC_JAX plasma-field references: volume Biot–Savart, fixed-Cartesian derivatives, axis integral, Hessian identity |
| `independent_checks/` | NumPy/SciPy/SymPy checks with no repository imports, and reproduction logs |
| `tests/` | study checks: VMEX pressure-family deck, circular-tokamak reference |
| `paper.mplstyle` | shared figure style |

## License

MIT. The material was developed in the ESSOS repository and moved here
with its history.
