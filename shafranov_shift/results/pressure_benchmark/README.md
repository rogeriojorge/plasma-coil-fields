# Fixed-coil pressure response: a resolved free-boundary benchmark (QH, September 2026)

The verification campaign (`../verification/README.md`) showed that at a = 18 mm and
β ~ 10⁻⁴ the QA free-boundary equilibria could not measure the first-order axis
displacement: every vacuum axis sat 0.47–0.69 mm from the directly traced coil axis, and the
answer depended on restarts and iteration counts. This folder is the benchmark that does
resolve it, on a different configuration and with a different force tolerance.

**Result.** For the four-period QH coils of Sec. 6 (a_b = 35 mm, a_b/R₀ = 0.029, ι = 1.14),
the VMEX free-boundary axis displacement per unit pressure is

    k₁ = VMEX / first-order theory = 1.037 ± 0.022

over a 16× range of on-axis β (6.75×10⁻⁴ to 1.08×10⁻²; displacement 0.26% to 4.1% of a_b).
The uncertainty is measured (budget below). The remaining 3.7% is not noise: it grows with the
boundary radius (1.017, 1.042, 1.061 at a_b/R₀ = 0.021, 0.029, 0.038) and extrapolates to
0.96 (linear in a_b) or 1.00 (quadratic in a_b) at a_b → 0, where the first-order near-axis
theory applies. The theory is therefore confirmed to within 4% by a full free-boundary
equilibrium, with a finite-a_b correction of +2% to +6% at the radii that converge.

![Fixed-coil pressure response of the QH axis](figures/qh_fixed_coil_pressure_response.png)

*(a) rms axis displacement / a_b against on-axis β: first-order theory (line), VMEX at the
standard FTOL 1e-10 (open) and continued to 1e-15 (filled), and the vacuum-axis bias at the
two tolerances (dotted, dashed). (b) δR and δZ against the toroidal angle at β = 5.4×10⁻³.
(c) Vacuum-axis bias against the traced coil axis vs the final force tolerance. (d) Response
ratio at β = 2.7×10⁻³ for three restart policies vs force tolerance. (e) Response ratio across
the β ladder: vacuum-subtracted (circles) and from the traced axis with a fitted offset
(squares, line = quadratic fit). (f) Response ratio against a_b/R₀ with linear and quadratic
extrapolations.*

## What controls signal and bias

- **Signal.** At fixed coils and profile shape p = p₀(1 − s), the first-order on-axis plasma
  field is proportional to p₂a_b² = −p₀, so the displacement depends on the on-axis β only,
  not on a_b (`pb.py theory`; checked: `pressure_axis_response` ∝ radius² at fixed p₂ to
  machine precision). In relative units, δ/a_b ∝ β/a_b.
- **Bias.** The vacuum-axis offset is a force-tolerance effect. The axis position is the
  softest direction of the free-boundary energy. VMEX's default ladder stops at FTOL 1e-10
  with the axis 278 µm (0.8% a_b) from the traced coil axis, and the pressure response is
  then only 0.73–0.77 of theory at every β. Tightening FTOL lowers the bias to 58 µm
  (1e-13) and 10–23 µm (1e-14 to 1e-16), and the ratio moves to 1.04 and stays there.
  More modes or radial surfaces do not do this: MPOL/NTOR 12 and NS 129 at FTOL 1e-10/1e-11
  leave 191–272 µm.
- **Stiffness and aspect ratio.** The QH axis (ι = 1.14, A ≈ 34) reaches FTOL 1e-16 in continuation
  jobs of under 10 minutes each. At a_b = 15 mm (A ≈ 80) the residual floors at 2×10⁻¹⁴ (vacuum) and
  7×10⁻¹⁴ (β = 2.7×10⁻³) over 54 000 iterations; the ratio drifts (0.73 → 0.69) and that
  radius is not used. The QA (ι = 0.21, A ≈ 61) behaves the same way: continued from its
  FTOL 1e-10 solve (vacuum axis 2.7 mm rms from the traced axis with these settings, PHIEDGE
  = π B₀ a², MPOL 8), six 9 000-iteration jobs bring the residual to 9.6×10⁻¹⁴ and then let it
  rise to 4.5×10⁻¹³, while the axis creeps to 1.45 mm and stops (`qa_a178_b0_cont14_hop*`).
  The earlier QA campaign was limited by exactly this: stiffness × tolerance, not the
  first-order source or operator.

## Protocol that resolves it

1. Cold VMEX ladder NS 17/33/65, FTOL 1e-8/1e-9/1e-10, MPOL = NTOR = 8, NZETA 32, DELT 0.5,
   direct Biot–Savart coils (480 segments), PHIEDGE = π B₀ a_b², p = p₀(1 − s), AC = 0,
   CURTOR = 0.
2. Restart from that WOUT on NS 65 to FTOL 1e-14, then again to 1e-15 (each a separate job
   under 600 s; `hop.sh`-style continuation when a job would exceed it).
3. Same protocol for β = 0. Trace the closed vacuum field line of the coils directly.
4. Theory: the periodic linear response of the traced closed field line to the near-axis
   plasma field per unit p₀ (`pb.py theory`), so xi = p₀ X. The ideal near-axis Frenet
   operator differs from it by 1.5% rms (the QH coils were fitted at finite β, so their
   vacuum axis is 0.87 mm from the near-axis curve).
5. Observable: the displacement of the VMEX axis on 101 toroidal planes per field period,
   projected on the theory shape, ratio = ⟨d, xi⟩/⟨xi, xi⟩, with the orthogonal residual kept.
   Headline fit: d measured from the traced coil axis, ratio·β = c₀ + k₁β + k₂β² over the
   ladder; c₀ absorbs the constant part of the vacuum bias (it corresponds to −9 µm).

## Numbers (a_b = 35 mm, NS 65, MPOL 8, FTOL 1e-15)

| on-axis β | theory rms (µm) | theory / a_b | VMEX/theory (vacuum run subtracted) | from traced axis | shape residual | iterations |
|---|---|---|---|---|---|---|
| 6.75e-4 | 90 | 0.26% | 1.037 | 0.940 | 4.5% | 6600 |
| 1.35e-3 | 180 | 0.51% | 1.039 | 0.990 | 4.6% | 6651 |
| 2.7e-3 | 360 | 1.03% | 1.042 | 1.017 | 4.7% | 7304 |
| 5.4e-3 | 721 | 2.06% | 1.047 | 1.035 | 5.0% | 10971 |
| 1.08e-2 | 1441 | 4.12% | 1.057 | 1.051 | 5.4% | 20614 |

Fits: k₁ = 1.0375 ± 0.0004 (traced axis, free offset; drop-one spread ±0.0009) and 1.0368
(vacuum-subtracted, zero offset). The quadratic term is 1.9% of the linear one at the largest
β, so the whole ladder is in the linear window. The traced-axis column at low β shows the
residual 20 µm vacuum bias directly (−6% at 90 µm of signal); the fit's offset removes it.

Vacuum-axis bias (rms, VMEX vacuum axis − traced coil axis):

| FTOL | 1e-10 | 1e-13 | 1e-14 | 1e-15 | 1e-16 |
|---|---|---|---|---|---|
| NS 65, MPOL 8 | 278 µm | 58 µm | 10.3–10.5 µm | 20.7–20.8 µm | 23.2 µm |
| NS 65, MPOL 12 | 272 µm | 54 µm | 17.7 µm | 25.4 µm | – |
| NS 129, MPOL 8 | 191 µm (1e-11) | 49 µm | – | 9.4 µm | – |

Bias/signal at FTOL 1e-15: 0.23 at β = 6.75×10⁻⁴, 0.058 at 2.7×10⁻³, 0.014 at 1.08×10⁻².

Restart dependence at β = 2.7×10⁻³ (VMEX/theory, vacuum-subtracted):

| start | FTOL 1e-14 | 1e-15 | 1e-16 |
|---|---|---|---|
| cold ladder, then continued | 1.040 | 1.042 | – |
| cold, one run to 1e-14 (β = 1.35×10⁻³ / 5.4×10⁻³; the 2.7×10⁻³ run hit the 600 s cap) | 1.038 / 1.044 | – | – |
| warm from the vacuum WOUT | 0.929 | 1.017 | 1.039 |
| warm from the β = 5.4×10⁻³ WOUT | 1.099 | 1.048 | 1.044 |

At 1e-14 the approach from below and from above still differ by 17%: the soft mode is not
yet relaxed. At 1e-16 they agree with the cold path to ±0.3%. Restart dependence is therefore
removed by tolerance, and it is the same mechanism as the bias.

Resolution at β = 2.7×10⁻³ (each with its own vacuum run): MPOL/NTOR 12, NZETA 64 gives
1.027 (shape residual 3.2%); NS 129 gives 1.047; base 1.042.

Boundary radius (FTOL 1e-15): 1.017 at a_b = 25 mm (β = 1.378×10⁻³, same p₂ as the 35 mm
β = 2.7×10⁻³ case; bias 9.3 µm, shape residual 2.6%), 1.042 at 35 mm, 1.061 at 45 mm
(β = 4.463×10⁻³; bias 55 µm, shape residual 6.6%). Extrapolation to a_b = 0: 0.962 (linear
in a_b/R₀) or 0.999 (quadratic); both fit the three points to 0.4%.

### Uncertainty budget for k₁ at a_b = 35 mm

| source | size |
|---|---|
| restart policy (half spread at FTOL ≤ 1e-15) | 0.3% |
| mode and radial resolution (MPOL 12, NS 129, in quadrature) | 1.5% |
| theory operator: traced-axis vs ideal near-axis Frenet | 1.5% |
| fit (σ and drop-one) | 0.1% |
| **combined, quadrature** | **2.2%** |
| (not in the headline: choice of vacuum run for the subtracted ratio, β = 2.7e-3) | 1.2% |

So k₁ = 1.037 ± 0.022 at a_b/R₀ = 0.029, and its excess over 1 scales with a_b. The
statement supported here: *the fixed-coil free-boundary axis displacement per unit β agrees
with the first-order theory to 2–6% at a_b/R₀ = 0.02–0.04, linear in β to 2% over a factor
16, with the discrepancy consistent with a finite-a_b correction that vanishes (0.96–1.00) as
a_b → 0.* The a_b → 0 limit rests on three radii and one β per radius at 25 and 45 mm.

## Not done

- **Hybrid (I₂ ≠ 0 design)**: not run. Its coils were fitted with the on-axis current, so
  their vacuum axis is 13.8 mm from the near-axis curve at a_b = 40 mm, and the current-free
  vacuum transform is 0.36. The theory is computed (`records/theory/theory_hybrid.json`); the
  VMEX ladder is the natural next case.
- **Axisymmetric check with an analytic Shafranov shift**: not run.
- **VMEC2000 cross-check at FTOL ≤ 1e-14**: not run (earlier campaign: VMEC2000 stalls above
  1e-13 on the QA).
- **More radii and β points at 25 and 45 mm**, and a QA re-run at FTOL ≤ 1e-14 with more
  hops: needed to pin the finite-a_b law and the QA floor.

## Reproduction

VMEX `b68807d8d51ec69d166cd336b1691454fd1c4f2f` (origin/main, 2026-09-28), ESSOS
`plasma-coil` `526ca70`, pyQSC_JAX `d42fb2e`, SOLVAX v0.27.0, JAX 0.9.2, Python 3.11,
float64, Apple M4 (shared; wall times 60–590 s per job). Coils: `../../pressure_benchmark/reference/qh_coils.json`
(SHA-256 `882012a2…`, the Sec. 6 second-pass QH coils).

```sh
export PYTHONPATH=/path/VMEX:/path/ESSOS:/path/pyQSC_JAX/src:/path/SOLVAX/src JAX_ENABLE_X64=1
cd shafranov_shift/pressure_benchmark
python pb.py theory qh --output RUNS/theory
python pb.py solve qh --radius 0.035 --beta 2.7e-3 --output RUNS/qh_a35_b2.7e-3
python pb.py solve qh --radius 0.035 --beta 2.7e-3 --ns 65 --ftol 1e-14 --niter 30000 \
  --restart RUNS/qh_a35_b2.7e-3/wout.nc --output RUNS/qh_a35_b2.7e-3_cont14
python pb.py solve qh --radius 0.035 --beta 2.7e-3 --ns 65 --ftol 1e-15 --niter 30000 \
  --restart RUNS/qh_a35_b2.7e-3_cont14/wout.nc --output RUNS/qh_a35_b2.7e-3_cont15
# ... the same for every beta and radius (run names in records/ encode radius, beta and protocol)
python collect.py --runs RUNS          # copies row.json files here, writes results.json and figures
```

`records/*/row.json` holds each solve's settings, convergence, iterations, WOUT SHA-256 and the
axis on 101 planes per period; `records/*/input.runtime` is the deck actually solved. A `_hopN`
suffix is one ≤ 9000-iteration continuation job; the unsuffixed name is the last hop.
Native WOUT files (≈120 MB) are outside Git.
