## 9. Fixed-coil pressure response of the magnetic axis

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

**A free-boundary equilibrium confirms it.** With p = p₀(1 − s) the on-axis
plasma field is proportional to p₂a_b² = −p₀, so the predicted displacement
depends on the on-axis β only. On the QH coils of Sec. 6 (a_b = 35 mm,
a_b/R₀ = 0.029, ι = 1.14), VMEX continued from its default ladder to
FTOL 10⁻¹⁵ gives

**VMEX / first-order theory = 1.037 ± 0.022**

for the axis displacement per unit β, linear over β = 6.75×10⁻⁴ to 1.08×10⁻²
(displacement 0.26% to 4.1% of a_b; the quadratic term is 1.9% at the top).
The uncertainty is measured: restart policy 0.3%, MPOL 12 and NS 129 1.5%,
traced-axis vs ideal near-axis operator 1.5%, fit 0.1%.

![Fixed-coil pressure response of the QH axis from VMEX](docs/figures/12_qh_fixed_coil_pressure_response.png)

*Figure 8. QH coils, fixed, p = p₀(1 − s), zero current. (a) rms axis displacement / a_b
against on-axis β: first-order theory, VMEX at the default FTOL 10⁻¹⁰ (open) and continued
to 10⁻¹⁵ (filled), and the vacuum-axis bias at the two tolerances. (b) δR and δZ along the
field period at β = 5.4×10⁻³. (c) Vacuum-axis bias against the directly traced coil axis vs
FTOL. (d) Response ratio for three restart policies vs FTOL. (e) Response ratio across the
β ladder, vacuum-subtracted and from the traced axis with a fitted offset. (f) Response
ratio against a_b/R₀, with linear and quadratic extrapolations to a_b = 0.*

What controls it is the force tolerance, and the axis stiffness behind it. The
axis position is the softest direction of the free-boundary energy. At VMEX's
default FTOL 10⁻¹⁰ the QH vacuum axis sits 278 µm (0.8% a_b) from the traced
coil axis and the response is only 0.73–0.77 of theory at every β; more modes
or surfaces do not change that. At 10⁻¹⁴–10⁻¹⁶ the bias is 10–23 µm, 1.4–6% of
the signal at β ≥ 2.7×10⁻³. Restarts that approach from below (the vacuum) and
from above (β = 5.4×10⁻³) differ by 17% at 10⁻¹⁴ and agree with the cold path
to ±0.3% at 10⁻¹⁶.

| a_b (mm) | a_b/R₀ | β | VMEX/theory | vacuum bias (µm) |
|---|---|---|---|---|
| 25 | 0.021 | 1.38×10⁻³ | 1.017 | 9 |
| 35 | 0.029 | 6.75×10⁻⁴ – 1.08×10⁻² | 1.037 (fit) | 10–23 |
| 45 | 0.038 | 4.46×10⁻³ | 1.061 | 55 |
| 15 | 0.013 | 2.7×10⁻³ | not converged (residual floor 7×10⁻¹⁴) | 37 |

The excess over 1 grows with a_b and extrapolates to 0.96 (linear in a_b) or
1.00 (quadratic) at a_b → 0, where the near-axis theory applies. The earlier
QA benchmark (a = 17.8 mm, ι = 0.21, aspect ratio 61) failed for the same
reason as the 15 mm QH: the solver's residual floors before the soft axis mode
relaxes. There every free-boundary vacuum axis sat 0.47–0.69 mm from the traced
coil axis with VMEX and VMEC2000, and the response depended on restarts and
iteration counts (details and figures in
[`shafranov_shift/results/verification/README.md`](shafranov_shift/results/verification/README.md)).

Numbers, protocol, run records and reproduction:
[`shafranov_shift/results/pressure_benchmark/README.md`](shafranov_shift/results/pressure_benchmark/README.md).
