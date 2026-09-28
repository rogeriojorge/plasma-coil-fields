# pyQSC_JAX validation figures and benchmarks

These are the figures and benchmarks from pyQSC_JAX's former `examples/publication/`
and `benchmarks/` folders. They moved here so that the package keeps only library code,
tests and short examples. Every script runs against the current (restructured)
pyQSC_JAX public API. The reference configurations that the package used to bundle
are now inlined in `common.py`, each with its source. Database entries are from
https://stellarator.physics.wisc.edu/app/plot/<id>.

## Running

```bash
export PYTHONPATH=<pyQSC_JAX>/src:<vmex>        # vmex is needed only for the VMEX items
export JAX_ENABLE_X64=1
python pyqsc_jax_validation/benchmarks/benchmark_core.py            # ~6 s
python pyqsc_jax_validation/benchmarks/benchmark_vmec_export.py     # ~3 s
python pyqsc_jax_validation/benchmarks/benchmark_b20_optimizers.py  # ~10 s
python pyqsc_jax_validation/benchmarks/benchmark_vmex_interface.py  # ~50 s, needs vmex
for f in pyqsc_jax_validation/figures/figure_*.py; do python "$f"; done
```

Figures are written to `figures/<name>.png` (300 dpi) and `.pdf`, along with a small
`<name>.json`. The JSON records the plotted numbers, the Python, JAX, pyQSC_JAX and
vmex versions, and the source commits. The three benchmark-based figures read the
newest matching report in `benchmarks/reports/`. The figures use the study's
`paper.mplstyle`. They are sized to journal widths (3.4 in for one column, 6.6 in for
two), and fields are shown relative to B0.

## Figures

| Figure | Shows |
|---|---|
| `axis_and_surfaces` | QA (nfp=2) and QH (nfp=4) surfaces at r = 0.05 m, from Landreman & Sengupta, JPP 85, 815850601 (2019), sections 5.1 and 5.5 |
| `QA_QH_branches` | First-order \|iota0\| and maximum elongation against etabar for the axes of database 139524 (QA) and 3 (QH); dotted lines mark the database etabar |
| `convergence` | Relative difference of iota0 and of the B20 residual from N=301, against grid points per period (database 139524, order r3); iota0 converges spectrally, B20 more slowly |
| `B20_optimization` | Database 57409 with exact B2c elimination (residual 3.1e-2 m^-2) against the stored eight-mode axis refinement (1.3e-10 m^-2, unchanged at N=243), plus the refined surface |
| `stellarator_gallery` | Four screened designs: iota0, surface radius and singular radius r_sing |
| `plasma_external_jet` | Plasma/external split of the on-axis field for database 52521 (finite p2, I2=0): plasma fraction about 0.2 %, tangential offsets, and transverse components that cancel |
| `core_performance` | First-order solve, JVP and vmap(64): warm times against compile-plus-first-call times, and cost per solve in a batch (`benchmark_core`) |
| `optimizer_comparison` | B20 flattening from database 57409 with L-BFGS-B, bounded least squares and low-budget differential evolution, plus the stored staged 8-mode result (`benchmark_b20_optimizers`) |
| `vmec_validation` | (a) VMEC on-axis iota error against boundary radius, following r^2. This is frozen VMEC 9.0 data in `reports/2026-07-30-b20-vmec-apple-m4.json`; VMEC is not rerun here. (b) VMEC export timing and reconstruction error (`benchmark_vmec_export`) |
| `vmex_radial_profiles` | VMEX fixed-boundary iota(s), quasisymmetry residual and magnetic well from near-axis boundaries at r = 0.02 m, with the implicit d W / d pres_scale |

## Benchmarks

`benchmarks/reports/2026-09-27-*.json` were measured on an Apple M3 Max (14 cores) with
Python 3.11, JAX 0.9.2 on CPU in float64, pyQSC_JAX 56e1f1c and vmex 0.11.4. Each report
records its own environment. The `2026-07-2x/30-*-apple-m4.json` files are the earlier
Apple M4 reports from pyQSC_JAX, kept for provenance.

Measured on 2026-09-27:
- First-order solve at N=121: 0.56 ms warm and about 0.3 s cold. A 64-case vmap batch
  costs 0.20 ms per case.
- VMEC export: 4.9 ms warm and 0.86 s cold.
- VMEX forward solve: 21 s, and value plus implicit gradient: 26 s. Both include
  compilation.
- Least squares reproduces the July residual of 1.025e-4 exactly.

## Retired or changed items

The old versions are on pyQSC_JAX branch `preserve/pr2-before-refactor-2026-09-27`.

- **Multistart Levenberg-Marquardt row** of the optimizer comparison, which used
  `search_axis`, `AxisSearchProblem` and `stellarator_symmetric_variable_indices`: retired.
  It depended on the removed axis-search framework, and a minimal re-implementation would
  not be the benchmarked method.
- **"Curvo profile: pass" labels** (`Criteria.from_curvo_2025`): dropped from the gallery
  and the B20 figure. The criteria module was removed, and the screening is not
  re-evaluated here.
- **`verify_B20_resolution`**: replaced by a direct B20 residual at N=121 and N=243,
  recorded in `B20_optimization.json`.
- **`solve_configuration` and `get_configuration`**: replaced by the inlined values in
  `common.py`.
