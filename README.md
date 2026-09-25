# plasma-coil-fields

Results, validation runs, benchmarks and figures for the study *Plasma
self-fields and pressure response in stellarator coil design*. The study
covers the free-space field of the plasma near a quasisymmetric magnetic axis,
its use as a source-subtracted target in finite-pressure coil optimization,
and the pressure-induced displacement of the magnetic axis at fixed coils.

The solvers are not here. This repository holds the study drivers, their
compact outputs and the figures. It runs against:

| Package | Branch / version | What it provides |
|---|---|---|
| [ESSOS](https://github.com/uwplasma/ESSOS) | `plasma-coil` ([PR #70](https://github.com/uwplasma/ESSOS/pull/70)) | coils, Biot–Savart, the finite-beta coil-optimization examples, `examples/coil_optimization/shafranov_shift.py` (pressure-response operator) and `nearaxis_finite_beta_helpers.py` |
| [pyQSC_JAX](https://github.com/uwplasma/pyQSC_JAX) | `refactor/pyqsc-jax-complete` ([PR #2](https://github.com/uwplasma/pyQSC_JAX/pull/2)) | near-axis equilibria and the plasma field, gradient and Hessian on axis (`pyqsc_jax.plasma`) |
| [VMEX](https://github.com/uwplasma/VMEX) | `main`, v0.11.2 (`926892ab`) | fixed- and free-boundary equilibria |
| VMEC2000 (optional) | STELLOPT `ee175502` | independent free-boundary comparison |

## Setup

```sh
git clone -b plasma-coil https://github.com/uwplasma/ESSOS.git
git clone -b refactor/pyqsc-jax-complete https://github.com/uwplasma/pyQSC_JAX.git
git clone https://github.com/uwplasma/VMEX.git && git -C VMEX checkout 926892ab
git clone https://github.com/rogeriojorge/plasma-coil-fields.git
export PYTHONPATH=$PWD/ESSOS:$PWD/pyQSC_JAX/src:$PWD/VMEX JAX_ENABLE_X64=1
cd plasma-coil-fields
```

The drivers find the ESSOS examples through the importable ESSOS checkout, or
through `ESSOS_EXAMPLE_DIR` if it is set (`essos_examples.py`). The
pressure-operator unit tests stay in ESSOS
(`pytest tests/test_shafranov_shift.py` from the ESSOS checkout).

## Layout

- `shafranov_shift/`: fixed-coil pressure response. It holds the fitted
  vacuum coil reference, the VMEX pressure-family driver (`scan.py`) and the
  September 2026 verification campaign in `results/verification/`. That
  folder has the source and closed-axis operator checks, the VMEX/VMEC2000
  resolution, restart and iteration studies, compact data with WOUT hashes,
  the three paper figures, and a drop-in manuscript section.
  **Start with [`shafranov_shift/results/verification/README.md`](shafranov_shift/results/verification/README.md).**
- Root scripts and `figures/`, `investigation/`, `*.jobs`: the archived 30 mm
  QA finite-pressure design, its refinement ladder, the Cartesian source
  tables, the vacuum-transform investigation, and the QH, hybrid and
  single-stage examples. They are described in
  [`archive_report.md`](archive_report.md).
- `independent_checks/`: NumPy/SciPy/SymPy calculations with no repository
  imports. They cover the source algebra, the vacuum-target obstruction and
  Hessian incompatibility identity, the circular and Shafranov-shift
  references, and the separate fractional-bootstrap reduced problem. It also
  holds their outputs and the 2026-09-24 reproduction logs.

## Status in one paragraph

The pressure source and the closed-axis response operator are verified
independently of any MHD solver. The near-axis source matches Biot–Savart
of its own current to 6.8e-4. The operator matches direct coil-field
tracing to 1.8e-8, and the near-axis Frenet form agrees to 0.22%. The
fixed-coil pressure derivative from free-boundary VMEX and VMEC2000 solves
is not resolved:

- The sign depends on the restart policy.
- The ratio to theory drifts with NS, NITER and the pressure step.
- Every free-boundary vacuum axis lies 474–688 µm from the directly traced
  coil axis. That is as large as the predicted shift at the screen
  amplitude (524 µm at a = 17.8 mm).

A resolved pressure coefficient, and the matched design and timing
comparison built on it, remain open.

## Reproducing the independent checks

```sh
python -m pip install -r independent_checks/requirements.txt
cd independent_checks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python independent_checks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python vacuum_target_obstruction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python fractional_bootstrap.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python original_checks/check_manuscript.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python original_checks/check_shafranov_shift.py
```

Native equilibrium outputs (WOUT, MAKEGRID) are not tracked; each accepted
run records its input and WOUT SHA-256 in the compact JSON.

## License

MIT (see `LICENSE`). The material was developed in the ESSOS repository and
moved here with its history.
