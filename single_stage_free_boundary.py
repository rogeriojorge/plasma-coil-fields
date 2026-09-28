"""One free-boundary run on the finished single-stage design (checkpoint in runs/single).

python single_stage_free_boundary.py NAME NS FTOL NITER DELT RESTART_WOUT|none  -> runs/single_fb/NAME
"""
import json, os, sys, time
from pathlib import Path
name, ns, ftol, niter, delt, restart = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
repo = Path(__file__).resolve().parent; D = repo / "drivers"; sys.path.insert(0, str(D)); os.chdir(D)
import matplotlib; matplotlib.use("Agg")
src = (D / "optimize_single_stage_nearaxis_finite_beta.py").read_text()
src = src.replace("RUN_VMEX = True", "RUN_VMEX = False", 1).replace(
    'OUTPUT_DIR = Path(__file__).resolve().parent / "output_single_stage_finite_beta"', f'OUTPUT_DIR = Path("{repo}/runs/single")', 1)
g = {"__file__": str(D / "x.py"), "__name__": "__main__"}
exec(compile(src, "single", "exec"), g)
g["VMEX"]["delt"] = delt
out = repo / "runs/single_fb" / name
rs = None if restart == "none" else Path(restart).resolve()
wout, rep = g["free_boundary_level"](g["states"]["optimized"]["solution"], g["states"]["optimized"]["field"], out, ns, ftol, niter, rs)
rep.update(delt=delt, restart=restart.rsplit("/", 1)[-1])
try:
    rep = g["benchmark_report"](wout, g["states"]["optimized"]["solution"], rep)
except Exception as e:
    rep["benchmark_error"] = repr(e)
(out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
print("REPORT", json.dumps({k: rep[k] for k in ("ns", "ftol", "delt", "iterations", "fsqr", "fsqz", "fsql", "converged", "betatotal", "iota_axis")}))
