#!/bin/zsh
# Usage: maxr_single.sh FRACTION : single-stage free-boundary ladder at a_b = FRACTION * a (one VMEX solve per job,
# each killed after 600 s): NS 17 FTOL 1e-10 (continued once), NS 33 and 65 at FTOL 1e-9, then NS 65 at FTOL 1e-10.
cd "$(dirname "$0")"
f=$1; fb=$PWD/runs/single4_fb; export SINGLE_RUN=single4
level() {  # name ns ftol niter restart
  perl -e 'alarm shift; exec @ARGV' 600 python -u single_stage_free_boundary.py ab${f}_$1 $2 $3 $4 0.5 $5 $f > $fb/ab${f}_$1.log 2>&1
  echo "single ab${f}_$1 exit=$? " >> runs/status.txt
  grep -q '"converged": true' $fb/ab${f}_$1.log
}
level ns17_f10 17 1e-10 3000 none || level ns17_f10_b 17 1e-10 3000 $fb/ab${f}_ns17_f10/wout_ns17_ftol1e-10.nc || exit 1
r17=$(ls -t $fb/ab${f}_ns17_f10*/wout_ns17_ftol1e-10.nc | head -1)
level ns33_f9 33 1e-9 4000 $r17 || exit 1
level ns65_f9 65 1e-9 2500 $fb/ab${f}_ns33_f9/wout_ns33_ftol1e-09.nc || exit 1
level ns65_f10 65 1e-10 3000 $fb/ab${f}_ns65_f9/wout_ns65_ftol1e-09.nc
