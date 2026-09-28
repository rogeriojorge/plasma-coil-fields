#!/bin/zsh
# hop.sh NAME SRC FTOL CASE ARGS...
# Continue RUNS/SRC/wout.nc on NS ${NS:-65} to FTOL in jobs of at most 9000 iterations, each
# killed after 600 s, until one converges (at most six); the last job is copied to RUNS/NAME.
# RUNS must be set; run from this directory with PYTHONPATH set (env.sh).
n=$1; src=$2; f=$3; shift 3
for h in 1 2 3 4 5 6; do
  out=$RUNS/${n}_hop$h
  mkdir -p $out
  perl -e 'alarm shift; exec @ARGV' 600 python3 pb.py solve "$@" --ns ${NS:-65} --ftol $f --niter 9000 \
    --restart $RUNS/$src/wout.nc --output $out > $out/stdout.log 2>&1
  [ -f $out/row.json ] || { echo "$n hop$h failed"; exit 1; }
  src=${n}_hop$h
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/row.json'))['converged'] else 1)" && break
done
rm -rf $RUNS/$n; cp -r $RUNS/$src $RUNS/$n; echo "$n from $src"
