#!/bin/zsh
# Usage: maxr_jobs.sh JOBFILE PARALLEL. Line: name<TAB>driver<TAB>args... (max-radius scan; every job killed after 600 s).
cd "$(dirname "$0")"
run_one() {
  local name=$1 driver=$2; shift 2; export EXAMPLE=$driver
  local t0=$(date +%s)
  eval perl -e "'alarm shift; exec @ARGV'" ${CAP:-600} python -u run.py runs/$name "$@" > runs/$name/run.log 2>&1
  echo "$name exit=$? seconds=$(( $(date +%s) - t0 ))" >> runs/status.txt
}
typeset -i n=0
while IFS=$'\t' read -r name driver args; do
  [[ -z $name || $name == \#* ]] && continue
  run_one $name $driver "$args" &
  n+=1
  if (( n % $2 == 0 )); then wait; fi
done < $1
wait
