#!/usr/bin/env bash
# OfficeBench batch under WSL.  Usage:
#   bash scripts/wsl_ob_batch.sh <out_dir> <budget> <method> [<method> ...] [-- extra run_officebench args]
# The Windows-side Bash tool expands $VARS inside commands it is given, so the loop
# lives in this file instead of an inline command.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/ob_venv/bin:$PATH" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
out="$1"; budget="$2"; shift 2
methods=()
extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  methods+=("$1"); shift
done
mkdir -p "$out"
for m in "${methods[@]}"; do
  python scripts/run_officebench.py --method "$m" --split test --out "$out" --budget "$budget" --workers 2 "${extra[@]}" 2>&1 \
    | grep -v "OBSERVATION\|+++\|====\|Command:\|Exit code\|STDOUT\|STDERR\|^(" > "$out/${m}_log.txt"
  echo DEV_DONE >> "$out/${m}_log.txt"
done
echo ALL_DONE > "$out/ALL_DONE"
