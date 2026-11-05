#!/usr/bin/env bash
# Usage: bash run.sh           full benchmark
#        bash run.sh --quick   a few minutes, to check everything works; timings are not comparable
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"  # the repo root, for `core`
if [ ! -x .venv/bin/python ] || [ ! -f data/val.npy ]; then
  echo "Run 'bash setup.sh' first." >&2
  exit 1
fi
exec .venv/bin/python run_all.py "$@"
