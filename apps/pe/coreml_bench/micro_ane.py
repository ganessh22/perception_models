"""Synthetic Neural Engine tests: `micro_ane.py [grid|sweep|mlp]`. They live in the aneopt package; see aneopt.probe."""
import sys

from aneopt.probe import find_step, run

mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
rows = run(mode)
if mode == "sweep":
    print("working-set step:", find_step(rows))
