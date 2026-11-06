"""Report CoreML's per-op device placement and estimated cost for a package.

  profile_plan.py <package-stem> [CPU_AND_NE|CPU_AND_GPU|ALL]
"""
import sys
from pathlib import Path

from aneopt.plan import print_plan

print_plan(Path(__file__).parent / "models" / f"{sys.argv[1]}.mlpackage", sys.argv[2] if len(sys.argv) > 2 else "CPU_AND_NE")
