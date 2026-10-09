"""Prepare the public slices, run Python and R, then write the comparison table."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def main() -> None:
    env = os.environ.copy()
    subprocess.run([sys.executable, str(ROOT / "bench/prepare.py")], check=True, env=env)
    subprocess.run([sys.executable, str(ROOT / "bench/run_python.py")], check=True, env=env)
    subprocess.run(["Rscript", str(ROOT / "bench/run_r.R")], check=True, env=env)
    subprocess.run([sys.executable, str(ROOT / "bench/compare.py")], check=True, env=env)


if __name__ == "__main__":
    main()
