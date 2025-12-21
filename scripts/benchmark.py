#!/usr/bin/env python3
"""
Run the main benchmark (command-line version).

Plain-language overview

- This script is a small “launcher” for the actual benchmark code in `lipid_benchmark/`.
- It reads a pairs CSV (a table that lists, for each target, the experimental structure and the prediction files).
- It then computes accuracy metrics (RMSD and contact overlap) and writes results under `analysis/`.

Most users should run:

`python scripts/benchmark.py`

If you want to see all options (e.g., choose a different pairs CSV or output folder):

`python scripts/benchmark.py --help`
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure Python can import the local package without requiring installation.
# (This adds the repository root to the import search path.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipid_benchmark.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
