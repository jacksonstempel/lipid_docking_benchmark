#!/usr/bin/env python3
"""
Run the main benchmark (command-line version).

Plain-language overview

- This script is a small “launcher” for the actual benchmark code in `lipid_benchmark/`.
- It reads a pairs CSV (a table that lists, for each target, the experimental structure and the prediction files).
- It then computes accuracy metrics (RMSD and contact overlap) and writes results under `analysis/`.

Most users should run:

`python scripts/benchmark.py`

If you prefer an interactive menu (TUI):

`python scripts/benchmark.py --tui`

If you want to see all options (e.g., choose a different pairs CSV or output folder):

`python scripts/benchmark.py --help`
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure Python can import the local package without requiring installation.
# (This adds the repository root to the import search path.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run() -> int:
    if "--tui" in sys.argv[1:]:
        argv = [a for a in sys.argv[1:] if a != "--tui"]
        if argv:
            raise SystemExit("--tui does not accept additional flags. Run without other options.")
        from lipid_benchmark.tui import main as tui_main

        return tui_main()

    from lipid_benchmark.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(_run())
