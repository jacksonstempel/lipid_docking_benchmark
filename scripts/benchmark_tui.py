#!/usr/bin/env python3
"""
Run the benchmark via a simple interactive menu (TUI).

Plain-language overview

- “TUI” means a text-based user interface: you pick actions from menus in the terminal.
- Under the hood, it runs the same benchmark pipeline as `scripts/benchmark.py`.
- This is useful if you prefer guided prompts over command-line flags.

Run:

`python scripts/benchmark_tui.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure Python can import the local package without requiring installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lipid_benchmark.tui import main


if __name__ == "__main__":
    raise SystemExit(main())
