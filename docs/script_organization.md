# Script Organization

This document defines script categories for maintainability and publication clarity.

## Categories

### A) Canonical scientific workflow

These scripts define manuscript-facing reproducibility:

- `scripts/benchmark.py`
- `scripts/build_benchmark_db.py`
- `scripts/analyze_benchmark_db.py`
- `scripts/verify_manuscript_numbers.py`
- `scripts/reproduce_paper.py`

### B) Scientific utilities

Utilities for analysis experiments or conversions. Useful, but not canonical unless explicitly referenced in manuscript methods.

Examples:

- `scripts/analyze_gnina_experiment.py`
- `scripts/resistant_case_analysis.py`
- `scripts/csv_to_sqlite.py`
- `scripts/compare_*`

### C) Operational/HPC scripts

Execution adapters for cluster environments and synchronization workflows.

Examples:

- `scripts/isaac_*.sbatch`
- `scripts/submit_*`
- `scripts/sync_*`
- `scripts/*smoketest*`

## Rules

1. Canonical scripts must be stable and documented in `README.md` and `docs/repro_contract.md`.
2. New one-off scripts should be clearly marked non-canonical in their docstring.
3. Script logic should delegate to `lipid_benchmark/` modules whenever possible.
4. Manuscript claims must cite canonical scripts and canonical outputs only.
