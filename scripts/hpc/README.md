# HPC Scripts (Operational)

This folder documents cluster-operational scripts used for ISAAC/Slurm execution.

## Scope

HPC submission and sync scripts are operational tooling. They are not required to reproduce the canonical publication tables/figures locally when benchmark outputs already exist.

## Canonical Publication Pipeline (non-HPC)

Use these scripts for manuscript-facing reproducibility:

- `scripts/benchmark.py`
- `scripts/build_benchmark_db.py`
- `scripts/analyze_benchmark_db.py`
- `scripts/verify_manuscript_numbers.py`
- `scripts/reproduce_paper.py`

## HPC-Operational Scripts

Scripts with `isaac_*`, `submit_*`, `sync_*`, or `*_smoketest_*` naming are cluster operations helpers.

Examples:

- ISAAC job templates (`*.sbatch`)
- submission wrappers (`submit_*`)
- sync helpers (`sync_*`)

## Policy

1. Keep cluster-specific assumptions out of canonical analysis logic.
2. Prefer script wrappers that call library code in `lipid_benchmark/`.
3. Treat HPC scripts as execution adapters, not scientific source-of-truth.
