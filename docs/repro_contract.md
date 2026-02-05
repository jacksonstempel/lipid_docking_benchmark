# Reproducibility Contract

This document defines the canonical, publication-relevant pipeline and output contract.

## Scope

The canonical workflow for manuscript results is:

1. Benchmark execution from structure pairs CSV
2. Unified SQLite database build
3. Database analysis + figure generation
4. Manuscript number verification

## Canonical Inputs

- Benchmark pair list: `structures/benchmark_entries.csv`
- Structure files referenced by each pair row:
  - experimental reference: `structures/experimental/*.cif`
  - Boltz prediction: `structures/boltz/*_model_0.cif`
  - Vina poses: `structures/vina/*.pdbqt`

## Canonical Commands

Run from repository root.

1. Benchmark stage:

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark
```

2. Database stage:

```bash
python scripts/build_benchmark_db.py \
  --out output/benchmark/benchmark_full.sqlite
```

3. Analysis stage:

```bash
python scripts/analyze_benchmark_db.py \
  --db output/benchmark/benchmark_full.sqlite \
  --out-dir output/analysis/db_pipeline \
  --fig-dir manuscript/figures
```

4. Verification stage:

```bash
python scripts/verify_manuscript_numbers.py
```

## Canonical Outputs

### Benchmark outputs

- `output/benchmark/benchmark_allposes.csv`
- `output/benchmark/benchmark_summary.csv`

### Unified database

- `output/benchmark/benchmark_full.sqlite`

### Analysis outputs

- `output/analysis/db_pipeline/per_target.csv`
- `output/analysis/db_pipeline/summary_table_numeric.csv`
- `output/analysis/db_pipeline/summary_table_formatted.csv`
- `output/analysis/db_pipeline/torsion_table_numeric.csv`
- `output/analysis/db_pipeline/torsion_table_formatted.csv`
- figure files in `manuscript/figures/`

## Acceptance Criteria

A run is considered reproducible when all of the following hold:

1. The four canonical commands complete without error.
2. The canonical output files exist at the expected paths.
3. `python scripts/verify_manuscript_numbers.py` returns success.
4. `python -m unittest` passes.

## Non-Canonical Tools

Cluster submission scripts, exploratory analysis scripts, and helper utilities are not part of the canonical publication workflow unless explicitly referenced in this document.
