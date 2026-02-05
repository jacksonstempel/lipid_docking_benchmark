# Workflows

This document lists supported repository workflows and their intended use.

## 1. Canonical Publication Workflow

Use this to regenerate manuscript-relevant benchmark tables and figures.

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark

python scripts/build_benchmark_db.py \
  --out output/benchmark/benchmark_full.sqlite

python scripts/analyze_benchmark_db.py \
  --db output/benchmark/benchmark_full.sqlite \
  --out-dir output/analysis/db_pipeline \
  --fig-dir manuscript/figures

python scripts/verify_manuscript_numbers.py
```

## 2. Benchmark-Only Workflow

Use this to regenerate only per-pose and summary benchmark outputs.

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark
```

## 3. Analysis-Only Workflow

Use this when `benchmark_full.sqlite` already exists.

```bash
python scripts/analyze_benchmark_db.py \
  --db output/benchmark/benchmark_full.sqlite \
  --out-dir output/analysis/db_pipeline \
  --fig-dir manuscript/figures
```

## 4. Validation Workflow

```bash
python -m unittest
python scripts/verify_manuscript_numbers.py
```
