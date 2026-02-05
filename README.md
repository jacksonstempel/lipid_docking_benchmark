# Lipid Docking Benchmark

Reproducible benchmark and analysis pipeline for evaluating lipid pose predictions against experimental structures.

## What This Repository Produces

- Per-pose and per-target benchmark metrics (RMSD and contact-overlap)
- Unified SQLite database for manuscript analysis
- Manuscript tables and figures
- Verification checks that manuscript-reported numbers match generated outputs

## Environment Setup

Use one of the following:

### Option A: Conda (pinned publication environment)

```bash
conda env create -f environment.yml
conda activate lipid-docking-benchmark
```

### Option B: pip (editable install)

```bash
pip install -e .
pip install -e ".[analysis]"
```

## Canonical Reproducibility Workflow

Run from repository root.

### One-command orchestrator

```bash
python scripts/reproduce_paper.py
```

### Equivalent staged commands

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

## Canonical Inputs

Primary benchmark entry table:

- `structures/benchmark_entries.csv`

Required columns:

- `pdbid`
- `ref`
- `boltz_pred`
- `vina_pred`

Paths can be absolute or repository-relative.

## Canonical Outputs

Benchmark:

- `output/benchmark/benchmark_allposes.csv`
- `output/benchmark/benchmark_summary.csv`

Database:

- `output/benchmark/benchmark_full.sqlite`

Analysis:

- `output/analysis/db_pipeline/per_target.csv`
- `output/analysis/db_pipeline/summary_table_numeric.csv`
- `output/analysis/db_pipeline/summary_table_formatted.csv`
- `output/analysis/db_pipeline/torsion_table_numeric.csv`
- `output/analysis/db_pipeline/torsion_table_formatted.csv`
- `manuscript/figures/*`

## Validation

```bash
python -m unittest
python scripts/verify_manuscript_numbers.py
```

## Reproducibility and Organization Docs

- Reproducibility contract: `docs/repro_contract.md`
- Workflow reference: `docs/workflows.md`
- Data/artifact policy: `docs/data_policy.md`
- Script taxonomy: `docs/script_organization.md`
- HPC operational guidance: `scripts/hpc/README.md`

## Repository Scope

Tracked source-of-truth inputs:

- `structures/experimental/`
- `structures/boltz/`
- `structures/vina/`
- `structures/benchmark_entries.csv`

Generated artifacts (not source inputs):

- `output/`
- `.cache/`

## Citation and License

- Citation metadata: `CITATION.cff`
- License: `LICENSE`
