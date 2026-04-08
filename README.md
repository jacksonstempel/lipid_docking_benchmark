# Lipid Docking Benchmark

Minimal public companion repository for the manuscript:
`Boltz-2 Outperforms AutoDock Vina on Lipid--Protein Complex Prediction`.

This repo contains the benchmark code, the curated 100-target structure set, the
manuscript/SI source, and the small archived data files needed to regenerate the
manuscript-ready figures and tables. It does not mirror the full local working
directory used during drafting.

## Public Scope

Included:

- benchmark code in `lipid_benchmark/` and `scripts/`
- curated benchmark inputs in `structures/`
- manuscript and SI source in `manuscript/`
- manuscript figure assets in `manuscript/figures/`
- small archived mutant-summary CSVs in `data/adversarial/`

Intentionally omitted:

- transient local drafting files
- figure-preparation scratch work
- large generated outputs under `output/`
- cluster run directories and other local-only remnants

## Reproducibility Levels

### 1. Core benchmark regeneration

This reproduces the main benchmark outputs from the tracked structures.

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark

python scripts/build_benchmark_db.py \
  --out output/benchmark/benchmark_full.sqlite
```

### 2. Manuscript tables and figures

This regenerates the manuscript-ready analysis tables and figures from the unified
database. The analysis script automatically uses the tracked adversarial mutant
summary archive in `data/adversarial/bs_mutagenesis_cutoff5A/` when present.

```bash
python scripts/analyze_benchmark_db.py \
  --db output/benchmark/benchmark_full.sqlite \
  --out-dir output/analysis/db_pipeline \
  --fig-dir manuscript/figures

python scripts/verify_manuscript_numbers.py
```

### 3. One-command publication workflow

```bash
python scripts/reproduce_paper.py
```

This runs:

1. benchmark generation
2. database build
3. manuscript analysis/figure generation
4. manuscript number verification

## Environment Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate lipid-docking-benchmark
```

### Option B: pip

```bash
pip install -e .
pip install -e ".[analysis]"
```

## Build the Manuscript

From the repo root:

```bash
make -C manuscript all
```

Outputs:

- `manuscript/dist/manuscript.pdf`
- `manuscript/dist_si/supporting_information.pdf`

More detail: [manuscript/README.md](manuscript/README.md)

## Canonical Inputs

- benchmark entry table: `structures/benchmark_entries.csv`
- experimental structures: `structures/experimental/*.cif`
- Boltz predictions: `structures/boltz/*_model_0.cif`
- Vina poses: `structures/vina/*.pdbqt`

## Canonical Generated Outputs

- `output/benchmark/benchmark_allposes.csv`
- `output/benchmark/benchmark_summary.csv`
- `output/benchmark/benchmark_full.sqlite`
- `output/analysis/db_pipeline/per_target.csv`
- `output/analysis/db_pipeline/summary_table_formatted.csv`
- `output/analysis/db_pipeline/torsion_table_formatted.csv`

## Validation

```bash
python -m unittest
python scripts/verify_manuscript_numbers.py
```

## Notes on the Adversarial Mutagenesis Figure

The public repo includes only the small archived summary CSVs needed to regenerate the
manuscript adversarial figure. It does not include the full mutant prediction workspace.
The compute-heavy mutant runs originally lived under ignored `output/` paths in the local
working repository.

## Documentation

- reproducibility contract: `docs/repro_contract.md`
- workflow reference: `docs/workflows.md`
- data layout: `docs/data_layout.md`
- adversarial experiment notes: `docs/adversarial_experiment_report.md`

## Citation and License

- citation metadata: `CITATION.cff`
- license: `LICENSE`
