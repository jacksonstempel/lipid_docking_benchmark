# Reproducibility Contract

This repository exposes two reproduction layers:

1. exact manuscript reproduction from the benchmark-result tables
2. structure-level reruns from the canonical prediction outputs and inputs

## 1. Manuscript Reproduction

The paper is reproduced from the archive in `data/reproducibility/`. The
canonical command is:

```bash
python scripts/reproduce_paper.py
```

It completes without manual intervention and produces:

- `output/benchmark/benchmark_full.sqlite`
- `output/analysis/db_pipeline/per_target.csv`
- `output/analysis/db_pipeline/summary_table_numeric.csv`
- `output/analysis/db_pipeline/summary_table_formatted.csv`
- `output/analysis/db_pipeline/torsion_table_numeric.csv`
- `output/analysis/db_pipeline/torsion_table_formatted.csv`
- refreshed manuscript figure files in `manuscript/figures/`

It also runs `python scripts/verify_manuscript_numbers.py`, which must return
success.

The archive contains:

- baseline Boltz + Vina benchmark tables
- GNINA CNN and no-CNN all-pose tables
- adversarial mutagenesis all-pose and summary tables
- Vina exhaustiveness-256 robustness tables
- higher-sampling Boltz-2 robustness tables

## 2. Structure-Level Reruns

The repository also includes canonical prediction outputs that can be
benchmarked directly:

- baseline Boltz + Vina outputs in `structures/`
- GNINA CNN and no-CNN raw outputs in `data/raw_predictions/gnina/`
- adversarial mutagenesis raw Boltz outputs in `data/raw_predictions/adversarial/`

Use the pair manifests to rerun those benchmarks:

```bash
python scripts/benchmark.py --pairs structures/benchmark_entries.csv --out-dir output/benchmark/baseline
python scripts/benchmark.py --pairs data/raw_predictions/gnina/cnn_rescore_exh8_cpu24/pairs.csv --out-dir output/benchmark/gnina_cnn
python scripts/benchmark.py --pairs data/raw_predictions/gnina/no_cnn_exh8_cpu24/pairs.csv --out-dir output/benchmark/gnina_nocnn
python scripts/benchmark.py --pairs data/raw_predictions/adversarial/bs_mutagenesis_cutoff5A/gly/pairs.csv --out-dir output/benchmark/adversarial_gly
python scripts/benchmark.py --pairs data/raw_predictions/adversarial/bs_mutagenesis_cutoff5A/phe/pairs.csv --out-dir output/benchmark/adversarial_phe
```

## Fresh-Inference Inputs

The curated model-input packages for the main inference stages live in:

- `prediction_inputs/boltz_inputs/`
- `prediction_inputs/vina_inputs/`

ISAAC Slurm submission helpers for Boltz, Vina, and GNINA are provided in
`scripts/`. Each helper takes its required paths and settings as command-line
arguments.

## Acceptance Criteria

The manuscript companion is considered valid when all of the following hold in
a clean checkout:

1. `python -m unittest` passes.
2. `python scripts/reproduce_paper.py` completes without error.
3. `python scripts/verify_manuscript_numbers.py` returns success.
4. The manuscript sources build with `make -C manuscript all` in an environment
   that provides the required LaTeX engine.
