# Lipid Docking Benchmark

Public manuscript companion for:
`Boltz-2 Outperforms AutoDock Vina on Lipid--Protein Complex Prediction`.

This repository contains the curated benchmark inputs, canonical prediction
outputs, archived benchmark-result tables, analysis code, and manuscript source
needed to reproduce the paper and rerun the benchmark workflows.

## What Is Tracked

- `structures/`
  - curated 100-target experimental structures
  - canonical Boltz baseline predictions
  - canonical Vina baseline predictions
  - canonical Vina exhaustiveness-256 robustness predictions
- `prediction_inputs/`
  - Boltz YAML inputs for the curated 100-target set
  - Vina/GNINA receptor, ligand, and box inputs for the curated 100-target set
  - `SHA256SUMS.txt` manifest for the tracked input bundle
- `data/raw_predictions/`
  - canonical GNINA CNN-rescored and no-CNN raw run outputs
  - canonical adversarial mutagenesis mutant inputs and raw Boltz outputs
  - pair manifests for benchmarking those tracked raw outputs directly
  - `SHA256SUMS.txt` manifest for the tracked raw-prediction bundle
- `data/reproducibility/`
  - archived benchmark-result tables used to rebuild the manuscript analysis bundle
- `lipid_benchmark/` and `scripts/`
  - benchmark library code and public CLI entry points
- `manuscript/`
  - main manuscript and supporting information source
  - final manuscript figure assets

## Exact Manuscript Reproduction

The tracked archive in `data/reproducibility/` contains the benchmark-result CSVs
used to regenerate the manuscript analysis bundle:

- baseline Boltz + Vina all-pose and summary tables
- GNINA CNN and no-CNN all-pose tables
- adversarial mutagenesis all-pose and summary tables
- robustness reruns for Vina exhaustiveness 256 and higher-sampling Boltz-2

Run:

```bash
python scripts/reproduce_paper.py
```

This rebuilds:

1. `output/benchmark/benchmark_full.sqlite`
2. `output/analysis/db_pipeline/*.csv`
3. the manuscript figure files in `manuscript/figures/`
4. the manuscript number audit in `scripts/verify_manuscript_numbers.py`

This is the canonical manuscript-reproduction path for the paper.

## Benchmark Reruns From Tracked Prediction Outputs

### Baseline Boltz + Vina

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark/baseline
```

### GNINA CNN-rescored

```bash
python scripts/benchmark.py \
  --pairs data/raw_predictions/gnina/cnn_rescore_exh8_cpu24/pairs.csv \
  --out-dir output/benchmark/gnina_cnn
```

### GNINA no-CNN

```bash
python scripts/benchmark.py \
  --pairs data/raw_predictions/gnina/no_cnn_exh8_cpu24/pairs.csv \
  --out-dir output/benchmark/gnina_nocnn
```

### Adversarial Mutagenesis

```bash
python scripts/benchmark.py \
  --pairs data/raw_predictions/adversarial/bs_mutagenesis_cutoff5A/gly/pairs.csv \
  --out-dir output/benchmark/adversarial_gly

python scripts/benchmark.py \
  --pairs data/raw_predictions/adversarial/bs_mutagenesis_cutoff5A/phe/pairs.csv \
  --out-dir output/benchmark/adversarial_phe
```

## Fresh Inference Reruns From Tracked Inputs

The curated Boltz and Vina/GNINA input packages are tracked under
`prediction_inputs/`. Example ISAAC Slurm submission helpers are included for:

- `scripts/submit_boltz_isaac_array.sh`
- `scripts/submit_vina_isaac_array.sh`
- `scripts/submit_gnina_isaac_array.sh`

These scripts are explicit about required paths and settings. They do not probe
private local workspaces.

## Environment Setup

### Conda

```bash
conda env create -f environment.yml
conda activate lipid-docking-benchmark
```

### pip

```bash
pip install -e ".[analysis]"
```

## Validation

```bash
python -m unittest
python scripts/reproduce_paper.py
```

The CI workflow runs both the test suite and the full public manuscript
reproduction path.

## Build the Manuscript PDFs

From the repository root:

```bash
make -C manuscript all
```

Outputs:

- `manuscript/dist/manuscript.pdf`
- `manuscript/dist_si/supporting_information.pdf`

More detail: [manuscript/README.md](manuscript/README.md)

## Repository Layout

- `data/raw_predictions/`: tracked canonical GNINA and adversarial raw outputs
- `data/reproducibility/`: tracked manuscript-analysis archive
- `docs/repro_contract.md`: precise public reproduction contract
- `docs/data_layout.md`: tracked versus generated data layout
- `lipid_benchmark/`: reusable benchmark library code
- `prediction_inputs/`: tracked Boltz and Vina/GNINA input packages
- `scripts/`: canonical CLI entry points and rerun helpers
- `structures/`: curated baseline structures and baseline prediction outputs
- `manuscript/`: main paper and SI source

## License

Code in this repository is released under the [MIT License](LICENSE). Bundled ACS
LaTeX support files in `manuscript/` retain their upstream licensing notices.
