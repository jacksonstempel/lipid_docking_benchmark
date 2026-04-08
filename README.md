# Lipid Docking Benchmark

Public manuscript companion for:
`Boltz-2 Outperforms AutoDock Vina on Lipid--Protein Complex Prediction`.

This repository is intentionally narrow. It contains:

- the benchmark code in `lipid_benchmark/` and `scripts/`
- the curated 100-target structure set in `structures/`
- the manuscript and supporting information source in `manuscript/`
- the final manuscript figure assets in `manuscript/figures/`
- a tracked benchmark-result archive in `data/reproducibility/`

It does not attempt to mirror the full local working directory used during
drafting, figure exploration, cluster submission, or manuscript review.

## What Reproducibility Means Here

This repo supports two distinct workflows.

### 1. Exact manuscript-analysis reproduction

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

This is the canonical public reproduction path for the paper.

### 2. Core baseline rerun from tracked structures

The repo also contains the curated experimental structures, Boltz predictions,
and Vina poses needed to rerun the main baseline benchmark directly:

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark
```

That command reproduces the baseline Boltz + Vina benchmark outputs from the
tracked structure inputs. It does not regenerate the GNINA or adversarial
results, which are provided as archived benchmark-result tables under
`data/reproducibility/`.

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

The CI workflow runs both the test suite and the full public manuscript-analysis
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

- `data/reproducibility/`: tracked manuscript-analysis archive
- `docs/repro_contract.md`: precise public reproduction contract
- `docs/data_layout.md`: local output layout conventions
- `lipid_benchmark/`: reusable benchmark library code
- `scripts/`: canonical CLI entry points used by the public workflow
- `structures/`: curated benchmark inputs
- `manuscript/`: main paper and SI source

## License

Code in this repository is released under the [MIT License](LICENSE). Bundled ACS
LaTeX support files in `manuscript/` retain their upstream licensing notices.
