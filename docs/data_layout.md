# Data Layout

This repository separates source data from generated outputs.

## Source data

- `structures/`
  - curated experimental structures
  - canonical baseline Boltz predictions
  - canonical baseline Vina predictions
  - canonical Vina exhaustiveness-256 robustness predictions
- `prediction_inputs/`
  - curated Boltz YAML inputs
  - curated Vina/GNINA receptor, ligand, and box inputs
- `data/raw_predictions/`
  - canonical GNINA raw runs
  - canonical adversarial mutagenesis raw runs and mutant inputs
  - pair manifests for benchmarking those raw outputs
- `data/reproducibility/`
  - benchmark-result tables used to rebuild the manuscript analysis bundle

## Generated outputs

Everything produced by reruns, reanalysis, or manuscript builds belongs under
`output/` or `manuscript/dist*`.

Recommended structure:

- `output/benchmark/`
  - benchmark CSV outputs and unified SQLite databases
- `output/analysis/db_pipeline/`
  - regenerated manuscript analysis tables
- `output/benchmark/<method_or_experiment>/`
  - benchmark reruns from raw outputs or fresh inference runs
