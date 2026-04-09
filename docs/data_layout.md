# Data Layout

This repository separates tracked source data from generated outputs.

## Tracked source data

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
  - pair manifests for benchmarking those tracked raw outputs
- `data/reproducibility/`
  - archived benchmark-result tables used to rebuild the manuscript analysis bundle

These tracked directories are versioned source data for the public repository.
They should not be overwritten by ad hoc local reruns.

## Generated outputs

Everything produced by reruns, reanalysis, or manuscript builds belongs under
`output/` or `manuscript/dist*`.

Recommended structure:

- `output/benchmark/`
  - benchmark CSV outputs and unified SQLite databases
- `output/analysis/db_pipeline/`
  - regenerated manuscript analysis tables
- `output/benchmark/<method_or_experiment>/`
  - benchmark reruns from tracked raw outputs or fresh inference runs

The public repo does not track `output/` or `.cache/`.
