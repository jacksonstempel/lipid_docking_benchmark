# Prediction Inputs

This directory contains the curated model-input packages needed to rerun the
main inference stages on the 100-target benchmark set.

Contents:

- `boltz_inputs/`
  - one Boltz YAML file per curated target
- `vina_inputs/`
  - one docking box file per curated target
  - one prepared receptor/ligand bundle per curated target for Vina and GNINA

These inputs are kept in sync with `structures/benchmark_entries.csv`.

Integrity:

- `SHA256SUMS.txt` records the tracked file hashes for this input bundle.
