# Tracked Raw Prediction Outputs

This directory contains the canonical raw prediction outputs that complement the
archived manuscript-result tables in `data/reproducibility/`.

Contents:

- `gnina/`
  - canonical GNINA CNN-rescored and no-CNN raw runs
  - per-target logs
  - pair manifests for benchmarking those runs directly
- `adversarial/`
  - canonical binding-site mutagenesis Boltz inputs and raw outputs
  - pair manifests for benchmarking those runs directly
  - mutation bookkeeping

These directories contain only the canonical run bundles used to support public
reruns. Exploratory analyses, subset runs, caches, and scratch work are not
tracked here.

Integrity:

- `SHA256SUMS.txt` records the tracked file hashes for this raw-prediction bundle.
