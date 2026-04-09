# GNINA Raw Runs

Canonical GNINA runs for the curated benchmark:

- `cnn_rescore_exh8_cpu24/`
  - GNINA 1.3.1
  - `cnn_scoring=rescore`
  - `scoring=vina`
  - `exhaustiveness=8`
  - `num_modes=20`
  - `cpu=24`
- `no_cnn_exh8_cpu24/`
  - GNINA 1.3.1
  - `cnn_scoring=none`
  - `scoring=vina`
  - `exhaustiveness=8`
  - `num_modes=20`
  - `cpu=24`

Each run directory contains:

- `flat/`: one multi-pose PDBQT per target
- `logs/`: one GNINA log per target
- `pairs.csv`: benchmark manifest for that run
