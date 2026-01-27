# Data layout conventions (local + ISAAC)

This repo produces large, derived artifacts (dock poses, per-target logs, benchmark CSVs, plots). To keep the repo tidy and avoid accidentally committing big files, we keep **all derived artifacts under `output/`**.

## Local (`lipid_docking_benchmark/`)

Recommended structure:

- `output/benchmark/`
  - `benchmark_allposes.csv` / `benchmark_allposes.sqlite`
  - `benchmark_summary.csv`
  - `benchmark_full.sqlite` (merged Boltz + Vina + GNINA all-poses DB)
- `output/gnina/runs/<run_name>/`
  - `flat/<PDBID>.pdbqt` (multi-pose PDBQT per target)
  - `logs/<PDBID>.log`
- `output/gnina/analysis/<analysis_name>/`
  - `per_target.csv`, `summary.csv`, `pose_failures.csv`, and any intermediate benchmark outputs
- `output/analysis/db_pipeline/`
  - `per_target.csv`, `summary_table_numeric.csv`, `summary_table_formatted.csv`,
    `torsion_table_numeric.csv`, `torsion_table_formatted.csv`
- `output/vina/runs/<run_name>/`
  - `flat/` and `logs/` (if stored locally)
- `output/archive/`
  - one-off debug artifacts, broken transfers, Windows `:Zone.Identifier` files, etc.

Naming tips:

- Use self-describing run names, e.g. `gnina_full_cnn_rescore_exh8_cpu24`.
- If you re-run with different parameters, make a new `<run_name>` rather than overwriting.

## ISAAC (`$SCRATCHDIR/`)

Recommended structure (mirrors local):

- `$SCRATCHDIR/gnina/runs/<run_name>/{flat,logs}/`
- `$SCRATCHDIR/gnina/tmp/` for job temp dirs (if needed)

Avoid writing large outputs to `$HOME` on ISAAC unless you have to.
