# Reproducibility Contract

This repository exposes one strict public reproduction contract for the paper and
one narrower rerun path for the baseline benchmark.

## Canonical Public Manuscript Workflow

The paper is reproduced from the tracked archive in `data/reproducibility/`.
That archive contains the exact benchmark-result tables needed for:

- baseline Boltz + Vina analysis
- GNINA CNN and no-CNN comparisons
- adversarial mutagenesis figures
- robustness analyses in the supporting information

The canonical command is:

```bash
python scripts/reproduce_paper.py
```

That command must complete without manual intervention and produces:

- `output/benchmark/benchmark_full.sqlite`
- `output/analysis/db_pipeline/per_target.csv`
- `output/analysis/db_pipeline/summary_table_numeric.csv`
- `output/analysis/db_pipeline/summary_table_formatted.csv`
- `output/analysis/db_pipeline/torsion_table_numeric.csv`
- `output/analysis/db_pipeline/torsion_table_formatted.csv`
- refreshed manuscript figure files in `manuscript/figures/`

It also runs `python scripts/verify_manuscript_numbers.py`, which must return
success.

## Canonical Archive Contents

The public manuscript workflow depends on these tracked files:

- `data/reproducibility/baseline/benchmark_allposes.csv`
- `data/reproducibility/baseline/benchmark_summary.csv`
- `data/reproducibility/gnina/benchmark_allposes_gnina_cnn.csv`
- `data/reproducibility/gnina/benchmark_allposes_gnina_nocnn.csv`
- `data/reproducibility/adversarial/benchmark_gly/benchmark_allposes.csv`
- `data/reproducibility/adversarial/benchmark_gly/benchmark_summary.csv`
- `data/reproducibility/adversarial/benchmark_phe/benchmark_allposes.csv`
- `data/reproducibility/adversarial/benchmark_phe/benchmark_summary.csv`
- `data/reproducibility/adversarial/mutation_summary.csv`
- `data/reproducibility/robustness/vina_exhaustiveness_256/benchmark_allposes.csv`
- `data/reproducibility/robustness/vina_exhaustiveness_256/benchmark_summary.csv`
- `data/reproducibility/robustness/boltz_high_sampling/benchmark_allposes.csv`
- `data/reproducibility/robustness/boltz_high_sampling/benchmark_summary.csv`

## Baseline Structure-Level Rerun

The repository also supports rerunning the main baseline benchmark directly from
the curated structures:

```bash
python scripts/benchmark.py \
  --pairs structures/benchmark_entries.csv \
  --out-dir output/benchmark
```

This reproduces the baseline Boltz + Vina benchmark outputs only. It does not
replace the archive-backed manuscript workflow above.

## Acceptance Criteria

The public manuscript companion is considered valid when all of the following
hold in a clean checkout:

1. `python -m unittest` passes.
2. `python scripts/reproduce_paper.py` completes without error.
3. `python scripts/verify_manuscript_numbers.py` returns success.
4. The manuscript sources build with `make -C manuscript all` in an environment
   that provides the required LaTeX engine.
