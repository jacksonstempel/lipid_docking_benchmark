# Lipid Docking Benchmark

Benchmark Boltz vs AutoDock Vina docking predictions on lipid–protein complexes against experimental structures.

## Scientific context

Lipids are flexible ligands whose biologically relevant binding modes are often defined by **headgroup chemistry** (specific polar/charged interactions) plus a highly mobile **hydrocarbon tail**. Because of this, “did we place the ligand correctly?” is not always captured by a single number.

This benchmark evaluates predictions in two complementary ways:

- **Geometry**: RMSD of the whole ligand and of the headgroup subset.
- **Interactions**: how well predicted headgroup contacts match the experimental structure.

The goal is to compare methods on a curated set of lipid–protein complexes using consistent, reproducible metrics.

## Install

```bash
pip install -e .
```

Optional plotting dependencies:

```bash
pip install -e ".[plot]"
```

## Run

```bash
python scripts/benchmark.py
```

Common flags:

```bash
python scripts/benchmark.py --pairs structures/benchmark_entries.csv --out-dir output --workers 4
```

The TUI wrapper is optional:

```bash
python scripts/benchmark.py --tui
```

## Inputs

This repo treats the pairs CSV as the source of truth for file locations.

- `structures/benchmark_entries.csv` lists one row per target: `pdbid,ref,boltz_pred,vina_pred`
- Paths in the CSV are resolved relative to the repo root (or may be absolute)
- `config.yaml` is only used to provide the default pairs CSV path (`paths.pairs`)

## How it works (high level)

For each benchmark target (one PDB ID), the pipeline:

1. **Loads the experimental complex** and selects the target lipid ligand.
2. **Loads predictions** (Boltz and a multi-pose Vina `*.pdbqt`).
3. **Measures RMSD per pose** by matching ligand atoms between prediction and reference, including a headgroup-only RMSD.
4. **Normalizes structures** into a consistent complex representation (used for contact analysis and caching).
5. **Extracts protein–ligand contacts** and summarizes interaction similarity as Jaccard overlaps.
6. **Writes CSV outputs** for downstream plotting/analysis.

For Vina, the benchmark always keeps per-pose rows (so you can evaluate “top‑K best” performance), and the summary file also includes the non-oracular **top‑1** pose (what a user would typically inspect first).

## Outputs

By default, outputs are written under `output/`:

- `output/benchmark_allposes.csv`: one row per method/pose with RMSD and contact metrics
- `output/benchmark_summary.csv`: one row per target for Boltz plus Vina top‑1 (non-oracular)

Generated caches (normalized PDBs and cached contacts) are stored under `.cache/lipid_benchmark/`.

## Plotting (optional)

`scripts/plot_results.py` generates figures from the CSVs:

```bash
python scripts/plot_results.py --summary output/benchmark_summary.csv --allposes output/benchmark_allposes.csv --out-dir plots
```

### Metrics at a glance

- `ligand_rmsd` / `headgroup_rmsd`: lower is better (Å).
- `*_jaccard`: higher is better (0–1 overlap of contact sets).
- “Vina top‑K best”: per target, take the best value among the first K ranked poses (min RMSD / max overlap).

## Tests

```bash
python -m unittest
```

## Layout

- `structures/experimental/`: experimental structures (`*.cif`)
- `structures/boltz/`: Boltz predictions (`*_model_0.cif`)
- `structures/vina/`: Vina predictions (`*.pdbqt`)
- `lipid_benchmark/`: benchmark library code
- `structures/benchmark_entries.csv`: benchmark entry list (paths to structure files)
- `scripts/`: CLIs
- `excluded_entries.txt`: excluded entries with reasons
