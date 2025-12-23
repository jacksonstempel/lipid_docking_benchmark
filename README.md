# Lipid Docking Benchmark

Benchmark Boltz vs AutoDock Vina docking predictions on lipid–protein complexes against experimental structures.

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

## Outputs

By default, outputs are written under `output/`:

- `output/benchmark_allposes.csv`: one row per method/pose with RMSD and contact metrics
- `output/benchmark_summary.csv`: best-per-target summary (Boltz + best Vina pose)

Generated caches (normalized PDBs and cached contacts) are stored under `.cache/lipid_benchmark/`.

## Plotting (optional)

`scripts/plot_results.py` generates figures from the CSVs:

```bash
python scripts/plot_results.py --summary output/benchmark_summary.csv --allposes output/benchmark_allposes.csv --out-dir plots
```

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
