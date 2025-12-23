# Repository Guidelines

This document outlines contribution practices for the lipid docking benchmark pipeline.

## Project Structure & Module Organization
Core entry points live in `scripts/` (thin CLI/TUI wrappers). Benchmark logic lives in `lipid_benchmark/` (library-style modules: RMSD, normalization, PandaMap contacts, metrics, and pipeline orchestration). Structure files are under `structures/` (`structures/experimental/*.cif`, `structures/boltz/*_model_0.cif`, `structures/vina/*.pdbqt`), and the benchmark entry list lives at `structures/benchmark_entries.csv`. Benchmark outputs are written under `output/`, and caches are stored under `.cache/`. Test coverage is in `tests/` and supporting notes in `docs/`.

## Build, Test, and Development Commands
```bash
pip install -e .                     # editable package install
pip install -e ".[plot]"             # (optional) plotting dependencies
python scripts/benchmark.py          # run full benchmark, writes CSVs under output/
python scripts/benchmark.py --tui    # interactive TUI runner
python -m unittest                   # run unit + integration tests
```
Update `config.yaml` if you relocate input/output folders; it defines the default paths used by batch scripts.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, type hints, and f-strings. Follow existing naming patterns: `snake_case` for modules/functions, `CamelCase` for classes, and descriptive constants in uppercase. There is no repo-level formatter or linter configuration, so keep formatting consistent with nearby code and avoid unrelated reflows.

## Testing Guidelines
Tests are written with `unittest` and discovered via `tests/test_*.py` (the `tests/` package includes `__init__.py` to enable discovery). Add or update tests when changing core geometry, ligand selection, or metrics logic, and keep fixtures lightweight.

## Commit & Pull Request Guidelines
Recent commits use short, sentence-case summaries without prefixes (e.g., “Prepare codebase for publication”). Keep commits focused and describe the impact on metrics or outputs. For pull requests, include: a brief problem/solution summary, the command(s) run (e.g., `python scripts/benchmark.py`), and any data or config changes. Generated outputs under `output/` are ignored by `.gitignore`; only include derived artifacts when explicitly needed.
