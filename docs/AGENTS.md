# Repository Guidelines

## Project Structure & Module Organization
`scripts/` contains all runnable entry points (`pose_benchmark.py` for single-target scoring, `benchmark_runner_min.py` for batch runs, plus utilities under `analysis/` and legacy variants). Shared helpers for config, path discovery, and aggregation live in `scripts/lib/`. Generated artifacts write into `analysis/` (`proteins/` per target and `aggregates/` summaries); review them locally but keep them out of commits. `metadata/` tracks lightweight lookup tables, while `model_inputs/benchmark_inputs/` hosts batch definitions and `model_inputs/test_inputs/` is ideal for small fixtures. Reference structures resolve from `raw_structures/benchmark_references/` by default via `config.yaml`, and predictions mirror under `model_outputs/`. Docking prep notebooks and third-party runners remain inside `docking/`.

## Build, Test, and Development Commands
- `mamba create -n ldb python=3.12.11 -y && mamba activate ldb` sets up the base environment.
- `mamba install -c conda-forge numpy scipy pandas gemmi biopython rdkit -y` installs core dependencies; add `pytest black` for QA/tooling.
- `python scripts/pose_benchmark.py 1FK1 --ref raw_structures/benchmark_references/1FK1/1FK1.cif --pred model_outputs/1FK1_output/.../1FK1_model_0.cif --full -v` runs a focused evaluation (adjust IDs and paths as needed).
- `python scripts/benchmark_runner_min.py --refs raw_structures/benchmark_references --preds model_outputs --pose scripts/pose_benchmark.py --full -v` executes the deterministic batch workflow and refreshes `analysis/aggregates/`.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, descriptive snake_case function names, and PascalCase classes; keep type hints and docstrings consistent with existing modules.
- Use `black` (line length 100) before opening a PR; if you script imports, align with the current ordering rather than introducing a new tool.
- Place shared utilities inside `scripts/lib/` and keep CLI modules thin; new config should either extend `config.yaml` or accept argparse overrides that round-trip through `PathResolver`.

## Testing Guidelines
- Prefer `pytest`, placing files under `tests/` with `test_*.py` naming; use the light fixtures in `model_inputs/test_inputs/` or create trimmed CIF snippets alongside the tests.
- Run `pytest -q` (or target-specific `pytest tests/test_paths.py -k aggregate`) before pushing; aim to cover new path resolution, aggregation, and RMSD logic with deterministic inputs.
- When adding expensive structural calculations, gate them behind markers (e.g., `@pytest.mark.slow`) so the default suite remains fast.

## Commit & Pull Request Guidelines
- Write imperative commit subjects (~50 chars) such as `bench: add locked RMSD threshold guard`; group related changes and describe the “what” first, “why” second in the commit body when needed.
- Scope pull requests narrowly and include: a brief summary, key validation steps (commands run), linked issues/tasks, and screenshots or CSV snippets if behavior changes.
- Flag new large data dependencies explicitly; confirm `analysis/` and `model_outputs/` artifacts stay untracked or are added to `.gitignore` before requesting review.

## Data & Configuration Practices
- Treat `config.yaml` as the single source of default paths, updating both the config file and any agent instructions when directories move.
- Keep generated CIF/PDB outputs and aggregate CSVs out of version control; store only reproducible scripts and minimal metadata.
- Sanitise external predictions before sharing, and avoid embedding credentials or proprietary structures in the repo.
