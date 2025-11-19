# Purpose

A lightweight tool to measure how well a predicted ligand pose matches a reference
structure. It aligns proteins (Cα), auto-selects the main ligand, pairs atoms via RDKit
(chem-only), and reports heavy-atom RMSD in the aligned frame.

# Quickstart

Prereqs: Python env with `numpy`, `pandas`, `gemmi`, `rdkit`.

Single pair:
```bash
python scripts/measure_ligand_pose.py \
  --ref benchmark_references/1B56.cif \
  --pred model_outputs/boltz/1B56_model_0.cif
```
(For multi-pose predictions, add `--max-poses N` to pick the best pose.)
Example output:
```
Reference:  /…/benchmark_references/1B56.cif
Prediction: /…/model_outputs/boltz/1B56_model_0.cif
Ligand:     A:PLM:136

Protein alignment:
  Cα pairs used: 133
  RMSD:          0.452 Å

Ligand:
  Pairing method: chem
  Heavy atoms:    18
  RMSD:           0.987 Å
```

Batch (CSV with `pdbid,ref,pred`):
```bash
python scripts/measure_ligand_pose_batch.py \
  --pairs scripts/boltz_pairs.csv \
  --out analysis/boltz_batch_results.csv
```
(Use `--kind vina` for the Vina defaults, or `--max-poses N` for multi-pose files.)
Status and errors are per-row in the output CSV.

# File Layout (simplified)

- `benchmark_references/` — reference CIFs.
- `model_outputs/boltz/*.cif` — Boltz predictions (flattened). Others in `model_outputs/boltz/extra_output/`.
- `model_outputs/vina/*.pdbqt` — Vina predictions (flattened). Others in `model_outputs/vina/extra_output/`.
- `scripts/`
  - `measure_ligand_pose.py` — single-pair CLI.
  - `measure_ligand_pose_batch.py` — batch runner.
  - `boltz_pairs.csv`, `vina_pairs.csv` — ready-made pair lists.
  - `lib/` — alignment, ligand handling, RMSD logic (chem-only pairing).
- `legacy_scripts/` — old/unused CLIs and helpers.
- `docking/` — prep templates (atom-name normalization) and Vina prep artifacts.
- `analysis/` — outputs from batch runs (CSV).

# Behavior & Assumptions

- Always aligns proteins (Cα, pruned Kabsch) and computes ligand RMSD in the aligned frame.
- Auto-selects a single significant ligand (ignores protein/water/common solvents/ions/small fragments).
- Atom pairing is RDKit chem-only; if RDKit can’t map ≥3 heavy atoms, the run fails for that pair.
- Hydrogens ignored.

# Repo Notes

- Legacy tools are sequestered under `legacy_scripts/`; the simple lane lives in `scripts/`.
- `metadata/` was removed (former curation lists). References are at `benchmark_references/`.
