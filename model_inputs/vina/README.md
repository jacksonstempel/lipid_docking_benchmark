## Vina Reproducibility Inputs

This folder stores the **inputs needed to reproduce the AutoDock Vina docking stage**
for the curated benchmark targets.

These files are **not used by the benchmark scoring code** directly. The benchmark reads
predictions from `model_outputs/vina/*.pdbqt` and compares them to the reference structures
listed in `scripts/pairs_curated.csv`.

### What’s Included

- `model_inputs/vina/box/`
  - One `<PDBID>.txt` file per curated target.
  - Format (one line): `center_x center_y center_z size_x size_y size_z`

- `model_inputs/vina/prep/<PDBID>/`
  - `receptor_no_ligand.pdb`: protein-only receptor structure used for docking
  - `receptor.pdbqt`: receptor converted to PDBQT
  - `ligand.pdb`: extracted lipid ligand
  - `ligand.pdbqt`: ligand converted to PDBQT
  - `run_manifest.json`: bookkeeping record tying the above files to the docking box

### Relationship to the Curated Benchmark

The curated entry list lives in `scripts/pairs_curated.csv`. This directory is kept in sync
with that list (i.e., only curated targets are included here).

