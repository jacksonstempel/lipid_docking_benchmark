# Vina Inputs

Inputs for reproducing the AutoDock Vina docking stage on the curated benchmark
targets.

## Contents

- `box/`
  - One `<PDBID>.txt` file per curated target.
  - Format (one line): `center_x center_y center_z size_x size_y size_z`

- `prep/<PDBID>/`
  - `receptor_no_ligand.pdb`: protein-only receptor structure used for docking
  - `receptor.pdbqt`: receptor converted to PDBQT
  - `ligand.pdb`: extracted lipid ligand
  - `ligand.pdbqt`: ligand converted to PDBQT
  - `run_manifest.json`: bookkeeping record tying the above files to the docking box

The curated entry list lives in `structures/benchmark_entries.csv`, and the
files in this directory correspond one-to-one with that list.
