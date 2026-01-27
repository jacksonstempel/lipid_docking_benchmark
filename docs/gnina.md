# GNINA on ISAAC (Apptainer)

This repo includes helper scripts to run [GNINA](https://github.com/gnina/gnina) on the curated benchmark targets using the existing Vina reproducibility inputs.

## Inputs

The scripts reuse `prediction_inputs/vina_inputs/` (same layout as the Vina Slurm scripts):

- `box/<PDBID>.txt` with one line: `center_x center_y center_z size_x size_y size_z`
- `prep/<PDBID>/receptor_no_ligand.pdb`
- `prep/<PDBID>/ligand.pdbqt`

## Build a GNINA image (one-time)

ISAAC-NG documentation recommends *pulling* container images (building typically requires elevated privileges). Pull an Apptainer image from the official Docker image:

```bash
apptainer pull "$HOME/gnina.sif" docker://gnina/gnina:latest
```

## Submit a run

Use the submit wrapper:

```bash
bash scripts/submit_gnina_isaac_array.sh \
  -A <ACCOUNT> -p <GPU_PARTITION> --qos <GPU_QOS> \
  --input-dir "$HOME/vina_inputs" \
  --out-root "$SCRATCHDIR/gnina/runs/gnina_isaac_run" \
  --gnina-sif "$HOME/gnina.sif" \
  --cuda-module cuda/12.2.0-binary \
  --cpus-per-task 8 --gpus 1 \
  --exhaustiveness 8 --num-modes 20 --seed 0
```

If you hit Slurm submission limits, use sharding:

```bash
bash scripts/submit_gnina_isaac_array.sh \
  -A <ACCOUNT> -p <GPU_PARTITION> --qos <GPU_QOS> \
  --input-dir "$HOME/vina_inputs" \
  --out-root "$SCRATCHDIR/gnina/runs/gnina_isaac_run" \
  --gnina-sif "$HOME/gnina.sif" \
  --shards 10
```

Outputs land under:

- `OUT_ROOT/flat/<PDBID>.pdbqt`
- `OUT_ROOT/logs/<PDBID>.log`

## Notes on formats

GNINA commonly uses SDF for ligands. The job script converts the prepared ligand PDBQT to SDF inside the container (OpenBabel) before docking, then writes GNINA output poses to PDBQT to keep downstream evaluation compatible with the existing pipeline.

## Evaluating GNINA outputs with this repo

1) Copy (or symlink) the GNINA outputs to a stable local location under `output/` (recommended):

- `output/gnina/runs/<run_name>/flat/`
- `output/gnina/runs/<run_name>/logs/`

2) Create a benchmark entry CSV that points at those GNINA PDBQTs. The simplest approach is to reuse the existing Vina helper script by pointing it at the GNINA directory:

```bash
python scripts/make_pairs_with_vina_dir.py \
  --vina-dir output/gnina/runs/<run_name>/flat \
  --out-csv structures/benchmark_entries_gnina.csv
```

3) Run the benchmark using that CSV (see `scripts/benchmark.py` help for an entry-list override flag, or temporarily swap the config path if you prefer).

## Reproducible ISAAC commands

We keep exact “run recipes” (submission commands used during development) in `docs/isaac/gnina_recipes.md`.
