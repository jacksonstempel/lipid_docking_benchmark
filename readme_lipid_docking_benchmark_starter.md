# lipid_docking_benchmark

**Author:** Jackson Stempel, UT/ORNL Center for Molecular Biophysics  
**Python:** 3.12.11 • **Env:** mamba/conda  

Short pitch: this repo benchmarks lipid **ligand pose** predictions against PDB ground truth. Given a reference mmCIF (protein+native lipid) and a predicted complex from Boltz, we compute protein RMSD and multiple ligand RMSDs (global vs pocket/locked), then aggregate one row **per protein** to track performance across the set.

---

## Repository layout (current)
```
lipid_docking_benchmark/
├─ analysis/                # per‑target reports + aggregate CSVs (generated)
├─ metadata/                # small static tables/configs (optional)
├─ model_inputs/            # YAML inputs for runs (not central to scoring)
├─ model_outputs/           # Boltz predictions (heavy; not meant for Git)
├─ raw_structures/          # (older path; superseded by benchmark_references/)
├─ benchmark_references/    # ground‑truth PDB/mmCIFs by PDB ID (preferred)
└─ scripts/                 # benchmarking + batch runner scripts
```
> Note: large artifacts (model_outputs, raw_structures, CIF/PDB files) should be **ignored in Git**; only code and small metadata live in the repo.

---

## Environment (mamba)
```bash
mamba create -n ldb python=3.12.11 -y
mamba activate ldb
# core deps (adjust as you finalize)
mamba install -c conda-forge numpy scipy pandas gemmi biopython rdkit -y
# optional: testing/formatting
# mamba install -c conda-forge pytest black -y
```

---

## Data locations (what scripts assume)
- **References (ground truth):** `benchmark_references/` with per‑ID files, e.g.
  - `~/lipid_docking_benchmark/benchmark_references/1FK1/1FK1.cif`
- **Predictions (Boltz outputs):** currently nested like:
  - `~/lipid_docking_benchmark/model_outputs/1FK1_output/boltz_results_1FK1/predictions/1FK1/1FK1_model_0.cif`

The batch runner knows this nesting and will find `*_model_0.cif` automatically (see **Auto‑discovery logic** below). A future cleanup will **flatten** prediction paths into a simpler canonical form.

> **WSL path tip:** Prefer Linux paths in commands (e.g., `~/lipid_docking_benchmark/...`) rather than Windows UNC like `\\wsl.localhost\Ubuntu\...` when running from a WSL terminal.

---

## Scripts

### `scripts/pose_benchmark.py`
Core evaluator for a **single** PDB ID. It
1) accepts a reference mmCIF and a predicted mmCIF,  
2) computes protein CA RMSD variants, ligand RMSDs (global vs pocket/locked), and counts (pairs, residues, atoms),  
3) writes a per‑target analysis folder under `analysis/<PDB>/<PDB>_analysis/` containing a CSV (and optionally JSON/plots depending on flags).

**Example (single target):**
```bash
python scripts/pose_benchmark.py 1FK1 \
  --ref  ~/lipid_docking_benchmark/benchmark_references/1FK1/1FK1.cif \
  --pred ~/lipid_docking_benchmark/model_outputs/1FK1_output/boltz_results_1FK1/predictions/1FK1/1FK1_model_0.cif \
  --full -v
```

### `scripts/benchmark_runner_min.py`
A **minimal, deterministic batch runner** that:
- discovers `(ref, pred)` pairs across your trees,
- calls `pose_benchmark.py` for each target,
- collects the per‑target CSVs, and
- writes a **condensed aggregate** CSV (one row per protein) to `analysis/aggregates/`.

**Usage (batch run):**
```bash
python scripts/benchmark_runner_min.py \
  --refs ~/lipid_docking_benchmark/benchmark_references \
  --preds ~/lipid_docking_benchmark/model_outputs \
  --pose ~/lipid_docking_benchmark/scripts/pose_benchmark.py \
  --full -v
```

**Optional:** restrict to a set of IDs with `--ids ids.txt` (one PDB ID per line).

**Where results land:**
```
analysis/aggregates/aggregate_{M}.{D}_{H}.{MIN}.csv
```

### `scripts/run_benchmark_cif_batch.py`
Legacy/alternate batch runner. Prefer `benchmark_runner_min.py` for a deterministic, slimmer pass and the condensed aggregate.

---

## Aggregate CSV schema (condensed, one row per protein)
The batch runner produces columns:

| column | meaning |
|---|---|
| `pdbid` | Protein/PDB identifier (uppercase) |
| `protein_rmsd` | Protein CA RMSD (prefers pruned/all‑under‑pruned if available) |
| `protein_rmsd_ca_allfit` | Protein CA RMSD (all‑fit alignment) |
| `ligand` | Ligand code/name (e.g., lipid CCD) for the **best** ligand row |
| `policy` | Pairing policy used during ligand matching |
| `rmsd_locked_global` | Ligand RMSD using global locked alignment (smaller is better) |
| `rmsd_locked_pocket` | Ligand RMSD using pocket‑locked alignment |
| `n_residues` | Protein residue count used in the pairs (from per‑target CSV) |
| `n_ligand_atoms` | Atoms in chosen ligand |
| `n_pocket_residues` | Residues in the identified pocket |

> When multiple ligand rows exist for a target, the runner picks the ligand with **minimum `rmsd_locked_global`**.

---

## Auto‑discovery logic (how targets are found)
**References:**
- Prefer `benchmark_references/<PDBID>/*.cif` (stem matching the ID). If multiple exist, the **shallower path** wins.

**Predictions:**
- Prefer the canonical pattern: `model_outputs/<ID>_output/boltz_results_<ID>/predictions/<ID>/<ID>_model_0.cif`.
- If multiples exist under `<ID>_output/`, choose by score:
  1) has a `/predictions/<ID>/` segment,
  2) shallower path (fewer segments),
  3) newer modification time,
  4) alphabetical tie‑break.

**Per‑target CSV collection:**
- Each successful run should create `analysis/<PDB>/<PDB>_analysis/<PDB>_analysis.csv`, which the batch runner ingests to build the aggregate.

---

## Typical workflow
1) Place/confirm reference files in `benchmark_references/<ID>/<ID>.cif`.
2) Generate Boltz predictions (they may land in the current nested layout under `model_outputs/`).
3) Run `benchmark_runner_min.py` with `--refs` and `--preds` (and `--ids` if needed).
4) Inspect `analysis/<ID>/...` for single‑target details; monitor `analysis/aggregates/aggregate_*.csv` for the study‑level view.

---

## Roadmap (near‑term)
- **Flatten predictions layout** to a simple canonical path (keep a back‑compat shim in the runner).
- Add unit tests with tiny CIF fixtures and run them in CI.
- Optional: a `Makefile` and VS Code tasks (`make run PDB=1FK1`, `make test`).

---

## A note on agents in VS Code
Use Copilot/ChatGPT inside VS Code to:
- refactor functions (pose pairing, RMSD calculators),
- add docstrings and type hints,
- write tests for the CSV aggregation and path discovery,
- propose the directory flattening PR.

Agents **do not** run code on your laptop; execute scripts from VS Code’s WSL terminal. Keep large files out of Git; use the repo for code and small config only.

