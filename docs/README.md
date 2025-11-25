# Purpose

This repo hosts a simple, end-to-end benchmark for lipid docking:

- RMSD-based pose evaluation for Boltz and Vina.
- Contact-level metrics (PandaMap) comparing ligand–protein interactions to the experimental reference.

The goal is a small, understandable pipeline: one command to run everything and a couple of CSVs with results.

# Quickstart

Prereqs: Python env with `numpy`, `pandas`, `gemmi`, `rdkit`, `pandamap`.

Run the full benchmark (RMSD + contacts) for all entries:
```bash
python scripts/run_full_benchmark.py
```

This will:
- Run RMSD benchmark for Boltz and Vina (20 poses per entry for Vina).
- Extract contacts for reference, Boltz, and all Vina poses via PandaMap.
- Compute contact metrics vs the reference.

Outputs:
- Per-pose metrics: `analysis/final/full_benchmark_allposes_<timestamp>.csv`
- Summary metrics: `analysis/final/full_benchmark_summary_<timestamp>.csv`

The summary CSV has exactly three rows per PDB:
- `method = boltz`
- `method = vina_best` (best RMSD pose among the 20)
- `method = vina_median` (median over the 20 poses per column)

Status and errors are captured per-row in the per-pose CSV (so failures are visible, not hidden).

# Directory layout (simplified)

- `benchmark_references/` — reference structures (`<pdbid>.cif`).
- `model_outputs/`
  - `boltz/*.cif` — Boltz predictions (flattened to top level). Extra files in `model_outputs/boltz/extra_output/`.
  - `vina/*.pdbqt` — Vina predictions (flattened). Extra files in `model_outputs/vina/extra_output/`.
- `scripts/`
  - `measure_ligand_pose.py` — single-pair CLI (one ref/pred structure).
  - `measure_ligand_pose_batch.py` — batch RMSD runner.
  - `run_full_benchmark.py` — one-command pipeline (RMSD + contacts + metrics).
  - `pairs.csv` — universal list of entries with columns: `pdbid, ref, boltz_pred, vina_pred`.
  - `lib/` — alignment, ligand handling, RMSD logic, and helpers.
- `contact_tools/`
  - `measure_contacts.py` — runs PandaMap for a single structure.
  - `run_batch_contacts.py` — runs contact extraction for all benchmark entries.
- `analysis/`
  - `final/` — final per-pose and summary CSVs from the full benchmark.
  - `plots/` — any analysis plots you generate manually.
- `docs/`
  - `README.md` — this file.
  - `removed_pdbs.txt` — list of PDB IDs removed from the benchmark and reasons.

# Behavior & assumptions

- Proteins are aligned (Cα, pruned Kabsch) and ligand RMSD is computed in the aligned frame.
- Ligand is auto-selected:
  - Ignores protein, water, common solvents, ions, and tiny fragments.
  - Uses simple size/name rules plus RDKit chemistry to pick the best-matching ligand if there are multiple candidates.
- Atom pairing is RDKit chem-only and **requires ≥90% of the reference ligand heavy atoms** to map:
  - If RDKit cannot achieve this, that PDB is excluded from contact-level analyses (and recorded explicitly).
  - Hydrogens are ignored everywhere.
- Vina:
  - Uses all 20 poses per complex.
  - `vina_best` is defined by the best ligand RMSD (consistent with the main RMSD benchmark).

# Contact-level metrics (PandaMap)

Contact extraction:
```bash
python contact_tools/run_batch_contacts.py
```
This writes three CSVs under `analysis/pandamap_contacts/`:
- `ref_contacts.csv`
- `boltz_contacts.csv`
- `vina_contacts.csv`

Each row is a single contact: `pdbid, pose_index, ligand_atom, residue, contact_type, distance`.

Contact metrics (if you want to run manually):
```bash
python scripts/compute_contact_metrics.py \
  --boltz-rmsd analysis/boltz_batch_results_YYYYMMDD_HHMMSS.csv \
  --vina-rmsd  analysis/vina_batch_results_YYYYMMDD_HHMMSS.csv
```

This writes:
- Per-pose contact metrics CSV (`analysis/contact_metrics_allposes_<timestamp>.csv`).
- Summary contact metrics CSV (`analysis/contact_metrics_summary_<timestamp>.csv`).

Columns include:
- Strict contact metrics (atom + residue + type):
  - `precision, recall, f1, jaccard, distance_mae`
  - `ref_contacts, pred_contacts, shared_contacts`
- Residue-level metrics:
  - `residue_precision, residue_recall, residue_f1, residue_jaccard`
  - `residue_shared, residue_ref_size, residue_pred_size`
- Plus ligand RMSD and protein RMSD carried through from the RMSD benchmark.

# Dataset notes

- The benchmark uses a curated set of entries where:
  - There is a clear primary lipid ligand.
  - RDKit can reliably map ligands between reference and predictions.
- PDB IDs that are ambiguous or problematic (multiple ligands, RDKit failures, poor coverage, mismatched ligands, missing outputs) are listed in `docs/removed_pdbs.txt` and removed from `scripts/pairs.csv`.

If you’re unsure why a particular PDB is missing from the benchmark, check `docs/removed_pdbs.txt`. The `status` and `error` fields in the per-pose CSVs should make any remaining failures explicit.
