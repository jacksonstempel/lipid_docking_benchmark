# Lipid Docking Benchmark – Pipeline Overview

This document explains how the benchmarking workflow is organised after the refactor. It
covers the end‑to‑end processing steps and the responsibility of each module or script
under `scripts/`.

## 1. End-to-End Pipeline

1. **Configuration & path resolution**
   - The CLI loads `config.yaml` via `scripts.lib.config.load_config`, then constructs a
     `scripts.lib.paths.PathResolver` to discover references, predictions, analysis, and
     aggregate locations. CLI overrides (e.g., `--preds`) are folded into this resolver.

2. **Structure loading**
   - `run_pose_benchmark` reads the reference structure (typically mmCIF) and the predicted
     structure using `scripts.lib.structures.load_structure`, which supports CIF, PDB, and
     PDBQT files.
   - If the prediction contains multiple models, `scripts.lib.structures.split_models`
     slices out the requested number of poses.

3. **Protein alignment**
   - Each pose is guaranteed to include a protein backbone: when the prediction is
     ligand-only, `scripts.lib.structures.ensure_protein_backbone` grafts the reference
     protein into the pose for alignment context.
   - `scripts.lib.alignment.extract_chain_sequences` turns each chain into a sequence and
     Cα coordinate series. The pruned Kabsch alignment (`chimera_pruned_fit`) finds the
     best rigid-body transform to superpose predicted Cα atoms with the reference while
     iteratively removing outliers.
  - The resulting transform is applied in-memory so subsequent ligand comparisons and
    pocket fits operate in the reference frame (no files are written at this stage).

4. **Ligand discovery & normalisation**
   - Ligands are collected from both structures using
     `scripts.lib.ligands.collect_ligands`, which deduplicates alternate conformations and
     ignores waters by default.
   - Optional atom-name templates (`docking/prep/<PDB>/ligand.pdb`) are loaded via
     `load_ligand_template_names` so predictions from different sources share identical
     atom naming for comparison.
   - Small ligands (fewer than `MIN_LIGAND_HEAVY_ATOMS` heavy atoms) are filtered unless the
     CLI requests to keep them.

5. **Ligand matching & scoring**
   - For every pose, the pipeline builds a cost matrix using locked global RMSD computed by
     `scripts.lib.ligands.locked_rmsd` over by-name heavy atom pairs. The Hungarian
     algorithm assigns predicted ligands to reference ligands optimally.
   - Optional pocket-only refinement (`scripts.lib.pockets.local_pocket_fit`) realigns the
     local protein environment to provide a second RMSD in the pocket frame.

6. **Reporting & aggregation**
   - The pipeline returns structured detail rows (protein fits, ligand matches, optional
     per-pair diagnostics) plus a summary dictionary capturing the best pose, counts, and
     protein RMSDs.
   - `scripts.lib.results_io.append_all_results` appends those detail rows to the shared
     `analysis/raw_data/all_results.csv`, preserving a single tidy table across runs.
   - `scripts.lib.results_io.build_and_write_summary` produces
     `analysis/aggregates/<label>/full_run_summary_<timestamp>.csv`, combining per-protein
     summaries with aggregate metrics (mean/median RMSDs, counts, etc.).
   - CLI callers receive the summary dict so they can log the best pose immediately.

7. **Batch processing (optional)**
   - `scripts.benchmark_runner_min` executes the pipeline in-process for each target,
     collects the returned rows, and emits the same aggregate outputs once per run
     without creating per-target artefacts.

## 2. Library Module Responsibilities

| Module | Purpose |
| --- | --- |
| `scripts/lib/__init__.py` | Re-exports core helpers for convenience when imported as a package. |
| `scripts/lib/constants.py` | Houses shared constants (three-letter to one-letter amino-acid map, recognised water residue names, minimum heavy-atom threshold). |
| `scripts/lib/config.py` | Loads `config.yaml`, resolves relative paths against the repository root, and exposes a strongly-typed `Config` object with nested path/script settings. |
| `scripts/lib/paths.py` | Implements `PathResolver`, which merges config defaults with CLI overrides, provides per-target analysis paths, and contains discovery helpers to locate reference and prediction files. |
| `scripts/lib/structures.py` | High-level structure utilities: loading CIF/PDB/PDBQT files, cloning/reslicing models, checking for protein content, applying rigid transforms, and writing aligned structures to disk. |
| `scripts/lib/alignment.py` | Converts Gemmi chains into alignable sequences (`ChainSeq`), matches chains by sequence identity, and runs the Chimera-style pruned Kabsch fit (`FitResult`) to obtain protein alignment metrics. |
| `scripts/lib/ligands.py` | Normalises ligand residues into lightweight data classes (`SimpleResidue`), deduplicates altLocs, applies optional atom-name templates, filters out small ligands, and provides locked RMSD computation utilities. |
| `scripts/lib/pockets.py` | Identifies protein residues around the reference ligand (within a configurable radius), maps them to the aligned prediction, and performs pocket-local rigid fits when enough Cα pairs exist. |
| `scripts/lib/results_io.py` | Defines stable schemas for the unified outputs, appends detail rows to `analysis/raw_data/all_results.csv`, infers source labels, and writes per-run summary CSVs with aggregate statistics. |
| `scripts/lib/pose_pipeline.py` | The orchestrator called by CLI scripts. Sequences the entire workflow from structure loading through ligand scoring and reporting, returning the summary dictionary for downstream use. |

## 3. Entry-Point & Utility Scripts

| Script | Description |
| --- | --- |
| `scripts/pose_benchmark.py` | Primary CLI. Parses arguments, resolves file paths (with CLI overrides), calls `run_pose_benchmark`, and logs the best pose metrics. Designed to handle predictions from Boltz, Vina, MOE, or any CIF/PDB exporter. |
| `scripts/benchmark_runner_min.py` | Batch driver. Discovers available targets, runs the pose pipeline in-process for each, and writes unified aggregate outputs (`all_results.csv` plus the per-run summary CSV). |
| `scripts/prep_vina_from_refs.py` | Generates AutoDock Vina preparation inputs (ligand and receptor PDBQT files) from the reference structures, simplifying downstream Vina runs. |
| `scripts/run_boltz.sh` | Shell helper invoked to launch Boltz sampling jobs with consistent arguments, typically used in larger automation pipelines. |
| `scripts/run_vina.py` | Python wrapper for running Vina across multiple targets using the prepared inputs, collecting resulting PDBQT poses under `model_outputs/vina/`. |

## 4. Suggested Reading Order

For contributors new to the project:

1. Start with `scripts/pose_benchmark.py` to see the CLI interface.
2. Skim `scripts/lib/pose_pipeline.py` to understand the orchestration.
3. Dive into the supporting modules (`structures`, `alignment`, `ligands`, `pockets`, `results_io`) as needed when modifying a specific stage.
4. Review batch utilities (`benchmark_runner_min.py`) when working on multi-target workflows or aggregate reports.

Keeping this document updated alongside code changes will ensure the benchmarking workflow
remains transparent to both domain scientists and developers.
