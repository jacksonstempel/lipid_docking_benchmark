# Lipid Docking Benchmark

A reproducible benchmark for evaluating AI-based molecular docking methods on lipid–protein complexes. Compares **Boltz** and **AutoDock Vina** predictions against experimental structures using RMSD and contact-level metrics.

## Overview

This pipeline evaluates docking predictions through two complementary lenses:

1. **Geometric accuracy** — Ligand heavy-atom RMSD after protein backbone alignment
2. **Interaction fidelity** — Precision, recall, and F1 of predicted ligand–protein contacts (via PandaMap)

One command runs the full benchmark and produces publication-ready CSV outputs.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd lipid_docking_benchmark

# Install Python dependencies (pinned for reproducibility)
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Software Versions

For exact reproducibility of published results:

| Software | Version |
|----------|---------|
| Boltz | 2.2.0 |
| AutoDock Vina | 23d1252-mod |
| Python dependencies | See `requirements.txt` |

## Quick Start

Run the complete benchmark:

```bash
python scripts/run_full_benchmark.py
```

This executes the full pipeline:
1. Computes ligand RMSD for Boltz (1 pose) and Vina (20 poses)
2. Extracts ligand–protein contacts via PandaMap
3. Calculates contact-level metrics against reference structures

**Outputs** (in `analysis/final/`):
- `full_benchmark_allposes_<timestamp>.csv` — Per-pose metrics
- `full_benchmark_summary_<timestamp>.csv` — Aggregated results (Boltz, Vina best, Vina median)

## Usage

### Full Benchmark (Recommended)

```bash
python scripts/run_full_benchmark.py
```

### Individual Components

**Single structure evaluation:**
```bash
python scripts/measure_ligand_pose.py \
    --ref benchmark_references/1ABC.cif \
    --pred model_outputs/boltz/1ABC_model_0.cif
```

**Batch RMSD calculation:**
```bash
# Boltz predictions
python scripts/measure_ligand_pose_batch.py --kind boltz

# Vina predictions (20 poses per target)
python scripts/measure_ligand_pose_batch.py --kind vina --max-poses 20
```

**Contact extraction only:**
```bash
python contact_tools/run_batch_contacts.py
```

**Contact metrics only:**
```bash
python scripts/compute_contact_metrics.py \
    --boltz-rmsd analysis/tmp/boltz_batch_results.csv \
    --vina-rmsd analysis/tmp/vina_batch_results.csv
```

## Output Format

### Per-Pose CSV

Each row represents one evaluated pose:

| Column | Description |
|--------|-------------|
| `pdbid` | PDB identifier |
| `method` | `boltz`, `vina_pose` |
| `pose_index` | Pose number (1-indexed) |
| `ligand_rmsd` | Heavy-atom RMSD (Å) |
| `protein_rmsd` | Backbone Cα RMSD (Å) |
| `precision` | Contact precision (strict) |
| `recall` | Contact recall (strict) |
| `f1` | Contact F1 score |
| `jaccard` | Jaccard similarity |
| `residue_f1` | Residue-level F1 |
| `status` | `ok` or `error` |
| `error` | Error message if failed |

### Summary CSV

Three rows per PDB ID:
- `boltz` — Single Boltz prediction
- `vina_best` — Vina pose with lowest RMSD
- `vina_median` — Median across all 20 Vina poses

## Methodology

### Protein Alignment

Structures are aligned using Cα atoms with ChimeraX-style iterative pruning:
1. Initial Kabsch superposition on all matched Cα pairs
2. Iteratively remove outliers beyond 2.0 Å
3. Final RMSD computed on pruned set

### Ligand Selection

The pipeline automatically identifies the primary ligand by:
1. Excluding protein residues, water, common solvents, and ions
2. Filtering fragments with fewer than 10 heavy atoms
3. Selecting the largest remaining candidate

### Atom Pairing

Ligand atoms are matched using RDKit's Maximum Common Substructure (MCS) algorithm. A minimum 90% coverage of reference heavy atoms is required to ensure meaningful comparisons.

### Contact Detection

Ligand–protein interactions are identified using PandaMap, which classifies contacts by type (hydrogen bonds, hydrophobic, etc.) and reports distances.

## Directory Structure

```
lipid_docking_benchmark/
├── benchmark_references/     # Experimental structures (*.cif)
├── model_outputs/
│   ├── boltz/               # Boltz predictions (*_model_0.cif)
│   └── vina/                # Vina predictions (*.pdbqt)
├── scripts/
│   ├── pairs.csv            # Input manifest (pdbid, ref, predictions)
│   ├── run_full_benchmark.py
│   ├── measure_ligand_pose.py
│   ├── measure_ligand_pose_batch.py
│   ├── compute_contact_metrics.py
│   └── lib/                 # Core algorithms
├── contact_tools/           # PandaMap integration
├── analysis/
│   └── final/              # Output CSVs
├── tests/                   # Unit and integration tests
└── docs/
    └── removed_pdbs.txt    # Excluded entries with reasons
```

## Running Tests

```bash
python -m unittest
```

