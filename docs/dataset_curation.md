# Dataset Curation Guidelines

This benchmark assumes that every reference structure represents a single, well-defined
protein–ligand complex. Targets that violate this expectation should be excluded in order
to keep the evaluation focused on ligand placement rather than pathological edge cases.

## Exclusion criteria

- **Ambiguous biological assemblies.** Structures that include multiple copies of the
  protein and ligand (e.g. symmetry mates with duplicated ligands) produce ambiguous
  “correct” poses. Example: 5C79 contains three PBU molecules sandwiched between two
  copies of the receptor; RMSD comparisons depend on which symmetry mate is chosen.
- **Large unresolved segments near the binding pocket.** When tens of residues are missing
  from the experimental model, co-folding methods (e.g. Boltz) have no template to align.
  The predicted pocket often drifts, producing large RMSDs unrelated to docking quality.
  Example: 5TCX includes a 17-residue break proximal to the ligand.
- **Ligand annotation inconsistencies.** Structures where the ligand identity or atom
  naming changes across repetitions (multiple hetero groups representing the same compound)
  should be fixed or dropped.
- **Non-canonical ligands or modified residues** that are not handled by the template
  renaming logic may require bespoke preprocessing.

Targets that meet any of the above criteria should be removed from:

1. `raw_structures/benchmark_references/`
2. `model_outputs/boltz/`
3. `model_outputs/vina/`
4. Any historical aggregate CSVs or detail logs.

## Current exclusions

As of the latest revision, the following PDB IDs have been removed:

| PDB ID | Reason |
| ------ | ------ |
| 5C79   | Multiple copies of the ligand between symmetry mates; ambiguous reference pose |
| 5TCX   | Large chain break (≈17 residues) near the binding site; co-folded protein drifts |
| 3QLM   | Ligand sits between symmetry-related proteins; multiple equivalent ligand copies |

## Recommendations

- Maintain a manifest (YAML/JSON) of vetted benchmark targets, checked in with the repo.
- Automate assembly checks (e.g. via Gemmi) to flag new entries with duplicated ligands or
  missing residues before they enter the benchmark.
- Log excluded targets during batch runs so downstream analyses are aware of dropped cases.
- When comparing to historical aggregates, regenerate baselines using the current code and
  the same curated dataset to avoid mixing incompatible runs.
