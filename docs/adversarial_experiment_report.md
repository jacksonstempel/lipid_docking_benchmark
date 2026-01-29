# Adversarial Binding-Site Mutagenesis Report

**Date**: 2026-01-28  
**Dataset**: 100 lipid–protein complexes (single-chain benchmark)  
**Pocket definition**: side-chain heavy atoms within 5.0 A of lipid headgroup heavy atoms (experimental structure)  
**Mutations**: Gly sweep (G) and Phe sweep (F); ligand identity unchanged  
**Fold filter (main)**: mutant protein RMSD <= 2.0 A

## Executive Summary

Mutating binding-site residues strongly disrupts Boltz-2 lipid headgroup placement, even when the mutant proteins remain close to the experimental fold. Among WT-correct cases (WT headgroup RMSD < 3 A), only **23%** (Gly) and **20%** (Phe) remain accurate after mutation under the fold filter, indicating strong dependence on binding-site chemistry/geometry rather than positional memorization alone.

## Main Results (protein RMSD <= 2.0 A)

### Fold preservation

- Gly fold-pass: **89/100 (89%)**
- Phe fold-pass: **91/100 (91%)**

### Outcome categories (headgroup RMSD; fold-passing mutants only)

Categories:

- Strong (“memorization-like”): < 3 A
- Partial: 3–6 A
- Physical response: > 6 A

Counts:

- WT (all 100): 88% <3 A; 6% 3–6 A; 6% >6 A
- Gly (N=89): **18** <3 A (20%); **36** 3–6 A (40%); **35** >6 A (39%)
- Phe (N=91): **16** <3 A (18%); **23** 3–6 A (25%); **52** >6 A (57%)

### Retention among WT-correct targets (Masters-style)

Define WT-correct as WT headgroup RMSD < 3 A (88 targets).

Among WT-correct targets that also pass the mutant fold filter:

- Gly retention: **18/79 = 23%**
- Phe retention: **16/81 = 20%**

## Interpretation (high-level)

- The large drop from WT accuracy (88% <3 A) to mutant accuracy (~20% <3 A) under a fold filter supports that Boltz-2’s headgroup placement is highly sensitive to binding-site residues.
- A minority (~20%) of targets remain accurate despite extreme mutation, motivating the resistant-case analysis (see `docs/resistant_case_analysis_report.md`).

## File Locations (Reproducibility)

Inputs:

- WT benchmark: `output/benchmark/benchmark_summary.csv`, `output/benchmark/benchmark_allposes.csv`
- Mutant benchmarks:
  - `output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly/benchmark_summary.csv`
  - `output/adversarial/bs_mutagenesis_cutoff5A/benchmark_phe/benchmark_summary.csv`
- Mutation log: `output/adversarial/bs_mutagenesis_cutoff5A/mutation_summary.csv`

Figures:

- Main adversarial figure (protein RMSD <= 2.0 A): `manuscript/figures/fig_adversarial_mutagenesis_prot2A.pdf`
- Sensitivity (protein RMSD <= 1.5 A): `manuscript/figures/fig_adversarial_mutagenesis_prot1.5A.pdf`

Unified database used for manuscript figures/tables:

- `output/benchmark/benchmark_full.sqlite`

## How to regenerate the paper figures/tables

```bash
python scripts/analyze_benchmark_db.py \
  --db output/benchmark/benchmark_full.sqlite \
  --out-dir output/analysis/db_pipeline \
  --fig-dir manuscript/figures
```
