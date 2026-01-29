# Adversarial Binding-Site Mutagenesis (Boltz-2, Lipid-Protein Complexes)

## Goal (biological question)

We want to test whether Boltz-2 places lipid headgroups correctly because it has learned **generalizable physical/chemical rules** (e.g., where polar headgroups should interact), or because it can sometimes **memorize training-like binding geometries** (placing ligands where it “expects” them even when local chemistry is disrupted).

The core idea:

- If we **destroy binding-site side-chain chemistry** in the input sequence, a chemistry/physics-sensitive model should often **move the lipid headgroup**.
- If the model still predicts the headgroup in essentially the **same place as the experimental structure**, that is “memorization-like” behavior (or a binding mode dominated by mutation-invariant backbone geometry).

This experiment is inspired by Masters et al. (2025), adapted to lipid–protein complexes.

## Dataset

- **100 lipid–protein complexes** (the same targets used in the baseline benchmark)
- Each target has:
  - experimental reference complex (structure)
  - WT Boltz-2 prediction
  - WT Boltz input YAML (protein sequence + ligand identity)

Important constraint:

- **One protein chain per entry**. If an entry has two protein chains in the Boltz input, it is an input error and must be fixed before running.

## Definitions

### Lipid headgroup atoms

We use the same headgroup definition as the benchmark (functional-group oriented). In short: identify the “polar/charged part” of the lipid (e.g., phosphate region, quaternary amine region, carbonyl/hetero region) and compute headgroup RMSD on those atoms.

### Binding-site (“headgroup-contacting”) residues

For each target, define binding-site residues using the **experimental structure**:

- Select all protein residues whose **side-chain heavy atoms** (non-hydrogen; excluding N/CA/C/O backbone atoms) are within **5.0 A** of any **lipid headgroup heavy atom**.

Rationale: for lipids, using all ligand atoms would make the long hydrophobic tail dominate the contact list and would mutate an unrealistically large fraction of the protein.

## Adversarial mutations (two arms)

For each target, create two mutant versions by changing only the **protein sequence** that Boltz-2 receives as input:

1. **Gly sweep (G)**: mutate every binding-site residue to glycine.
   - Interpretation: removes side-chain chemistry while preserving backbone.
2. **Phe sweep (F)**: mutate every binding-site residue to phenylalanine.
   - Interpretation: adds steric bulk that can occlude the pocket.

Notes:

- The ligand identity is unchanged (no ligand modifications).
- Record both:
  - **Targeted residues** (selected by the 5.0 A rule)
  - **Actual substitutions** (positions that truly change; e.g., Phe->Phe is a no-op in the Phe arm)

## Boltz-2 prediction runs (ISAAC)

Run Boltz-2 twice over the full dataset:

- One run for the Gly mutant set
- One run for the Phe mutant set

Run settings must match the baseline Boltz configuration (same Boltz version and inference parameters).

## Evaluation (benchmark pipeline)

Evaluate WT, Gly-mutant, and Phe-mutant predictions against the experimental structures using the same benchmark code.

For each target, compute:

- **Protein RMSD** (protein fold similarity to experimental; after alignment)
- **Headgroup RMSD** (primary readout; headgroup-only)
- **Ligand RMSD** (supporting readout; full lipid)

### Fold-quality filter (exclude refolds)

To avoid confounding by global refolding, apply a fold filter:

- Main filter: **protein RMSD <= 2.0 A**
- Sensitivity: also report a stricter filter **protein RMSD <= 1.5 A**

Targets failing the fold filter are excluded from mutant outcome summaries.

## Primary readouts

### A) Outcome categories (Panel A)

Classify each prediction using headgroup RMSD:

- **Strong (“memorization-like”)**: headgroup RMSD < 3.0 A
- **Partial response**: 3.0 A <= headgroup RMSD <= 6.0 A
- **Physical response**: headgroup RMSD > 6.0 A

Reporting:

- WT category fractions are computed over all 100 targets.
- Mutant category fractions are computed over **fold-passing** targets only.

### B) Retention among WT-correct targets (Panel B; Masters-style)

Among targets that are WT-correct (WT headgroup RMSD < 3.0 A), ask what fraction remain correct after mutation:

- Gly retention = fraction with mutant headgroup RMSD < 3.0 A among WT-correct AND Gly fold-pass
- Phe retention = fraction with mutant headgroup RMSD < 3.0 A among WT-correct AND Phe fold-pass

Report Wilson 95% confidence intervals for retention proportions.

## Outputs and “single source of truth”

All experiment artifacts live under:

- `output/adversarial/bs_mutagenesis_cutoff5A/`

Key files:

- `benchmark_gly/benchmark_summary.csv` and `benchmark_gly/benchmark_allposes.csv`
- `benchmark_phe/benchmark_summary.csv` and `benchmark_phe/benchmark_allposes.csv`
- `mutation_summary.csv` (mutation counts and residue lists)
- Resistant-case analysis outputs (Gly arm; 5.0 A contacts):
  - `resistant_case_analysis_contact5A_prot2A/`
  - `resistant_case_analysis_contact5A_prot1.5A/` (sensitivity)

The unified database used for manuscript figures/tables is:

- `output/benchmark/benchmark_full.sqlite`

## How to reproduce (commands)

Baseline benchmark:

```bash
python scripts/benchmark.py
```

Adversarial evaluation (after Boltz mutant structures are available and paired):

```bash
python scripts/benchmark.py --pairs output/adversarial/bs_mutagenesis_cutoff5A/pairs_gly.csv \\
  --out-dir output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly

python scripts/benchmark.py --pairs output/adversarial/bs_mutagenesis_cutoff5A/pairs_phe.csv \\
  --out-dir output/adversarial/bs_mutagenesis_cutoff5A/benchmark_phe
```

Resistant-case analysis (Gly arm):

```bash
python scripts/resistant_case_analysis.py \\
  --wt-summary output/benchmark/benchmark_summary.csv \\
  --gly-summary output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly/benchmark_summary.csv \\
  --mutation-summary output/adversarial/bs_mutagenesis_cutoff5A/mutation_summary.csv \\
  --structures-dir structures/experimental \\
  --protein-rmsd-cutoff 2.0 \\
  --headgroup-rmsd-cutoff 3.0 \\
  --contact-cutoff 5.0 \\
  --out-dir output/adversarial/bs_mutagenesis_cutoff5A/resistant_case_analysis_contact5A_prot2A
```

Figures/tables from the unified database:

```bash
python scripts/analyze_benchmark_db.py \\
  --db output/benchmark/benchmark_full.sqlite \\
  --out-dir output/analysis/db_pipeline \\
  --fig-dir manuscript/figures
```
