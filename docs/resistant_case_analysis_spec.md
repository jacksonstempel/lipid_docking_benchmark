# Memorization-Resistant Case Analysis

## Objective

Investigate why ~20% of lipid-protein complexes retain accurate headgroup placement despite binding-site mutagenesis, while ~80% show substantial degradation.

---

## Case Definitions

Using the Gly arm results (cleaner interpretation than Phe):

- **Fold filter (main)**: mutant protein RMSD \(\le\) **2.0 Å**
- **Resistant**: Headgroup RMSD < 3.0 Å after Gly mutation AND fold-pass
- **Sensitive**: Headgroup RMSD ≥ 3.0 Å after Gly mutation AND fold-pass

Optional sensitivity analysis:

- Repeat the same analysis with a stricter fold filter (protein RMSD \(\le\) 1.5 Å) to confirm conclusions do not depend on borderline refolding cases.

---

## Analyses

### 1. Lipid Class Breakdown

**Question**: Are resistant cases enriched for specific lipid classes?

**Method**: Tabulate lipid class (fatty acid, phospholipid, glycolipid, sterol, etc.) for resistant vs sensitive groups. Fisher's exact test if any class appears notably enriched.

**Output**: Contingency table with counts and percentages per group.

---

### 2. Mutation Count Comparison

**Question**: Do resistant cases have fewer mutated residues?

**Method**: Compare mean and median mutation count between resistant and sensitive groups. Mann-Whitney U test for significance.

**Output**: 

| Group | N | Mean | Median | p-value |
|-------|---|------|--------|---------|

---

### 3. Total Contact Count

**Question**: Do resistant cases have more contacts (redundancy) or fewer (less to lose)?

**Method**: For each complex, count protein heavy atoms within 5.0 Å of any headgroup heavy atom in the experimental reference structure. Compare distributions between groups.

**Output**:

| Group | Mean contacts | Median | p-value |
|-------|---------------|--------|---------|

---

### 4. Backbone vs Side-Chain Contact Fraction

**Question**: Do resistant cases rely more on backbone contacts (which are mutation-invariant)?

**Method**: For each contact identified in Analysis 3, classify as backbone (atom name N, CA, C, or O) or side-chain (everything else). Calculate the fraction of contacts that are backbone for each complex. Compare fractions between groups.

**Output**:

| Group | Mean backbone fraction | Median | p-value |
|-------|------------------------|--------|---------|

---

### 5. Wild-Type Prediction Accuracy

**Question**: Were resistant cases already marginal (near 3 Å threshold) or highly accurate?

**Method**: Pull wild-type headgroup RMSD for both groups. Compare distributions. Report fraction with WT RMSD < 1.5 Å (excellent) vs 1.5–3.0 Å (good but marginal).

**Output**:

| Group | Mean WT RMSD | Median | Fraction < 1.5 Å |
|-------|--------------|--------|------------------|

---

### 6. Headgroup Chemistry Breakdown

**Question**: Are certain headgroup chemistries enriched in resistant cases?

**Method**: Classify each lipid by dominant headgroup polar feature:
- Phosphate-containing
- Carboxylate
- Hydroxyl-dominant  
- Amine/quaternary nitrogen

Compare distribution between groups.

**Output**: Contingency table with counts per headgroup type.

---

## Data Sources

| Data needed | Source |
|-------------|--------|
| Gly mutant headgroup RMSD | `benchmark_gly/benchmark_summary.csv` |
| Gly mutant protein RMSD | `benchmark_gly/benchmark_summary.csv` |
| Wild-type headgroup RMSD | `benchmark/benchmark_summary.csv` |
| Mutation counts | `mutation_summary.csv` |
| Lipid class/identity | Original benchmark metadata |
| Contact analysis | Compute from experimental structures |

---

## Expected Output

Final summary table:

| Metric | Resistant (N≈21) | Sensitive (N≈76) | p-value | 
|--------|------------------|------------------|---------|
| Mutation count | | | |
| Total contacts | | | |
| Backbone contact fraction | | | |
| WT headgroup RMSD | | | |

Plus contingency tables for lipid class and headgroup chemistry.

---

## Interpretation Guide

- **Higher backbone fraction in resistant cases**: Suggests these lipids are positioned by backbone geometry, which mutations don't affect. Not memorization—correct physical reasoning.

- **Fewer mutations in resistant cases**: Suggests insufficient perturbation rather than true resistance.

- **More total contacts in resistant cases**: Suggests redundancy—even after losing some interactions, enough remain.

- **Marginal WT accuracy in resistant cases**: Suggests "retention" is noise around the 3 Å threshold rather than true robustness.

- **Lipid class enrichment**: Suggests class-specific binding modes or training data density effects.

- **No distinguishing features**: Suggests the ~20% is simply variance in the dataset.
