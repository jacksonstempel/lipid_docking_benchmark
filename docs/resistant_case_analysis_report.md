# Memorization-Resistant Case Analysis Report (Gly arm)

**Date**: 2026-01-28  
**Experiment context**: Adversarial binding-site mutagenesis (5.0 A pocket definition; Gly sweep)  
**Dataset**: 100 lipid--protein complexes (single-chain benchmark)

## Executive Summary

In the adversarial mutagenesis experiment, a minority of targets (~20%) retained accurate lipid headgroup placement after binding-site residues were mutated to glycine. Here we analyze those **“resistant”** targets to ask whether they are explained by (i) weaker perturbation (fewer mutations), (ii) more interaction redundancy (more contacts), or (iii) binding modes that rely more on **protein backbone geometry** (mutation-invariant).

**Key finding**: Resistant targets are **not** explained by fewer mutations or more headgroup contacts overall. Instead, they show a **significantly higher fraction of backbone-mediated headgroup contacts** in the experimental structures, consistent with binding modes that are less dependent on side-chain chemistry.

## Definitions (Gly arm)

- **Fold filter (main)**: keep targets with mutant protein RMSD <= 2.0 A (to isolate binding-site effects from global refolding)
- **Resistant**: Gly headgroup RMSD < 3.0 A and fold-pass
- **Sensitive**: Gly headgroup RMSD >= 3.0 A and fold-pass

**Cohort sizes (protein RMSD <= 2.0 A)**

- Total targets: 100
- Excluded for fold failure (protein RMSD > 2.0 A): 11 (`1H6H`, `1LV2`, `1PZ4`, `2WEW`, `2WEX`, `3FYS`, `3PEG`, `4X9X`, `7Z6W`, `8QJZ`, `8T5T`)
- Included (fold-pass): 89
  - Resistant: 18
  - Sensitive: 71

## Methods (high-level)

### Mutation count
We use the logged number of actual Gly substitutions per target (`mutation_summary.csv`).

### Headgroup contact analysis (5.0 A)
From the experimental structure for each target:

1. Identify the lipid headgroup atoms (same functional-group heuristic used in the benchmark).
2. Count **protein heavy atoms** within **5.0 A** of any headgroup heavy atom.
3. Classify each contacting atom as:
   - **Backbone**: N, CA, C, O
   - **Side chain**: everything else
4. Compute:
   - **Total headgroup contacts**
   - **Backbone contact fraction**

### Statistical comparisons
For continuous metrics (mutation count, contacts, fractions, WT RMSD), we compare resistant vs sensitive using a two-sided Mann-Whitney U test.

## Results (protein RMSD <= 2.0 A)

| Metric | Resistant (N=18) | Sensitive (N=71) | p-value |
|---|---:|---:|---:|
| Mutation count (Gly) | mean 6.22 (median 6) | mean 6.62 (median 7) | 0.56 |
| Headgroup contacts within 5.0 A | mean 22.67 (median 25) | mean 22.13 (median 20) | 0.83 |
| Backbone contact fraction | mean 0.290 (median 0.290) | mean 0.157 (median 0.00) | **0.013** |
| WT headgroup RMSD (Boltz) | mean 1.13 A (median 1.00 A) | mean 2.66 A (median 1.76 A) | 0.034 |

Interpretation:

- **No evidence of under-perturbation**: resistant targets do not have fewer mutations on average.
- **No evidence of contact redundancy**: resistant targets do not have more headgroup contacts overall at 5.0 A.
- **Clear structural signature**: resistant targets have a higher **backbone fraction** of headgroup contacts, suggesting a binding geometry that is less dependent on side-chain chemistry.
- **WT accuracy trend**: resistant targets tend to be more accurate already in WT (better WT headgroup RMSD), consistent with a genuine retention effect rather than pure threshold noise.

## Sensitivity to a Stricter Fold Filter

Repeating the same analysis with protein RMSD <= 1.5 A yields similar conclusions:

- Resistant: 16, Sensitive: 63
- Backbone contact fraction remains higher in Resistant (p = 0.0068)
- Mutation count and total contacts remain non-different (p = 0.94 and p = 0.57, respectively)

## File Locations (Reproducibility)

Inputs:

- WT summary: `output/benchmark/benchmark_summary.csv`
- Gly mutant summary: `output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly/benchmark_summary.csv`
- Mutation log: `output/adversarial/bs_mutagenesis_cutoff5A/mutation_summary.csv`
- Experimental structures: `structures/experimental/`

Outputs:

- Main analysis (protein RMSD <= 2.0 A): `output/adversarial/bs_mutagenesis_cutoff5A/resistant_case_analysis_contact5A_prot2A/`
- Sensitivity (protein RMSD <= 1.5 A): `output/adversarial/bs_mutagenesis_cutoff5A/resistant_case_analysis_contact5A_prot1.5A/`
