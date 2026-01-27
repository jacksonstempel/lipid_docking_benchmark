# TODO / Plan (Next Steps)

This document is the working plan for bringing the manuscript + analysis pipeline to a clean, reproducible, publication-ready state.

## 0) Immediate “stop-the-line” correctness issues

### GNINA contact metrics wiring bug (must fix before submission)
- Symptom: `gnina_cnn_top1_head_env_jaccard` and `gnina_nocnn_top1_head_env_jaccard` (and typed analogs) are identical across all targets in `output/analysis/db_pipeline/per_target.csv`, while GNINA CNN/no-CNN RMSDs differ.
- Why it matters: It can look like a copy/paste error or a flawed pipeline; reviewers will flag internal inconsistency immediately.
- Likely cause: Contact metrics are being computed from the wrong ligand coordinates (e.g., accidentally reusing one method’s pose for both CNN/no-CNN, or using an input ligand rather than the GNINA output pose).
- Concrete tasks:
  - Trace the GNINA “contact metric” computation path end-to-end (from ISAAC output `flat/*.pdbqt` and `logs/*.log` through benchmarking CSVs into the DB and `per_target.csv`).
  - Confirm whether the GNINA allposes CSVs used for DB build contain distinct CNN/no-CNN contact metrics for pose 1. If not, fix at the benchmark step; if yes, fix at the DB/analysis step.
  - Add a small regression test:
    - Assert CNN vs no-CNN contact metrics differ for at least one target when RMSDs differ (or assert the underlying per-pose files differ and metrics are computed from those).
  - Rebuild the DB and regenerate:
    - `output/analysis/db_pipeline/per_target.csv`
    - Figure 4 and Table 1 numbers
    - Any manuscript text that references GNINA contact metrics

## 1) Manuscript: scientific framing + reviewer-risk mitigation

### A) Problem statement clarity (re-docking vs complex prediction)
- Ensure the wording consistently reflects:
  - Vina/GNINA: binding-site-known re-docking into the experimental receptor (best-case classical docking setup).
  - Boltz-2: sequence+ligand complex prediction (not “docking into a fixed receptor”).
- Verify the same framing is used in:
  - Title/Abstract/Introduction
  - Methods
  - Results captions
  - Discussion + Limitations + Conclusion

### B) Training set leakage / generalization
- Keep claims scoped: “in this benchmark under these conditions”.
- Strengthen Limitations wording so it reads as a boundary of interpretation (not a soft caveat).
- Optional (if feasible): add at least one quantitative “good-faith” check (SI):
  - date-based split by PDB release date, OR
  - sequence-identity clustering / hold-out by protein families

### C) RMSD/atom-mapping caveats for lipids
- Add a brief limitation note about MCS degeneracy for long aliphatic chains and symmetric substructures.
- Optional SI robustness check:
  - Alternative mapping / symmetry-aware RMSD for a small representative subset of lipids.

### D) Headgroup definition robustness
- Add a short justification in Methods.
- Optional SI:
  - Alternative headgroup definition (e.g., all heteroatoms + 1-bond neighborhood) and show trends don’t change.
  - Small table: headgroup atom counts by lipid class + 2–3 illustrated examples.

### E) Statistics clarity
- Keep “primary hypothesis test” explicit (Boltz-2 vs Vina top-1 ligand RMSD).
- Treat other method comparisons as descriptive OR add a simple multiple-comparisons correction (Holm) in SI.

### F) Add a qualitative figure panel (optional but high value)
- One figure (or SI) with 2–3 representative cases:
  - Vina: good top-20 pose but ranked poorly.
  - Boltz-2: headgroup correct, tail off (or another instructive failure mode).
  - GNINA: ranking improvement example.

## 2) Analysis pipeline & database: reproducibility-by-construction

### A) “Single source of truth” database file
- Goal: one `.sqlite` that contains all per-pose metrics needed to reproduce all plots/tables.
- Requirements:
  - Includes all poses for: Boltz, Vina, GNINA CNN, GNINA no-CNN.
  - Contains metadata tables for:
    - run configuration / tool versions (as best as possible)
    - target-level attributes (e.g., TORSDOF bins)
  - Has a clear schema documented in `docs/`:
    - table names and column definitions
    - method labels and how they map to “Boltz/Vina/GNINA CNN/no-CNN”

### B) Transparent, testable analysis code
- Ensure `scripts/analyze_benchmark_db.py` can:
  - regenerate per-target tables
  - regenerate summary tables used in manuscript
  - regenerate all manuscript figures
- Add tests (unittest) for:
  - top-1 selection
  - best-of-K selection
  - metric summaries matching expected small fixtures

### C) Deterministic outputs
- Fix seeds where randomness exists (bootstrap, subsampling, etc.).
- Ensure that all figure generation is deterministic given the DB.

## 3) Plots: final manuscript-quality and consistent styling

### A) Ensure plots match manuscript narrative
- Verify each figure answers a “why does this matter?” question:
  - Fig 1: headline performance gap.
  - Fig 2: per-target paired comparisons to show consistency and outliers.
  - Fig 3: sampling vs ranking story.
  - Fig 4: interaction fidelity (env distribution + typed success-rate).

### B) Style consistency
- Color palette consistency across all figures.
- Axis limits consistent where comparisons are made (Fig 2 already standardized).
- Titles/labels consistent with manuscript phrasing (avoid “top-1” in series labels if title already conveys it).

### C) Figure placement robustness
- Continue using global float tuning and local `\FloatBarrier` where needed.
- Avoid excessive `[H]` usage except where it is necessary to prevent mid-paragraph placement.

## 4) GNINA operational notes (ISAAC + reproducibility)
- Consolidate ISAAC workflow + GNINA recipes into stable docs:
  - partitions/QoS limits, shards strategy
  - scratch layout conventions
  - container best practices (`apptainer pull`, tmp dir handling)
  - rsync/ssh multiplexing workflow
- Ensure scripts in `scripts/` are:
  - minimal, robust, and readable (avoid huge `sbatch --wrap` one-liners)
  - parameterized and documented

## 5) Git hygiene: preparing a clean public commit

### A) Decide what is public vs internal-only
- Public:
  - `lipid_benchmark/` code changes
  - `scripts/` analysis + plotting scripts
  - `manuscript/` LaTeX source + figures needed for compilation
  - `docs/` (including this TODO, data layout notes, ISAAC notes)
  - small fixture data if needed for tests
- Not public:
  - large derived outputs (`output/`, raw GNINA runs, scratch exports)
  - caches (`.cache/`, `.apptainer/`, conda env metadata)
  - any credentials or cluster-specific secrets

### B) Audit ignore rules
- Confirm `.gitignore` covers:
  - `output/`, `build/`, `dist/`, `.cache/`
  - large GNINA/Vina run directories
  - PDF build artifacts if not intended for versioning

### C) Selective staging workflow
- Use:
  - `git status -s` (inventory)
  - `git add -p` (stage by hunk)
  - `git diff --cached` (review staged patch)
- Only after staged patch is correct:
  - run tests
  - build manuscript PDF
  - commit with a descriptive message

## 6) Final pre-commit checklist
- Fix GNINA contact-metric wiring bug (Section 0).
- Regenerate all figures from the DB pipeline and confirm they appear correctly in the PDF.
- `python -m unittest` passes.
- Manuscript builds cleanly (`make -C manuscript dist/manuscript.pdf`).
- `git diff --cached` reviewed line-by-line.

