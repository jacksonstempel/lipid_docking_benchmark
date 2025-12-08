# Code Review TODO List

This document tracks issues identified during the comprehensive code review for publication preparation.

---

## CRITICAL — Must Fix Before Publication

### ✅ 1. No Dependency Management (COMPLETED)
**Status:** Fixed
**What was done:**
- Created `requirements.txt` with pinned versions of all dependencies
- Added installation instructions to README
- Documented software versions (Boltz v2.2.0, AutoDock Vina 23d1252-mod)

**Files changed:**
- `requirements.txt` (created)
- `docs/README.md` (updated)

---

### ✅ 2. Hardcoded Absolute Paths (COMPLETED)
**Status:** Fixed
**What was done:**
- Converted `scripts/pairs.csv` from absolute paths to relative paths
- Updated `measure_ligand_pose_batch.py` to resolve relative paths from project root
- Verified all 112 entries work correctly with new path handling

**Files changed:**
- `scripts/pairs.csv`
- `scripts/measure_ligand_pose_batch.py`

**Note:** 467 files in `model_outputs/vina/extra_output/` and `analysis/` still have hardcoded paths in their content, but these are output/log files (not source code) and don't affect portability.

---

### ✅ 3. Silent Exception Swallowing (COMPLETED)
**Status:** Fixed
**What was done:**
- Modified `measure_ligand_pose_all()` to record failed poses in output CSV with status="error"
- Added `status` and `error` columns to all result entries
- Errors logged at DEBUG level (only visible with --verbose) to keep terminal clean
- Updated batch scripts to preserve status/error information

**Files changed:**
- `scripts/lib/ligand_pose_core.py`
- `scripts/measure_ligand_pose_batch.py`

**Impact:** CSV outputs now show every pose attempted (both successful and failed) for complete transparency.

---

### ✅ 4. Missing Random Seed Control (COMPLETED)
**Status:** Fixed
**What was done:**
- Verified `randomized_copy()` and `_random_rotation()` functions were completely unused
- Deleted both functions from codebase
- Confirmed benchmark pipeline has zero randomness (fully deterministic)

**Files changed:**
- `scripts/lib/ligands.py`

**Result:** Benchmark is now 100% reproducible with no non-deterministic operations.

---

## HIGH — Should Fix

### ⬜ 5. Magic Numbers Without Scientific Justification
**Issue:** Several constants lack documentation of their scientific rationale

**Locations:**
- `MIN_LIGAND_HEAVY_ATOMS = 10` (constants.py:32) — Why 10 atoms?
- `MIN_LIGAND_COVERAGE = 0.9` (ligand_pose_core.py:25) — Why 90% coverage?
- `cutoff = 2.0` (alignment.py:128) — Why 2.0 Å for pruning?
- Bond tolerance `0.5` (ligands.py:237) — Why 0.5 Å?

**Fix needed:** Add docstrings/comments explaining the scientific rationale for each threshold.

---

### ⬜ 6. Duplicated Constants
**Issue:** Same constant defined in multiple places

**Locations:**
- `VINA_MAX_POSES = 20` in `compute_contact_metrics.py:35`
- `VINA_POSES = 20` in `run_batch_contacts.py:38`
- `--max-poses 20` hardcoded in `run_full_benchmark.py:74`

**Fix needed:** Define once in `constants.py` and import everywhere.

---

### ⬜ 7. No Input Validation
**Issue:** Functions don't check if files exist before attempting to read

**Location:** `scripts/lib/structures.py:25-33` (load_structure)

**Fix needed:** Add `if not path.exists(): raise FileNotFoundError(...)` checks.

---

### ⬜ 8. Broad Exception Catching
**Issue:** Many `except Exception:` blocks that could mask bugs

**Locations:**
- `ligands.py:71, 252-254, 257-259, 303`
- `compute_contact_metrics.py:53-54, 123-124, 296, 306, 317, 319`
- `measure_contacts.py:46-47, 83-84`

**Fix needed:** Catch specific exceptions where possible (ValueError, KeyError, etc.).

---

### ⬜ 9. Unused Function
**Issue:** `_residue_id()` function defined but never called

**Location:** `contact_tools/measure_contacts.py:40-48`

**Fix needed:** Either use this function or remove it.

---

## MEDIUM — Recommended Improvements

### ⬜ 10. README Location
**Issue:** README is at `docs/README.md` instead of project root

**Fix needed:** Move to root or create root README linking to docs.

---

### ⬜ 11. No Unit Tests
**Issue:** No test files found; key calculations (RMSD, Kabsch) should be validated

**Fix needed:** Add basic tests for critical functions:
```python
# tests/test_alignment.py
def test_kabsch_identity()
def test_rmsd_known_value()
```

---

### ⬜ 12. Duplicate sys Import
**Location:** `scripts/measure_ligand_pose_batch.py:19` and `26`

**Fix needed:** Remove the duplicate import.

---

### ⬜ 13. Inconsistent Logging vs Print
**Issue:** `run_full_benchmark.py` uses `print("[INFO] ...")` while other modules use `logging.info(...)`

**Fix needed:** Use logging consistently throughout.

---

### ⬜ 14. Missing External Tool Versions
**Issue:** Software versions now documented (✅ completed in Issue #1)

**Status:** Actually completed as part of Issue #1 fix.

---

## LOW — Polish

### ⬜ 15. Minimal .gitignore
**Current:** Only has `__pycache__/`, `*.py[cod]`, `.DS_Store`, `*.swp`

**Missing:**
```
.env
*.csv
analysis/tmp/
analysis/pandamap_contacts/
.vscode/
.idea/
*.egg-info/
```

**Fix needed:** Expand .gitignore to cover more common files.

---

### ⬜ 16. Empty __init__.py
**Location:** `contact_tools/__init__.py`

**Fix needed:** Export main functions:
```python
from .measure_contacts import extract_contacts
from .run_batch_contacts import main as run_batch
```

---

### ⬜ 17. Repeated sys.path Manipulation
**Issue:** Bootstrap pattern repeated in 6+ files

**Fix needed:** Create a bootstrap module or use proper package installation with `pip install -e .`

---

## Notes

### RMSD Calculation Review
**Status:** ✅ Verified correct
The RMSD implementation in `scripts/lib/ligands.py:307-321` uses the standard formula: `√(Σ(d²)/N)` and is mathematically sound. No changes needed.

### Coordinate Units
**Note:** Code doesn't explicitly verify units, but gemmi uses Angstroms by default. Consider adding an assertion or comment confirming this assumption.

---

## Progress Summary

- **CRITICAL issues:** 4/4 completed (100%)
- **HIGH priority:** 0/5 completed (0%)
- **MEDIUM priority:** 0/5 completed (0%)
- **LOW priority:** 0/3 completed (0%)

**Next steps:** Address HIGH priority issues (#5-9) before publication.
