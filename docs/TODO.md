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

### ✅ 5. Magic Numbers Without Scientific Justification (COMPLETED)
**Status:** Fixed
**What was done:** Added inline rationale/comments for ligand size cutoff, RDKit coverage threshold, Chimera-style 2.0 Å pruning cutoff, and bond tolerance cushion.

**Files changed:**
- `scripts/lib/constants.py`
- `scripts/lib/ligand_pose_core.py`
- `scripts/lib/alignment.py`
- `scripts/lib/ligands.py`

---

### ✅ 6. Duplicated Constants (COMPLETED)
**Status:** Fixed
**What was done:** Centralized Vina pose count as `VINA_MAX_POSES` in `scripts/lib/constants.py` and reused everywhere (compute_contact_metrics, run_batch_contacts, run_full_benchmark).

**Files changed:**
- `scripts/lib/constants.py`
- `scripts/compute_contact_metrics.py`
- `contact_tools/run_batch_contacts.py`
- `scripts/run_full_benchmark.py`

---

### ✅ 7. No Input Validation (COMPLETED)
**Status:** Fixed
**What was done:** `load_structure` now checks for existence and raises `FileNotFoundError` before parsing.

**Files changed:**
- `scripts/lib/structures.py`

---

### ✅ 8. Broad Exception Catching (COMPLETED)
**Status:** Fixed
**What was done:** Narrowed broad `except Exception` blocks to specific error types in ligands, contact metrics, and measure_contacts to avoid masking real bugs.

**Files changed:**
- `scripts/lib/ligands.py`
- `scripts/compute_contact_metrics.py`
- `contact_tools/measure_contacts.py`

---

### ✅ 9. Unused Function (COMPLETED)
**Status:** Fixed
**What was done:** Removed unused `_residue_id` helper from `contact_tools/measure_contacts.py`.

**Files changed:**
- `contact_tools/measure_contacts.py`

---

## MEDIUM — Recommended Improvements

### ✅ 10. README Location (COMPLETED)
**Status:** Fixed
**What was done:** Moved main README to project root.

**Files changed:**
- `README.md` (moved from `docs/README.md`)

---

### ✅ 11. No Unit Tests (COMPLETED)
**Status:** Fixed
**What was done:** Added `unittest` coverage for core geometry (Kabsch/pruned alignment/RMSD), ligand selection and naming, structure helpers, metrics math, and RMSD CSV parsing. Made `tests/` a package for easy discovery.

**Files added:**
- `tests/__init__.py`
- `tests/test_alignment.py`
- `tests/test_rmsd.py`
- `tests/test_ligand_selection.py`
- `tests/test_structures.py`
- `tests/test_metrics.py`

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
- **HIGH priority:** 5/5 completed (100%)
- **MEDIUM priority:** 2/5 completed (40%)
- **LOW priority:** 0/3 completed (0%)

**Next steps:** Address remaining MEDIUM/LOW items (#12-17) as needed.
