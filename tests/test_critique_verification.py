"""
Tests to verify code review concerns before including them in critique document.
These tests check whether identified issues are real problems or false positives.
"""

import math
import unittest
from pathlib import Path
from unittest.mock import patch
import inspect

import numpy as np


class TestRMSDEdgeCases(unittest.TestCase):
    """Verify concerns about RMSD calculation edge cases."""

    def test_single_atom_rmsd(self):
        """Test: Does locked_rmsd work correctly with a single atom pair?

        Concern: Single-atom RMSD is mathematically valid but may be noisy.
        This test verifies the function handles this case without error.
        """
        from lipid_benchmark.ligands import SimpleAtom, locked_rmsd

        ref_atom = SimpleAtom(name="P1", element="P", xyz=np.array([0.0, 0.0, 0.0]))
        pred_atom = SimpleAtom(name="P1", element="P", xyz=np.array([1.0, 0.0, 0.0]))

        pairs = [(0, 0)]
        rmsd, n = locked_rmsd([pred_atom], [ref_atom], pairs, np.eye(3), np.zeros(3))

        # Should return valid RMSD of 1.0 for 1 atom separated by 1 Angstrom
        self.assertEqual(n, 1)
        self.assertAlmostEqual(rmsd, 1.0, places=7)

    def test_empty_pairs_rmsd(self):
        """Test: Does locked_rmsd handle empty pairs correctly?

        Concern: Division by zero if P.shape[0] is 0.
        """
        from lipid_benchmark.ligands import SimpleAtom, locked_rmsd

        ref_atom = SimpleAtom(name="P1", element="P", xyz=np.array([0.0, 0.0, 0.0]))

        pairs = []  # Empty pairs
        rmsd, n = locked_rmsd([ref_atom], [ref_atom], pairs, np.eye(3), np.zeros(3))

        # Should return infinity for empty pairs, not crash
        self.assertEqual(n, 0)
        self.assertEqual(rmsd, float("inf"))


class TestSilentExceptionHandling(unittest.TestCase):
    """Verify concerns about silent exception swallowing."""

    def test_best_pred_ligand_catches_exceptions_silently(self):
        """Test: Does _best_pred_ligand_by_rmsd silently swallow exceptions?

        Concern: `except Exception: continue` may mask real bugs.
        This test verifies that invalid ligands are skipped without logging.
        """
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue
        from lipid_benchmark.rmsd import _best_pred_ligand_by_rmsd, AtomPairingError

        # Create a valid reference ligand
        ref_atoms = [
            SimpleAtom(name=f"C{i}", element="C", xyz=np.array([float(i), 0.0, 0.0]))
            for i in range(15)
        ]
        ref_ligand = SimpleResidue(chain_id="A", res_name="LIP", res_id="1", atoms=ref_atoms)

        # Create an "invalid" candidate that will fail RDKit pairing
        # (single atom ligand that won't meet coverage threshold)
        invalid_atoms = [SimpleAtom(name="X1", element="X", xyz=np.array([0.0, 0.0, 0.0]))]
        invalid_ligand = SimpleResidue(chain_id="B", res_name="BAD", res_id="2", atoms=invalid_atoms)

        # Create a valid candidate
        valid_atoms = [
            SimpleAtom(name=f"C{i}", element="C", xyz=np.array([float(i) + 0.1, 0.0, 0.0]))
            for i in range(15)
        ]
        valid_ligand = SimpleResidue(chain_id="C", res_name="LIP", res_id="3", atoms=valid_atoms)

        # This should succeed by selecting the valid ligand, silently skipping invalid
        result = _best_pred_ligand_by_rmsd(ref_ligand, [invalid_ligand, valid_ligand])

        # Verify we got the valid ligand back
        self.assertEqual(result[0].res_id, "3")

    def test_all_candidates_fail_raises_error(self):
        """Test: Does the function raise an error when ALL candidates fail?"""
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue
        from lipid_benchmark.rmsd import _best_pred_ligand_by_rmsd, AtomPairingError

        ref_atoms = [
            SimpleAtom(name=f"C{i}", element="C", xyz=np.array([float(i), 0.0, 0.0]))
            for i in range(15)
        ]
        ref_ligand = SimpleResidue(chain_id="A", res_name="LIP", res_id="1", atoms=ref_atoms)

        # All candidates are invalid (won't match)
        invalid_atoms = [SimpleAtom(name="X1", element="X", xyz=np.array([0.0, 0.0, 0.0]))]
        invalid1 = SimpleResidue(chain_id="B", res_name="BAD", res_id="2", atoms=invalid_atoms)
        invalid2 = SimpleResidue(chain_id="C", res_name="BAD", res_id="3", atoms=invalid_atoms)

        with self.assertRaises(AtomPairingError):
            _best_pred_ligand_by_rmsd(ref_ligand, [invalid1, invalid2])


class TestNaNHandling(unittest.TestCase):
    """Verify concerns about string-based NaN/inf handling."""

    def test_nan_string_variations(self):
        """Test: Does the string-based NaN check handle all variations?

        Concern: Check only handles "nan", not "NaN", "NAN", or float('nan').
        """
        # The check in pipeline.py is:
        # if str(value).strip() not in ("", "nan", "inf")

        test_cases = [
            ("nan", True),      # Should be treated as NA
            ("NaN", False),     # May NOT be caught - POTENTIAL BUG
            ("NAN", False),     # May NOT be caught - POTENTIAL BUG
            ("inf", True),      # Should be treated as NA
            ("Inf", False),     # May NOT be caught - POTENTIAL BUG
            ("-inf", False),    # May NOT be caught - POTENTIAL BUG
            ("", True),         # Should be treated as NA
            ("1.5", False),     # Valid number
        ]

        check_strings = ("", "nan", "inf")

        bugs_found = []
        for value, should_be_na in test_cases:
            is_na = value.strip().lower() in check_strings or value.strip() in check_strings
            actual_check = value.strip() in check_strings

            # Check if the actual implementation would miss this
            if should_be_na and not actual_check:
                # This is a case that SHOULD be NA but the check misses it
                if value.lower() in ("nan", "inf", "-inf"):
                    bugs_found.append(f"'{value}' should be NA but check misses it")

        # Record findings - this test documents the behavior
        self.bugs_found = bugs_found

    def test_float_nan_handling(self):
        """Test: How does the check handle actual float('nan')?"""
        value = float('nan')
        check = str(value).strip() not in ("", "nan", "inf")

        # str(float('nan')) returns 'nan' on most platforms
        # So this should work correctly
        self.assertFalse(check, "float('nan') should be caught by the check")

    def test_numpy_nan_handling(self):
        """Test: How does the check handle np.nan?"""
        value = np.nan
        check = str(value).strip() not in ("", "nan", "inf")

        # str(np.nan) returns 'nan'
        self.assertFalse(check, "np.nan should be caught by the check")


class TestHeadgroupDetection(unittest.TestCase):
    """Verify concerns about headgroup detection for different lipid types."""

    def test_phospholipid_headgroup(self):
        """Test: Phospholipids should have headgroup detected via P atoms."""
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue, headgroup_indices_functional

        # Simplified phosphatidylcholine-like structure
        atoms = [
            SimpleAtom(name="P1", element="P", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="O1", element="O", xyz=np.array([1.4, 0.0, 0.0])),
            SimpleAtom(name="O2", element="O", xyz=np.array([-1.4, 0.0, 0.0])),
            SimpleAtom(name="O3", element="O", xyz=np.array([0.0, 1.4, 0.0])),
            SimpleAtom(name="O4", element="O", xyz=np.array([0.0, -1.4, 0.0])),
            SimpleAtom(name="C1", element="C", xyz=np.array([2.8, 0.0, 0.0])),
            SimpleAtom(name="C2", element="C", xyz=np.array([4.2, 0.0, 0.0])),
            SimpleAtom(name="N1", element="N", xyz=np.array([5.6, 0.0, 0.0])),
            # Tail carbons (should not be in headgroup)
            SimpleAtom(name="C10", element="C", xyz=np.array([0.0, -2.8, 0.0])),
            SimpleAtom(name="C11", element="C", xyz=np.array([0.0, -4.2, 0.0])),
            SimpleAtom(name="C12", element="C", xyz=np.array([0.0, -5.6, 0.0])),
        ]
        residue = SimpleResidue(chain_id="L", res_name="PC", res_id="1", atoms=atoms)

        head_indices = headgroup_indices_functional(residue)

        # P and atoms within 2 bonds should be included
        self.assertGreater(len(head_indices), 0, "Phospholipid should have headgroup atoms")
        # P atom (index 0) should definitely be in headgroup
        self.assertIn(0, head_indices, "Phosphorus should be in headgroup")

    def test_sphingolipid_headgroup(self):
        """Test: Sphingolipids (no P) should still detect headgroup via N.

        Concern: Lipids without phosphate may not be detected.
        """
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue, headgroup_indices_functional

        # Simplified ceramide-like structure (no phosphate, has amide N)
        atoms = [
            SimpleAtom(name="N1", element="N", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="C1", element="C", xyz=np.array([1.4, 0.0, 0.0])),  # carbonyl C
            SimpleAtom(name="O1", element="O", xyz=np.array([2.1, 1.2, 0.0])),  # carbonyl O
            SimpleAtom(name="C2", element="C", xyz=np.array([-1.4, 0.0, 0.0])),
            SimpleAtom(name="O2", element="O", xyz=np.array([-2.1, 1.2, 0.0])),  # hydroxyl
            SimpleAtom(name="C3", element="C", xyz=np.array([-1.4, -1.4, 0.0])),
            SimpleAtom(name="O3", element="O", xyz=np.array([-2.8, -1.4, 0.0])),  # hydroxyl
            # Tail
            SimpleAtom(name="C10", element="C", xyz=np.array([2.8, 0.0, 0.0])),
            SimpleAtom(name="C11", element="C", xyz=np.array([4.2, 0.0, 0.0])),
        ]
        residue = SimpleResidue(chain_id="L", res_name="CER", res_id="1", atoms=atoms)

        head_indices = headgroup_indices_functional(residue)

        # N with degree >= 3 triggers the secondary heuristic
        # But amide N typically has degree 2 (bonded to C and H)
        # This tests whether the fallback to heteroatoms works
        self.assertGreater(len(head_indices), 0, "Sphingolipid should have headgroup detected")

    def test_fatty_acid_headgroup(self):
        """Test: Fatty acids (only COOH) should detect carboxyl as headgroup.

        Concern: Simple fatty acids may fall through to heteroatom fallback.
        """
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue, headgroup_indices_functional

        # Simplified fatty acid (palmitic acid-like)
        atoms = [
            SimpleAtom(name="C1", element="C", xyz=np.array([0.0, 0.0, 0.0])),   # carboxyl C
            SimpleAtom(name="O1", element="O", xyz=np.array([1.2, 0.6, 0.0])),   # carbonyl O
            SimpleAtom(name="O2", element="O", xyz=np.array([0.0, 1.4, 0.0])),   # hydroxyl O
            # Tail
            SimpleAtom(name="C2", element="C", xyz=np.array([-1.4, 0.0, 0.0])),
            SimpleAtom(name="C3", element="C", xyz=np.array([-2.8, 0.0, 0.0])),
            SimpleAtom(name="C4", element="C", xyz=np.array([-4.2, 0.0, 0.0])),
        ]
        residue = SimpleResidue(chain_id="L", res_name="PLM", res_id="1", atoms=atoms)

        head_indices = headgroup_indices_functional(residue)

        # Should detect via "carbon with >=2 oxygen neighbors" heuristic
        # or fall back to heteroatoms (O)
        self.assertGreater(len(head_indices), 0, "Fatty acid should have headgroup detected")
        # Oxygens should be in headgroup
        o_indices = [i for i, a in enumerate(atoms) if a.element == "O"]
        for oi in o_indices:
            self.assertIn(oi, head_indices, f"Oxygen at index {oi} should be in headgroup")


class TestCovalentRadii(unittest.TestCase):
    """Verify concerns about missing covalent radii."""

    def test_selenium_handling(self):
        """Test: How does the code handle selenomethionine (Se)?"""
        from lipid_benchmark.ligands import _element_radius, _ELEMENT_TO_Z

        # Se is common in proteins (selenomethionine)
        se_radius = _element_radius("Se")

        # Default is 0.77 if not in table
        self.assertEqual(se_radius, 0.77, "Se uses default radius")

        # Check if Se is in element-to-Z table
        self.assertNotIn("SE", _ELEMENT_TO_Z, "Se is not in element table - uses carbon fallback")

    def test_metal_handling(self):
        """Test: How does the code handle common metals (Mg, Ca, Zn)?"""
        from lipid_benchmark.ligands import _element_radius, _ELEMENT_TO_Z

        metals = ["MG", "CA", "ZN", "FE", "MN"]
        for metal in metals:
            radius = _element_radius(metal)
            # These use default 0.77
            self.assertEqual(radius, 0.77, f"{metal} uses default radius")
            self.assertNotIn(metal, _ELEMENT_TO_Z, f"{metal} not in element table")


class TestDuplicateCode(unittest.TestCase):
    """Verify concerns about duplicate code."""

    def test_clone_residue_duplicates(self):
        """Test: Are there duplicate _clone_residue functions?"""
        from lipid_benchmark import structures, normalization

        # Check if both modules have _clone_residue
        has_in_structures = hasattr(structures, '_clone_residue')
        has_in_normalization = hasattr(normalization, '_clone_residue')

        self.assertTrue(has_in_structures, "_clone_residue exists in structures")
        self.assertTrue(has_in_normalization, "_clone_residue exists in normalization")

        # Both exist - this is indeed duplicate code
        if has_in_structures and has_in_normalization:
            # Check if implementations are identical
            src_structures = inspect.getsource(structures._clone_residue)
            src_normalization = inspect.getsource(normalization._clone_residue)

            # They may have slight differences in implementation
            self.duplicates_found = True
            self.same_implementation = (src_structures == src_normalization)


class TestUnusedConfigPaths(unittest.TestCase):
    """Verify concerns about unused configuration paths."""

    def test_unused_config_keys(self):
        """Test: Are analysis_proteins and analysis_aggregates used anywhere?"""
        import yaml

        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config.yaml"

        if not config_path.exists():
            self.skipTest("config.yaml not found")

        config = yaml.safe_load(config_path.read_text())
        paths = config.get("paths", {})

        # Check which paths exist in config
        defined_paths = set(paths.keys())

        # Search codebase for usage of these paths
        lipid_benchmark_dir = project_root / "lipid_benchmark"
        scripts_dir = project_root / "scripts"

        used_paths = set()
        search_dirs = [lipid_benchmark_dir, scripts_dir]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for py_file in search_dir.glob("**/*.py"):
                content = py_file.read_text()
                for path_key in defined_paths:
                    if path_key in content:
                        used_paths.add(path_key)

        unused = defined_paths - used_paths
        self.unused_paths = unused


if __name__ == "__main__":
    unittest.main(verbosity=2)
