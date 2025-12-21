import unittest
from collections import Counter
from pathlib import Path

from lipid_benchmark.residue_mapping import build_residue_id_map, remap_typed_ids
from lipid_benchmark.structures import load_structure


class TestV2ResidueMapping(unittest.TestCase):
    def test_boltz_residue_offset_mapping_1b56(self):
        project_root = Path(__file__).resolve().parent.parent
        ref = load_structure(project_root / "benchmark_references" / "1B56.cif")
        pred = load_structure(project_root / "model_outputs" / "boltz" / "1B56_model_0.cif")

        mapping = build_residue_id_map(pred, ref)
        self.assertGreater(len(mapping), 10)

        diffs = Counter()
        for k, v in mapping.items():
            try:
                k_num = int(k.split(":")[2])
                v_num = int(v.split(":")[2])
            except Exception:
                continue
            diffs[v_num - k_num] += 1

        # 1B56 is a canonical offset case: reference residue numbers are +2 vs Boltz.
        self.assertEqual(diffs.most_common(1)[0][0], 2)

    def test_chain_mapping_3stm(self):
        project_root = Path(__file__).resolve().parent.parent
        ref = load_structure(project_root / "benchmark_references" / "3STM.cif")
        pred = load_structure(project_root / "model_outputs" / "boltz" / "3STM_model_0.cif")

        mapping = build_residue_id_map(pred, ref)
        self.assertGreater(len(mapping), 10)

        any_a_to_x = any(k.startswith("A:") and v.startswith("X:") for k, v in mapping.items())
        self.assertTrue(any_a_to_x, "Expected at least one A:* residue to map to X:* for 3STM.")

    def test_remap_typed_ids(self):
        mapping = {"A:GLY:1": "B:GLY:10"}
        typed = {"A:GLY:1|hydrogen_bonds", "X:ASP:2|ionic"}
        remapped = remap_typed_ids(typed, mapping)
        self.assertIn("B:GLY:10|hydrogen_bonds", remapped)
        self.assertIn("X:ASP:2|ionic", remapped)


if __name__ == "__main__":
    unittest.main()
