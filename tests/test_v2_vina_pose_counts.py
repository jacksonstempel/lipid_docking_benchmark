import unittest
from pathlib import Path

from lipid_benchmark.structures import load_structure, split_models


class TestV2VinaPoseCounts(unittest.TestCase):
    def test_vina_pose_count_can_be_less_than_20(self):
        project_root = Path(__file__).resolve().parent.parent
        vina_path = project_root / "model_outputs" / "vina" / "3PL5.pdbqt"
        if not vina_path.exists():
            self.skipTest("Expected Vina file not present: 3PL5.pdbqt")
        structure = load_structure(vina_path)
        poses = split_models(structure, 20)
        self.assertEqual(len(poses), 11)


if __name__ == "__main__":
    unittest.main()
