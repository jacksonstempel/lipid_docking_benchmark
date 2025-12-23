import unittest
from pathlib import Path
import tempfile


class TestV2EndToEnd(unittest.TestCase):
    def test_runs_single_entry(self):
        try:
            import rdkit  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is required for the end-to-end test.")
        try:
            import pandamap  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("PandaMap is required for the end-to-end test.")

        from lipid_benchmark.pipeline import run_benchmark
        from lipid_benchmark.io import PairEntry

        project_root = Path(__file__).resolve().parent.parent
        entry = PairEntry(
            pdbid="1B56",
            ref_path=project_root / "structures" / "experimental" / "1B56.cif",
            boltz_path=project_root / "structures" / "boltz" / "1B56_model_0.cif",
            vina_path=project_root / "structures" / "vina" / "1B56.pdbqt",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            allposes, summary = run_benchmark(
                [entry],
                vina_max_poses=1,
                normalized_dir=Path(tmpdir) / "normalized",
                quiet=True,
            )
        self.assertEqual(len(allposes), 2)
        self.assertEqual({row["method"] for row in summary}, {"boltz", "vina_top1"})

        boltz = next(row for row in allposes if row["method"] == "boltz")
        self.assertIsInstance(boltz["ligand_rmsd"], float)
        self.assertGreaterEqual(boltz["ligand_rmsd"], 0.0)
        self.assertIn("headgroup_typed_jaccard", boltz)
        self.assertIn("head_env_jaccard", boltz)


if __name__ == "__main__":
    unittest.main()
