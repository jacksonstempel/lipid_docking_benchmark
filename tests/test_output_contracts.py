import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lipid_benchmark.analysis_db import build_per_target
from lipid_benchmark.pipeline import BENCHMARK_FIELDNAMES
from scripts.build_benchmark_db import build_db


class OutputContractTests(unittest.TestCase):
    def test_benchmark_fieldnames_contract(self) -> None:
        expected = [
            "pdbid",
            "method",
            "pose_index",
            "ref_ligand_id",
            "pred_ligand_id",
            "pairing_method",
            "ligand_heavy_atoms",
            "ligand_rmsd",
            "headgroup_atoms",
            "headgroup_rmsd",
            "protein_pairs",
            "protein_rmsd",
            "headgroup_contacts_ref",
            "headgroup_contacts_pred",
            "headgroup_types_ref",
            "headgroup_types_pred",
            "head_env_precision",
            "head_env_recall",
            "head_env_f1",
            "head_env_jaccard",
            "head_env_shared",
            "head_env_ref_size",
            "head_env_pred_size",
            "headgroup_typed_precision",
            "headgroup_typed_recall",
            "headgroup_typed_f1",
            "headgroup_typed_jaccard",
            "headgroup_typed_shared",
            "headgroup_typed_ref_size",
            "headgroup_typed_pred_size",
        ]
        self.assertEqual(BENCHMARK_FIELDNAMES, expected)

    def test_per_target_columns_contract(self) -> None:
        rows = [
            {
                "pdbid": "AAAA",
                "method": "boltz",
                "pose_index": 1,
                "ligand_rmsd": 1.0,
                "headgroup_rmsd": 1.1,
                "head_env_jaccard": 0.8,
                "headgroup_typed_jaccard": 0.7,
            },
            {
                "pdbid": "AAAA",
                "method": "vina_pose",
                "pose_index": 1,
                "ligand_rmsd": 5.0,
                "headgroup_rmsd": 5.5,
                "head_env_jaccard": 0.3,
                "headgroup_typed_jaccard": 0.2,
            },
            {
                "pdbid": "AAAA",
                "method": "vina_pose",
                "pose_index": 2,
                "ligand_rmsd": 2.0,
                "headgroup_rmsd": 2.2,
                "head_env_jaccard": 0.4,
                "headgroup_typed_jaccard": 0.3,
            },
            {
                "pdbid": "AAAA",
                "method": "gnina_cnn_pose",
                "pose_index": 1,
                "ligand_rmsd": 3.0,
                "headgroup_rmsd": 3.1,
                "head_env_jaccard": 0.5,
                "headgroup_typed_jaccard": 0.4,
            },
            {
                "pdbid": "AAAA",
                "method": "gnina_nocnn_pose",
                "pose_index": 1,
                "ligand_rmsd": 4.0,
                "headgroup_rmsd": 4.1,
                "head_env_jaccard": 0.45,
                "headgroup_typed_jaccard": 0.35,
            },
        ]
        per = build_per_target(pd.DataFrame(rows), k=2)
        required = {
            "pdbid",
            "boltz_top1_ligand_rmsd",
            "boltz_top1_headgroup_rmsd",
            "boltz_top1_head_env_jaccard",
            "boltz_top1_headgroup_typed_jaccard",
            "vina_top1_ligand_rmsd",
            "vina_bestK_ligand_rmsd",
            "gnina_cnn_top1_ligand_rmsd",
            "gnina_nocnn_top1_ligand_rmsd",
            "vina_gap_ligand_rmsd",
            "gnina_cnn_gap_ligand_rmsd",
            "gnina_nocnn_gap_ligand_rmsd",
        }
        self.assertTrue(required.issubset(set(per.columns)))

    def test_unified_db_build_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Minimal baseline (boltz + vina_pose)
            baseline = pd.DataFrame(
                [
                    {
                        "pdbid": "AAAA",
                        "method": "boltz",
                        "pose_index": 1,
                        "ligand_rmsd": 1.0,
                        "headgroup_rmsd": 1.1,
                        "head_env_jaccard": 0.8,
                        "headgroup_typed_jaccard": 0.7,
                    },
                    {
                        "pdbid": "AAAA",
                        "method": "vina_pose",
                        "pose_index": 1,
                        "ligand_rmsd": 5.0,
                        "headgroup_rmsd": 5.5,
                        "head_env_jaccard": 0.3,
                        "headgroup_typed_jaccard": 0.2,
                    },
                    {
                        "pdbid": "AAAA",
                        "method": "vina_pose",
                        "pose_index": 2,
                        "ligand_rmsd": 2.0,
                        "headgroup_rmsd": 2.2,
                        "head_env_jaccard": 0.4,
                        "headgroup_typed_jaccard": 0.3,
                    },
                ]
            )

            # GNINA inputs intentionally use method=vina_pose and are remapped by build_db.
            gnina_cnn = pd.DataFrame(
                [
                    {
                        "pdbid": "AAAA",
                        "method": "vina_pose",
                        "pose_index": 1,
                        "ligand_rmsd": 3.0,
                        "headgroup_rmsd": 3.1,
                        "head_env_jaccard": 0.5,
                        "headgroup_typed_jaccard": 0.4,
                    }
                ]
            )
            gnina_nocnn = pd.DataFrame(
                [
                    {
                        "pdbid": "AAAA",
                        "method": "vina_pose",
                        "pose_index": 1,
                        "ligand_rmsd": 4.0,
                        "headgroup_rmsd": 4.2,
                        "head_env_jaccard": 0.45,
                        "headgroup_typed_jaccard": 0.35,
                    }
                ]
            )

            baseline_path = tmp / "benchmark_allposes.csv"
            gnina_cnn_path = tmp / "gnina_cnn_allposes.csv"
            gnina_nocnn_path = tmp / "gnina_nocnn_allposes.csv"
            db_path = tmp / "benchmark_full.sqlite"

            baseline.to_csv(baseline_path, index=False)
            gnina_cnn.to_csv(gnina_cnn_path, index=False)
            gnina_nocnn.to_csv(gnina_nocnn_path, index=False)

            vina_dir = tmp / "vina"
            vina_dir.mkdir(parents=True, exist_ok=True)
            (vina_dir / "AAAA.pdbqt").write_text("TORSDOF 12\n")

            build_db(
                out_path=db_path,
                baseline_allposes=baseline_path,
                gnina_cnn_allposes=gnina_cnn_path,
                gnina_nocnn_allposes=gnina_nocnn_path,
                adversarial_gly_allposes=None,
                adversarial_phe_allposes=None,
                adversarial_gly_summary=None,
                adversarial_phe_summary=None,
                vina_dir=vina_dir,
            )

            self.assertTrue(db_path.exists())

            con = sqlite3.connect(db_path)
            try:
                tables = {
                    row[0]
                    for row in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertIn("allposes", tables)
                self.assertIn("targets", tables)
                self.assertIn("meta", tables)

                methods = {
                    row[0]
                    for row in con.execute("SELECT DISTINCT method FROM allposes")
                }
                self.assertIn("boltz", methods)
                self.assertIn("vina_pose", methods)
                self.assertIn("gnina_cnn_pose", methods)
                self.assertIn("gnina_nocnn_pose", methods)

                torsdof = con.execute(
                    "SELECT torsdof FROM targets WHERE pdbid='AAAA'"
                ).fetchone()
                self.assertIsNotNone(torsdof)
                self.assertEqual(int(torsdof[0]), 12)
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()
