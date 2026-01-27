import unittest

import pandas as pd

from lipid_benchmark.analysis_db import build_per_target, summarize_methods


def _toy_allposes() -> pd.DataFrame:
    rows = [
        {"pdbid": "AAAA", "method": "boltz", "pose_index": 1, "ligand_rmsd": 1.0, "headgroup_rmsd": 1.2, "head_env_jaccard": 0.8, "headgroup_typed_jaccard": 0.7},
        {"pdbid": "BBBB", "method": "boltz", "pose_index": 1, "ligand_rmsd": 2.0, "headgroup_rmsd": 2.5, "head_env_jaccard": 0.6, "headgroup_typed_jaccard": 0.5},
        {"pdbid": "AAAA", "method": "vina_pose", "pose_index": 1, "ligand_rmsd": 5.0, "headgroup_rmsd": 5.1, "head_env_jaccard": 0.3, "headgroup_typed_jaccard": 0.2},
        {"pdbid": "AAAA", "method": "vina_pose", "pose_index": 2, "ligand_rmsd": 2.0, "headgroup_rmsd": 2.2, "head_env_jaccard": 0.4, "headgroup_typed_jaccard": 0.3},
        {"pdbid": "BBBB", "method": "vina_pose", "pose_index": 1, "ligand_rmsd": 6.0, "headgroup_rmsd": 6.5, "head_env_jaccard": 0.2, "headgroup_typed_jaccard": 0.1},
        {"pdbid": "BBBB", "method": "vina_pose", "pose_index": 2, "ligand_rmsd": 3.0, "headgroup_rmsd": 3.2, "head_env_jaccard": 0.35, "headgroup_typed_jaccard": 0.25},
        {"pdbid": "AAAA", "method": "gnina_cnn_pose", "pose_index": 1, "ligand_rmsd": 3.0, "headgroup_rmsd": 3.1, "head_env_jaccard": 0.5, "headgroup_typed_jaccard": 0.4},
        {"pdbid": "BBBB", "method": "gnina_cnn_pose", "pose_index": 1, "ligand_rmsd": 4.0, "headgroup_rmsd": 4.1, "head_env_jaccard": 0.45, "headgroup_typed_jaccard": 0.35},
        {"pdbid": "AAAA", "method": "gnina_nocnn_pose", "pose_index": 1, "ligand_rmsd": 4.0, "headgroup_rmsd": 4.2, "head_env_jaccard": 0.4, "headgroup_typed_jaccard": 0.3},
        {"pdbid": "BBBB", "method": "gnina_nocnn_pose", "pose_index": 1, "ligand_rmsd": 5.0, "headgroup_rmsd": 5.2, "head_env_jaccard": 0.3, "headgroup_typed_jaccard": 0.2},
    ]
    return pd.DataFrame(rows)


class AnalysisDbTests(unittest.TestCase):
    def test_build_per_target_bestk_gap(self) -> None:
        per = build_per_target(_toy_allposes(), k=2)
        row_a = per.set_index("pdbid").loc["AAAA"]
        self.assertEqual(row_a["vina_bestK_ligand_rmsd"], 2.0)
        self.assertEqual(row_a["vina_gap_ligand_rmsd"], 3.0)

    def test_summarize_methods_format(self) -> None:
        per = build_per_target(_toy_allposes(), k=2)
        summary = summarize_methods(per)
        self.assertIn("Boltz-2", summary.formatted["method"].values)
