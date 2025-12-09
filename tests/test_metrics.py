import io
import unittest

import numpy as np

from scripts.compute_contact_metrics import _load_contacts, _metrics_residue, _metrics_strict, _parse_res, _read_rmsd_csv


class TestContactMetrics(unittest.TestCase):
    def test_metrics_strict(self):
        # ref: A,B,C ; pred: B,C,D => TP={B,C}, FP={D}, FN={A}
        ref = {("L1", "A", "contact"), ("L1", "B", "contact"), ("L1", "C", "contact")}
        pred = {("L1", "B", "contact"), ("L1", "C", "contact"), ("L1", "D", "contact")}
        ref_set = type("CSet", (), {"contacts": ref, "distances": {c: 1.0 for c in ref}})()
        pred_set = type("CSet", (), {"contacts": pred, "distances": {c: 1.0 for c in pred}})()
        metrics = _metrics_strict(ref_set, pred_set)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)))
        self.assertAlmostEqual(metrics["jaccard"], 2 / 4)
        self.assertEqual(metrics["ref_contacts"], 3)
        self.assertEqual(metrics["pred_contacts"], 3)
        self.assertEqual(metrics["shared_contacts"], 2)

    def test_metrics_residue(self):
        ref = type("CSet", (), {"contacts": {("L1", "A:GLY:1", "x"), ("L1", "B:GLY:2", "x")}})()
        pred = type("CSet", (), {"contacts": {("L1", "B:GLY:2", "x"), ("L1", "C:GLY:3", "x")}})()
        metrics = _metrics_residue(ref, pred)
        self.assertAlmostEqual(metrics["residue_precision"], 1 / 2)
        self.assertAlmostEqual(metrics["residue_recall"], 1 / 2)
        self.assertEqual(metrics["residue_shared"], 1)
        self.assertEqual(metrics["residue_ref_size"], 2)
        self.assertEqual(metrics["residue_pred_size"], 2)

    def test_parse_res(self):
        self.assertEqual(_parse_res("A:GLY:42"), ("A", "GLY", 42))
        self.assertIsNone(_parse_res("bad_format"))

    def test_read_rmsd_csv(self):
        csv_data = """pdbid,pose_index,ligand_rmsd,status
TEST,1,1.0,ok
TEST,2,2.0,ok
TEST,3,5.0,error
"""
        buf = io.StringIO(csv_data)
        # _read_rmsd_csv expects a Path, but we can patch by writing to a temp file via NamedTemporaryFile
        import tempfile, pathlib

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write(csv_data)
            tmp.flush()
            path = pathlib.Path(tmp.name)
            data = _read_rmsd_csv(path)
        self.assertIn("TEST", data)
        poses = data["TEST"]["poses"]
        self.assertEqual(set(poses.keys()), {1, 2})  # error row skipped
        self.assertEqual(data["TEST"]["best"]["pose_index"], "1")


if __name__ == "__main__":
    unittest.main()
