import csv
import tempfile
import unittest
from pathlib import Path

from lipid_benchmark.io import PairEntry, find_project_root, read_pairs_csv


class TestProjectRoot(unittest.TestCase):
    def test_find_project_root_walks_up_to_config_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.yaml").write_text("paths: {}\n", encoding="utf-8")
            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)

            found = find_project_root(start=nested)
            self.assertEqual(found, root.resolve())


class TestPairsCsv(unittest.TestCase):
    def test_read_pairs_csv_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.yaml").write_text("paths: {}\n", encoding="utf-8")

            ref = root / "ref.cif"
            boltz = root / "boltz.cif"
            vina = root / "vina.pdbqt"
            for p in (ref, boltz, vina):
                p.write_text("x", encoding="utf-8")

            pairs = root / "pairs.csv"
            with pairs.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["pdbid", "ref", "boltz_pred", "vina_pred"], lineterminator="\n")
                w.writeheader()
                w.writerow({"pdbid": "1b56", "ref": "ref.cif", "boltz_pred": "boltz.cif", "vina_pred": "vina.pdbqt"})

            entries = read_pairs_csv(root, pairs)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertIsInstance(entry, PairEntry)
            self.assertEqual(entry.pdbid, "1B56")
            self.assertEqual(entry.ref_path, ref.resolve())
            self.assertEqual(entry.boltz_path, boltz.resolve())
            self.assertEqual(entry.vina_path, vina.resolve())

    def test_read_pairs_csv_requires_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pairs = root / "pairs.csv"
            pairs.write_text("pdbid,ref\n1B56,x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_pairs_csv(root, pairs)
