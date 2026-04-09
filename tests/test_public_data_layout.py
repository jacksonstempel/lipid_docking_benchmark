import csv
import hashlib
import unittest
from pathlib import Path

from lipid_benchmark.public_archive import canonical_prediction_inputs, canonical_raw_predictions


def _benchmark_target_set() -> set[str]:
    root = Path(__file__).resolve().parent.parent
    entries = root / "structures" / "benchmark_entries.csv"
    with entries.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row["pdbid"] or "").strip().upper() for row in reader}


def _csv_pdbids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row["pdbid"] or "").strip().upper() for row in reader}


def _read_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, relpath = line.split("  ", 1)
        entries[relpath] = digest
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestPublicDataLayout(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent.parent
        self.targets = _benchmark_target_set()
        self.inputs = canonical_prediction_inputs(self.root)
        self.raw = canonical_raw_predictions(self.root)

    def test_prediction_inputs_match_curated_target_set(self) -> None:
        boltz_ids = {path.stem.upper() for path in self.inputs.boltz_inputs_root.glob("*.yaml")}
        box_ids = {path.stem.upper() for path in self.inputs.vina_box_root.glob("*.txt")}
        prep_ids = {path.name.upper() for path in self.inputs.vina_prep_root.iterdir() if path.is_dir()}

        self.assertEqual(boltz_ids, self.targets)
        self.assertEqual(box_ids, self.targets)
        self.assertEqual(prep_ids, self.targets)
        self.assertNotIn("3L1N", boltz_ids | box_ids | prep_ids)

    def test_vina_prep_directories_have_exact_expected_files(self) -> None:
        expected = {
            "receptor_no_ligand.pdb",
            "receptor.pdbqt",
            "ligand.pdb",
            "ligand.pdbqt",
            "run_manifest.json",
        }
        for pdbid in self.targets:
            prep_dir = self.inputs.vina_prep_root / pdbid
            self.assertTrue(prep_dir.is_dir(), prep_dir)
            files = {path.name for path in prep_dir.iterdir() if path.is_file()}
            self.assertEqual(files, expected, prep_dir)

    def test_gnina_raw_runs_match_curated_target_set(self) -> None:
        for run_root, pairs_path in (
            (self.raw.gnina_cnn_root, self.raw.gnina_cnn_pairs),
            (self.raw.gnina_nocnn_root, self.raw.gnina_nocnn_pairs),
        ):
            flat_ids = {path.stem.upper() for path in (run_root / "flat").glob("*.pdbqt")}
            log_ids = {path.stem.upper() for path in (run_root / "logs").glob("*.log")}
            pair_ids = _csv_pdbids(pairs_path)
            self.assertEqual(flat_ids, self.targets)
            self.assertEqual(log_ids, self.targets)
            self.assertEqual(pair_ids, self.targets)
            self.assertNotIn("3L1N", flat_ids | log_ids | pair_ids)

    def test_adversarial_raw_runs_match_curated_target_set(self) -> None:
        for arm_root, pairs_path in (
            (self.raw.adversarial_gly_root, self.raw.adversarial_gly_pairs),
            (self.raw.adversarial_phe_root, self.raw.adversarial_phe_pairs),
        ):
            input_ids = {path.stem.upper() for path in (arm_root / "inputs").glob("*.yaml")}
            cif_ids = {
                path.name[:-len("_model_0.cif")].upper()
                for path in (arm_root / "flat").glob("*_model_0.cif")
            }
            confidence_ids = {
                path.name[len("confidence_"):-len("_model_0.json")].upper()
                for path in (arm_root / "flat").glob("confidence_*_model_0.json")
            }
            pair_ids = _csv_pdbids(pairs_path)

            self.assertEqual(input_ids, self.targets)
            self.assertEqual(cif_ids, self.targets)
            self.assertEqual(confidence_ids, self.targets)
            self.assertEqual(pair_ids, self.targets)

        mutation_ids = _csv_pdbids(self.raw.adversarial_mutation_summary)
        self.assertEqual(mutation_ids, self.targets)

    def test_sha256_manifests_match_tracked_files(self) -> None:
        for root in (self.inputs.root, self.raw.root):
            manifest = root / "SHA256SUMS.txt"
            self.assertTrue(manifest.is_file(), manifest)
            recorded = _read_sha256_manifest(manifest)
            expected_paths = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS.txt"
            )
            self.assertEqual(sorted(recorded.keys()), expected_paths, root)
            for relpath, digest in recorded.items():
                self.assertEqual(_sha256(root / relpath), digest, relpath)


if __name__ == "__main__":
    unittest.main()
