import csv
import unittest

from lipid_benchmark.public_archive import canonical_repro_archive, require_paths


class TestReproArchive(unittest.TestCase):
    def test_required_archive_files_exist(self) -> None:
        archive = canonical_repro_archive()
        require_paths(
            [
                archive.baseline_allposes,
                archive.baseline_summary,
                archive.gnina_cnn_allposes,
                archive.gnina_nocnn_allposes,
                archive.adversarial_gly_allposes,
                archive.adversarial_gly_summary,
                archive.adversarial_phe_allposes,
                archive.adversarial_phe_summary,
                archive.mutation_summary,
                archive.vina_exh256_allposes,
                archive.vina_exh256_summary,
                archive.boltz_high_sampling_allposes,
                archive.boltz_high_sampling_summary,
            ]
        )

    def test_archive_tables_cover_expected_target_counts(self) -> None:
        archive = canonical_repro_archive()

        def _count_unique_pdbids(path) -> int:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                return len({(row.get("pdbid") or "").strip().upper() for row in reader if row.get("pdbid")})

        self.assertEqual(_count_unique_pdbids(archive.baseline_summary), 100)
        self.assertEqual(_count_unique_pdbids(archive.gnina_cnn_allposes), 100)
        self.assertEqual(_count_unique_pdbids(archive.gnina_nocnn_allposes), 100)
        self.assertEqual(_count_unique_pdbids(archive.adversarial_gly_summary), 100)
        self.assertEqual(_count_unique_pdbids(archive.adversarial_phe_summary), 100)
        self.assertEqual(_count_unique_pdbids(archive.vina_exh256_summary), 100)
        self.assertEqual(_count_unique_pdbids(archive.boltz_high_sampling_summary), 99)
