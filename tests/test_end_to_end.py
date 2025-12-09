import unittest
from pathlib import Path
from unittest import mock


class TestEndToEndBenchmark(unittest.TestCase):
    def test_single_pdb_contact_and_metrics_pipeline(self):
        # This is a heavier, end-to-end test that exercises the real
        # contact-extraction + metrics pipeline on a single PDB ID.
        # Skip gracefully if optional heavy dependencies are missing.
        try:
            import rdkit  # type: ignore  # noqa: F401
            import pandamap  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("RDKit and PandaMap are required for the end-to-end test.")

        import contact_tools.run_batch_contacts as run_batch_contacts
        from scripts import compute_contact_metrics

        project_root = Path(__file__).resolve().parent.parent
        ref_dir = project_root / "benchmark_references"
        boltz_dir = project_root / "model_outputs" / "boltz"
        vina_dir = project_root / "model_outputs" / "vina"

        ref_ids, boltz_ids, vina_ids, common = run_batch_contacts._collect_ids(ref_dir, boltz_dir, vina_dir)
        if not common:
            self.skipTest("No common PDB IDs found for end-to-end test.")
        pdbid = common[0]

        original_collect = run_batch_contacts._collect_ids

        def fake_collect_ids(ref_dir2, boltz_dir2, vina_dir2):
            # Reuse original ID sets but restrict the common list to a single PDB ID.
            _, _, _, all_common = original_collect(ref_dir2, boltz_dir2, vina_dir2)
            if pdbid in all_common:
                subset = [pdbid]
            else:
                subset = all_common[:1] if all_common else []
            return ref_ids, boltz_ids, vina_ids, subset

        # Run contact extraction for just one PDB ID.
        with mock.patch.object(run_batch_contacts, "_collect_ids", side_effect=fake_collect_ids):
            rc = run_batch_contacts.main(quiet=True)
        self.assertEqual(rc, 0, "Contact extraction failed for the selected PDB ID.")

        analysis_root = project_root / "analysis"
        ref_csv = analysis_root / "pandamap_contacts" / "ref_contacts.csv"
        boltz_csv = analysis_root / "pandamap_contacts" / "boltz_contacts.csv"
        vina_csv = analysis_root / "pandamap_contacts" / "vina_contacts.csv"

        for path in (ref_csv, boltz_csv, vina_csv):
            self.assertTrue(path.exists(), f"Expected contacts CSV not found: {path}")

        # Compute contact-level metrics using the generated contacts.
        full_rows, summary_rows = compute_contact_metrics.compute_metrics(
            project_root,
            ref_csv,
            boltz_csv,
            vina_csv,
            boltz_rmsd_path=None,
            vina_rmsd_path=None,
            quiet=True,
        )

        self.assertGreater(len(full_rows), 0, "No per-pose rows produced by compute_metrics.")
        pdbids_seen = {row["pdbid"] for row in full_rows}
        self.assertIn(pdbid, pdbids_seen, "Selected PDB ID not present in metrics output.")


if __name__ == "__main__":
    unittest.main()
