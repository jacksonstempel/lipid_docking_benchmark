import sqlite3
import unittest
from pathlib import Path

from lipid_benchmark.rmsd import measure_ligand_pose_all


class TestAlignmentValidity(unittest.TestCase):
    """Sanity checks that protein-backbone alignment is functioning as intended.

    These tests are aimed at catching a failure mode where sequence mutations
    cause incorrect chain/residue pairing, which would invalidate ligand/headgroup RMSDs.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.db_path = cls.root / "output" / "benchmark" / "benchmark_full.sqlite"
        if not cls.db_path.is_file():
            raise unittest.SkipTest("benchmark_full.sqlite not found; run benchmark pipeline first.")

        cls.ref_dir = cls.root / "structures" / "experimental"
        cls.base_boltz_dir = cls.root / "structures" / "boltz"
        cls.gly_boltz_dir = (
            cls.root
            / "output"
            / "adversarial"
            / "bs_mutagenesis_cutoff5A"
            / "isaac_runs"
            / "boltz2_bs_mutagenesis_cutoff5A_gly_v1"
            / "flat"
        )
        cls.phe_boltz_dir = (
            cls.root
            / "output"
            / "adversarial"
            / "bs_mutagenesis_cutoff5A"
            / "isaac_runs"
            / "boltz2_bs_mutagenesis_cutoff5A_phe_v1"
            / "flat"
        )

        # These files are derived; skip if not present.
        if not cls.gly_boltz_dir.is_dir() or not cls.phe_boltz_dir.is_dir():
            raise unittest.SkipTest("Mutant CIFs not found under output/adversarial/.../isaac_runs.")

    def test_protein_pairs_identical_between_baseline_and_mutants(self) -> None:
        """If sequence mutations were breaking alignment pairing, we'd see shifts in protein_pairs."""
        con = sqlite3.connect(self.db_path)
        try:
            base = {
                pdbid: int(pairs)
                for pdbid, pairs in con.execute(
                    "SELECT pdbid, protein_pairs FROM allposes WHERE method='boltz' AND pose_index=1"
                )
            }
            gly = {
                pdbid: int(pairs)
                for pdbid, pairs in con.execute(
                    "SELECT pdbid, protein_pairs FROM allposes WHERE method='boltz_bs5A_gly' AND pose_index=1"
                )
            }
            phe = {
                pdbid: int(pairs)
                for pdbid, pairs in con.execute(
                    "SELECT pdbid, protein_pairs FROM allposes WHERE method='boltz_bs5A_phe' AND pose_index=1"
                )
            }
        finally:
            con.close()

        self.assertEqual(set(base), set(gly))
        self.assertEqual(set(base), set(phe))
        self.assertGreaterEqual(len(base), 50)  # should be ~100 targets

        for pdbid, n_pairs in base.items():
            self.assertEqual(gly[pdbid], n_pairs, f"{pdbid}: GLY protein_pairs differs from baseline")
            self.assertEqual(phe[pdbid], n_pairs, f"{pdbid}: PHE protein_pairs differs from baseline")
            self.assertGreaterEqual(n_pairs, 50, f"{pdbid}: unexpectedly low protein_pairs={n_pairs}")

    def test_alignment_is_applied_and_matches_recorded_metrics(self) -> None:
        """Spot-check that align_protein=True reproduces stored RMSDs, and align_protein=False does not."""
        # A small panel of targets with typical sizes.
        targets = ["1B56", "7E25", "8T5T"]

        con = sqlite3.connect(self.db_path)
        try:
            def fetch_row(method: str, pdbid: str) -> dict[str, float]:
                row = con.execute(
                    "SELECT ligand_rmsd, headgroup_rmsd, protein_rmsd, protein_pairs "
                    "FROM allposes WHERE method=? AND pose_index=1 AND pdbid=?",
                    (method, pdbid),
                ).fetchone()
                if row is None:
                    raise AssertionError(f"Missing DB row for {method} {pdbid}")
                return {
                    "ligand_rmsd": float(row[0]),
                    "headgroup_rmsd": float(row[1]),
                    "protein_rmsd": float(row[2]),
                    "protein_pairs": int(row[3]),
                }

            for pdbid in targets:
                ref = self.ref_dir / f"{pdbid}.cif"
                base_pred = self.base_boltz_dir / f"{pdbid}_model_0.cif"
                gly_pred = self.gly_boltz_dir / f"{pdbid}_model_0.cif"

                if not ref.is_file() or not base_pred.is_file() or not gly_pred.is_file():
                    continue  # keep test robust to partial checkouts

                # Baseline: align on protein backbone and compare to DB.
                db_base = fetch_row("boltz", pdbid)
                got_base = measure_ligand_pose_all(ref, base_pred, max_poses=1, align_protein=True)[0]
                self.assertAlmostEqual(float(got_base["ligand_rmsd"]), db_base["ligand_rmsd"], places=6)
                self.assertAlmostEqual(float(got_base["headgroup_rmsd"]), db_base["headgroup_rmsd"], places=6)
                self.assertAlmostEqual(float(got_base["protein_rmsd"]), db_base["protein_rmsd"], places=6)
                self.assertEqual(int(got_base["protein_pairs"]), db_base["protein_pairs"])

                # Mutant GLY: align on protein backbone and compare to DB.
                db_gly = fetch_row("boltz_bs5A_gly", pdbid)
                got_gly = measure_ligand_pose_all(ref, gly_pred, max_poses=1, align_protein=True)[0]
                self.assertAlmostEqual(float(got_gly["ligand_rmsd"]), db_gly["ligand_rmsd"], places=6)
                self.assertAlmostEqual(float(got_gly["headgroup_rmsd"]), db_gly["headgroup_rmsd"], places=6)
                self.assertAlmostEqual(float(got_gly["protein_rmsd"]), db_gly["protein_rmsd"], places=6)
                self.assertEqual(int(got_gly["protein_pairs"]), db_gly["protein_pairs"])

                # Demonstrate that skipping alignment produces very large RMSDs for Boltz outputs.
                no_align = measure_ligand_pose_all(ref, base_pred, max_poses=1, align_protein=False)[0]
                self.assertGreater(float(no_align["ligand_rmsd"]), 10.0, f"{pdbid}: unexpected small RMSD without alignment")

        finally:
            con.close()

