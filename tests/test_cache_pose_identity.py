import tempfile
import unittest
from pathlib import Path


class TestPoseAwareContactCaching(unittest.TestCase):
    def _shift_pdbqt(self, in_path: Path, out_path: Path, *, dx: float, dy: float, dz: float) -> None:
        lines = []
        for line in in_path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    lines.append(line)
                    continue
                line = f"{line[:30]}{x + dx:8.3f}{y + dy:8.3f}{z + dz:8.3f}{line[54:]}"
            lines.append(line)
        out_path.write_text("\n".join(lines) + "\n")

    def test_contact_cache_changes_with_pose_geometry(self):
        try:
            import rdkit  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is required for the cache behavior test.")
        try:
            import pandamap  # type: ignore  # noqa: F401
        except ImportError:
            self.skipTest("PandaMap is required for the cache behavior test.")

        from lipid_benchmark.pipeline import run_benchmark
        from lipid_benchmark.io import PairEntry

        project_root = Path(__file__).resolve().parent.parent
        ref_path = project_root / "structures" / "experimental" / "1B56.cif"
        boltz_path = project_root / "structures" / "boltz" / "1B56_model_0.cif"
        vina_path = project_root / "structures" / "vina" / "1B56.pdbqt"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            shifted_path = tmpdir_path / "1B56_shifted.pdbqt"
            self._shift_pdbqt(vina_path, shifted_path, dx=50.0, dy=0.0, dz=0.0)

            entry_a = PairEntry(pdbid="1B56", ref_path=ref_path, boltz_path=boltz_path, vina_path=vina_path)
            entry_b = PairEntry(pdbid="1B56", ref_path=ref_path, boltz_path=boltz_path, vina_path=shifted_path)

            normalized_dir = tmpdir_path / "normalized"
            allposes_a, _ = run_benchmark(
                [entry_a],
                vina_max_poses=1,
                normalized_dir=normalized_dir,
                quiet=True,
                cache_normalized=True,
                cache_contacts=True,
            )
            allposes_b, _ = run_benchmark(
                [entry_b],
                vina_max_poses=1,
                normalized_dir=normalized_dir,
                quiet=True,
                cache_normalized=True,
                cache_contacts=True,
            )

            row_a = next(r for r in allposes_a if r["method"] == "vina_pose" and r["pose_index"] == 1)
            row_b = next(r for r in allposes_b if r["method"] == "vina_pose" and r["pose_index"] == 1)

            self.assertNotEqual(float(row_a["ligand_rmsd"]), float(row_b["ligand_rmsd"]))
            head_env_differs = float(row_a["head_env_jaccard"]) != float(row_b["head_env_jaccard"])
            typed_differs = float(row_a["headgroup_typed_jaccard"]) != float(row_b["headgroup_typed_jaccard"])
            self.assertTrue(
                head_env_differs or typed_differs,
                "Contact metrics should change when the pose geometry changes, even with caching enabled.",
            )


if __name__ == "__main__":
    unittest.main()
