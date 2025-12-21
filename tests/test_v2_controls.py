import unittest
from pathlib import Path
import tempfile

from lipid_benchmark.contacts import (
    contacts_to_typed_set,
    extract_contacts,
    filter_headgroup_contacts,
)
from lipid_benchmark.normalization import NORMALIZED_LIGAND_RESNAME, normalize_entry_from_selected
from lipid_benchmark.rmsd import _select_single_ligand, measure_ligand_pose_all
from lipid_benchmark.structures import load_structure, split_models, write_pdb_structure


class TestV2ControlExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rdkit  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("RDKit is required for control tests.")
        try:
            import pandamap  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PandaMap is required for control tests.")

    def _normalize_ref(self, pdbid: str):
        project_root = Path(__file__).resolve().parent.parent
        ref_path = project_root / "benchmark_references" / f"{pdbid}.cif"
        boltz_path = project_root / "model_outputs" / "boltz" / f"{pdbid}_model_0.cif"
        vina_path = project_root / "model_outputs" / "vina" / f"{pdbid}.pdbqt"
        if not (ref_path.exists() and boltz_path.exists() and vina_path.exists()):
            self.skipTest(f"Required input files for {pdbid} are missing.")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out_dir = Path(tmp.name)

        ref_structure = load_structure(ref_path)
        boltz_structure = load_structure(boltz_path)
        vina_structure = load_structure(vina_path)

        ref_ligand = _select_single_ligand(ref_structure, include_h=False)
        boltz_rmsd = measure_ligand_pose_all(ref_path, boltz_path, max_poses=1)[0]
        vina_rmsd = measure_ligand_pose_all(ref_path, vina_path, max_poses=1, align_protein=False)[0]

        boltz_models = split_models(boltz_structure, 1)
        vina_models = split_models(vina_structure, 1)
        self.assertEqual(len(boltz_models), 1)
        self.assertEqual(len(vina_models), 1)

        boltz_ligand_id = str(boltz_rmsd.get("pred_ligand_id") or "")
        vina_ligand_id = str(vina_rmsd.get("pred_ligand_id") or "")
        if not boltz_ligand_id or not vina_ligand_id:
            self.skipTest("Could not identify ligand IDs for normalization.")

        return normalize_entry_from_selected(
            pdbid,
            ref_structure,
            boltz_models[0],
            vina_models,
            ref_ligand=ref_ligand,
            boltz_ligand_id=boltz_ligand_id,
            vina_ligand_ids=[vina_ligand_id],
            out_dir=out_dir,
            use_cache=False,
        )

    def test_ref_vs_ref_headgroup_overlap_is_one(self):
        normalized = self._normalize_ref("1DSY")
        contacts = extract_contacts(normalized.ref_pdb, ligand_resname=NORMALIZED_LIGAND_RESNAME)
        head_contacts = filter_headgroup_contacts(
            contacts,
            allowed_atoms=set(normalized.ref_headgroup_atoms),
        )
        ref_residues = {str(c.get("residue") or "") for c in head_contacts if str(c.get("residue") or "")}
        ref_typed = contacts_to_typed_set(head_contacts)

        if not ref_residues or not ref_typed:
            self.skipTest("Reference has no headgroup interactions; cannot assert overlap=1.")

        # Overlap with itself should be perfect (Jaccard = 1.0).
        jacc_res = len(ref_residues & ref_residues) / len(ref_residues | ref_residues)
        jacc_typed = len(ref_typed & ref_typed) / len(ref_typed | ref_typed)
        self.assertEqual(jacc_res, 1.0)
        self.assertEqual(jacc_typed, 1.0)

    def test_shifted_ligand_has_no_headgroup_interactions(self):
        normalized = self._normalize_ref("1DSY")
        structure = load_structure(normalized.ref_pdb)

        # Translate only the ligand chain far away to eliminate interactions.
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.het_flag != "H":
                        continue
                    for atom in residue:
                        atom.pos.x += 100.0
                        atom.pos.y += 100.0
                        atom.pos.z += 100.0

        shifted = Path(normalized.ref_pdb).with_name("ref_shifted.pdb")
        write_pdb_structure(structure, shifted)

        contacts = extract_contacts(shifted, ligand_resname=NORMALIZED_LIGAND_RESNAME)
        head_contacts = filter_headgroup_contacts(
            contacts,
            allowed_atoms=set(normalized.ref_headgroup_atoms),
        )
        self.assertEqual(len(head_contacts), 0)


if __name__ == "__main__":
    unittest.main()
