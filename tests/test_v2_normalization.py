import json
import unittest
from pathlib import Path
import tempfile

from lipid_benchmark.contacts import HEADGROUP_INTERACTION_TYPES, extract_contacts, filter_headgroup_contacts
from lipid_benchmark.normalization import NORMALIZED_LIGAND_RESNAME, normalize_entry_from_selected
from lipid_benchmark.rmsd import _select_single_ligand, measure_ligand_pose_all
from lipid_benchmark.ligands import collect_ligands, find_ligand_by_id
from lipid_benchmark.structures import load_structure, split_models


class TestV2Normalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rdkit  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("RDKit is required for normalization tests.")
        try:
            import pandamap  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PandaMap is required for normalization tests.")

    def test_normalized_ligand_atom_count_and_unique_names(self):
        project_root = Path(__file__).resolve().parent.parent
        ref_path = project_root / "benchmark_references" / "1B56.cif"
        boltz_path = project_root / "model_outputs" / "boltz" / "1B56_model_0.cif"
        vina_path = project_root / "model_outputs" / "vina" / "1B56.pdbqt"
        if not (ref_path.exists() and boltz_path.exists() and vina_path.exists()):
            self.skipTest("Required input files for 1B56 are missing.")

        ref_structure = load_structure(ref_path)
        boltz_structure = load_structure(boltz_path)
        vina_structure = load_structure(vina_path)

        ref_ligand = _select_single_ligand(ref_structure, include_h=False)
        boltz_rmsd = measure_ligand_pose_all(ref_path, boltz_path, max_poses=1)[0]
        vina_rmsd = measure_ligand_pose_all(ref_path, vina_path, max_poses=1)[0]

        boltz_ligand_id = str(boltz_rmsd.get("pred_ligand_id") or "")
        vina_ligand_id = str(vina_rmsd.get("pred_ligand_id") or "")
        self.assertTrue(boltz_ligand_id)
        self.assertTrue(vina_ligand_id)

        boltz_models = split_models(boltz_structure, 1)
        vina_models = split_models(vina_structure, 1)
        self.assertEqual(len(boltz_models), 1)
        self.assertEqual(len(vina_models), 1)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out_dir = Path(tmp.name)
        normalized = normalize_entry_from_selected(
            "1B56",
            ref_structure,
            boltz_models[0],
            vina_models,
            ref_ligand=ref_ligand,
            boltz_ligand_id=boltz_ligand_id,
            vina_ligand_ids=[vina_ligand_id],
            out_dir=out_dir,
        )

        # Check normalized Vina ligand
        norm_vina_struct = load_structure(normalized.vina_pdbs[0])
        norm_ligands = collect_ligands(norm_vina_struct, include_h=False)
        self.assertEqual(len(norm_ligands), 1)
        norm_lig = norm_ligands[0]
        self.assertEqual(norm_lig.res_name, NORMALIZED_LIGAND_RESNAME)
        self.assertEqual(len({a.name for a in norm_lig.atoms}), len(norm_lig.atoms))

        orig_ligand = find_ligand_by_id(vina_models[0], vina_ligand_id)
        self.assertEqual(norm_lig.heavy_atom_count(), orig_ligand.heavy_atom_count())

        audit = json.loads(normalized.audit_json.read_text())
        self.assertIn("vina_ligand_ids", audit)
        self.assertEqual(audit["vina_ligand_ids"][0], vina_ligand_id)

    def test_pandamap_headgroup_interactions_nonzero(self):
        project_root = Path(__file__).resolve().parent.parent
        ref_path = project_root / "benchmark_references" / "1DSY.cif"
        boltz_path = project_root / "model_outputs" / "boltz" / "1DSY_model_0.cif"
        vina_path = project_root / "model_outputs" / "vina" / "1DSY.pdbqt"
        if not (ref_path.exists() and boltz_path.exists() and vina_path.exists()):
            self.skipTest("Required input files for 1DSY are missing.")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out_dir = Path(tmp.name)

        ref_structure = load_structure(ref_path)
        boltz_structure = load_structure(boltz_path)
        vina_structure = load_structure(vina_path)

        ref_ligand = _select_single_ligand(ref_structure, include_h=False)
        boltz_rmsd = measure_ligand_pose_all(ref_path, boltz_path, max_poses=1)[0]
        vina_rmsd = measure_ligand_pose_all(ref_path, vina_path, max_poses=1, align_protein=False)[0]
        boltz_ligand_id = str(boltz_rmsd.get("pred_ligand_id") or "")
        vina_ligand_id = str(vina_rmsd.get("pred_ligand_id") or "")
        if not boltz_ligand_id or not vina_ligand_id:
            self.skipTest("Could not identify ligand IDs for normalization.")

        boltz_models = split_models(boltz_structure, 1)
        vina_models = split_models(vina_structure, 1)
        self.assertEqual(len(boltz_models), 1)
        self.assertEqual(len(vina_models), 1)

        normalized = normalize_entry_from_selected(
            "1DSY",
            ref_structure,
            boltz_models[0],
            vina_models,
            ref_ligand=ref_ligand,
            boltz_ligand_id=boltz_ligand_id,
            vina_ligand_ids=[vina_ligand_id],
            out_dir=out_dir,
            use_cache=False,
        )

        contacts = extract_contacts(normalized.ref_pdb, ligand_resname=NORMALIZED_LIGAND_RESNAME)
        head_contacts = filter_headgroup_contacts(
            contacts,
            allowed_atoms=set(normalized.ref_headgroup_atoms),
        )
        self.assertGreater(len(head_contacts), 0)
        for contact in head_contacts:
            self.assertIn(str(contact.get("contact_type") or ""), HEADGROUP_INTERACTION_TYPES)

    def test_headgroup_atoms_mapped_from_reference(self):
        project_root = Path(__file__).resolve().parent.parent
        pdbid = "1IKT"
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
        boltz_ligand_id = str(boltz_rmsd.get("pred_ligand_id") or "")
        vina_ligand_id = str(vina_rmsd.get("pred_ligand_id") or "")
        if not boltz_ligand_id or not vina_ligand_id:
            self.skipTest("Could not identify ligand IDs for normalization.")

        boltz_models = split_models(boltz_structure, 1)
        vina_models = split_models(vina_structure, 1)
        self.assertEqual(len(boltz_models), 1)
        self.assertEqual(len(vina_models), 1)

        normalized = normalize_entry_from_selected(
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
        self.assertEqual(len(normalized.ref_headgroup_atoms), len(normalized.boltz_headgroup_atoms))
        self.assertEqual(len(normalized.ref_headgroup_atoms), len(normalized.vina_headgroup_atoms[0]))


class TestV2PipelineHeadgroupConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rdkit  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("RDKit is required for v2 pipeline tests.")
        try:
            import pandamap  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PandaMap is required for v2 pipeline tests.")

    def test_headgroup_fields_consistent(self):
        from lipid_benchmark.pipeline import run_benchmark
        from lipid_benchmark.io import PairEntry

        project_root = Path(__file__).resolve().parent.parent
        entry = PairEntry(
            pdbid="1B56",
            ref_path=project_root / "benchmark_references" / "1B56.cif",
            boltz_path=project_root / "model_outputs" / "boltz" / "1B56_model_0.cif",
            vina_path=project_root / "model_outputs" / "vina" / "1B56.pdbqt",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            normalized_dir = Path(tmpdir) / "normalized"
            allposes, _summary = run_benchmark(
                [entry],
                vina_max_poses=1,
                normalized_dir=normalized_dir,
                quiet=True,
            )
        self.assertEqual(len(allposes), 2)

        ref_counts = {row["headgroup_contacts_ref"] for row in allposes}
        self.assertEqual(len(ref_counts), 1)

        ref_types = {row["headgroup_types_ref"] for row in allposes}
        self.assertEqual(len(ref_types), 1)
        for t in next(iter(ref_types)).split(";"):
            if not t:
                continue
            ctype = t.split("=")[0]
            self.assertIn(ctype, HEADGROUP_INTERACTION_TYPES)

        ref_contact_count = int(next(iter(ref_counts)))
        if ref_contact_count > 0:
            self.assertGreater(allposes[0]["head_env_ref_size"], 0)
            self.assertGreater(allposes[0]["headgroup_typed_ref_size"], 0)


if __name__ == "__main__":
    unittest.main()
