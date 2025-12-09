import unittest

import numpy as np
import gemmi

from scripts.lib.structures import ensure_protein_backbone, load_structure, split_models


class TestStructures(unittest.TestCase):
    def test_split_models(self):
        # Build a structure with two models, each with one atom.
        s = gemmi.Structure()
        for i in range(2):
            model = gemmi.Model(str(i + 1))
            chain = gemmi.Chain("A")
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(str(i + 1))
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res.add_atom(atom)
            chain.add_residue(res)
            model.add_chain(chain)
            s.add_model(model)

        splits = split_models(s, 2)
        self.assertEqual(len(splits), 2)
        # Verify coordinates were copied
        coords0 = [(atom.pos.x, atom.pos.y, atom.pos.z) for chain in splits[0][0] for res in chain for atom in res]
        coords1 = [(atom.pos.x, atom.pos.y, atom.pos.z) for chain in splits[1][0] for res in chain for atom in res]
        self.assertEqual(coords0, [(0.0, 0.0, 0.0)])
        self.assertEqual(coords1, [(1.0, 0.0, 0.0)])

    def test_ensure_protein_backbone_grafts_from_reference(self):
        # Prediction has only ligand; reference has protein + ligand.
        ref = gemmi.Structure()
        ref_model = gemmi.Model("1")
        ref_chain = gemmi.Chain("A")
        ref_res = gemmi.Residue()
        ref_res.name = "ALA"
        ref_res.seqid = gemmi.SeqId("1")
        ca = gemmi.Atom()
        ca.name = "CA"
        ca.element = gemmi.Element("C")
        ca.pos = gemmi.Position(0, 0, 0)
        ref_res.add_atom(ca)
        ref_chain.add_residue(ref_res)
        ref_model.add_chain(ref_chain)
        ref.add_model(ref_model)

        pred = gemmi.Structure()
        pred_model = gemmi.Model("1")
        pred_chain = gemmi.Chain("L")
        pred_res = gemmi.Residue()
        pred_res.name = "LIG"
        pred_res.seqid = gemmi.SeqId("1")
        lig_atom = gemmi.Atom()
        lig_atom.name = "C1"
        lig_atom.element = gemmi.Element("C")
        lig_atom.pos = gemmi.Position(1, 1, 1)
        pred_res.add_atom(lig_atom)
        pred_chain.add_residue(pred_res)
        pred_model.add_chain(pred_chain)
        pred.add_model(pred_model)

        combined = ensure_protein_backbone(pred, ref)
        # Should contain both the ligand and the protein chain.
        chain_names = sorted(chain.name for chain in combined[0])
        self.assertEqual(chain_names, ["A", "L"])

    def test_load_structure_missing_raises(self):
        from pathlib import Path
        with self.assertRaises(FileNotFoundError):
            load_structure(Path("definitely_missing_file_12345.pdb"))


if __name__ == "__main__":
    unittest.main()
