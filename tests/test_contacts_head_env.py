import tempfile
import unittest
from pathlib import Path

import gemmi

from lipid_benchmark.contacts import headgroup_environment_residues
from lipid_benchmark.structures import write_pdb_structure


class TestHeadgroupEnvironmentResidues(unittest.TestCase):
    def test_empty_headgroup_atom_names_returns_empty_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "x.pdb"
            s = gemmi.Structure()
            s.add_model(gemmi.Model("1"))
            write_pdb_structure(s, path)
            self.assertEqual(headgroup_environment_residues(path, headgroup_atom_names=[]), set())

    def test_returns_nearby_protein_residue_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "complex.pdb"

            s = gemmi.Structure()
            s.add_model(gemmi.Model("1"))
            m = s[0]

            # Protein chain A with one residue ALA:1 near origin.
            chain_a = m.add_chain("A")
            res_ala = gemmi.Residue()
            res_ala.name = "ALA"
            res_ala.seqid = gemmi.SeqId("1")
            ca = gemmi.Atom()
            ca.name = "CA"
            ca.element = gemmi.Element("C")
            ca.pos = gemmi.Position(0.0, 0.0, 0.0)
            res_ala.add_atom(ca)
            chain_a.add_residue(res_ala)

            # Ligand chain L with residue LIG:1 and headgroup atom O1 close to CA.
            chain_l = m.add_chain("L")
            res_lig = gemmi.Residue()
            res_lig.name = "LIG"
            res_lig.seqid = gemmi.SeqId("1")
            res_lig.het_flag = "H"
            o1 = gemmi.Atom()
            o1.name = "O1"
            o1.element = gemmi.Element("O")
            o1.pos = gemmi.Position(0.5, 0.0, 0.0)
            res_lig.add_atom(o1)
            chain_l.add_residue(res_lig)

            write_pdb_structure(s, path)
            residues = headgroup_environment_residues(path, headgroup_atom_names=["O1"], cutoff_a=1.0)
            self.assertIn("A:ALA:1", residues)


if __name__ == "__main__":
    unittest.main()
