import unittest

import numpy as np

from lipid_benchmark.ligands import SimpleAtom, SimpleResidue
from lipid_benchmark.rmsd import LigandSelectionError, _filter_ligands, _select_single_ligand
from lipid_benchmark.rmsd import MIN_LIGAND_HEAVY_ATOMS
import gemmi


def _make_residue(name: str, chain: str, res_id: str, heavy_atoms: int) -> SimpleResidue:
    atoms = [SimpleAtom(name=f"C{i}", element="C", xyz=np.zeros(3)) for i in range(heavy_atoms)]
    return SimpleResidue(chain_id=chain, res_name=name, res_id=res_id, atoms=atoms)


class TestLigandFiltering(unittest.TestCase):
    def test_filters_solvents_ions_and_small(self):
        # Water
        water = _make_residue("HOH", "A", "1", 1)
        # Ion
        ion = _make_residue("NA", "A", "2", 1)
        # Tiny fragment below threshold
        tiny = _make_residue("LIG", "A", "3", MIN_LIGAND_HEAVY_ATOMS - 1)
        # Valid ligand above threshold
        valid = _make_residue("LIP", "A", "4", MIN_LIGAND_HEAVY_ATOMS + 5)
        filtered = _filter_ligands([water, ion, tiny, valid])
        self.assertEqual(filtered, [valid])

    def test_selects_heaviest_and_raises_when_none(self):
        lig_small = _make_residue("LIP", "A", "1", MIN_LIGAND_HEAVY_ATOMS)
        lig_big = _make_residue("LIP", "B", "2", MIN_LIGAND_HEAVY_ATOMS + 10)
        # Build a minimal gemmi structure with two ligand residues
        structure = gemmi.Structure()
        structure.add_model(gemmi.Model("1"))
        # Put ligands on separate chains to avoid merging
        for lig in (lig_small, lig_big):
            chain = structure[0].add_chain(lig.chain_id)
            res = gemmi.Residue()
            res.name = lig.res_name
            res.seqid = gemmi.SeqId(lig.res_id)
            for idx, atom in enumerate(lig.atoms):
                a = gemmi.Atom()
                a.name = atom.name
                a.element = gemmi.Element(atom.element)
                a.pos = gemmi.Position(float(idx), 0.0, 0.0)
                res.add_atom(a)
            chain.add_residue(res)
        selected = _select_single_ligand(structure, include_h=False)
        self.assertEqual(selected.res_name, lig_big.res_name)
        self.assertEqual(selected.res_id, lig_big.res_id)

        # Now test raising when no ligands survive filtering
        empty_structure = gemmi.Structure()
        empty_structure.add_model(gemmi.Model("1"))
        with self.assertRaises(LigandSelectionError):
            _select_single_ligand(empty_structure, include_h=False)


if __name__ == "__main__":
    unittest.main()
