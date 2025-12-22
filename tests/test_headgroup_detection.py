import unittest

import numpy as np


class TestHeadgroupDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rdkit  # type: ignore  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("RDKit is required for headgroup detection tests.")

    def test_phospholipid_headgroup_detects_phosphorus(self):
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue, headgroup_indices_functional

        atoms = [
            SimpleAtom(name="P1", element="P", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="O1", element="O", xyz=np.array([1.4, 0.0, 0.0])),
            SimpleAtom(name="O2", element="O", xyz=np.array([-1.4, 0.0, 0.0])),
            SimpleAtom(name="O3", element="O", xyz=np.array([0.0, 1.4, 0.0])),
            SimpleAtom(name="O4", element="O", xyz=np.array([0.0, -1.4, 0.0])),
            SimpleAtom(name="C1", element="C", xyz=np.array([2.8, 0.0, 0.0])),
            SimpleAtom(name="C2", element="C", xyz=np.array([4.2, 0.0, 0.0])),
            SimpleAtom(name="N1", element="N", xyz=np.array([5.6, 0.0, 0.0])),
            # Tail carbons
            SimpleAtom(name="C10", element="C", xyz=np.array([0.0, -2.8, 0.0])),
            SimpleAtom(name="C11", element="C", xyz=np.array([0.0, -4.2, 0.0])),
            SimpleAtom(name="C12", element="C", xyz=np.array([0.0, -5.6, 0.0])),
        ]
        residue = SimpleResidue(chain_id="L", res_name="PC", res_id="1", atoms=atoms)
        head_indices = headgroup_indices_functional(residue)
        self.assertIn(0, head_indices)

    def test_sphingolipid_like_headgroup_is_nonempty(self):
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue, headgroup_indices_functional

        atoms = [
            SimpleAtom(name="N1", element="N", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="C1", element="C", xyz=np.array([1.4, 0.0, 0.0])),
            SimpleAtom(name="O1", element="O", xyz=np.array([2.1, 1.2, 0.0])),
            SimpleAtom(name="C2", element="C", xyz=np.array([-1.4, 0.0, 0.0])),
            SimpleAtom(name="O2", element="O", xyz=np.array([-2.1, 1.2, 0.0])),
            SimpleAtom(name="C3", element="C", xyz=np.array([-1.4, -1.4, 0.0])),
            SimpleAtom(name="O3", element="O", xyz=np.array([-2.8, -1.4, 0.0])),
            # Tail
            SimpleAtom(name="C10", element="C", xyz=np.array([2.8, 0.0, 0.0])),
            SimpleAtom(name="C11", element="C", xyz=np.array([4.2, 0.0, 0.0])),
        ]
        residue = SimpleResidue(chain_id="L", res_name="CER", res_id="1", atoms=atoms)
        head_indices = headgroup_indices_functional(residue)
        self.assertGreater(len(head_indices), 0)

    def test_fatty_acid_like_headgroup_contains_oxygens(self):
        from lipid_benchmark.ligands import SimpleAtom, SimpleResidue, headgroup_indices_functional

        atoms = [
            SimpleAtom(name="C1", element="C", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="O1", element="O", xyz=np.array([1.2, 0.6, 0.0])),
            SimpleAtom(name="O2", element="O", xyz=np.array([0.0, 1.4, 0.0])),
            # Tail
            SimpleAtom(name="C2", element="C", xyz=np.array([-1.4, 0.0, 0.0])),
            SimpleAtom(name="C3", element="C", xyz=np.array([-2.8, 0.0, 0.0])),
            SimpleAtom(name="C4", element="C", xyz=np.array([-4.2, 0.0, 0.0])),
        ]
        residue = SimpleResidue(chain_id="L", res_name="PLM", res_id="1", atoms=atoms)
        head_indices = headgroup_indices_functional(residue)

        oxygen_indices = [i for i, atom in enumerate(atoms) if atom.element == "O"]
        for idx in oxygen_indices:
            self.assertIn(idx, head_indices)


if __name__ == "__main__":
    unittest.main()

