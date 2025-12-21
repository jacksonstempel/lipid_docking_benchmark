import unittest

import numpy as np

from lipid_benchmark.ligands import SimpleAtom, locked_rmsd


class TestLockedRMSD(unittest.TestCase):
    def test_zero_rmsd_for_identical(self):
        atoms = [
            SimpleAtom(name="C1", element="C", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="C2", element="C", xyz=np.array([1.0, 0.0, 0.0])),
            SimpleAtom(name="C3", element="C", xyz=np.array([0.0, 1.0, 0.0])),
        ]
        pairs = [(0, 0), (1, 1), (2, 2)]
        rmsd, n = locked_rmsd(atoms, atoms, pairs, np.eye(3), np.zeros(3))
        self.assertEqual(n, 3)
        self.assertAlmostEqual(rmsd, 0.0, places=7)

    def test_translation_only(self):
        ref_atoms = [
            SimpleAtom(name="C1", element="C", xyz=np.array([0.0, 0.0, 0.0])),
            SimpleAtom(name="C2", element="C", xyz=np.array([1.0, 0.0, 0.0])),
            SimpleAtom(name="C3", element="C", xyz=np.array([0.0, 1.0, 0.0])),
        ]
        pred_atoms = [
            SimpleAtom(name="C1", element="C", xyz=np.array([1.0, 2.0, 3.0])),
            SimpleAtom(name="C2", element="C", xyz=np.array([2.0, 2.0, 3.0])),
            SimpleAtom(name="C3", element="C", xyz=np.array([1.0, 3.0, 3.0])),
        ]
        pairs = [(0, 0), (1, 1), (2, 2)]
        # Translation aligns pred to ref; rotation is identity.
        translation = np.array([-1.0, -2.0, -3.0])
        rmsd, n = locked_rmsd(pred_atoms, ref_atoms, pairs, np.eye(3), translation)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(rmsd, 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
