import math
import unittest

import numpy as np

from lipid_benchmark.alignment import chimera_pruned_fit, kabsch, extract_chain_sequences, pair_chains
import gemmi


class TestKabsch(unittest.TestCase):
    def test_identity_alignment(self):
        # Coordinates are identical, so optimal R is identity and t is zero.
        coords = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0], [-3.0, 0.5, 4.0]])
        R, t = kabsch(coords, coords)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-7)

    def test_known_rotation_translation(self):
        # Apply a 90° rotation about Z and a translation; Kabsch should recover the inverse transform.
        theta = math.pi / 2
        R_true = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t_true = np.array([1.0, -2.0, 0.5])
        P = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [3.0, -1.0, 1.0]])
        Q = P @ R_true + t_true
        R, t = kabsch(P, Q)
        np.testing.assert_allclose(P @ R + t, Q, atol=1e-7)
        # Recovered rotation should match the forward transform, translation should match t_true.
        np.testing.assert_allclose(R, R_true, atol=1e-7)
        np.testing.assert_allclose(t, t_true, atol=1e-7)


class TestChimeraPrunedFit(unittest.TestCase):
    def test_prunes_single_outlier(self):
        # Three inliers match exactly; one outlier is far away and should be dropped by the 2 Å cutoff.
        P = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 10.0, 10.0],  # outlier
            ]
        )
        Q = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [20.0, 20.0, 20.0],  # mismatched outlier
            ]
        )
        fit = chimera_pruned_fit(P, Q, cutoff=2.0)
        self.assertEqual(fit.n_pruned, 3)  # three inliers retained
        self.assertAlmostEqual(fit.rmsd_pruned, 0.0, places=7)
        # Outlier should be excluded; full-set RMSD should be non-zero.
        self.assertGreater(fit.rmsd_allfit, 0.0)

    def test_keeps_all_within_cutoff(self):
        # All pairs are within the cutoff; nothing should be pruned.
        P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Q = P + 0.5  # small shift, still well within 2 Å
        fit = chimera_pruned_fit(P, Q, cutoff=2.0)
        self.assertEqual(fit.n_pruned, 3)
        self.assertEqual(fit.n_all, 3)
        # Because points differ by a constant shift, Kabsch should align them with low RMSD.
        self.assertAlmostEqual(fit.rmsd_pruned, 0.0, places=7)
        self.assertAlmostEqual(fit.rmsd_allfit, fit.rmsd_pruned, places=7)


class TestChainPairing(unittest.TestCase):
    def test_pair_chains_simple_match(self):
        # Build two structures with one chain each, same sequence ALA-CYS-ASP.
        def make_structure(chain_name: str) -> gemmi.Structure:
            s = gemmi.Structure()
            s.add_model(gemmi.Model("1"))
            c = s[0].add_chain(chain_name)
            for i, resname in enumerate(["ALA", "CYS", "ASP"], start=1):
                res = gemmi.Residue()
                res.name = resname
                res.seqid = gemmi.SeqId(str(i))
                ca = gemmi.Atom()
                ca.name = "CA"
                ca.element = gemmi.Element("C")
                ca.pos = gemmi.Position(float(i), 0.0, 0.0)
                res.add_atom(ca)
                c.add_residue(res)
            return s

        pred_s = make_structure("A")
        ref_s = make_structure("X")
        pred_chains = extract_chain_sequences(pred_s)
        ref_chains = extract_chain_sequences(ref_s)
        pairs = pair_chains(pred_chains, ref_chains)
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        # Sequences should be identical and residue pairs aligned 0-0,1-1,2-2.
        self.assertEqual(pair.pred.seq, pair.ref.seq)
        self.assertEqual(pair.res_pairs, [(0, 0), (1, 1), (2, 2)])


if __name__ == "__main__":
    unittest.main()
