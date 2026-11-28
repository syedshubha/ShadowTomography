"""Checks for the shadow simulation, the feature map, and one training run."""

import os
import sys
import unittest

import numpy as np
import torch
from qutip import basis, ket2dm, qeye, tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import neural_shadows as ns  # noqa: E402

BELL = ket2dm((tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit())
BELL_VEC = np.array([1, 0, 0, 1]) / np.sqrt(2)


class Physics(unittest.TestCase):
    def test_concurrence_limits(self):
        self.assertAlmostEqual(ns.concurrence(BELL), 1.0, places=6)
        self.assertAlmostEqual(ns.concurrence(qeye([2, 2]) / 4), 0.0, places=6)

    def test_werner_boundary(self):
        # C(p) = max(0, 1 - 3p/2), so the state turns separable at p = 2/3.
        for p in (0.1, 0.4, 0.6, 0.7, 0.9):
            rho = ns.werner_state_channel(BELL, p)
            self.assertAlmostEqual(ns.concurrence(rho), max(0.0, 1 - 1.5 * p), places=6)


class Shadows(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)

    def test_snapshot_format(self):
        s = ns.generate_shadow_pair(BELL, 2, 50)
        self.assertEqual(s.shape, (50, 2, 2))
        self.assertTrue(set(np.unique(s[:, :, 0])) <= {0, 1, 2})
        self.assertTrue(set(np.unique(s[:, :, 1])) <= {-1, 1})

    def test_estimate_has_unit_trace(self):
        feats = ns.shadow_to_features(ns.generate_shadow_pair(BELL, 2, 5)[None])
        self.assertAlmostEqual(np.trace(feats[0].reshape(4, 4)), 1.0, places=9)

    def test_estimate_converges_to_bell_state(self):
        feats = ns.shadow_to_features(ns.generate_shadow_pair(BELL, 2, 4000)[None])
        rho_hat = feats[0].reshape(4, 4)
        self.assertAlmostEqual(BELL_VEC @ rho_hat @ BELL_VEC, 1.0, delta=0.15)


class Pipeline(unittest.TestCase):
    def test_dataset_shape_and_labels(self):
        X, y = ns.generate_dataset(40, 5, seed=7)
        self.assertEqual(X.shape, (40, 5, 2, 2))
        self.assertEqual(set(np.unique(y)), {0, 1})

    def test_single_trial_scores(self):
        np.random.seed(7)
        torch.manual_seed(7)
        ns.N_SAMPLES, ns.T_SNAPSHOTS = 60, 10
        *scores, X_feat, y = ns.run_single_trial(verbose=False)
        self.assertEqual(X_feat.shape, (60, 16))
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
