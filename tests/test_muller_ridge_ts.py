import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_transition_states as pts


class MullerRidgeTsTest(unittest.TestCase):

    def test_top_k_nonpositive_returns_original_mask(self):
        mask = np.array([True, False, True, False])
        scores = np.array([1.0, 9.0, 3.0, 8.0])
        out = pts.select_ridge_ts_mask(mask, scores, top_k=0)
        np.testing.assert_array_equal(out, mask)
        out = pts.select_ridge_ts_mask(mask, scores, top_k=-1)
        np.testing.assert_array_equal(out, mask)

    def test_empty_mask_returns_empty(self):
        mask = np.zeros(5, dtype=bool)
        scores = np.arange(5, dtype=float)
        out = pts.select_ridge_ts_mask(mask, scores, top_k=20)
        np.testing.assert_array_equal(out, mask)
        self.assertEqual(int(out.sum()), 0)

    def test_per_pair_keeps_highest_scores(self):
        mask = np.zeros(8, dtype=bool)
        mask[[0, 1, 2, 4, 5]] = True
        scores = np.array([1.0, 5.0, 4.0, 99.0, 2.0, 8.0, 7.0, 6.0])
        top1 = np.array([0, 0, 0, 9, 1, 1, 1, 2])
        top2 = np.array([1, 1, 1, 9, 2, 2, 2, 0])
        out = pts.select_ridge_ts_mask(
            mask, scores, top1=top1, top2=top2, top_k=1)
        self.assertTrue(out[1])
        self.assertTrue(out[5])
        self.assertEqual(int(out.sum()), 2)
        self.assertFalse(out[0])
        self.assertFalse(out[2])
        self.assertFalse(out[4])

    def test_global_ranking_without_pairs(self):
        mask = np.array([True, True, True, False])
        scores = np.array([1.0, 3.0, 2.0, 9.0])
        out = pts.select_ridge_ts_mask(mask, scores, top_k=2)
        np.testing.assert_array_equal(out, np.array([False, True, True, False]))

    def test_load_decoder_top_pairs_from_file(self):
        top1 = np.arange(4)
        top2 = np.array([1, 0, 3, 2])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "run_ts0_top2.npy")
            np.save(path, np.stack([top1, top2], axis=1))
            loaded1, loaded2 = pts.load_decoder_top_pairs(None, path)
            np.testing.assert_array_equal(loaded1, top1)
            np.testing.assert_array_equal(loaded2, top2)


if __name__ == "__main__":
    unittest.main()
