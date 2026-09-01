import tempfile
import unittest
from pathlib import Path

import numpy as np

import SPIB_training
import plot_spib_plus


class ProteinTransitionStateTest(unittest.TestCase):

    def test_cross_state_mask_matches_neighborhood_definition(self):
        hard = np.array([0, 0, 1, 1, 2, 2, 2, 0])
        for window in (0, 1, 2, 3):
            expected = np.zeros(len(hard), dtype=bool)
            for frame in range(len(hard)):
                lo = max(0, frame - window)
                hi = min(len(hard), frame + window + 1)
                expected[frame] = np.unique(hard[lo:hi]).size >= 2
            actual = SPIB_training._cross_state_neighborhood_mask(
                hard, window)
            np.testing.assert_array_equal(actual, expected)

    def test_latent_summary_writes_trpcage_ts_overlays(self):
        rng = np.random.RandomState(7)
        z = np.vstack([
            rng.normal(loc=(-1.0, 0.0), scale=0.2, size=(120, 2)),
            rng.normal(loc=(1.0, 0.0), scale=0.2, size=(120, 2)),
        ])
        labels = np.zeros((len(z), 2), dtype=np.float32)
        labels[:120, 0] = 1.0
        labels[120:, 1] = 1.0
        ts_mask = np.zeros(len(z), dtype=bool)
        ts_mask[[118, 119, 120, 121]] = True

        with tempfile.TemporaryDirectory() as tmp:
            results = plot_spib_plus.plot_plus_summary(
                z, labels, [[1, 3, 2]], tmp, "trpcage_test",
                method_label="HSIC-SPIB+", ts_mask=ts_mask, dpi=40)

            kinds = {kind for kind, _ in results}
            self.assertEqual(
                kinds,
                {
                    "latent_FE",
                    "latent_labels",
                    "latent_FE_with_TS",
                    "latent_labels_with_TS",
                    "state_number",
                },
            )
            for _, path in results:
                output = Path(path)
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)

    def test_latent_ts_subsampling_preserves_total_count(self):
        z = np.column_stack([np.arange(20), np.arange(20)])
        mask = np.ones(20, dtype=bool)
        points, n_ts = plot_spib_plus._latent_transition_points(
            z, mask, max_points=5)
        self.assertEqual(n_ts, 20)
        self.assertEqual(points.shape, (5, 2))
        np.testing.assert_array_equal(points[[0, -1]], z[[0, -1]])


if __name__ == "__main__":
    unittest.main()
