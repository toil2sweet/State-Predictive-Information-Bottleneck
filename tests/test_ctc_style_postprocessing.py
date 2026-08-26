import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

# The function under test is NumPy-only, while SPIB_training also contains the
# PyTorch training loop. Permit this focused unit test to run in lightweight
# local environments where torch is intentionally absent.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda function: function)
    sys.modules["torch"] = torch_stub
    sys.modules["hsic_utils"] = types.ModuleType("hsic_utils")

import SPIB_training


class CTCStylePostprocessingTest(unittest.TestCase):

    def setUp(self):
        self.traj = np.zeros((30, 2), dtype=float)
        self.traj[8:15] = (1.0, 1.0)
        self.traj[20:23] = (2.0, 2.0)

        ts_mask = np.zeros(30, dtype=bool)
        ts_mask[[3, 4, 10, 11, 20, 21]] = True
        margin = np.ones(30, dtype=float)
        margin[[3, 4, 10, 11, 20, 21]] = [0.01, 0.02, 0.03, 0.04, 0.02, 0.03]
        top1 = np.zeros(30, dtype=int)
        top2 = np.ones(30, dtype=int)
        top1[20:22] = 1
        top2[20:22] = 2
        self.ts_result = {
            "ts_mask": ts_mask,
            "margin": margin,
            "top1": top1,
            "top2": top2,
        }

    def test_global_selection_and_saved_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "run_ts0")
            result = SPIB_training.select_ctc_style_ts_representatives(
                self.traj, self.ts_result, output_prefix=prefix,
                top_k=2, event_max_gap=0, density_bins=3,
                selection_scope="global")

            np.testing.assert_array_equal(result["event_indices"], [3, 10, 20])
            np.testing.assert_array_equal(result["selected_indices"], [3, 10])
            self.assertEqual(int(result["event_mask"].sum()), 3)
            self.assertEqual(int(result["selected_mask"].sum()), 2)
            self.assertTrue(os.path.isfile(prefix + "_ctc_top2_mask.npy"))
            with open(prefix + "_ctc_metadata.json") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["n_event_representatives"], 3)
            self.assertEqual(metadata["n_selected_representatives"], 2)

    def test_per_pair_selection(self):
        result = SPIB_training.select_ctc_style_ts_representatives(
            self.traj, self.ts_result, top_k=1, event_max_gap=0,
            density_bins=3, selection_scope="per_pair")

        np.testing.assert_array_equal(result["selected_indices"], [3, 20])
        np.testing.assert_array_equal(result["selected_pairs"], [[0, 1], [1, 2]])

    def test_state_pair_filter(self):
        result = SPIB_training.select_ctc_style_ts_representatives(
            self.traj, self.ts_result, top_k=10, event_max_gap=0,
            density_bins=3, selection_scope="global", state_pairs=[[1, 2]])

        np.testing.assert_array_equal(result["event_indices"], [20])
        np.testing.assert_array_equal(result["selected_indices"], [20])
        self.assertEqual(result["metadata"]["n_filtered_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
