import os
import sys
import types
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda function: function)
    sys.modules["torch"] = torch_stub
    sys.modules["hsic_utils"] = types.ModuleType("hsic_utils")

import SPIB_training


class PopulationMarginTsTest(unittest.TestCase):

    def setUp(self):
        # Two-state decoder rows: equal K vs K proportional to ρ=(0.8, 0.2).
        self.K = np.array([
            [0.50, 0.50],
            [0.80, 0.20],
            [0.51, 0.49],
            [0.79, 0.21],
        ], dtype=float)
        self.rho = np.array([0.8, 0.2], dtype=float)

    def test_raw_mode_keeps_unweighted_isocommittor(self):
        result = SPIB_training.identify_transition_states(
            self.K, population=self.rho, eps_ts=0.05,
            require_cross_state=False, ts_margin_mode="raw")
        np.testing.assert_array_equal(
            result["ts_mask"], np.array([True, False, True, False]))
        self.assertEqual(result["ts_margin_mode"], "raw")

    def test_population_mode_shifts_to_rho_weighted_isocommittor(self):
        result = SPIB_training.identify_transition_states(
            self.K, population=self.rho, eps_ts=0.05,
            require_cross_state=False, ts_margin_mode="population")
        np.testing.assert_array_equal(
            result["ts_mask"], np.array([False, True, False, True]))
        np.testing.assert_allclose(result["margin"][1], 0.0, atol=1e-12)
        self.assertGreater(result["margin"][0], 0.5)
        self.assertEqual(result["ts_margin_mode"], "population")

    def test_balanced_population_matches_raw(self):
        K = np.array([[0.50, 0.50], [0.70, 0.30]], dtype=float)
        rho = np.array([0.5, 0.5], dtype=float)
        raw = SPIB_training.identify_transition_states(
            K, population=rho, eps_ts=0.05,
            require_cross_state=False, ts_margin_mode="raw")
        pop = SPIB_training.identify_transition_states(
            K, population=rho, eps_ts=0.05,
            require_cross_state=False, ts_margin_mode="population")
        np.testing.assert_allclose(raw["margin"], pop["margin"], atol=1e-12)
        np.testing.assert_array_equal(raw["ts_mask"], pop["ts_mask"])

    def test_alias_and_invalid_mode(self):
        result = SPIB_training.identify_transition_states(
            self.K, population=self.rho, eps_ts=0.05,
            require_cross_state=False, ts_margin_mode="rho")
        self.assertEqual(result["ts_margin_mode"], "population")
        with self.assertRaises(ValueError):
            SPIB_training.identify_transition_states(
                self.K, ts_margin_mode="committor")

    def test_single_state_returns_empty_mask(self):
        K = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=float)
        result = SPIB_training.identify_transition_states(
            K, active_indices=[0], eps_ts=0.05,
            require_cross_state=False, ts_margin_mode="population",
            population=np.array([1.0, 0.0]))
        self.assertEqual(result["n_transition_states"], 0)
        self.assertEqual(int(result["ts_mask"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
