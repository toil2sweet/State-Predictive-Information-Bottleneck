import configparser
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import SPIB_training
except ModuleNotFoundError:
    torch = None
    SPIB_training = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None, "PyTorch is required for SPIB split test")
class Tutorial3SplitTest(unittest.TestCase):
    def test_contiguous_block_split_keeps_lagged_pairs_within_blocks(self):
        # tutorial3 splits an ultra-long trajectory into 100 contiguous
        # trajectories, then applies the lag independently within each one.
        data = torch.arange(100, dtype=torch.float32).reshape(-1, 1)
        labels = torch.nn.functional.one_hot(
            torch.arange(100) % 4, num_classes=4).float()

        np.random.seed(0)
        result = SPIB_training.data_init(
            0, 3, data, labels, None,
            split_mode="trajectory_blocks", num_blocks=10)
        (_, train_past, train_future, train_labels, train_weights,
         test_past, test_future, test_labels, test_weights) = result

        # 9 train blocks and 1 test block, each contributing 10 - dt pairs.
        self.assertEqual(train_past.shape, (63, 1))
        self.assertEqual(test_past.shape, (7, 1))
        self.assertTrue(torch.equal(train_future - train_past,
                                    torch.full_like(train_past, 3)))
        self.assertTrue(torch.equal(test_future - test_past,
                                    torch.full_like(test_past, 3)))
        self.assertTrue(torch.equal(train_labels.argmax(dim=1),
                                    (train_future[:, 0].long() % 4)))
        self.assertTrue(torch.equal(test_labels.argmax(dim=1),
                                    (test_future[:, 0].long() % 4)))
        self.assertIsNone(train_weights)
        self.assertIsNone(test_weights)


class Tutorial3ConfigTest(unittest.TestCase):
    def test_trpcage_baseline_uses_tutorial_loss_convergence(self):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(ROOT / "examples" / "TrpCage_sample_plus_config.ini")

        self.assertEqual(
            config.get("Training Parameters", "convergence_mode"), "loss")
        self.assertEqual(
            config.getfloat("Training Parameters", "tolerance"), 0.002)
        self.assertEqual(
            config.get("Data", "split_mode"), "trajectory_blocks")
        self.assertEqual(
            config.getint("Data", "split_num_blocks"), 100)

    def test_hsic_trpcage_uses_the_same_stable_training_protocol(self):
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(ROOT / "examples" / "TrpCage_hsic_plus_config.ini")

        self.assertEqual(
            config.get("Model Parameters", "dt"), "[500]")
        self.assertEqual(
            config.get("Training Parameters", "convergence_mode"), "loss")
        self.assertEqual(
            config.getfloat("Training Parameters", "tolerance"), 0.005)
        self.assertEqual(
            config.getint("Training Parameters", "batch_size"), 1024)
        self.assertEqual(
            config.get("Data", "split_mode"), "trajectory_blocks")
        self.assertEqual(
            config.getint("Data", "split_num_blocks"), 100)
        self.assertEqual(
            config.getfloat("HSIC-SPIB", "beta_x"), 0.01)
        self.assertEqual(
            config.getfloat("HSIC-SPIB", "lambda_y"), 0.1)

    def test_hsic_trpcage_mtl_hku_uses_three_lags_and_the_same_protocol(self):
        hsic = configparser.ConfigParser(allow_no_value=True)
        hsic.read(ROOT / "examples" / "TrpCage_hsic_plus_hku_config.ini")
        mtl = configparser.ConfigParser(allow_no_value=True)
        mtl.read(ROOT / "examples" / "TrpCage_hsic_plus_MTL_hku_config.ini")

        self.assertEqual(mtl.get("Model Parameters", "dt"), "[50, 100, 500]")
        self.assertEqual(mtl.get("Model Parameters", "d"), "[2]")
        self.assertEqual(mtl.get("Model Parameters", "encoder_type"), "Nonlinear")
        self.assertEqual(
            mtl.get("Training Parameters", "convergence_mode"),
            hsic.get("Training Parameters", "convergence_mode"))
        self.assertEqual(
            mtl.getfloat("Training Parameters", "tolerance"),
            hsic.getfloat("Training Parameters", "tolerance"))
        self.assertEqual(
            mtl.getint("Training Parameters", "batch_size"),
            hsic.getint("Training Parameters", "batch_size"))
        self.assertEqual(
            mtl.get("Data", "split_mode"), hsic.get("Data", "split_mode"))
        self.assertEqual(
            mtl.getint("Data", "split_num_blocks"),
            hsic.getint("Data", "split_num_blocks"))
        self.assertEqual(
            mtl.get("Data", "traj_data"), hsic.get("Data", "traj_data"))
        self.assertEqual(
            mtl.get("Data", "initial_labels"), hsic.get("Data", "initial_labels"))
        self.assertEqual(
            mtl.getfloat("HSIC-SPIB", "beta_x"),
            hsic.getfloat("HSIC-SPIB", "beta_x"))
        self.assertEqual(
            mtl.getfloat("HSIC-SPIB", "lambda_y"),
            hsic.getfloat("HSIC-SPIB", "lambda_y"))
        self.assertEqual(
            mtl.get("HSIC-SPIB", "kernel_x"), hsic.get("HSIC-SPIB", "kernel_x"))
        self.assertEqual(
            mtl.get("HSIC-SPIB", "ts_potential"), "trpcage")
        self.assertFalse(mtl.has_section("CTC-Style TS"))


if __name__ == "__main__":
    unittest.main()
