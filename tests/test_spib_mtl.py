import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ModuleNotFoundError:
    torch = None

import plot_mtl
import plot_spib_plus


def _one_hot(idx, n_classes):
    return torch.nn.functional.one_hot(idx.long(), num_classes=n_classes).float()


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class SPIBMTLModelTest(unittest.TestCase):

    def setUp(self):
        import SPIB
        import SPIB_training
        self.SPIB = SPIB
        self.SPIB_training = SPIB_training

    def test_dt_tag(self):
        self.assertEqual(self.SPIB.dt_tag([50, 200, 1000]), "50-200-1000")
        self.assertEqual(self.SPIB.dt_key(50), "50")

    def test_aligned_random_frames(self):
        t = torch.arange(20, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
        labels = _one_hot(torch.arange(20) % 4, 4)
        split = self.SPIB_training.data_init_mtl(0, [2, 5], t, labels, None)
        n_train = split["past_train"].shape[0]
        n_test = split["past_test"].shape[0]
        self.assertEqual(n_train + n_test, 15)
        for dt in (2, 5):
            self.assertEqual(split["future_train"][dt].shape[0], n_train)
            self.assertEqual(split["label_train"][dt].shape[0], n_train)
            self.assertEqual(split["future_test"][dt].shape[0], n_test)

    def test_independent_head_pruning(self):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.randn(40, 2)
        y4 = _one_hot(torch.arange(40) % 4, 4)
        y2 = _one_hot(torch.arange(40) % 2, 4)
        model = self.SPIB.SPIBMTL(
            "Linear", 1, 4, (2,), torch.device("cpu"), [50, 200],
            UpdateLabel=True, neuron_num1=8, neuron_num2=8)
        model.init_representative_inputs(x, {50: y4, 200: y2})
        y2_train, y2_test = model.update_model(
            x, None, y2, y2.clone(), batch_size=8, dt=200, eps_rho=0.0,
            update_representative=False)
        self.assertEqual(model.output_dim[50], 4)
        self.assertEqual(model.output_dim[200], 2)
        self.assertEqual(y2_train.shape[1], 2)
        self.assertEqual(model.decode(torch.zeros(3, 1), 50).shape, (3, 4))
        self.assertEqual(model.decode(torch.zeros(3, 1), 200).shape, (3, 2))
        self.assertEqual(y2_test.shape[1], 2)

    def test_joint_loss_and_short_train(self):
        torch.manual_seed(0)
        np.random.seed(0)
        n = 48
        x = torch.randn(n, 2)
        y = {50: _one_hot(torch.arange(n) % 4, 4),
             200: _one_hot(torch.arange(n) % 3, 4)}
        split = {
            "past_train": x[:40],
            "past_test": x[40:],
            "future_train": {50: x[:40], 200: x[:40]},
            "future_test": {50: x[40:], 200: x[40:]},
            "label_train": {50: y[50][:40], 200: y[200][:40]},
            "label_test": {50: y[50][40:], 200: y[200][40:]},
        }
        model = self.SPIB.SPIBMTL(
            "Linear", 1, 4, (2,), torch.device("cpu"), [50, 200],
            UpdateLabel=True, neuron_num1=8, neuron_num2=8)
        model.init_representative_inputs(split["past_train"], split["label_train"])
        cfg = self.SPIB_training.default_hsic_config()
        cfg.update({"loss_mode": "hsic_spib", "lambda_y": 1.0, "beta_x": 1.0,
                    "decoder_on_mean": True})
        loss, ce, kl, zx, zy = self.SPIB_training.calculate_loss_mtl(
            model, split["past_train"][:8],
            {50: split["label_train"][50][:8], 200: split["label_train"][200][:8]},
            None, [50, 200], beta=0.01, hsic_config=cfg)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(ce))
        self.assertTrue(torch.isfinite(zx))
        self.assertTrue(torch.isfinite(zy))
        nan = self.SPIB_training.train_mtl(
            model, 0.01, split["past_train"], split["future_train"],
            split["label_train"], None, split["past_test"], split["future_test"],
            split["label_test"], None, [50, 200], 1e-3, 5, 1, 8, 10.0, 0, 0,
            os.path.join(tempfile.mkdtemp(), "mtl"), 10000, torch.device("cpu"),
            0, hsic_config=cfg, epoch_sample_size=16)
        self.assertFalse(nan)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "run")
            model.save_traj_results(x, 8, prefix, True, 0, 0)
            self.assertTrue(os.path.isfile(prefix + "_t=50_traj0_labels0.npy"))
            self.assertTrue(os.path.isfile(prefix + "_t=200_traj0_labels0.npy"))
            self.assertTrue(os.path.isfile(prefix + "_traj0_mean_representation0.npy"))


class SPIBMTLPlotTest(unittest.TestCase):

    def test_combined_plots(self):
        rng = np.random.default_rng(0)
        traj = rng.normal(size=(80, 2))
        labels50 = np.eye(4, dtype=np.float64)[rng.integers(0, 4, size=80)]
        labels200 = np.eye(3, dtype=np.float64)[rng.integers(0, 3, size=80)]
        mask50 = np.zeros(80, dtype=bool)
        mask200 = np.zeros(80, dtype=bool)
        mask50[::10] = True
        mask200[::15] = True
        payload = [
            {"dt": 50, "labels": labels50, "ts_mask": mask50},
            {"dt": 200, "labels": labels200, "ts_mask": mask200},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            figs = plot_mtl.plot_all_mtl_figures(
                traj, payload, tmp, "demo", potential="four_well")
            kinds = [kind for kind, _ in figs]
            self.assertIn("mtl_labels", kinds)
            self.assertIn("mtl_labels_with_TS", kinds)
            self.assertIn("mtl_free_energy_with_TS", kinds)
            self.assertIn("mtl_potential_with_TS", kinds)
            for _, path in figs:
                self.assertTrue(os.path.isfile(path))

    def test_protein_plots_use_latent_and_skip_potential(self):
        rng = np.random.default_rng(1)
        distances = rng.normal(size=(60, 8))
        latent = rng.normal(size=(60, 2))
        labels50 = np.eye(4, dtype=np.float64)[rng.integers(0, 4, size=60)]
        labels100 = np.eye(3, dtype=np.float64)[rng.integers(0, 3, size=60)]
        payload = [
            {"dt": 50, "labels": labels50, "ts_mask": np.zeros(60, dtype=bool)},
            {"dt": 100, "labels": labels100, "ts_mask": np.zeros(60, dtype=bool)},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            latent_path = os.path.join(tmp, "run_traj0_mean_representation0.npy")
            np.save(latent_path, latent)
            coords, xlabel, ylabel, skip_pot = plot_mtl.plot_coordinates(
                distances, latent_path=latent_path, ts_potential="trpcage")
            self.assertEqual(coords.shape, (60, 2))
            self.assertEqual(xlabel, r"$IB_0$")
            self.assertEqual(ylabel, r"$IB_1$")
            self.assertTrue(skip_pot)
            figs = plot_mtl.plot_all_mtl_figures(
                coords, payload, tmp, "trpcage", potential=None,
                xlabel=xlabel, ylabel=ylabel, style="latent",
                fe_beta=1.0, fe_vmax=None)
            kinds = [kind for kind, _ in figs]
            self.assertIn("mtl_labels", kinds)
            self.assertIn("mtl_labels_with_TS", kinds)
            self.assertIn("mtl_free_energy_with_TS", kinds)
            self.assertNotIn("mtl_potential_with_TS", kinds)
            toy_coords, toy_x, toy_y, toy_skip = plot_mtl.plot_coordinates(
                latent, ts_potential="four_well")
            np.testing.assert_array_equal(toy_coords, latent)
            self.assertEqual(toy_x, "x")
            self.assertEqual(toy_y, "y")
            self.assertFalse(toy_skip)

    def test_latent_fe_reuses_shared_grid_without_vmax_clip(self):
        rng = np.random.default_rng(2)
        z = np.vstack([
            rng.normal(scale=0.05, size=(800, 2)),
            rng.normal(loc=(4.0, 4.0), scale=0.05, size=(8, 2)),
        ])
        _, _, _, fe = plot_spib_plus.latent_free_energy_grid(z, fe_beta=1.0)
        self.assertGreater(float(np.nanmax(fe)), 3.0)
        n = z.shape[0]
        payload = [
            {"dt": 50, "labels": np.eye(2)[rng.integers(0, 2, size=n)],
             "ts_mask": np.zeros(n, dtype=bool)},
            {"dt": 500, "labels": np.eye(2)[rng.integers(0, 2, size=n)],
             "ts_mask": np.zeros(n, dtype=bool)},
        ]
        payload[0]["ts_mask"][::40] = True
        payload[1]["ts_mask"][::80] = True
        with tempfile.TemporaryDirectory() as tmp:
            figs = plot_mtl.plot_all_mtl_figures(
                z, payload, tmp, "shared_z", potential=None,
                fe_beta=1.0, fe_vmax=None, style="latent",
                xlabel=r"$IB_0$", ylabel=r"$IB_1$")
            kinds = [kind for kind, _ in figs]
            self.assertEqual(
                kinds,
                ["mtl_labels", "mtl_labels_with_TS", "mtl_free_energy_with_TS"])
            for _, path in figs:
                self.assertTrue(os.path.isfile(path))
                self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
