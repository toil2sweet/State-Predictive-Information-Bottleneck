import os
import unittest
from pathlib import Path

import numpy as np

import trpcage.prepare_spib_data as prepare


DESRES_ROOT = Path(os.environ.get(
    "TRPCAGE_DESRES_ROOT", str(prepare.DEFAULT_DESRES_ROOT)))


class TrpCageContactCountTest(unittest.TestCase):

    def test_trp_cage_has_153_minimal_contacts(self):
        self.assertEqual(prepare.n_minimal_contacts(20), 153)
        self.assertEqual(prepare.EXPECTED_N_CONTACTS, 153)


@unittest.skipUnless(
    DESRES_ROOT.is_dir(),
    "local DESRES Trp-cage distribution is not present",
)
class TrpCageDesresSmokeTest(unittest.TestCase):

    def test_mae_and_first_dcd_yield_153_contacts(self):
        protein_dir = prepare.find_protein_dir(DESRES_ROOT)
        atoms = prepare.parse_mae_atoms(prepare.find_mae(protein_dir))
        groups, pairs, resids = prepare.residue_heavy_index_groups(atoms)
        self.assertEqual(len(resids), 20)
        self.assertEqual(len(pairs), 153)
        self.assertEqual(len(atoms), 272)

        first_dcd = prepare.list_protein_dcds(protein_dir)[0]
        xyz = prepare.load_dcd_xyz(first_dcd)[:4]
        self.assertEqual(xyz.shape[1], 272)
        contacts = prepare.closest_heavy_contacts(xyz, groups, pairs)
        self.assertEqual(contacts.shape, (4, 153))
        self.assertTrue(np.isfinite(contacts).all())
        self.assertGreater(float(contacts.min()), 0.0)
        self.assertLess(float(contacts.max()), 80.0)


if __name__ == "__main__":
    unittest.main()
