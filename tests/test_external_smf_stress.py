import unittest
import numpy as np

from acquisition.prepare_external_smf_stress import select_smf
from finetuning.analyze_external_smf_stress import seam_control


class ExternalSMFSelectionTests(unittest.TestCase):
    def test_seam_control_separates_target_energy_from_relative_error(self):
        raw = np.random.default_rng(8).normal(size=(4, 2000)).astype(np.float32)
        raw[:, 1000] += 100
        result = seam_control(raw, .8 * raw, np.ones(raw.shape, dtype=bool))
        self.assertAlmostEqual(result['seam']['relative_difference_rmse'], .2, places=5)
        self.assertAlmostEqual(result['interior']['relative_difference_rmse'], .2, places=5)

    def test_selection_is_metadata_fixed_and_never_reads_EDAS(self):
        files = []
        for fiber in ('SMF', 'EDAS Acrylate', 'EDAS Polyimide'):
            for orientation in ('Horizontal', 'Vertical'):
                for frequency in (200, 1000, 10000):
                    files.append({'id': f'{fiber}_{orientation}_{frequency}', 'name': f'Copy of {frequency}Hz.h5',
                                  'candidate_folder': {'fiber': fiber, 'orientation': orientation}})
        selected = select_smf({'candidate_files': files})
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(r['candidate_folder']['fiber'] == 'SMF' for r in selected))
        self.assertEqual([r['stimulus_label_hz'] for r in selected], [200, 1000, 10000, 200, 1000, 10000])

    def test_missing_cell_fails_without_picking_another_frequency(self):
        with self.assertRaises(ValueError):
            select_smf({'candidate_files': []})
