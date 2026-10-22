import unittest
from pathlib import Path

import numpy as np

from finetuning.audit_checkpoint_selection import choose, group_scores, sample_indices
from finetuning.reconstruction_objective import stratified_validation


class CheckpointSelectionAuditTests(unittest.TestCase):
    def test_guard_checks_correlation_and_channel_tail_and_keeps_baseline(self):
        scores = np.array([[[1., .5, 1.]], [[.8, .4, .9]], [[.9, .6, 1.1]]])
        self.assertEqual(choose(scores, np.ones(1), False)[0], 1)
        self.assertEqual(choose(scores, np.ones(1), True)[0], 0)
        scores[2, 0, 2] = .9
        self.assertEqual(choose(scores, np.ones(1), True)[0], 2)

    def test_sampling_is_paired_deterministic_and_temporally_distributed(self):
        rows = [{"drive_id": "a", "sample_path": f"/{i:03}.npz", "block_index": "0",
                 "time_start": str(i)} for i in range(20)]
        indices = sample_indices(rows, 4, 2027, "time_stratified")
        self.assertEqual([i // 5 for i in indices], [0, 1, 2, 3])
        np.testing.assert_array_equal(indices, sample_indices(rows, 4, 2027, "time_stratified"))
        self.assertEqual(len(set(sample_indices(rows, 4, 2027, "random"))), 4)
        with self.assertRaises(ValueError):
            sample_indices(rows, 0, 2027, "random")

    def test_group_scores_preserve_configuration_weighting(self):
        values = np.array([[[1., 1., 1.], [3., 3., 3.], [9., 9., 9.]]])
        scores = group_scores(values, [0, 1, 2], np.array([0, 0, 1]))
        np.testing.assert_array_equal(scores[0, :, 0], [2, 9])
        with self.assertRaises(ValueError):
            group_scores(values, [0, 1], np.array([0, 0, 1]))

    def test_random_design_matches_existing_trainer_sampler(self):
        rows = [{"drive_id": file_id, "sample_path": f"/{file_id}_{i:02}.npz",
                 "block_index": "0", "time_start": str(i)}
                for file_id in ("a", "b") for i in range(20)]
        paths = [Path(row["sample_path"]) for row in rows]
        metadata = {path: row for path, row in zip(paths, rows)}
        expected = stratified_validation(paths, metadata, 2, 2027)
        selected = sample_indices(rows, 2, 2027, "random")
        self.assertEqual([paths[i] for i in selected], expected)
