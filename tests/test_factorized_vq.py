import unittest

import torch

from finetuning.factorized_vq import ProductEmbedding, long_training_gate, training_check_metadata
from finetuning.model_factory import architecture
from finetuning.run_index_search import nearest


class ProductEmbeddingTests(unittest.TestCase):
    def test_internal_check_metadata_does_not_accept_validation_or_test_rows(self):
        row = {'sample_path': 'data/train/example.npz', 'split': 'train'}
        result = training_check_metadata([row], '/repo')
        self.assertEqual(str(next(iter(result))), '/repo/data/train/example.npz')
        for split in ('val', 'test'):
            with self.assertRaises(ValueError):
                training_check_metadata([{**row, 'split': split}], '/repo')
        with self.assertRaises(ValueError):
            training_check_metadata([row, row], '/repo')

    def test_composite_ids_reconstruct_both_subspaces_exactly(self):
        left, right = torch.randn(32, 64), torch.randn(64, 64)
        embedding = ProductEmbedding(left, right, torch.eye(128), torch.zeros(128))
        ids = torch.tensor([0, 63, 64, 2047])
        expected = torch.cat([left[ids // 64], right[ids % 64]], dim=1)
        self.assertTrue(torch.equal(embedding(ids), expected))
        self.assertEqual(embedding.weight.shape, (2048, 128))
        self.assertEqual(sum(p.numel() for p in embedding.parameters()), 6144)
        embedding(ids).square().mean().backward()
        self.assertGreater(float(embedding.left.grad.abs().sum()), 0.)
        self.assertGreater(float(embedding.right.grad.abs().sum()), 0.)
        self.assertFalse(embedding.basis.requires_grad)

    def test_nonorthogonal_transform_and_legacy_loader_are_rejected(self):
        with self.assertRaises(ValueError):
            ProductEmbedding(torch.zeros(32, 64), torch.zeros(64, 64), torch.ones(128, 128), torch.zeros(128))
        with self.assertRaises(ValueError):
            architecture({'architecture': 'factorized-vq-study-v1'})

    def test_full_book_nearest_matches_independent_subspace_assignments(self):
        torch.manual_seed(12)
        left, right = torch.randn(32, 64), torch.randn(64, 64)
        basis = torch.eye(128)[:, torch.randperm(128)]
        center = torch.randn(128)
        book = ProductEmbedding(left, right, basis, center)
        original = torch.randn(8, 128)
        transformed = (original - center) @ basis
        left_ids = torch.cdist(transformed[:, :64], left).argmin(dim=1)
        right_ids = torch.cdist(transformed[:, 64:], right).argmin(dim=1)
        z = original.reshape(1, 2, 4, 128).permute(0, 3, 1, 2)
        self.assertTrue(torch.equal(nearest(z, book.weight), left_ids * 64 + right_ids))

    def test_long_training_gate_requires_learning_rate_proximity_and_noncollapse(self):
        good = dict(initial_product=.72, previous_product=.70, final_product=.695,
                    flat_rmse=.68, product_bytes=1020, flat_bytes=1000, perplexity=200)
        self.assertTrue(long_training_gate(**good))
        for key, value in [('final_product', .71), ('final_product', -1), ('product_bytes', 1100), ('perplexity', 1), ('flat_rmse', .60)]:
            with self.subTest(key=key):
                self.assertFalse(long_training_gate(**{**good, key: value}))
