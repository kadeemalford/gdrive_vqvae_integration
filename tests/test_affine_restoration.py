import unittest
import importlib.util

import numpy as np
import torch

from compression.affine_restoration import affine_side, candidates, restore, AFFINE_MAGIC
from finetuning.robust_escape import encode_robust
from finetuning.reversible_normalization import decode, parse_side_stream
from finetuning.affine_index_search import paired_search
from finetuning.run_index_search import FrozenIndexStream
from finetuning.run_affine_index_search import excluded_windows


class AffineRestorationTests(unittest.TestCase):
    def test_constrained_ols_respects_the_source_rms_feasible_interval(self):
        rng = np.random.default_rng(12)
        raw = rng.normal(size=(10, 1000)).astype(np.float32)
        packet = encode_robust(raw)
        prediction = .7 * packet.values + .7 * rng.normal(size=raw.shape)
        choices = candidates(packet.values, prediction, packet.valid, packet.side_stream, baseline_rms=.95)
        bounded = next(v for v in choices if v['kind'] == 'bounded_ols')
        fitted = prediction * bounded['gain'] + bounded['offset']
        ratio = np.linalg.norm(fitted) / np.linalg.norm(packet.values)
        self.assertGreaterEqual(ratio, .95)
        self.assertLessEqual(ratio, 1.05)
        self.assertEqual(len(bounded['side']), len(packet.side_stream))

    def test_historical_protocol_selection_string_is_not_a_window_map(self):
        self.assertEqual(excluded_windows({'selection': 'train criterion', 'validation': ['a']}), {'a'})
        self.assertEqual(excluded_windows({'selection': {'val': ['b']}, 'excluded_validation_windows': ['a']}), {'a', 'b'})

    @unittest.skipIf(importlib.util.find_spec('constriction') is None, 'Optional entropy dependency not installed')
    def test_paired_search_replays_nested_candidates_without_weight_updates(self):
        torch.set_num_threads(1)
        torch.manual_seed(2087)
        model = torch.nn.Module()
        model.encoder = torch.nn.Conv2d(1, 4, 1)
        model.decoder = torch.nn.Conv2d(4, 1, 1)
        model.quantizer = torch.nn.Module()
        model.quantizer.embedding = torch.nn.Embedding(8, 4)
        model.eval().requires_grad_(False)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        raw = np.random.default_rng(12).normal(size=(4, 20)).astype(np.float32)
        packet = encode_robust(raw)
        stream = FrozenIndexStream(np.ones(2048, dtype=np.int64), '0' * 64)
        source, before, best, _ = paired_search(model, torch.from_numpy(packet.values)[None, None],
            torch.from_numpy(packet.valid)[None, None], {'family': 'a', 'configuration': 'c'}, raw, packet, stream, .05, steps=4, bounded_ols=True)
        self.assertLessEqual(best['affine']['loss'], best['strict']['loss'])
        self.assertLessEqual(len(best['affine']['frame']), len(source))
        self.assertLessEqual(best['bounded']['loss'], best['affine']['loss'])
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, initial[k], rtol=0, atol=0)

    def test_layout_cost_escapes_and_original_contract_remain_explicit(self):
        raw = np.arange(100, dtype=np.float32).reshape(10, 10)
        raw[0, 0] = 1e30
        packet = encode_robust(raw)
        side = affine_side(packet.side_stream, 1.2, .1)
        self.assertEqual(len(side), len(packet.side_stream))
        self.assertEqual(side[:4], AFFINE_MAGIC)
        with self.assertRaises(ValueError):
            parse_side_stream(side)
        actual = restore(packet.values, side)
        self.assertEqual(actual[0, 0].tobytes(), raw[0, 0].tobytes())
        expected = decode(packet.values * np.float32(1.2) + np.float32(.1), packet.side_stream)
        np.testing.assert_allclose(actual[packet.valid], expected[packet.valid], rtol=1e-6, atol=1e-5)

    def test_ols_recovers_affine_distortion_without_more_side_bytes(self):
        raw = np.arange(100, dtype=np.float32).reshape(10, 10)
        packet = encode_robust(raw)
        prediction = packet.values * .7 + .2
        fitted = candidates(packet.values, prediction, packet.valid, packet.side_stream)
        self.assertEqual([v['kind'] for v in fitted], ['ols', 'unit_rms'])
        np.testing.assert_allclose(restore(prediction, fitted[0]['side']), raw, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(fitted[0]['side']), len(packet.side_stream))

    def test_constant_or_invalid_inputs_are_not_fabricated(self):
        raw = np.zeros((2, 20), dtype=np.float32)
        packet = encode_robust(raw)
        self.assertEqual(candidates(packet.values, packet.values, packet.valid, packet.side_stream), [])
        for gain in [0., -1., float('nan')]:
            with self.assertRaises(ValueError):
                affine_side(packet.side_stream, gain, 0.)

    def test_classic_side_roundtrip_is_unchanged(self):
        packet = encode_robust(np.arange(12, dtype=np.float32).reshape(3, 4))
        np.testing.assert_array_equal(restore(packet.values, packet.side_stream), decode(packet.values, packet.side_stream))
