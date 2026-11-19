import copy
import unittest

import numpy as np
import torch
from torch import nn

from compression.conditioned_vq_stream import HEADER, ConditionedVQWire, pack_context, unpack_context
from finetuning.physical_conditioning import ConditionalGroupNorm, condition, metadata_plan, set_condition
from tests.test_joint_anchor_vq import ToyCodec


class PhysicalConditioningTests(unittest.TestCase):
    def plan(self):
        fields = ("dx_m", "dt_s", "GaugeLength", "PulseWidth", "PulseRateFreq", "SamplingRate",
                  "SamplingRes", "Oversampling", "AmpliPower", "DerivationTime")
        groups = ["doped/5m SMF", "multicore/C1 and C4", "multicore/C2 and C3",
                  "multimode/Configuration 7"] + [f"single_mode/Configuration {n}" for n in (11, 12, 3, 4, 7, 8)]
        records = [{"drive_id": str(i), "group_id": group} for i, group in enumerate(groups)]
        summary = {"numeric_fields_by_file": {
            str(i): {key: float(i + 1) if j == 0 else 1. for j, key in enumerate(fields)}
            for i in range(len(groups))}}
        return metadata_plan(summary, records), summary, records

    def test_training_normalizer_excludes_heldout_combinations(self):
        plan, summary, records = self.plan()
        changed = copy.deepcopy(summary)
        changed["numeric_fields_by_file"]["4"]["dx_m"] = 1000.
        after = metadata_plan(changed, records)
        self.assertEqual(plan["log_center"], after["log_center"])
        self.assertEqual(plan["log_scale"], after["log_scale"])
        self.assertEqual(plan["heldout"], ["single_mode/Configuration 11", "single_mode/Configuration 8"])
        unknown = condition(plan, "category", plan["heldout"][0])
        self.assertTrue(np.all(unknown == 0))
        self.assertEqual(condition(plan, "hardware", plan["heldout"][0], True)[-1], 1)

    def test_initial_function_exact_and_modulation_gradients_live(self):
        torch.manual_seed(105)
        original = nn.GroupNorm(2, 4)
        layer = ConditionalGroupNorm(original)
        x = torch.randn(2, 4, 3, 17)
        model = nn.Sequential(layer)
        set_condition(model, np.arange(11, dtype=np.float32) / 10)
        expected = original(x)
        actual = layer(x)
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
        actual.square().mean().backward()
        self.assertGreater(float(layer.modulator[-1].weight.grad.abs().sum()), 0)

    def test_context_cost_crc_and_modes(self):
        plan, _, _ = self.plan()
        group = plan["seen"][0]
        for method, size in (("constant", 0), ("hardware", 40), ("category", 1)):
            vector = condition(plan, method, group)
            frame = pack_context(b"inner", method, vector, 0)
            self.assertEqual(len(frame), HEADER.size + size + 5)
            restored, inner = unpack_context(frame, method)
            np.testing.assert_array_equal(vector.view(np.uint32), restored.view(np.uint32))
            self.assertEqual(inner, b"inner")
            corrupt = bytearray(frame)
            corrupt[-1] ^= 1
            with self.assertRaises(ValueError):
                unpack_context(bytes(corrupt), method)
            missing = condition(plan, method, group, True)
            restored, _ = unpack_context(pack_context(b"inner", method, missing, 0, True), method)
            np.testing.assert_array_equal(missing, restored)

    def test_uncharged_context_is_rejected(self):
        with self.assertRaises(ValueError):
            pack_context(b"inner", "constant", np.ones(11, np.float32))

    def test_conditioned_codec_independent_receiver_uses_only_paid_context(self):
        plan, _, _ = self.plan()
        torch.manual_seed(105)

        def make():
            model = ToyCodec()
            model.encoder = nn.Sequential(model.encoder, ConditionalGroupNorm(nn.GroupNorm(1, 1)))
            model.decoder = nn.Sequential(model.decoder, ConditionalGroupNorm(nn.GroupNorm(1, 1)))
            return model.eval()

        sender_model, receiver_model = make(), make()
        with torch.no_grad():
            for module in sender_model.modules():
                if isinstance(module, ConditionalGroupNorm):
                    module.modulator[-1].weight.fill_(.02)
        receiver_model.load_state_dict(sender_model.state_dict())
        raw = np.random.default_rng(105).normal(size=(3, 17)).astype(np.float32)
        for method in ("constant", "hardware", "category"):
            config = {"method": method, "plan": plan}
            sender = ConditionedVQWire.create(sender_model, np.ones(2048, np.int64), "0" * 64, config)
            receiver = ConditionedVQWire.from_setup(receiver_model, sender.setup_bytes(), config)
            for group in (plan["seen"][0], plan["heldout"][0]):
                for missing in (False, True):
                    frame = sender.encode(raw, group, missing)
                    _, inner = unpack_context(frame, method)
                    expected = sender.inner.decode(inner)
                    set_condition(sender_model, np.full(11, 123., np.float32))
                    self.assertEqual(receiver.decode(frame).tobytes(), expected.tobytes())
