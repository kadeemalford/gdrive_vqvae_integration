import unittest

import torch

from compression.untied_synthesis import SynthesisLookup
from finetuning.run_index_search import vectors
from finetuning.run_untied_synthesis_probe import weighted_macro_distortion


class UntiedSynthesisTests(unittest.TestCase):
    def test_initial_synthesis_is_exact_and_does_not_alias_analysis(self):
        analysis = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(8, 3))
        ids = torch.tensor([0, 1, 7, 4, 3, 2])
        synthesis = SynthesisLookup(analysis)
        self.assertTrue(torch.equal(synthesis(ids, (2, 3)), vectors(ids, analysis, (2, 3))))
        self.assertNotEqual(synthesis.weight.data_ptr(), analysis.data_ptr())
        before = analysis.detach().clone()
        optimizer = torch.optim.Adam([synthesis.weight], lr=1e-3)
        synthesis(ids, (2, 3)).square().mean().backward(); optimizer.step()
        self.assertIsNone(analysis.grad)
        self.assertTrue(torch.equal(analysis, before))
        self.assertFalse(torch.equal(synthesis.weight, analysis))

    def test_invalid_ids_and_grids_fail_closed(self):
        synthesis = SynthesisLookup(torch.ones(8, 3))
        for ids, shape in [(torch.tensor([8]), (1, 1)), (torch.tensor([-1]), (1, 1)),
                           (torch.tensor([1., 2.]), (1, 2)), (torch.tensor([1, 2]), (2, 2)),
                           (torch.tensor([True]), (1, 1)), (torch.tensor([1+2j]), (1, 1))]:
            with self.assertRaises(ValueError):
                synthesis(ids, shape)

    def test_macro_uses_cells_within_configuration(self):
        rows=[('a','c',1.,100),('a','c',10.,1),('b','d',2.,50)]
        self.assertAlmostEqual(weighted_macro_distortion(rows),((100+10)/101+2)/2)
