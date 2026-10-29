"""Controls for tile weighting, region coverage and bounded extension decisions."""
from pathlib import Path
import unittest

from finetuning.reconstruction_objective import aggregate_validation, stratified_validation
from finetuning.run_spatial_abc_study import extension_decision


class SpatialStudyProtocolTests(unittest.TestCase):
    def test_tiny_tile_does_not_equal_large_tile_in_macro(self):
        rows = [dict(family="one",configuration="a",relative_rmse=.2,correlation=.9,_aggregation_weight=512000.),
                dict(family="one",configuration="a",relative_rmse=2.,correlation=.1,_aggregation_weight=1000.)]
        macro, groups = aggregate_validation(rows)
        self.assertAlmostEqual(macro["macro_relative_rmse"],(.2*512+2)/513)
        self.assertNotIn("macro__aggregation_weight",macro)
        self.assertEqual(groups[0]["windows"],2)

    def test_pilot_covers_each_retained_region(self):
        paths = [Path(f"{region}_{i}.npz") for region in range(3) for i in range(4)]
        metadata = {p:{"drive_id":"same","spatial_region_index":p.stem.split("_")[0]} for p in paths}
        selected = stratified_validation(paths,metadata,2,2031)
        self.assertEqual(len(selected),6)
        for region in range(3):
            self.assertEqual(sum(p.stem.startswith(str(region)+"_") for p in selected),2)

    def test_extension_is_bounded_and_trend_based_not_winner_based(self):
        improving = [{"epoch":i+1,"macro_relative_rmse":1.-i*.002} for i in range(16)]
        result = extension_decision(improving)
        self.assertTrue(result["extend"])
        self.assertEqual(result["extra_epochs"],8)
        plateau = [{"epoch":i+1,"macro_relative_rmse":.5} for i in range(16)]
        self.assertFalse(extension_decision(plateau)["extend"])


if __name__=="__main__":
    unittest.main()
