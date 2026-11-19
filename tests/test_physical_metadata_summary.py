import unittest

from finetuning.summarize_physical_metadata import summarize


class MetadataSummaryTests(unittest.TestCase):
    def record(self):
        return {"source_exists": True, "drive_id": "id", "group_id": "family/config",
                "dataset": "zone/data", "prepared_dx_m": [".2"], "prepared_dt_s": [".00005"],
                "attributes": {"/zone": {"Spacing": [.2, .05, 1.], "GaugeLength": [1.],
                    "PulseWidth": [1], "PulseRateFreq": [100], "SamplingRate": [500],
                    "SamplingRes": [20], "Oversampling": [1], "AmpliPower": [26],
                    "DerivationTime": [.01], "Hostname": "device", "AcqStartTime": [123]}}}

    def test_preparation_spacing_and_header_provenance_not_independence(self):
        result = summarize([self.record()])
        self.assertEqual(result["prepared_spacing_conflict_files"], [])
        self.assertEqual(result["header_hostname_start_combinations"], 1)
        self.assertFalse(result["dataset_values_read"])
        record = self.record()
        record["prepared_dt_s"] = [".001"]
        self.assertEqual(summarize([record])["prepared_spacing_conflict_files"], ["id"])
