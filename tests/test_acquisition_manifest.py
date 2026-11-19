import unittest

from tools.validate_acquisition_manifest import FIELDS, validate


class AcquisitionManifestTests(unittest.TestCase):
    def record(self):
        row = {field: "documented" for field in FIELDS}
        row.update(source_id="new-source", session_id="session-a", parent_source_id="",
                   is_derived="false", acquired_at_utc="2026-11-15T00:00:00Z", dt_s="0.00005",
                   dx_m=".2", gauge_length_m="1", pulse_width_value="10", pulse_width_unit="ns",
                   derivation_time_s=".01", partition_role="development", source_sha256="0" * 64)
        return row

    def test_metadata_validation_is_not_independence_certification(self):
        result = validate([self.record()])
        self.assertTrue(result["metadata_valid"])
        self.assertFalse(result["independence_certified"])
        self.assertFalse(result["waveforms_opened"])

    def test_empty_or_unresolved_units_cannot_be_admitted(self):
        with self.assertRaises(ValueError):
            validate([])
        with self.assertRaises(ValueError):
            validate([{**self.record(), "pulse_width_unit": "native_ticks"}])

    def test_same_session_and_known_derivatives_cannot_cross_roles(self):
        row = self.record()
        with self.assertRaises(ValueError):
            validate([row, {**row, "source_id": "other", "partition_role": "independent_confirmation"}])
        with self.assertRaises(ValueError):
            validate([row, {**row, "source_id": "derivative", "parent_source_id": row["source_id"],
                           "session_id": "other", "is_derived": "true", "partition_role": "reserved"}])
