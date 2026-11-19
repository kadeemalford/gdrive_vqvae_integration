import unittest

from finetuning.audit_feasibility_deployment import compare


def point(transport, installation):
    return {"cells": 1000, "transport_bytes": transport, "installation_bytes": installation,
            "total_bytes": transport + installation}


class DeploymentAuditTests(unittest.TestCase):
    def test_zero_installation_counterfactual_does_not_change_transport(self):
        result = compare(point(990, 10), point(920, 30))
        self.assertAlmostEqual(result["measured_net_saving_fraction"], .05)
        self.assertAlmostEqual(result["saving_fraction_with_contextual_installation_free"], .08)
        self.assertEqual(result["additional_bytes_needed_for_10pct_at_measured_installation"], 50)
        self.assertEqual(result["additional_transport_bytes_needed_for_10pct_even_with_free_installation"], 20)

    def test_nonreconciling_ledger_fails(self):
        with self.assertRaises(ValueError):
            compare(point(990, 10), {**point(920, 30), "total_bytes": 940})
