import unittest

from finetuning.joint_anchor_gate import extension_gate


class JointAnchorGateTests(unittest.TestCase):
    def test_extension_not_promotion(self):
        point = lambda error, size=100: {"macro_families": error, "stream_bytes": size, "cells": 1000}
        result = extension_gate(point(1.), point(.81), point(.8), point(.8), point(.79), 32., 32.)
        self.assertTrue(result["eligible"])
        self.assertFalse(result["final_acceptance"])
        self.assertFalse(extension_gate(point(1.), point(.81), point(.8), point(.8, 106),
                                        point(.79), 32., 32.)["eligible"])
        self.assertFalse(extension_gate(point(1.), point(.81), point(.8), point(.8),
                                        point(.79), 32., 1.)["eligible"])
