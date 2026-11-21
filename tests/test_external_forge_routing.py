import struct
import unittest

import numpy as np

from finetuning.run_external_forge_routing import DT_US, NS, STRIDE, parse_block


class ForgeTraceTests(unittest.TestCase):
    def test_endian_orientation_payload_and_gap(self):
        b = bytearray(2*STRIDE)
        for i in range(2):
            pos = i*STRIDE
            struct.pack_into(">i", b, pos, i+1)
            struct.pack_into(">HH", b, pos+114, NS, DT_US)
            struct.pack_into(">i", b, pos+196, 1000*i)
            struct.pack_into(">II", b, pos+240, 0x80000000+i, 0x7fa12345+i)
        raw, headers = parse_block(b, 0, 2)
        np.testing.assert_array_equal(raw[:2].view("<u4"),
                                      [[0x80000000,0x80000001],[0x7fa12345,0x7fa12346]])
        self.assertEqual(len(headers), 480)
        struct.pack_into(">i", b, STRIDE, 7)
        with self.assertRaises(ValueError):
            parse_block(b, 0, 2)
