import io
import unittest
from unittest.mock import patch

from acquisition.public_hdf5_ranges import PublicRanges


class RangeTests(unittest.TestCase):
    def open_source(self, block=8, limit=64):
        patcher = patch("acquisition.public_hdf5_ranges.requests.Session")
        session = patcher.start().return_value
        self.addCleanup(patcher.stop)
        content = bytes(range(32))
        def get(url, *, params, headers, stream, timeout):
            first, last = map(int, headers["Range"][6:].split("-"))
            response = session.reply
            response.__enter__.return_value = response
            response.status_code = 206
            response.headers = {"ETag": '"v1"', "Content-Range": f"bytes {first}-{last}/32"}
            response.raw = io.BytesIO(content[first:last+1])
            return response
        session.get.side_effect = get
        src = PublicRanges("https://example.test/object", block=block, limit=limit)
        self.addCleanup(src.close)
        return src, session

    def test_seek_across_blocks_and_eof(self):
        src, session = self.open_source()
        src.seek(7)
        self.assertEqual(src.read(5), bytes(range(7, 12)))
        self.assertEqual(src.seek(-2, 2), 30)
        self.assertEqual(src.read(9), b"\x1e\x1f")
        self.assertEqual(src.read(1), b"")
        self.assertEqual(session.get.call_args.kwargs["headers"]["If-Match"], '"v1"')

    def test_enforces_budget_before_request(self):
        src, session = self.open_source(limit=8)
        with self.assertRaisesRegex(ValueError, "budget"):
            src.read(8)
        self.assertEqual(session.get.call_count, 1)

    def test_server_must_respect_range_and_object_version(self):
        for status, tag in ((200, '"v1"'), (206, '"v2"')):
            with self.subTest(status=status, tag=tag):
                src, session = self.open_source()
                response = session.reply
                response.status_code = status
                response.headers = {"ETag": tag, "Content-Range": "bytes 0-7/32"}
                session.get.side_effect = None
                session.get.return_value = response
                with self.assertRaisesRegex(ValueError, "partial HTTP"):
                    src.read(1)
