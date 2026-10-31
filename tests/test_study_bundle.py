from pathlib import Path
import tempfile
import unittest

from compression.study_bundle import write_bundle, read_bundle


class StudyBundleTests(unittest.TestCase):
    def test_assets_and_order_preserved_without_source_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = []
            for i in range(3):
                p = root / f'{i}.frame';p.write_bytes(bytes([i]) * (i + 10));frames.append(p)
            assets = {'geometry.json': b'{"x":1}', 'sequence.json': b'[0,1,2]'}
            path = root / 'codec.fsb'
            write_bundle(path, assets, frames)
            contents = list(read_bundle(path))
            self.assertEqual(contents[0], assets)
            self.assertEqual(contents[1:], [p.read_bytes() for p in frames])
            with self.assertRaises(FileExistsError):
                write_bundle(path, assets, frames)

    def test_truncation_and_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);frame = root / 'x';frame.write_bytes(b'abc')
            path = root / 'ok';write_bundle(path, {'x': b'y'}, [frame])
            data = path.read_bytes()
            for value in (data[:-1], data + b'junk', b'bad'):
                damaged = root / 'bad';damaged.write_bytes(value)
                with self.assertRaises(ValueError):
                    list(read_bundle(damaged))


if __name__ == '__main__':
    unittest.main()
