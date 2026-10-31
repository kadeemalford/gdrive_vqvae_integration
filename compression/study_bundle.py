"""Length-framed, lossless transport bundle for already coded neural windows."""
import io
import struct
import zipfile

HEADER = struct.Struct('<4sIQ')
LENGTH = struct.Struct('<I')
MAGIC = b'FSB1'


def metadata_archive(assets):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(assets.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    return stream.getvalue()


def write_bundle(path, assets, frame_paths):
    """Frame order must be declared by the metadata manifest, not external filenames."""
    metadata = metadata_archive(assets)
    with open(path, 'xb') as stream:
        stream.write(HEADER.pack(MAGIC, len(frame_paths), len(metadata)))
        stream.write(metadata)
        for frame in frame_paths:
            data = frame.read_bytes()
            if len(data) >= 2**32:
                raise ValueError('Frame exceeds transport size limit')
            stream.write(LENGTH.pack(len(data)))
            stream.write(data)
    return {'metadata_archive_bytes': len(metadata),
            'transport_header_and_lengths_bytes': HEADER.size + LENGTH.size * len(frame_paths),
            'frames': len(frame_paths)}


def read_bundle(path):
    """Yield assets first, then frame bytes. No archive paths are extracted to disk."""
    with open(path, 'rb') as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError('Truncated bundle header')
        magic, count, metadata_size = HEADER.unpack(raw)
        if magic != MAGIC or count > 10_000_000 or metadata_size > 64 * 2**20:
            raise ValueError('Invalid transport header')
        metadata = stream.read(metadata_size)
        if len(metadata) != metadata_size:
            raise ValueError('Truncated metadata')
        with zipfile.ZipFile(io.BytesIO(metadata)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 128 * 2**20:
                raise ValueError('Metadata exceeds decoded size limit')
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError('Duplicate metadata asset')
            yield {name: archive.read(name) for name in names}
        for _ in range(count):
            length = stream.read(LENGTH.size)
            if len(length) != LENGTH.size:
                raise ValueError('Missing frame length')
            size = LENGTH.unpack(length)[0]
            if size > 64 * 2**20:
                raise ValueError('Frame exceeds read size limit')
            data = stream.read(size)
            if len(data) != size:
                raise ValueError('Truncated transport frame')
            yield data
        if stream.read(1):
            raise ValueError('Trailing transport bytes')
