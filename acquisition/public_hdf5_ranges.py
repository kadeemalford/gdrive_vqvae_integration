"""Seekable, ETag-pinned public HDF5 input, with a hard transfer budget."""
import io
import re
from functools import lru_cache

import requests


class PublicRanges(io.RawIOBase):
    def __init__(self, url, *, block=65536, limit=96 << 20, etag=None):
        self.url, self.block, self.limit = url, block, limit
        self.pos = self.transferred = 0
        self.session = requests.Session()
        self.etag, self.size = etag, None
        self._get(0, 0)
        self.cached = lru_cache(maxsize=128)(self._block)

    def _get(self, first, last):
        count = last - first + 1
        if self.transferred + count > self.limit:
            raise ValueError("Range transfer budget exceeded")
        headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        with self.session.get(self.url, params={"range_id": f"{first}-{last}"},
                              headers=headers, stream=True, timeout=(10, 45)) as response:
            response.raise_for_status()
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
            tag = response.headers.get("ETag")
            if (response.status_code != 206 or not match or not tag
                    or (self.etag and self.etag != tag)
                    or tuple(map(int, match.groups()[:2])) != (first, last)):
                raise ValueError("Unpinned or invalid partial HTTP response")
            size = int(match[3])
            if self.size is not None and size != self.size:
                raise ValueError("Remote size changed")
            data = response.raw.read(count + 1)
            if len(data) != count:
                raise ValueError("Incomplete or oversized range")
            self.etag, self.size = tag, size
            self.transferred += count
            return data

    def _block(self, index):
        return self._get(index * self.block, min((index + 1) * self.block, self.size) - 1)

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = (0, self.pos, self.size)[whence] + offset
        if pos < 0:
            raise ValueError("Negative seek")
        self.pos = pos
        return pos

    def readinto(self, buf):
        count = min(len(buf), max(self.size - self.pos, 0))
        if count > self.block:
            buf[:count] = self._get(self.pos, self.pos + count - 1)
            self.pos += count
            return count
        done = 0
        while done < count:
            index, offset = divmod(self.pos, self.block)
            data = self.cached(index)
            n = min(len(data) - offset, count - done)
            buf[done:done+n] = data[offset:offset+n]
            self.pos += n
            done += n
        return count

    def close(self):
        if hasattr(self, "cached"):
            self.cached.cache_clear()
        self.session.close()
        super().close()
