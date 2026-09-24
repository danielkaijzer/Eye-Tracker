"""Reader for a UVC camera's V4L2 metadata node (Linux, format 'UVCH').

Every UVC camera exposes a second /dev/video node that delivers the raw UVC
payload headers of the frames streamed on its capture node. Those headers carry
the device-clock timestamps needed for cross-camera sync:

- PTS (dwPresentationTime): device clock (STC) at a capture point of the frame.
- SCR (source clock reference): a (STC, SOF) pair, i.e. the device clock
  sampled together with the device's view of the USB frame counter.

The kernel wraps each header in a `struct uvc_meta_buf` (packed):
    __u64 ns;        host CLOCK_MONOTONIC when the packet was processed
    __u16 sof;       host USB frame number at that moment
    __u8  length;    UVC header length (bHeaderLength)
    __u8  flags;     bmHeaderInfo
    __u8  buf[length - 2];   PTS (4 B, if flags & PTS), then SCR (6 B, if SCR)
It only records headers that carry PTS/SCR and skips exact SCR repeats. One
dequeued metadata buffer corresponds to one video frame (same V4L2 sequence
number).

Requires uvcvideo `nodrop=1`. Both rig cams put a fresh SCR in nearly every
packet, so each frame's headers overflow the kernel's fixed 1 KiB metadata
buffer after 46 entries (46 x 22 B). The driver marks the buffer as corrupted;
with nodrop=0 (the Jetson default) it silently requeues every such buffer, so
the node appears to deliver nothing. With nodrop=1 it's delivered holding the
frame's first 46 headers (no V4L2 error flag), which is plenty: PTS is the
same in every header of a frame and 46 SCR samples per frame feed the clock fit.

The node only produces data while its capture node is streaming, and the two
can be opened independently. Talks to V4L2 directly via ioctl + mmap.
"""
import ctypes
import fcntl
import mmap
import os
import select
import struct
from dataclasses import dataclass
from typing import List, Optional

# ---- V4L2 ABI (64-bit) -------------------------------------------------------

V4L2_BUF_TYPE_META_CAPTURE = 13
V4L2_MEMORY_MMAP = 1
V4L2_BUF_FLAG_ERROR = 0x40

# bmHeaderInfo bits
HDR_FID = 0x01
HDR_EOF = 0x02
HDR_PTS = 0x04
HDR_SCR = 0x08
HDR_ERR = 0x40


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _Timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _BufM(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                ("planes", ctypes.c_void_p), ("fd", ctypes.c_int32)]


class _V4l2Buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", _Timeval),
                ("timecode", _Timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m", _BufM),
                ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("request_fd", ctypes.c_int32)]


class _V4l2RequestBuffers(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32)]


def _ioc(direction: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("V") << 8) | nr


_IOW, _IOWR = 1, 3
VIDIOC_REQBUFS = _ioc(_IOWR, 8, ctypes.sizeof(_V4l2RequestBuffers))
VIDIOC_QUERYBUF = _ioc(_IOWR, 9, ctypes.sizeof(_V4l2Buffer))
VIDIOC_QBUF = _ioc(_IOWR, 15, ctypes.sizeof(_V4l2Buffer))
VIDIOC_DQBUF = _ioc(_IOWR, 17, ctypes.sizeof(_V4l2Buffer))
VIDIOC_STREAMON = _ioc(_IOW, 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _ioc(_IOW, 19, ctypes.sizeof(ctypes.c_int))

_META_HDR = struct.Struct("<QHBB")   # ns, sof, length, flags


# ---- parsed records ----------------------------------------------------------

@dataclass
class MetaEntry:
    """One UVC payload header, as recorded by the kernel."""
    host_ns: int                   # host CLOCK_MONOTONIC at packet processing
    host_sof: int                  # host USB frame number at that moment
    flags: int                     # bmHeaderInfo
    pts: Optional[int] = None      # device clock (STC ticks) at capture
    scr_stc: Optional[int] = None  # device clock sampled with...
    scr_sof: Optional[int] = None  # ...this device-side USB SOF (11-bit)


@dataclass
class MetaFrame:
    """All headers for one video frame (one dequeued metadata buffer)."""
    sequence: int                  # V4L2 sequence (matches the video buffer)
    buf_ts_ns: int                 # V4L2 buffer timestamp (host), ns
    buf_flags: int                 # V4L2 buffer flags (ERROR = 0x40)
    entries: List[MetaEntry]

    @property
    def pts(self) -> Optional[int]:
        return next((e.pts for e in self.entries if e.pts is not None), None)

    @property
    def errored(self) -> bool:
        return bool(self.buf_flags & V4L2_BUF_FLAG_ERROR)


def parse_meta_buffer(data: bytes) -> List[MetaEntry]:
    """Split one metadata buffer into its uvc_meta_buf entries."""
    entries = []
    off = 0
    while off + _META_HDR.size <= len(data):
        ns, sof, length, flags = _META_HDR.unpack_from(data, off)
        if length < 2:
            break
        body = data[off + _META_HDR.size: off + _META_HDR.size + length - 2]
        off += _META_HDR.size + length - 2
        e = MetaEntry(host_ns=ns, host_sof=sof, flags=flags)
        pos = 0
        if flags & HDR_PTS and len(body) >= pos + 4:
            e.pts = struct.unpack_from("<I", body, pos)[0]
            pos += 4
        if flags & HDR_SCR and len(body) >= pos + 6:
            e.scr_stc, sof_raw = struct.unpack_from("<IH", body, pos)
            e.scr_sof = sof_raw & 0x7FF
        entries.append(e)
    return entries


# ---- reader ------------------------------------------------------------------

class UvcMetaReader:
    """Streams one UVC metadata node. open() -> read() per frame -> close()."""

    def __init__(self, device: str, num_buffers: int = 8):
        self.device = device
        self.num_buffers = num_buffers
        self._fd: Optional[int] = None
        self._maps: List[mmap.mmap] = []

    def open(self) -> None:
        self._fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        req = _V4l2RequestBuffers(count=self.num_buffers,
                                  type=V4L2_BUF_TYPE_META_CAPTURE,
                                  memory=V4L2_MEMORY_MMAP)
        fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req)
        for i in range(req.count):
            buf = _V4l2Buffer(index=i, type=V4L2_BUF_TYPE_META_CAPTURE,
                              memory=V4L2_MEMORY_MMAP)
            fcntl.ioctl(self._fd, VIDIOC_QUERYBUF, buf)
            self._maps.append(mmap.mmap(self._fd, buf.length, mmap.MAP_SHARED,
                                        mmap.PROT_READ, offset=buf.m.offset))
            fcntl.ioctl(self._fd, VIDIOC_QBUF, buf)
        fcntl.ioctl(self._fd, VIDIOC_STREAMON,
                    ctypes.c_int(V4L2_BUF_TYPE_META_CAPTURE))

    def read(self, timeout_s: float = 1.0) -> Optional[MetaFrame]:
        """Next frame's headers, or None on timeout."""
        r, _, _ = select.select([self._fd], [], [], timeout_s)
        if not r:
            return None
        buf = _V4l2Buffer(type=V4L2_BUF_TYPE_META_CAPTURE, memory=V4L2_MEMORY_MMAP)
        fcntl.ioctl(self._fd, VIDIOC_DQBUF, buf)
        data = self._maps[buf.index][:buf.bytesused]
        frame = MetaFrame(
            sequence=buf.sequence,
            buf_ts_ns=buf.timestamp.tv_sec * 1_000_000_000 + buf.timestamp.tv_usec * 1000,
            buf_flags=buf.flags,
            entries=parse_meta_buffer(data))
        fcntl.ioctl(self._fd, VIDIOC_QBUF, buf)
        return frame

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.ioctl(self._fd, VIDIOC_STREAMOFF,
                        ctypes.c_int(V4L2_BUF_TYPE_META_CAPTURE))
        except OSError:
            pass
        for m in self._maps:
            m.close()
        self._maps = []
        os.close(self._fd)
        self._fd = None
