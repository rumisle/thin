"""Memory, buffers, tags and the LDS as Python sees them."""

from __future__ import annotations

import numpy as np

from thin import _native

__all__ = [
    "LDS",
    "PENDING",
    "POISON",
    "POISON64",
    "TAG_SHIFT",
    "Buffer",
    "Memory",
    "tag_buf",
    "tag_make",
    "tag_off",
]

TAG_SHIFT = 40
"""The shift of a memory tag's buffer field."""
POISON = 0x7FC17FC1
"""Every register's value at dispatch."""
POISON64 = POISON | POISON << 32
"""POISON in both dwords."""
PENDING = -2
"""The LDS epoch of a DMA not landed."""


class Buffer:
    """One buffer of a run: a view of the memory's buffer at an index."""

    def __init__(self, mem: Memory, index: int, meta: dict):
        self.mem, self.index, self.meta = mem, index, meta

    def _info(self):
        return self.mem._native.info(self.index)

    @property
    def name(self) -> str:
        """Its name (the newest buffer of a name wins a lookup by name)."""
        return self._info()[0]

    @property
    def base(self) -> int:
        """Its base address."""
        return self._info()[1]

    @property
    def size(self) -> int:
        """Its size in bytes."""
        return self._info()[2]

    @property
    def data(self) -> bytes:
        """A copy of its bytes (``write`` changes them)."""
        return self.mem._native.read(self.index, 0, self.size)

    def read(self, offset: int, n: int) -> bytes:
        """N bytes at an offset."""
        return self.mem._native.read(self.index, offset, n)

    def write(self, offset: int, data) -> None:
        """Writes bytes at an offset (its version counts the write)."""
        self.mem._native.write(self.index, offset, bytes(data))

    @property
    def tags(self) -> np.ndarray:
        """A copy of every byte's tag."""
        return self.mem._native.tags(self.index)

    def get_tags(self, offsets) -> np.ndarray:
        """The tags of these offsets."""
        return self.tags[np.asarray(offsets, dtype=np.int64)]

    def set_tags(self, offsets, tags) -> None:
        """Replaces the tags of these offsets."""
        o = np.ascontiguousarray(offsets, dtype=np.int64)
        t = np.ascontiguousarray(tags, dtype=np.int64)
        self.mem._native.set_tags(self.index, o, t)

    @property
    def tbits(self) -> int:
        """The tag width in bits."""
        return 64

    @property
    def tcount(self) -> int:
        """The bytes carrying a nonzero tag."""
        return int(np.count_nonzero(self.tags))


class _Flight:
    def __init__(self, mem):
        self.mem = mem

    @property
    def events(self) -> list[str]:
        """The stale reads served."""
        return self.mem._native.events()

    def end_launch(self, rank: int) -> None:
        """Ends a rank's launch in the flight model."""
        self.mem._native.end_launch(rank)


class Memory:
    """The buffers of a run, shared by every rank."""

    def __init__(self, straddle=1 << 32):
        """Empty memory; buffers straddle 4 GiB boundaries."""
        self._native = _native.EmuMemory(straddle)
        self._meta: list[dict] = []
        self.flight = _Flight(self)

    def alloc(self, name, size, tagged=True, fill=0, **meta) -> Buffer:
        """Allocates a buffer: tagged with its own memory tags, or untagged."""
        flags = (bool(meta.get("ipc")), bool(meta.get("cached")), bool(meta.get("uncached")))
        index = self._native.alloc(name, size, tagged, fill, *flags)
        self._meta.append(meta)
        return Buffer(self, index, meta)

    def release(self, b: Buffer) -> None:
        """Frees a buffer and unmaps its range."""
        self._native.release(b.index)

    @property
    def bufs(self) -> list[Buffer]:
        """The buffers in allocation (address) order."""
        return [Buffer(self, i, self._meta[i]) for i in range(self._native.count())]

    def by_name(self, name: str) -> Buffer:
        """The buffer of a name."""
        return Buffer(self, self.index(name), {})

    def index(self, name: str) -> int:
        """The index of a buffer by name (KeyError for none)."""
        found = self._native.by_name(name)
        return {None: lambda: _missing(name)}.get(found, lambda: found)()

    @property
    def version(self) -> list[int]:
        """Each buffer's store count."""
        return self._native.versions()

    def resolve(self, addr: int):
        """(buffer index, offset) of an address, or EmuError."""
        return self._native.resolve(addr)

    def resolve_many(self, addrs):
        """(index, offset) of each address; index -1 for none."""
        return self._native.resolve_many([int(a) for a in addrs])

    def track_traffic(self) -> None:
        """Starts the traffic counts."""
        self._native.track_traffic()

    @property
    def scope_model(self) -> bool:
        """Whether the scope model is on (in a cluster)."""
        return self._native.scope_model()

    @property
    def ipc(self) -> bool:
        """True once a buffer is ipc."""
        return self._native.ipc()


def _missing(name):
    raise KeyError(name)


class LDS:
    """A workgroup's LDS bytes: its size; the race state is the machine's."""

    def __init__(self, size):
        self.size = size
        self.data = bytearray(size)


def tag_make(buf, off):
    """The memory tag of a buffer byte."""
    return _native.emu_tag_make(buf, off)


def tag_buf(tag):
    """A memory tag's buffer."""
    return _native.emu_tag_buf(tag)


def tag_off(tag):
    """A memory tag's offset."""
    return _native.emu_tag_off(tag)
