"""Devices: buffers between guards, loading, dispatch, the end of a stream.

A concrete device subclasses Device and implements the device-implementer
protocol of (alloc_bytes, write_bytes, read_bytes, copy_bytes,
fill_bytes, sync, free, load, run, and its guard); everything else is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thin.runtime.errors import DeviceFault, LaunchError

__all__ = [
    "ALIGN",
    "GUARD",
    "GUARD_FILL",
    "POISON",
    "Buffer",
    "Device",
    "Program",
    "dispatch",
    "program",
]

GUARD = 4096
"""The default guard: bytes of GUARD_FILL each side of a buffer."""

GUARD_FILL = 0x5B
"""What a guard holds, so a write past a tensor changes it."""

POISON = 0xFF
"""What an output holds before a launch: NaN in bf16 and f32, never zero."""

ALIGN = 16
"""A slice's offset alignment."""


def _dtype(dtype):
    from thin import formats

    return formats.DTYPES[getattr(dtype, "name", dtype)]


def _width(dtype) -> int:
    return _dtype(dtype).width


@dataclass(eq=False)
class Allocation:
    """An allocation: its handle, its address and the launches given it since the last end."""

    handle: object
    address: int
    nbytes: int
    launches: list = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class Buffer:
    """A device buffer between two guards.

    Its user-visible fields are dtype, nbytes and ptr; the rest is the
    device's own record.
    """

    dtype: object
    """Its dtype."""
    nbytes: int
    """Its bytes."""
    ptr: int
    """The device address of its first byte."""
    _device: Device
    _allocation: Allocation
    _offset: int
    """Its offset past the guard within the allocation: 0 for a tensor."""
    _whole: int
    """The allocation's bytes between its guards."""


@dataclass(frozen=True, eq=False)
class Program:
    """Loaded code: device, key, block, kernarg bytes and the device's own loaded code."""

    device: Device
    key: str
    block: tuple
    kernarg_bytes: int
    code: object


class Device:
    """The generic device; a concrete one overrides the hooks below."""

    guard = GUARD
    """Guard bytes each side of a buffer."""

    def __init__(self, rank=None):
        self.rank = rank
        """None alone, else its rank in a group."""
        self.closed = False
        self._pending: list = []
        self._allocations: list[Allocation] = []

    # The device-implementer protocol: a concrete device
    # implements these hooks; the generic class calls nothing else.

    def alloc_bytes(self, n: int, fill: int, meta: dict) -> tuple[object, int]:
        """Allocates n bytes, each ``fill``; returns (allocation, device address)."""
        raise NotImplementedError(f"{type(self).__name__}.alloc_bytes")

    def write_bytes(self, allocation, offset: int, data: bytes) -> None:
        """Writes data at offset."""
        raise NotImplementedError(f"{type(self).__name__}.write_bytes")

    def read_bytes(self, allocation, offset: int, n: int) -> bytes:
        """Reads n bytes at offset."""
        raise NotImplementedError(f"{type(self).__name__}.read_bytes")

    def copy_bytes(self, dst, dst_offset: int, src, src_offset: int, n: int) -> None:
        """Copies n bytes between allocations, in stream order."""
        raise NotImplementedError(f"{type(self).__name__}.copy_bytes")

    def fill_bytes(self, allocation, offset: int, n: int, byte: int) -> None:
        """Sets n bytes to byte, in stream order."""
        raise NotImplementedError(f"{type(self).__name__}.fill_bytes")

    def sync(self, what: str) -> None:
        """Ends what is enqueued; ``what`` names the launches it ends."""
        raise NotImplementedError(f"{type(self).__name__}.sync")

    def free(self) -> None:
        """Releases everything (called once, by close)."""
        raise NotImplementedError(f"{type(self).__name__}.free")

    def load(self, image: bytes, code_object, asm, block, key: str) -> object:
        """Loads a code object; returns the device's code. May refuse (LaunchError)."""
        raise NotImplementedError(f"{type(self).__name__}.load")

    def run(self, code, key: str, grid, block, kernarg: bytes, what: str, cooperative: bool):
        """Enqueues one launch of loaded code at grid with the kernarg bytes."""
        raise NotImplementedError(f"{type(self).__name__}.run")

    # The device's calls.

    def _open(self) -> None:
        if self.closed:
            raise LaunchError("a closed device: every call refused")

    def _mine(self, t) -> Buffer:
        if not isinstance(t, Buffer) or t._device is not self:
            raise LaunchError(f"a buffer of another device ({getattr(t, 'nbytes', '?')} bytes)")
        return t

    def tensor(self, dtype, data=None, *, nbytes=None, fill=0, meta=None) -> Buffer:
        """A buffer of data, or of nbytes filled."""
        self._open()
        if (data is None) == (nbytes is None):
            raise LaunchError("a tensor of its data or of nbytes, one of them")
        if data is not None:
            self.wait()
            data = bytes(data)
            nbytes = len(data)
        name = getattr(dtype, "name", dtype)
        if not (type(nbytes) is int and nbytes >= 0 and nbytes * 8 % _width(name) == 0):
            raise LaunchError(f"a {name} tensor of {nbytes!r} bytes: whole elements")
        g = self.guard
        handle, base = self.alloc_bytes(g + nbytes + g, GUARD_FILL, dict(meta or {}))
        a = Allocation(handle, base, g + nbytes + g)
        self._allocations.append(a)
        if data is not None:
            self.write_bytes(handle, g, data)
        else:
            self.fill_bytes(handle, g, nbytes, fill)
        return Buffer(_dtype(name), nbytes, base + g, self, a, 0, nbytes)

    def slice(self, t, offset, nbytes) -> Buffer:
        """Bytes of t at offset: whole elements, at a multiple of 16."""
        self._open()
        t = self._mine(t)
        w = _width(t.dtype)
        ok = (
            type(offset) is int
            and type(nbytes) is int
            and offset >= 0
            and nbytes >= 0
            and offset + nbytes <= t.nbytes
            and offset % ALIGN == 0
            and nbytes * 8 % w == 0
        )
        if not ok:
            raise LaunchError(
                f"a slice of {nbytes!r} bytes at {offset!r} of a {t.dtype} buffer of {t.nbytes}: "
                "inside it, "
                "whole elements, at a multiple of 16"
            )
        return Buffer(
            t.dtype, nbytes, t.ptr + offset, self, t._allocation, t._offset + offset, t._whole
        )

    def read(self, t) -> bytes:
        """Ends what is enqueued, then t's bytes."""
        self._open()
        t = self._mine(t)
        self.wait()
        return self.read_bytes(t._allocation.handle, self.guard + t._offset, t.nbytes)

    def copy(self, dst, src) -> None:
        """Enqueues a copy of equal sizes."""
        self._open()
        dst, src = self._mine(dst), self._mine(src)
        if dst.nbytes != src.nbytes:
            raise LaunchError(f"a copy of {src.nbytes} bytes into {dst.nbytes}: equal sizes")
        g = self.guard
        dst_at, src_at = g + dst._offset, g + src._offset
        self.copy_bytes(dst._allocation.handle, dst_at, src._allocation.handle, src_at, src.nbytes)

    def fill(self, t, byte) -> None:
        """Enqueues a fill of one byte value."""
        self._open()
        t = self._mine(t)
        if not (type(byte) is int and 0 <= byte <= 255):
            raise LaunchError(f"a fill of {byte!r}: a byte")
        self.fill_bytes(t._allocation.handle, self.guard + t._offset, t.nbytes, byte)

    def wait(self) -> None:
        """Ends what is enqueued, then checks the guards of its launches."""
        self._open()
        pending, self._pending = self._pending, []
        names = list(dict.fromkeys(p[0] for p in pending))
        self.sync("; ".join(names) if names else "the stream")
        g = self.guard
        if not g:
            return
        seen: set = set()
        for _, buffers in pending:
            for name, b in buffers.items():
                a = b._allocation
                if id(a) in seen:
                    continue
                seen.add(id(a))
                launches = list(dict.fromkeys(a.launches))
                a.launches.clear()
                head = self.read_bytes(a.handle, 0, g)
                tail = self.read_bytes(a.handle, a.nbytes - g, g)
                if head != bytes([GUARD_FILL]) * g or tail != bytes([GUARD_FILL]) * g:
                    raise DeviceFault(
                        f"{'; '.join(launches)}: a guard of {name} written", f"bounds:{name}"
                    )

    def close(self) -> None:
        """Waits, then (whatever the wait said) marks the device closed and frees its memory."""
        self._open()
        try:
            self.wait()
        finally:
            self.closed = True
            self._allocations.clear()
            self.free()


def program(device, co, asm, key=None) -> Program:
    """Loads a code object on a device."""
    from thin import formats

    device._open()
    try:
        obj = formats.read(co)
    except ValueError as e:
        raise LaunchError(f"{key or 'a kernel'}: the code object: {e}") from None
    key = key or obj.symbol
    kernels = obj.metadata.get("amdhsa.kernels") or [{}]
    block = (kernels[0].get(".max_flat_workgroup_size", 1), 1)
    code = device.load(bytes(co), obj, asm, block, key)
    return Program(device, key, block, obj.kernarg, code)


def dispatch(program, grid, kernarg, buffers, what=None, cooperative=False) -> None:
    """Enqueues one run of a Program at grid on its device."""
    device = program.device
    device._open()
    if len(kernarg) != program.kernarg_bytes:
        raise LaunchError(
            f"{program.key}: a kernarg of {len(kernarg)} bytes, the code object's "
            f"{program.kernarg_bytes}"
        )
    buffers = {n: device._mine(b) for n, b in buffers.items()}
    what = what or program.key
    grid = tuple(grid)
    device.run(program.code, program.key, grid, program.block, bytes(kernarg), what, cooperative)
    for b in buffers.values():
        b._allocation.launches.append(what)
    device._pending.append((what, buffers))
