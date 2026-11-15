"""HIP serving: load a code object on the current GPU, launch it on a stream.

Serving has no guards, no sync and keeps nothing. This module loads only
itself, ctypes and the error classes: no numpy, no torch (``tensor``
imports torch inside the call).
"""

from __future__ import annotations

import contextlib
import ctypes
import weakref

from thin.runtime.errors import LaunchError

__all__ = [
    "alloc",
    "close_ipc",
    "enable_peers",
    "enqueue",
    "found",
    "free",
    "gpus",
    "here",
    "ipc_handle",
    "lib",
    "module",
    "open_ipc",
    "runtime",
    "tensor",
]

_NAME = "libamdhip64.so"
_PROPS = 8192
"""Bytes read for a device's properties: more than the record."""
_ARCH_AT = 1160
"""The arch name's offset in the properties record."""
_UNCACHED = 0x3
"""hipDeviceMallocUncached."""
_PEER_ENABLED = 704
"""hipErrorPeerAccessAlreadyEnabled."""
_IPC_LAZY = 1
"""hipIpcMemLazyEnablePeerAccess."""
_EXTRA = (1, 2, 3)
"""HIP_LAUNCH_PARAM_BUFFER_POINTER, _BUFFER_SIZE, _END."""


def runtime() -> str:
    """The HIP library already loaded (a caller's, as torch's), else the linker's name."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                at = line.find("/")
                if at >= 0 and f"/{_NAME}" in line:
                    return line[at:].strip()
    except OSError:
        pass
    return _NAME


_LIB: list = []

_vp, _u32, _i32, _sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_size_t


def lib():
    """The runtime library, its launch signatures declared once a process."""
    if not _LIB:
        h = ctypes.CDLL(runtime())
        sigs = {
            "hipModuleLoadData": [ctypes.POINTER(_vp), ctypes.c_char_p],
            "hipModuleGetFunction": [ctypes.POINTER(_vp), _vp, ctypes.c_char_p],
            "hipModuleLaunchKernel": [_vp, _u32, _u32, _u32, _u32, _u32, _u32, _u32, _vp, _vp, _vp],
            "hipModuleLaunchCooperativeKernel": [
                _vp,
                _u32,
                _u32,
                _u32,
                _u32,
                _u32,
                _u32,
                _u32,
                _vp,
                _vp,
            ],
            "hipGetErrorString": [_i32],
            "hipExtMallocWithFlags": [ctypes.POINTER(_vp), _sz, _u32],
            "hipMemset": [_vp, _i32, _sz],
            "hipDeviceSynchronize": [],
            "hipFree": [_vp],
            "hipGetDevice": [ctypes.POINTER(_i32)],
            "hipSetDevice": [_i32],
            "hipDeviceEnablePeerAccess": [_i32, _u32],
            "hipGetLastError": [],
            "hipIpcGetMemHandle": [ctypes.c_char_p, _vp],
            "hipIpcOpenMemHandle": [ctypes.POINTER(_vp), _IpcHandle, _u32],
            "hipIpcCloseMemHandle": [_vp],
        }
        for name, argtypes in sigs.items():
            fn = getattr(h, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_char_p if name == "hipGetErrorString" else _i32
        _LIB.append(h)
    return _LIB[0]


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _error(code: int) -> str:
    try:
        text = lib().hipGetErrorString(code)
    except OSError:
        return f"HIP error {code}"
    return text.decode(errors="replace") if text else f"HIP error {code}"


def _clear() -> None:
    """Clears HIP's sticky last error once it is raised as a LaunchError.

    The tensor library's next check then does not report it again as its own.
    """
    with contextlib.suppress(OSError, AttributeError):
        lib().hipGetLastError()


def _check(code: int, what: str) -> None:
    if code:
        text = _error(code)
        _clear()
        raise LaunchError(f"{what}: {text} ({code})")


def gpus() -> tuple[tuple[str, ...], int]:
    """(each GPU's arch, the runtime version) by a handle of its own; ``((), 0)`` without HIP."""
    try:
        h = ctypes.CDLL(runtime())
    except OSError:
        return ((), 0)
    count, version = ctypes.c_int(0), ctypes.c_int(0)
    try:
        if h.hipGetDeviceCount(ctypes.byref(count)) or h.hipRuntimeGetVersion(
            ctypes.byref(version)
        ):
            return ((), 0)
    except AttributeError:
        return ((), 0)
    # The arch name's offset is that of the properties record (ROCm 6
    # and 7): another record version is not read. A device whose record
    # cannot be read keeps its index, as "unknown".
    props = getattr(h, "hipGetDevicePropertiesR0600", None)
    archs = []
    for i in range(count.value):
        buf = ctypes.create_string_buffer(_PROPS)
        if props is None or props(buf, i):
            archs.append("unknown")
            continue
        name = buf.raw[_ARCH_AT:].split(b"\0", 1)[0].decode(errors="replace")
        archs.append(name.split(":", 1)[0])
    return (tuple(archs), version.value)


def found() -> str:
    """``"<arch> x<count>, …, HIP <version>"``, ``"no HIP"`` or ``"no GPU, HIP <version>"``."""
    archs, version = gpus()
    if not version:
        return "no HIP"
    if not archs:
        return f"no GPU, HIP {version}"
    parts = [f"{a} x{archs.count(a)}" for a in sorted(set(archs))]
    return ", ".join([*parts, f"HIP {version}"])


def here(arch: str | None = None) -> bool:
    """Whether HIP's first GPU (where a test runs) is that arch; by default the default target."""
    if arch is None:
        from thin.target import target

        arch = target().name
    archs, _ = gpus()
    return bool(archs) and archs[0] == arch


def module(co: bytes, symbol: str, key: str):
    """(module, function): the code object loaded on the current GPU, its function by symbol."""
    h = lib()
    mod, fn = _vp(), _vp()
    _check(h.hipModuleLoadData(ctypes.byref(mod), bytes(co)), f"{key}: load")
    _check(
        h.hipModuleGetFunction(ctypes.byref(fn), mod, symbol.encode()), f"{key}: function {symbol}"
    )
    return mod, fn


def enqueue(fn, grid, block, kernarg: bytes, stream, what: str, cooperative: bool = False) -> None:
    """Launches fn on the stream; the kernarg by the extra array, or as the one argument."""
    h = lib()
    gx, gy, gz = grid
    bx, by = block[0], block[1]
    buf = ctypes.create_string_buffer(bytes(kernarg), max(len(kernarg), 1))
    stream = _vp(stream or 0)
    if cooperative:
        params = (_vp * 1)(ctypes.cast(buf, _vp)) if kernarg else None
        code = h.hipModuleLaunchCooperativeKernel(fn, gx, gy, gz, bx, by, 1, 0, stream, params)
        _check(code, f"{what}: cooperative launch")
        return
    size = _sz(len(kernarg))
    extra = (_vp * 5)(
        _vp(_EXTRA[0]),
        _vp(ctypes.addressof(buf)),
        _vp(_EXTRA[1]),
        _vp(ctypes.addressof(size)),
        _vp(_EXTRA[2]),
    )
    _check(
        h.hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, 1, 0, stream, None, extra),
        f"{what}: launch",
    )


def _call(code: int, call: str) -> None:
    if code:
        _clear()
        raise LaunchError(f"{call}: HIP error {code}")


def alloc(nbytes: int) -> int:
    """A zeroed uncached allocation of max(nbytes, 256) bytes; the device synchronized."""
    h = lib()
    ptr, n = _vp(), max(nbytes, 256)
    _call(h.hipExtMallocWithFlags(ctypes.byref(ptr), n, _UNCACHED), "hipExtMallocWithFlags")
    _call(h.hipMemset(ptr, 0, n), "hipMemset")
    _call(h.hipDeviceSynchronize(), "hipDeviceSynchronize")
    return ptr.value


def free(ptr: int) -> None:
    """Frees an allocation of ``alloc``."""
    _call(lib().hipFree(_vp(ptr)), "hipFree")


class _Owner:
    """The array-interface object a tensor of ``alloc`` keeps alive."""

    def __init__(self, ptr: int, nbytes: int, device: int):
        self.ptr, self.device = ptr, device
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
            "strides": None,
        }
        weakref.finalize(self, _free_on, ptr, device)


def _free_on(ptr: int, device: int) -> None:
    with contextlib.suppress(LaunchError):
        free(ptr)


def tensor(nbytes: int, device: int):
    """A zeroed u8 torch tensor over an uncached allocation, freed when the tensor is."""
    import torch

    with torch.cuda.device(device):
        ptr = alloc(nbytes)
        owner = _Owner(ptr, nbytes, device)
        t = torch.as_tensor(owner, device=f"cuda:{device}")
    t._thin_owner = owner
    return t


def enable_peers(devices) -> None:
    """Peer access from every GPU of the set to every other (each ordered pair)."""
    h = lib()
    before = ctypes.c_int(0)
    _call(h.hipGetDevice(ctypes.byref(before)), "hipGetDevice")
    try:
        for a in devices:
            _call(h.hipSetDevice(a), "hipSetDevice")
            for b in devices:
                if a == b:
                    continue
                code = h.hipDeviceEnablePeerAccess(b, 0)
                if code == _PEER_ENABLED:
                    h.hipGetLastError()
                elif code:
                    _check(code, f"hipDeviceEnablePeerAccess({a} -> {b})")
    finally:
        h.hipSetDevice(before.value)


def ipc_handle(ptr: int) -> bytes:
    """The 64-byte IPC handle of an allocation of ``alloc``."""
    out = ctypes.create_string_buffer(64)
    _check(lib().hipIpcGetMemHandle(out, _vp(ptr)), "hipIpcGetMemHandle")
    return out.raw


def open_ipc(handle: bytes) -> int:
    """A peer process's allocation, by its handle: its device address here (lazy peer access)."""
    h, ptr = _IpcHandle(), _vp()
    h.reserved = bytes(handle)
    _check(lib().hipIpcOpenMemHandle(ctypes.byref(ptr), h, _IPC_LAZY), "hipIpcOpenMemHandle")
    return ptr.value


def close_ipc(ptr: int) -> None:
    """Closes an opened handle."""
    _check(lib().hipIpcCloseMemHandle(_vp(ptr)), "hipIpcCloseMemHandle")
