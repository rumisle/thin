"""The device layer and the backends a race runs on.

The HIP runtime is the one the tensor library loaded: found mapped
into the process, else loaded by its default name, bound once. ``program``
and ``launch`` load and enqueue a code object's kernel (graph-capturable,
cooperative on request). ``uncached`` allocates visible host memory;
``peers`` manages peer access.

A race reaches the device through a **backend**: allocation, streams, graph
capture and timed replay, and the own kernels' launches (the copy compare,
the invalidate, the empty kernel). ``GpuBackend`` is the device itself;
``HostBackend`` keeps every allocation in host memory, runs the copy compare
by its specified result and times repeated calls with the host clock.
This lets tests run without a GPU. ``using(backend)`` selects
one for the races in its block; the default is the GPU.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import time
import weakref

__all__ = [
    "GpuBackend",
    "HostBackend",
    "backend",
    "launch",
    "peers",
    "program",
    "runtime",
    "uncached",
    "using",
]

_NAME = "libamdhip64.so"
_UNCACHED = 0x3
"""hipDeviceMallocUncached."""
_PEER_ENABLED = 704
"""hipErrorPeerAccessAlreadyEnabled."""
_EXTRA = (1, 2, 3)
"""HIP_LAUNCH_PARAM_BUFFER_POINTER, _BUFFER_SIZE, _END."""

_vp, _u32, _i32, _sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_size_t
_LIB: list = []
_PROGRAMS: dict = {}


def runtime() -> str:
    """The HIP runtime library mapped into the process (the tensor library's), else its name."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                at = line.find("/")
                if at >= 0 and f"/{_NAME}" in line:
                    return line[at:].strip()
    except OSError:
        pass
    return _NAME


def _lib():
    if not _LIB:
        import torch  # noqa: F401 - the library first, so its runtime is the one found

        h = ctypes.CDLL(runtime())
        sigs = {
            "hipModuleLoadData": [ctypes.POINTER(_vp), ctypes.c_char_p],
            "hipModuleGetFunction": [ctypes.POINTER(_vp), _vp, ctypes.c_char_p],
            "hipModuleLaunchKernel": [_vp, _u32, _u32, _u32, _u32, _u32, _u32, _u32, _vp, _vp, _vp],
            "hipModuleLaunchCooperativeKernel": [_vp, *([_u32] * 7), _vp, _vp],
            "hipGetErrorString": [_i32],
            "hipExtMallocWithFlags": [ctypes.POINTER(_vp), _sz, _u32],
            "hipFree": [_vp],
            "hipGetDevice": [ctypes.POINTER(_i32)],
            "hipSetDevice": [_i32],
            "hipDeviceEnablePeerAccess": [_i32, _u32],
            "hipGetLastError": [],
        }
        for name, argtypes in sigs.items():
            fn = getattr(h, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_char_p if name == "hipGetErrorString" else _i32
        _LIB.append(h)
    return _LIB[0]


def hip_version() -> int:  # pragma: no cover - hardware
    """The HIP runtime's version (hipRuntimeGetVersion) the tensor library loaded; 0 without."""
    try:
        h = _lib()
    except (OSError, ImportError):
        return 0
    v = ctypes.c_int(0)
    fn = h.hipRuntimeGetVersion
    fn.argtypes, fn.restype = [ctypes.POINTER(_i32)], _i32
    return v.value if fn(ctypes.byref(v)) == 0 else 0


def _error(code: int) -> str:
    text = _lib().hipGetErrorString(code)
    return text.decode(errors="replace") if text else f"HIP error {code}"


def _check(code: int, what: str) -> None:
    """Raises a HIP error once: the runtime's sticky last error is cleared first."""
    if code:
        text = _error(code)
        _lib().hipGetLastError()
        raise RuntimeError(f"{what}: {text} ({code})")


class Program:
    """A kernel loaded on a device."""

    __slots__ = ("block", "fn", "module", "symbol")

    def __init__(self, module, fn, symbol, block):
        self.module, self.fn, self.symbol, self.block = module, fn, symbol, block


def program(co: bytes, symbol: str, block=(256, 1)) -> Program:  # pragma: no cover - hardware
    """Kernel `symbol` of `co` loaded on the current device.

    The module is loaded once per process per (device, sha256(co), symbol);
    each block gets its own Program over it.
    """
    h = _lib()
    dev = ctypes.c_int(0)
    _check(h.hipGetDevice(ctypes.byref(dev)), f"{symbol}: device")
    loaded = (dev.value, hashlib.sha256(co).hexdigest(), symbol)
    key = (*loaded, tuple(block))
    if key not in _PROGRAMS:
        same = next((p for k, p in _PROGRAMS.items() if k[:3] == loaded), None)
        if same is not None:
            mod, fn = same.module, same.fn
        else:
            mod, fn = _vp(), _vp()
            _check(h.hipModuleLoadData(ctypes.byref(mod), bytes(co)), f"{symbol}: load")
            _check(
                h.hipModuleGetFunction(ctypes.byref(fn), mod, symbol.encode()),
                f"{symbol}: function",
            )
        _PROGRAMS[key] = Program(mod, fn, symbol, tuple(block))
    return _PROGRAMS[key]


def _stream_handle(stream) -> int:
    return int(getattr(stream, "cuda_stream", stream) or 0)


def launch(p: Program, grid, kernarg: bytes, stream, cooperative=False) -> None:  # pragma: no cover
    """Enqueues one run of p.

    The argument block goes by the extra-parameters buffer, or (cooperative) as
    the parameter array's sole entry.
    """
    h = _lib()
    gx, gy, gz = (*grid, 1, 1)[:3]
    bx, by = p.block[0], p.block[1]
    buf = ctypes.create_string_buffer(bytes(kernarg), max(len(kernarg), 1))
    s = _vp(_stream_handle(stream))
    if cooperative:
        params = (_vp * 1)(ctypes.cast(buf, _vp)) if kernarg else None
        code = h.hipModuleLaunchCooperativeKernel(p.fn, gx, gy, gz, bx, by, 1, 0, s, params)
        _check(code, f"{p.symbol}: cooperative launch")
        return
    size = _sz(len(kernarg))
    extra = (_vp * 5)(
        _vp(_EXTRA[0]),
        _vp(ctypes.addressof(buf)),
        _vp(_EXTRA[1]),
        _vp(ctypes.addressof(size)),
        _vp(_EXTRA[2]),
    )
    _check(h.hipModuleLaunchKernel(p.fn, gx, gy, gz, bx, by, 1, 0, s, None, extra), p.symbol)


class _Owner:
    """The array interface a tensor over an uncached allocation keeps alive."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
            "strides": None,
        }
        weakref.finalize(self, _free, ptr)


def _free(ptr: int) -> None:
    with contextlib.suppress(Exception):
        _lib().hipFree(_vp(ptr))


def uncached(nbytes: int):  # pragma: no cover - hardware
    """A u8 tensor of nbytes over a fresh uncached allocation.

    The allocation is at least 256 bytes, its contents undefined, freed with the tensor.
    """
    import torch

    h, ptr = _lib(), _vp()
    _check(
        h.hipExtMallocWithFlags(ctypes.byref(ptr), max(nbytes, 256), _UNCACHED), "uncached alloc"
    )
    owner = _Owner(ptr.value, max(nbytes, 256))
    t = torch.as_tensor(owner, device=f"cuda:{torch.cuda.current_device()}")[:nbytes]
    t._metron_owner = owner
    return t


def peers(world: int) -> None:  # pragma: no cover - hardware
    """Peer access for every ordered pair of GPUs 0.. world - 1."""
    h = _lib()
    before = ctypes.c_int(0)
    _check(h.hipGetDevice(ctypes.byref(before)), "peers: device")
    try:
        for a in range(world):
            _check(h.hipSetDevice(a), f"peers: set device {a}")
            for b in range(world):
                if a == b:
                    continue
                code = h.hipDeviceEnablePeerAccess(b, 0)
                if code == _PEER_ENABLED:
                    h.hipGetLastError()
                elif code:
                    _check(code, f"peers: {a} -> {b}")
    finally:
        h.hipSetDevice(before.value)


# --------------------------------------------------------------- backends


class Graph:
    """A captured sequence of calls (the GPU's graph, or the host's list)."""

    __slots__ = ("calls", "g", "n")

    def __init__(self, n, g=None, calls=None):
        self.n, self.g, self.calls = n, g, calls


class HostBackend:
    """The host backend: races on CPU tensors.

    Every allocation is in host memory; the compare gives its specified result;
    graphs are call lists; replays are timed by the host clock.
    """

    name = "host"

    def __init__(self, cus=1, arch="host", cold=None, hip=0):
        """A backend of `cus` compute units reporting `arch`.

        `cold`: the bytes the rotated copies span (default the device's COLD);
        host memory has no cache to defeat, so a test may span less. `hip`: the
        HIP version it reports (0: none, so graph_replays_ok() is true).
        """
        from metron._memory import COLD

        self.cus, self.arch, self.hip = cus, arch, hip
        self.captured = 0
        """Graphs captured."""
        self.cold = COLD if cold is None else cold
        self.alive = self.peak_alive = 0
        """Launches in graphs not yet released, and their most."""

    def device(self, rank):
        """The torch device of a rank's allocations."""
        return "cpu"

    def alloc(self, rank, nbytes):
        """A u8 buffer of nbytes."""
        import torch

        return torch.empty(nbytes, dtype=torch.uint8)

    def uncached(self, rank, nbytes):
        """An uncached u8 buffer (host memory here)."""
        return self.alloc(rank, nbytes)

    def stream(self, rank):
        """The rank's stream (none on the host)."""
        return None

    def sync(self, rank):
        """Waits for the rank's work (nothing is pending on the host)."""

    def differ(self, rank, base, ref, stride, n, flags):
        """flags[y] = 1 where copy y's stride bytes differ from ref's."""
        import torch

        rows = base[: n * stride].view(n, stride)
        out = (rows != ref[:stride].view(1, stride)).any(dim=1).to(torch.int32)
        flags[: 4 * n].view(torch.int32).copy_(out)

    def invalidate(self, rank, stream):
        """The instruction-cache invalidate (nothing to invalidate on the host)."""

    def empty(self, rank, stream):
        """The empty kernel."""

    def hip_version(self) -> int:
        """The HIP version it reports."""
        return self.hip

    def rank_arch(self, rank) -> str:
        """A rank's arch name."""
        return self.arch

    def run(self, rank, thunks):
        """A pass: the calls launched one by one."""
        for c in thunks:
            c()

    def timed_pass(self, rank, thunks) -> tuple:
        """An untimed pass, then a timed one: (us per call, host us per call)."""
        self.run(rank, thunks)
        t = time.perf_counter()
        self.run(rank, thunks)
        us = (time.perf_counter() - t) * 1e6 / max(len(thunks), 1)
        return us, us

    def capture(self, rank, thunks):
        """A graph of the calls, run in order at each replay."""
        self.captured += 1
        self.alive += len(thunks)
        self.peak_alive = max(self.peak_alive, self.alive)
        return Graph(len(thunks), calls=list(thunks))

    def release(self, rank, graph):
        """Drops a graph: its launches are no longer alive."""
        if graph.calls is not None:
            self.alive -= graph.n
            graph.calls = None

    def replay(self, rank, graph):
        """Runs the graph's calls."""
        for c in graph.calls:
            c()

    def timed(self, rank, graph) -> float:
        """An untimed replay, then a timed one: microseconds per call, by the host clock."""
        self.replay(rank, graph)
        t = time.perf_counter()
        self.replay(rank, graph)
        return (time.perf_counter() - t) * 1e6 / max(graph.n, 1)

    def on(self, rank):
        """The block's work in the rank's order (host work is in order already)."""
        return contextlib.nullcontext()

    def enable_peers(self, world):
        """Peer access (every rank shares host memory)."""

    def describe(self) -> dict:
        """The device a record names: the host, its fields of null."""
        keys = ("pci", "amdgpu", "vbios", "firmware", "exclusive")
        return {"name": "host", "arch": self.arch, "cus": self.cus, **dict.fromkeys(keys)}


class GpuBackend:  # pragma: no cover - hardware
    """The device backend.

    torch's allocator, streams and graphs; the own kernels launched through the
    runtime the library loaded.
    """

    name = "gpu"

    def __init__(self):
        """Binds the runtime."""
        from metron._memory import COLD

        _lib()
        self.cold = COLD
        self._kernels: dict = {}
        self._streams: dict = {}
        self.captured = 0
        """Graphs captured."""

    def device(self, rank):
        """Rank r on GPU r (a one-rank race on the current device)."""
        import torch

        return torch.device(f"cuda:{rank}")

    @property
    def arch(self) -> str:
        """The current device's arch name up to its first ':'."""
        import torch

        return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]

    @property
    def cus(self) -> int:
        """Its compute units."""
        import torch

        return torch.cuda.get_device_properties(0).multi_processor_count

    def alloc(self, rank, nbytes):
        """A u8 tensor of the rank's GPU."""
        import torch

        return torch.empty(nbytes, dtype=torch.uint8, device=self.device(rank))

    def uncached(self, rank, nbytes):
        """An uncached u8 tensor of the rank's GPU."""
        import torch

        with torch.cuda.device(rank):
            return uncached(nbytes)

    def stream(self, rank):
        """The rank's own stream."""
        import torch

        if rank not in self._streams:
            self._streams[rank] = torch.cuda.Stream(device=self.device(rank))
        return self._streams[rank]

    def sync(self, rank):
        """Waits for the rank's stream."""
        self.stream(rank).synchronize()

    @contextlib.contextmanager
    def on(self, rank):
        """The rank's GPU and stream current for the block.

        Copies, fills and launches of the block are then ordered with the rank's
        other work: the copies restored before a replay, the flags zeroed before
        the compare.
        """
        import torch

        with torch.cuda.device(rank), torch.cuda.stream(self.stream(rank)):
            yield

    def _kernel(self, rank, name):
        from metron import _kernels

        key = (rank, name)
        if key not in self._kernels:
            import torch

            with torch.cuda.device(rank):
                desc, co = _kernels.kept(self.arch, name)
                self._kernels[key] = (desc, program(co, desc["symbol"], desc["block"]))
        return self._kernels[key]

    def _run(self, rank, name, grid, stream, **args):
        from metron import _kernels

        desc, p = self._kernel(rank, name)
        launch(p, grid, _kernels.pack(desc, args), stream)

    def differ(self, rank, base, ref, stride, n, flags):
        """The copy compare: flags zeroed, then one launch."""
        s = self.stream(rank)
        flags[: 4 * n].zero_()
        desc, _ = self._kernel(rank, "differ")
        chunk = desc["meta"]["chunk"]
        grid = (-(-stride // chunk), n)
        self._run(
            rank, "differ", grid, s,
            base=base.data_ptr(), ref=ref.data_ptr(), flags=flags.data_ptr(), stride=stride,
        )  # fmt: skip

    def invalidate(self, rank, stream):
        """The invalidate on every compute unit."""
        desc, _ = self._kernel(rank, "icache_inv")
        self._run(rank, "icache_inv", (self.cus * desc["meta"]["per_cu"],), stream, unused=0)

    def empty(self, rank, stream):
        """The empty kernel."""
        self._run(rank, "empty", (1,), stream, unused=0)

    def hip_version(self) -> int:
        """The HIP runtime's version."""
        return hip_version()

    def rank_arch(self, rank) -> str:
        """Rank r's GPU arch name up to its first ':'."""
        import torch

        return torch.cuda.get_device_properties(rank).gcnArchName.split(":", 1)[0]

    def run(self, rank, thunks):
        """A pass: the calls launched one by one on the rank's stream."""
        with self.on(rank):
            for c in thunks:
                c()

    def timed_pass(self, rank, thunks) -> tuple:
        """An untimed pass, then one between two timing events.

        (us per call on the device, us per call the host spent launching it).
        """
        import torch

        s = self.stream(rank)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with self.on(rank):
            for c in thunks:
                c()
            start.record(s)
            t = time.perf_counter()
            for c in thunks:
                c()
            host = time.perf_counter() - t
            end.record(s)
        end.synchronize()
        n = max(len(thunks), 1)
        return start.elapsed_time(end) * 1000 / n, host * 1e6 / n

    def capture(self, rank, thunks):
        """A graph of the calls, captured on the rank's stream."""
        import torch

        self.captured += 1
        g = torch.cuda.CUDAGraph()
        with torch.cuda.device(rank), torch.cuda.graph(g, stream=self.stream(rank)):
            for c in thunks:
                c()
        return Graph(len(thunks), g=g)

    def release(self, rank, graph):
        """Destroys a graph once the rank's stream is done with it."""
        if graph.g is not None:
            self.sync(rank)
            graph.g.reset()
            graph.g = None

    def replay(self, rank, graph):
        """One replay on the rank's stream."""
        import torch

        with torch.cuda.stream(self.stream(rank)):
            graph.g.replay()

    def timed(self, rank, graph) -> float:
        """An untimed replay, then one between two timing events: us per call."""
        import torch

        s = self.stream(rank)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s):
            graph.g.replay()
            start.record(s)
            graph.g.replay()
            end.record(s)
        end.synchronize()
        return start.elapsed_time(end) * 1000 / max(graph.n, 1)

    def enable_peers(self, world):
        """Peer access among GPUs 0.. world - 1."""
        peers(world)


_CURRENT: list = []


def backend():
    """The backend races run on: the innermost ``using``, else the GPU."""
    if not _CURRENT:
        _CURRENT.append(GpuBackend())
    return _CURRENT[-1]


@contextlib.contextmanager
def using(b):
    """Runs the block's races on backend b."""
    _CURRENT.append(b)
    try:
        yield b
    finally:
        _CURRENT.remove(b)
