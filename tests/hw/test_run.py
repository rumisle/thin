"""Compare gfx1201 execution and timing with CPU results and HIP events."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tests import notation
from tests.golden import load

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")

KERNELS = {c["name"]: c for c in load("codegen", "kernels.json")["kernels"]}


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def kernel_of(case):
    """A runtime Kernel of a kernels.json case: its rendered.s and code object."""
    from thin import codegen
    from thin.runtime.kernel import Kernel, KernelDef

    _, art = notation.compile_case(case)
    _, manifest = codegen.split_manifest(art.text)
    d = KernelDef(case["symbol"], None, None, case["wgsize"], tuple(case["max_grid"]))
    co = codegen.code_object(art.text)
    return Kernel(manifest.symbol, art.text, manifest, co, {}, f"hw:{case['name']}", d)


def _on_gpu(args):
    return {n: torch.from_numpy(v.copy()).cuda() if isinstance(v, np.ndarray) else v
            for n, v in args.items()}  # fmt: skip


def both(k, grid, args, outs):
    """(emulated, GPU) outputs by name, as numpy arrays, for the same inputs."""
    emu_args = {n: v.copy() if isinstance(v, np.ndarray) else v for n, v in args.items()}
    got = k.bind(grid, emu_args).emulate()
    emulated = {n: np.asarray(got[n] if n in got else emu_args[n]) for n in outs}
    gpu_args = _on_gpu(args)
    k.bind(grid, gpu_args).run()
    torch.cuda.synchronize()
    ran = {n: gpu_args[n].cpu().numpy() for n in outs}
    return emulated, ran


def same_bits(emulated, ran):
    for n in emulated:
        a, b = emulated[n].view(np.uint8), ran[n].view(np.uint8)
        wrong = np.flatnonzero(a != b)
        assert not len(wrong), f"{n}: {len(wrong)} bytes differ, first at {wrong[0]}"


def _poison(n, dtype=np.uint32):
    return np.full(n, 0xFFFFFFFF, np.uint32).view(dtype)


@pytest.mark.parametrize("n", [1, 31, 256, 257, 1000, 4096, 1 << 18])
def test_vadd_on_the_gpu_is_its_emulation(n):
    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    x[: min(n, 4)] = [np.float32(1e-40), -0.0, np.inf, np.nan][: min(n, 4)]  # subnormal etc.
    args = {"x": x, "y": y, "z": _poison(n, np.float32), "n": n}
    emulated, ran = both(k, (-(-n // 256),), args, ["z"])
    same_bits(emulated, ran)


def test_vadd_at_2_24_on_the_gpu_is_numpys():
    """Past what the emulator runs in a test's budget: numpy's IEEE add."""
    import thin
    from examples.vadd import build_vadd

    n = (1 << 24) + 3
    k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    rng = np.random.default_rng(0)
    x, y = (rng.standard_normal(n).astype(np.float32) for _ in range(2))
    tx, ty = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    tz = torch.full((n,), float("nan"), device="cuda")
    k(x=tx, y=ty, z=tz, n=n).run()
    torch.cuda.synchronize()
    assert np.array_equal(tz.cpu().numpy().view(np.uint32), (x + y).view(np.uint32))


def _case_args(name, rng):
    """(grid, args, outputs) of a kernels.json kernel."""
    if name in ("scalar_reuse_b64", "guarded_permute_b64"):
        n = 1000
        return (-(-n // 64),), {"out": _poison(n), "count": n}, ["out"]
    if name == "matrix_recurrence_b32":
        # Integer-valued f16: every order of the sums gives the same f32.
        a = rng.integers(-4, 5, 256).astype(np.float16)
        b = rng.integers(-4, 5, 256).astype(np.float16)
        b = np.concatenate((np.zeros(8, dtype=np.float16), b))
        return (1,), {"lhs": a, "rhs": b, "out": _poison(512, np.float32)}, ["out"]
    if name == "loop_filter_b64":
        g = 8
        x = rng.standard_normal(64 * g).astype(np.float32)
        return (g,), {"src": x, "dst": _poison(64 * g, np.float32), "rounds": 37}, ["dst"]
    if name == "lds_exchange_b96":
        g = 24
        return (g,), {"out": _poison(96 * g)}, ["out"]
    if name == "plane_store_b32":
        w, h = 7, 5
        size = (h - 1) * 1024 + (w - 1) * w * 32 + 64
        return (w, h), {"out": _poison(size), "width": w}, ["out"]
    raise AssertionError(name)


@pytest.mark.parametrize("name", [n for n in KERNELS if n != "scaled_merge_b128"])
def test_every_kernel_of_kernels_json_runs_as_it_emulates(name):
    from thin import emu

    k = kernel_of(KERNELS[name])
    grid, args, outs = _case_args(name, np.random.default_rng(1))
    emulated, ran = both(k, grid, args, outs)
    if name == "guarded_permute_b64":
        # Lane 13 reads inactive lane 5. FI = BC = 0 leaves shuffled unwritten;
        # adding the destination lane transforms its poison/stale-register bits.
        assert emulated["out"][13] == (emu.POISON + 13) & 0xFFFFFFFF
        emulated["out"][13] = ran["out"][13]
    same_bits(emulated, ran)


def test_scaled_merge_b128_of_kernels_json_runs_as_it_emulates():
    k = kernel_of(KERNELS["scaled_merge_b128"])
    n = 3000
    rng = np.random.default_rng(2)
    x, y = (rng.standard_normal(n).astype(np.float32) for _ in range(2))
    emulated, ran = both(
        k,
        (-(-n // 128),),
        {"left": x, "right": y, "result": _poison(n, np.float32), "count": n},
        ["result"],
    )
    same_bits(emulated, ran)


# --- time against a HIP event pair --------------------------------------------


def _hip_events_us(fn, grid, block, kernarg, iters):
    """Mean microseconds of `iters` launches between two HIP events of our own."""
    from thin.runtime import hip

    h = hip.lib()
    vp = ctypes.c_void_p
    for name, args in {
        "hipEventCreate": [ctypes.POINTER(vp)],
        "hipEventRecord": [vp, vp],
        "hipEventSynchronize": [vp],
        "hipEventElapsedTime": [ctypes.POINTER(ctypes.c_float), vp, vp],
        "hipEventDestroy": [vp],
    }.items():
        getattr(h, name).argtypes, getattr(h, name).restype = args, ctypes.c_int
    start, end = vp(), vp()
    assert not h.hipEventCreate(ctypes.byref(start)) and not h.hipEventCreate(ctypes.byref(end))
    stream = torch.cuda.current_stream().cuda_stream
    hip.enqueue(fn, grid, block, kernarg, stream, "warm")
    assert not h.hipEventRecord(start, vp(stream))
    for _ in range(iters):
        hip.enqueue(fn, grid, block, kernarg, stream, "timed")
    assert not h.hipEventRecord(end, vp(stream))
    assert not h.hipEventSynchronize(end)
    ms = ctypes.c_float()
    assert not h.hipEventElapsedTime(ctypes.byref(ms), start, end)
    h.hipEventDestroy(start), h.hipEventDestroy(end)
    return ms.value * 1000 / iters


def test_time_agrees_with_a_hip_event_pair():
    import thin
    from examples.vadd import build_vadd
    from thin.runtime import gpu

    n = 1 << 26  # 768 MiB moved a call: the kernel, not the launch, dominates
    k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    z = torch.empty(n, device="cuda")
    b = k(x=x, y=y, z=z, n=n)
    ours = b.time(iters=20)
    fn = gpu._function(k, torch.cuda.current_device())
    addrs = {"x": x.data_ptr(), "y": y.data_ptr(), "z": z.data_ptr()}
    theirs = _hip_events_us(fn, b.grid, (256, 1), b.kernarg(addrs.__getitem__), 20)
    print(f"vadd n=2^26: time() {ours:.1f} us, HIP events {theirs:.1f} us")
    assert 0.8 < ours / theirs < 1.25, (ours, theirs)
    # 768 MiB at the 9070 XT's 640 GB/s is at least 1.26 ms.
    assert ours > 1000, ours
