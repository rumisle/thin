"""Check uncached allocations, visibility and retained memory on gfx1201."""

from __future__ import annotations

import ctypes
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _hip(name, argtypes):
    from thin.runtime import hip

    fn = getattr(hip.lib(), name)
    fn.argtypes, fn.restype = argtypes, ctypes.c_int
    return fn


@pytest.mark.parametrize("nbytes", [1, 100, 256, 257, 4096])
def test_an_uncached_allocation_has_its_floor_and_is_zero(nbytes):
    from thin.runtime import hip

    info = _hip("hipMemPtrGetInfo", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)])
    t = hip.tensor(nbytes, torch.cuda.current_device())
    size = ctypes.c_size_t()
    assert not info(ctypes.c_void_p(t.data_ptr()), ctypes.byref(size))
    assert size.value >= max(nbytes, 256), size.value
    assert t.numel() == nbytes and int(t.sum()) == 0


def gen_flag():
    """flag[0] = 1, then `spin` loads of the flag, then flag[0] = 2."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        flag, spin = k.load()
        zero = V.u32(0, name="zero")
        with b.lanes(k.tid == 0):
            b.emit(O.global_store(zero, V.u32(1, name="one"), flag))
            for _ in b.loop(spin, name="k"):
                c = V.u32(name="c")
                b.emit(O.global_load(c, zero, flag, offset=4))
                b.wait(c, "a load to pass time")
            b.emit(O.global_store(zero, V.u32(2, name="two"), flag))

    return KernelDef("flag", Args(flag=ptr, spin=u32.within(1, 1 << 30)), body, wgsize=32)


def _seen_while_running(flag, spin=4_000_000, wait_s=0.05):
    """What a copy on another stream reads of flag[0] while the kernel spins."""
    import thin

    copy = _hip(
        "hipMemcpyAsync",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
    )
    create = _hip("hipStreamCreateWithFlags", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint])
    sync = _hip("hipStreamSynchronize", [ctypes.c_void_p])
    other = ctypes.c_void_p()
    assert not create(ctypes.byref(other), 1)  # hipStreamNonBlocking
    k = thin.compile(gen_flag)
    out = (ctypes.c_uint32 * 1)()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        start = time.perf_counter()
        k.bind((1,), {"flag": flag, "spin": spin}).run()
    time.sleep(wait_s)
    assert not copy(ctypes.addressof(out), ctypes.c_void_p(flag.data_ptr()), 4, 2, other)
    assert not sync(other)
    seen_at = time.perf_counter() - start
    s.synchronize()
    ran = time.perf_counter() - start
    return int(out[0]), seen_at, ran, int(flag.view(torch.int32)[0])


def test_a_running_kernels_uncached_store_reaches_a_copy():
    from thin.runtime import hip

    flag = hip.tensor(256, torch.cuda.current_device())
    seen, seen_at, ran, after = _seen_while_running(flag)
    print(f"uncached: copy read {seen} at {seen_at * 1e3:.0f} ms; kernel ran {ran * 1e3:.0f} ms")
    assert seen_at < ran, "the copy did not overlap the kernel: spin longer"
    assert seen == 1 and after == 2


def test_a_running_kernels_cached_store_for_comparison():
    """Recorded, not asserted: whether plain device memory shows the same (on
    gfx1201 the copy waited for the kernel to end, so it shows nothing)."""
    flag = torch.zeros(64, dtype=torch.uint8, device="cuda")
    seen, seen_at, ran, after = _seen_while_running(flag)
    print(f"cached: copy read {seen} at {seen_at * 1e3:.0f} ms; kernel ran {ran * 1e3:.0f} ms")
    assert after == 2


def gen_count(uncached: bool = False):
    """c[lane] += 1 for 64 lanes of held (carried) memory."""
    from thin import KernelDef
    from thin.dsl import Args, O, carried

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        (c,) = k.load()
        off = V.u32(k.tid * 4, name="off")
        x = V.u32(name="x")
        b.emit(O.global_load(x, off, c))
        b.wait(x, "the count")
        x += 1
        b.emit(O.global_store(off, x, c))

    kind = carried.uncached if uncached else carried
    return KernelDef(f"count{int(uncached)}", Args(c=kind), body, wgsize=64)


@pytest.mark.parametrize("uncached", [False, True])
def test_held_memory_reads_back_after_calls_and_a_graph_replay(uncached):
    import thin

    k = thin.compile(gen_count, uncached=uncached)
    b = k.bind((1,), {"c": 256})
    for _ in range(3):
        b.run()
    torch.cuda.synchronize()
    assert b.owned["c"].view(torch.int32)[:64].tolist() == [3] * 64
    assert hasattr(b.owned["c"], "_thin_owner") == uncached  # an allocation of
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        b.run()
    for _ in range(4):
        g.replay()
    torch.cuda.synchronize()
    assert b.owned["c"].view(torch.int32)[:64].tolist() == [7] * 64


def test_a_first_held_allocation_inside_a_capture_is_refused():
    import thin
    from thin.runtime.errors import LaunchError

    k = thin.compile(gen_count, uncached=True)
    b = k.bind((1,), {"c": 512})  # another size: another held buffer, not yet made
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with (
        pytest.raises(LaunchError, match="first allocated inside a graph capture"),
        torch.cuda.stream(s),
        torch.cuda.graph(g, stream=s),
    ):
        b.run()


def test_a_buffer_between_two_calls():
    import thin
    from examples.vadd import build_vadd
    from thin.runtime.bound import buffer

    n = 5000
    k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    z = torch.empty(n, device="cuda")
    mid = buffer(n * 4, like=z)
    k(x=x, y=y, z=mid, n=n).run()
    k(x=mid, y=y, z=z, n=n).run()
    torch.cuda.synchronize()
    want = (x.cpu().numpy() + y.cpu().numpy()) + y.cpu().numpy()
    assert np.array_equal(z.cpu().numpy().view(np.uint32), want.view(np.uint32))


def test_an_overread_past_a_tensors_storage_is_padded_and_copied_back():
    import thin
    from examples.vadd import build_vadd

    n = 1000
    k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    z = torch.empty(n, device="cuda")
    b = k.bind(-(-n // 256), {"x": x, "y": y, "z": z, "n": n}, overread={"z": 4096})
    b.run()
    torch.cuda.synchronize()
    assert torch.equal(z, x + y)
