"""CPU tests of operations used by shipped GEMM kernels."""

from __future__ import annotations

import numpy as np
import pytest

from tests.hw import _emu_ops as ops

GRID = 2  # workgroups emulated here
N = 32 * GRID


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _emulate(k, args, words=1):
    """The kernel's out over GRID workgroups, every input padded to the
    kernels' size."""
    full = {}
    for name, v in args.items():
        pad = np.zeros(ops.N * words if name == "out" else ops.N, v.dtype)
        pad[: len(v)] = v
        full[name] = pad
    k.bind((GRID,), full).emulate()
    return full["out"][: N * words]


def _rng():
    return np.random.default_rng(5)


def _valu(op, n, srcs, old, wide=False):
    k = ops.compiled(ops.valu_kernel, op=op, n=n, wide=wide)
    dt = np.uint64 if wide else np.uint32
    srcs = [*srcs, *[np.zeros(N, dt)] * (3 - len(srcs))]
    return _emulate(k, dict(zip("xyz", srcs, strict=True), w=old, out=np.zeros(N, dt)))


def test_and_or_bfe_i32_and_mul_i32_i24():
    rng = _rng()
    x, y, z = (ops.u32_values(rng, N) for _ in range(3))
    got = _valu("v_and_or_b32", 3, [x, y, z], x)
    assert np.array_equal(got, (x & y) | z)
    off, width = rng.integers(0, 40, N).astype(np.uint32), rng.integers(0, 40, N).astype(np.uint32)
    got = _valu("v_bfe_i32", 3, [x, off, width], x)
    o, w = (off & 31).astype(np.int64), (width & 31).astype(np.int64)
    field = (x.view(np.int32).astype(np.int64) >> o) & ((1 << w) - 1)
    want = np.where(w == 0, 0, np.where(field >> np.maximum(w - 1, 0) & 1, field - (1 << w), field))
    assert np.array_equal(got, want.astype(np.int64).astype(np.uint32))
    got = _valu("v_mul_i32_i24", 2, [x, y], x)
    i24 = [((v << 8).view(np.int32) >> 8).astype(np.int64) for v in (x, y)]
    assert np.array_equal(got, (i24[0] * i24[1] & 0xFFFFFFFF).astype(np.uint32))


def test_lshlrev_b16_and_max_num_f16_keep_the_high_half():
    rng = _rng()
    s, v, old = rng.integers(0, 40, N).astype(np.uint32), *(ops.u32_values(rng, N) for _ in "ab")
    got = _valu("v_lshlrev_b16", 2, [s, v], old)
    assert np.array_equal(got, old & 0xFFFF0000 | (v << (s & 15)) & 0xFFFF)
    a = ops.f16_values(rng, N)
    b = ops.shuffled(rng, ops.f16_values(rng, N))
    finite = ~np.isnan(a.astype(np.uint16).view(np.float16)) & ~np.isnan(
        b.astype(np.uint16).view(np.float16)
    )
    got = _valu("v_max_num_f16", 2, [a | 0xABCD0000, b], old)
    assert np.array_equal(got >> 16, old >> 16)
    fa, fb = (u.astype(np.uint16).view(np.float16) for u in (a, b))
    want = np.where(fa > fb, a, np.where(fb > fa, b, np.where(a & 0x8000, b, a)))
    assert np.array_equal((got & 0xFFFF)[finite], want[finite])


def test_pack_b32_f16_and_mul_f64_with_the_gpus_nans():
    rng = _rng()
    a, b = ops.f16_values(rng, N), ops.shuffled(rng, ops.f16_values(rng, N))
    got = _valu("v_pack_b32_f16", 2, [a | 0x12340000, b], a)
    nan = [(h & 0x7FFF) > 0x7C00 for h in (a, b)]  # a NaN half is quieted
    quiet = [np.where(n, h | 0x200, h) for n, h in zip(nan, (a, b), strict=True)]
    assert np.array_equal(got, quiet[0] | quiet[1] << 16)
    x, y = ops.f64_values(rng, N), ops.shuffled(rng, ops.f64_values(rng, N))
    got = _valu("v_mul_f64", 2, [x, y], x, wide=True)
    with np.errstate(all="ignore"):
        want = (x.view(np.float64) * y.view(np.float64)).view(np.uint64)
    # The GPU's NaN rules, written out rather than taken from the host (an
    # Arm host's NaNs differ): the first NaN operand quieted, else an invalid
    # product 0xFFF8000000000000.
    x_nan, y_nan = (np.isnan(v.view(np.float64)) for v in (x, y))
    quiet = np.uint64(1 << 51)
    want = np.where(np.isnan(want.view(np.float64)), np.uint64(0xFFF8000000000000), want)
    want = np.where(y_nan, y | quiet, want)
    want = np.where(x_nan, x | quiet, want)
    assert np.array_equal(got, want)


def test_cmp_class_names_each_lanes_class():
    rng = _rng()
    x = ops.f32_values(rng, N)
    y = rng.integers(0, 1 << 10, N).astype(np.uint32)
    k = ops.compiled(ops.class_kernel, e64=True)
    got = _emulate(k, {"x": x, "y": y, "out": np.zeros(2 * N, np.uint32)}, words=2)
    sign, exp, man = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
    bit = np.select(
        [
            (exp == 255) & (man != 0) & (man & 0x400000 == 0),
            (exp == 255) & (man != 0),
            exp == 255,
            exp > 0,
            man != 0,
        ],
        [0, 1, np.where(sign, 2, 9), np.where(sign, 3, 8), np.where(sign, 4, 7)],
        np.where(sign, 5, 6),
    )
    assert np.array_equal(got[0::2], (y >> bit) & 1)
    assert set(got[1::2].tolist()) == {0x1234}


@pytest.mark.parametrize(
    "op", ["buffer_load_d16_b16", "buffer_load_d16_hi_i8", "global_load_d16_hi_b16"]
)
def test_a_d16_load_merges_into_its_half(op):
    rng = _rng()
    x, w = ops.u32_values(rng, N), ops.u32_values(rng, N)
    k = ops.compiled(ops.d16_kernel, op=op, buffer=op.startswith("buffer"))
    got = _emulate(k, {"x": x, "w": w, "out": np.zeros(N, np.uint32)})
    tid, i = np.arange(N) % 32, np.arange(N)
    short = op.endswith("b16")
    at = 4 * i + np.where(short, 2 * (tid & 1), tid & 3)
    data = np.zeros(ops.N, np.uint32)
    data[:N] = x
    raw = data.view(np.uint8)
    v = raw[at].astype(np.uint32) | (raw[at + 1].astype(np.uint32) << 8 if short else 0)
    if op.endswith("i8"):
        v = v.astype(np.uint8).view(np.int8).astype(np.int16).view(np.uint16).astype(np.uint32)
    merged = (w & 0xFFFF) | v << 16 if "_hi_" in op else (w & 0xFFFF0000) | v
    assert np.array_equal(got, np.where(tid % 3 == 1, w, merged))


def test_a_d16_hi_ds_store_stores_the_high_half():
    rng = _rng()
    x, w = ops.u32_values(rng, N), ops.u32_values(rng, N)
    k = ops.compiled(ops.ds_hi_kernel, op="ds_store_b16_d16_hi")
    got = _emulate(k, {"x": x, "w": w, "out": np.zeros(N, np.uint32)})
    tid = np.arange(N) % 32
    shift = 16 * (tid & 1)
    want = w & ~(np.uint32(0xFFFF) << shift) | (x >> 16) << shift
    assert np.array_equal(got, np.where(tid % 3 == 1, w, want))


def test_a_64_bit_exec_and_or_saveexec():
    rng = _rng()
    x, y = ops.u32_values(rng, N), ops.u32_values(rng, N)
    x[0], y[0], x[32], y[32] = 0xFFFF, 0x12345678, 0, 1
    k = ops.edited(ops.compiled(ops.exec_kernel, keep_dead=True), ops.exec_text, ":exec")
    got = _emulate(k, {"x": x, "y": y, "out": np.zeros(4 * N, np.uint32)}, words=4)
    lane = np.arange(64) % 32
    ran = (np.arange(64) < 32) & (lane < 16)  # EXEC_LO 0xffff, then 0
    assert np.array_equal(got[0::4], np.where(ran, 7, lane))
    assert np.array_equal(got[1::4], np.repeat([0xFFFF, 0], 32))
    assert np.array_equal(got[2::4], np.repeat([0x12345678, 1], 32))
    assert np.array_equal(got[3::4], got[2::4])
    z, w = ops.u32_values(rng, N), ops.u32_values(rng, N)
    x[0], y[0], z[0], x[32], y[32], z[32] = 0xF0, 0x0F00, 7, 0, 0, 0
    k = ops.compiled(ops.saveexec_kernel, op="s_or_saveexec_b32", wide=False)
    args = {"x": x, "y": y, "z": z, "w": w, "out": np.zeros(4 * N, np.uint32)}
    got = _emulate(k, args, words=4)
    assert got[[0, 1, 2, 3, 128, 129, 130, 131]].tolist() == [0xF0, 0xFF0, 1, 7, 0, 0, 0, 0]
