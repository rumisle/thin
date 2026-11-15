"""The ops and operands of shipped kernels (see docs/dev/lift.md) on the GPU,
bit for bit against their emulation: each op over random operands and the
shift edges 0, 31, 32; vcc_hi as a plain scalar beside the wave32 mask; a
buffer load with soffset null. v_rcp_iflag_f32 is fed powers of two and
special values (the emulator's rcp is correctly rounded, the hardware's an
approximation, exact on those), as tests/hw/test_hazards.py does for rcp."""

from __future__ import annotations

import numpy as np
import pytest

from tests.hw.test_hazards import _emulated, _gpu

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")

GROUPS = 64
N = 32 * GROUPS
EDGES = [0, 1, 31, 32, 33, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]

# (mnemonic, sources, file): v ops read three lane inputs, s ops the first lane's.
VECTOR = [("v_add_lshl_u32", 3), ("v_add_nc_i32", 2), ("v_rcp_iflag_f32", 1)]
SCALAR = [
    ("s_cmov_b32", 1),
    ("s_mul_hi_i32", 2),
    ("s_abs_i32", 1),
    ("s_sub_co_i32", 2),
    ("s_bfm_b32", 2),
    ("s_sext_i32_i16", 1),
]


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def op_kernel(op: str, n: int, scalar: bool):
    """out[2 i]: the op's result in lane i; out[2 i + 1]: SCC after it (scalar ops).
    x, y, z: the sources (a scalar op takes its workgroup's first lane's)."""
    from thin import KernelDef
    from thin.dsl import Args, O, first_lane, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, y, z, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        srcs = []
        for name, p in zip("xyz", (x, y, z), strict=True):
            v = V.u32(name=name)
            b.emit(O.global_load(v, V.u32(i * 4, name=f"{name}o"), p))
            srcs.append(v)
        b.wait(srcs[-1], "the sources landed")
        o = V.u32(i * 8, name="o")
        if not scalar:
            d = V.u32(name="d")
            b.emit(O.op(op, d, *srcs[:n]))
            b.emit(O.global_store(o, d, out))
            return
        s = [S.u32(first_lane(v), name=f"s{j}") for j, v in enumerate(srcs)]
        d = S.u32(s[1] ^ 0xA5A5A5A5, name="d")  # s_cmov's old value
        c = S.u32(0, name="c")
        b.emit(O.op("s_cmp_lg_u32", s[2], 0))  # SCC in: z's first lane nonzero
        b.emit(O.op(op, d, *s[:n]))
        b.emit(O.op("s_cbranch_scc0", ".Lscc0"))  # SCC out, by a branch
        c[...] = 1
        b.put_label(".Lscc0")
        b.emit(O.global_store(o, V.u32(d, name="dv"), out))
        b.emit(O.global_store(o, V.u32(c, name="cv"), out, offset=4))

    args = Args(x=ptr, y=ptr, z=ptr, out=ptr)
    return KernelDef(f"op_{op}", args, body, wgsize=32, max_grid=(GROUPS,))


def _inputs(op, seed=0):
    rng = np.random.default_rng(seed)
    xs = [rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32) for _ in range(3)]
    for a in xs:  # the edges in the first lanes of every workgroup, and lane by lane
        a[: len(EDGES)] = EDGES
        a[32 * np.arange(GROUPS)] = rng.choice(EDGES, GROUPS)
    xs[2][32 * np.arange(GROUPS // 2)] = 0  # half the workgroups SCC in 0
    if op == "v_rcp_iflag_f32":
        e = rng.integers(-100, 101, N)
        f = np.ldexp(np.where(rng.random(N) < 0.5, -1.0, 1.0), e).astype(np.float32)
        f[:6] = [0.0, -0.0, np.inf, -np.inf, 1.0, -2.0]
        xs[0] = f.view(np.uint32)
    return {"x": xs[0], "y": xs[1], "z": xs[2], "out": np.full(2 * N, 0x7FC17FC1, np.uint32)}


@pytest.mark.parametrize(("op", "n", "scalar"), [(o, n, False) for o, n in VECTOR]
                         + [(o, n, True) for o, n in SCALAR])  # fmt: skip
def test_an_op_of_shipped_kernels_runs_as_it_emulates(op, n, scalar):
    import thin

    k = thin.compile(op_kernel, op=op, n=n, scalar=scalar)
    args = _inputs(op)
    emu = _emulated(k, (GROUPS,), args, ["out"])["out"]
    gpu = _gpu(k, (GROUPS,), args, ["out"])["out"]
    wrong = np.flatnonzero(emu != gpu)
    assert not len(wrong), (
        f"{len(wrong)} words differ; first {wrong[0]}: {emu[wrong[0]]:#x} vs {gpu[wrong[0]]:#x}"
    )


def vcc_hi_kernel():
    """out[2 i]: v_cndmask by the wave32 mask (lanes above 15) after vcc_hi is
    written; out[2 i + 1]: vcc_hi read back (its workgroup's id)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        (out,) = k.load()
        o = V.u32((k.wgid * 32 + k.tid) * 8, name="o")
        b.emit(O.op("v_cmp_lt_u32", O.VCC_LO, 15, k.tid))
        b.emit(O.op("s_mov_b32", O.VCC_HI, k.wgid))
        t = S.u32(name="t")
        b.emit(O.op("s_mov_b32", t, O.VCC_HI))
        r = V.u32(name="r")
        b.emit(O.op("v_cndmask_b32", r, 1, 2, O.VCC_LO))
        b.emit(O.global_store(o, r, out))
        b.emit(O.global_store(o, V.u32(t, name="tv"), out, offset=4))

    return KernelDef("vcc_hi", Args(out=ptr), body, wgsize=32, max_grid=(GROUPS,))


def test_vcc_hi_is_a_plain_scalar_beside_the_wave32_mask():
    import thin

    k = thin.compile(vcc_hi_kernel)
    args = {"out": np.full(2 * N, 0x7FC17FC1, np.uint32)}
    emu = _emulated(k, (GROUPS,), args, ["out"])["out"]
    lanes = np.arange(N) % 32
    assert np.array_equal(emu[0::2], np.where(lanes > 15, 2, 1))
    assert np.array_equal(emu[1::2], np.arange(N) // 32)
    assert np.array_equal(_gpu(k, (GROUPS,), args, ["out"])["out"], emu)


def buffer_kernel():
    """out[i] = x[i] by a buffer load with soffset `null` (its voffset i * 4,
    offen; num_records 4 N). `off` there is refused by the encoder; the
    emulator's reading of it is tests/test_emu_shipped.py's."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, out = k.load()
        o = V.u32((k.wgid * 32 + k.tid) * 4, name="o")
        d = S.u32(n=4, name="desc")
        b.emit(O.op("s_mov_b32", d[0], x.lo))
        b.emit(O.op("s_and_b32", d[1], x.hi, 0xFFFF))
        b.emit(O.op("s_mov_b32", d[2], 4 * N))
        b.emit(O.op("s_mov_b32", d[3], 0x30020000))  # shipped GEMMs' word 3 (0: no load)
        v = V.u32(name="v")
        b.emit(O.buffer_load(v, o, d, O.NULL))
        b.wait(v, "x landed")
        b.emit(O.global_store(o, v, out))

    return KernelDef("buf_null", Args(x=ptr, out=ptr), body, wgsize=32, max_grid=(GROUPS,))


def test_a_null_buffer_soffset_adds_nothing():
    import thin

    k = thin.compile(buffer_kernel)
    x = np.random.default_rng(3).integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
    args = {"x": x, "out": np.full(N, 0x7FC17FC1, np.uint32)}
    emu = _emulated(k, (GROUPS,), args, ["out"])["out"]
    assert np.array_equal(emu, x)
    assert np.array_equal(_gpu(k, (GROUPS,), args, ["out"])["out"], x)
