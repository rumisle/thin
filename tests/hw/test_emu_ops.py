"""Compare operations used by shipped GEMM kernels with gfx1201 hardware."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from tests.hw import _emu_ops as ops
from tests.hw import _wmma
from tests.hw.test_hazards import _emulated, _gpu

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _both(k, args, outs=("out",)):
    grid = (ops.GROUPS,)
    return _emulated(k, grid, args, list(outs)), _gpu(k, grid, args, list(outs))


def _check(emu, gpu, kind="bits"):
    """Every word equal; `kind` (the inputs' format) is for the message."""
    bad = ops.differences(emu, gpu)
    assert not bad, (
        f"{len(bad)} of {len(gpu)} {kind} words differ; first {bad[0]}: emulator "
        f"{int(emu[bad[0]]):#x}, GPU {int(gpu[bad[0]]):#x}"
    )


# (id, mnemonic, sources, fields, source modifiers, wide, comparison, inputs)
def _ints(rng):
    return [ops.u32_values(rng, ops.N), ops.shuffled(rng, ops.u32_values(rng, ops.N)),
            ops.shuffled(rng, ops.u32_values(rng, ops.N))]  # fmt: skip


def _bfe(rng):
    x = ops.u32_values(rng, ops.N)
    return [
        x,
        rng.integers(0, 40, ops.N).astype(np.uint32),
        rng.integers(0, 40, ops.N).astype(np.uint32),
    ]


def _shift16(rng):
    return [rng.integers(0, 40, ops.N).astype(np.uint32), ops.u32_values(rng, ops.N)]


def _halves(rng):
    """Two f16 values a dword (low: an edge or a random value; high: another)."""
    return ops.f16_values(rng, ops.N) | ops.shuffled(rng, ops.f16_values(rng, ops.N)) << 16


def _f16s(rng):
    return [_halves(rng), ops.shuffled(rng, _halves(rng))]


def _mix(rng):
    """Each source an f32 value or two f16 halves, by lane parity."""
    out = []
    for _ in range(3):
        f, h = ops.f32_values(rng, ops.N), _halves(rng)
        out.append(np.where(np.arange(ops.N) % 2 == 0, h, ops.shuffled(rng, f)).astype(np.uint32))
    return out


def _f64s(rng):
    return [ops.f64_values(rng, ops.N), ops.shuffled(rng, ops.f64_values(rng, ops.N)),
            ops.shuffled(rng, ops.f64_values(rng, ops.N))]  # fmt: skip


VALU = [
    ("and_or", "v_and_or_b32", 3, (), (), False, "bits", _ints),
    ("bfe_i32", "v_bfe_i32", 3, (), (), False, "bits", _bfe),
    ("mul_i24", "v_mul_i32_i24", 2, (), (), False, "bits", _ints),
    ("mul_i24_e64", "v_mul_i32_i24_e64", 2, (), (), False, "bits", _ints),
    ("mul_u24", "v_mul_u32_u24", 2, (), (), False, "bits", _ints),
    ("mul_u24_e64", "v_mul_u32_u24_e64", 2, (), (), False, "bits", _ints),
    ("mad_u24", "v_mad_u32_u24", 3, (), (), False, "bits", _ints),
    ("lshlrev_b16", "v_lshlrev_b16", 2, (), (), False, "bits", _shift16),
    ("max_f16", "v_max_num_f16", 2, (), (), False, "f16x2", _f16s),
    ("max_f16_e64", "v_max_num_f16_e64", 2, (), (), False, "f16x2", _f16s),
    ("pack", "v_pack_b32_f16", 2, (), (), False, "f16x2", _f16s),
    ("mix_f32", "v_fma_mix_f32", 3, (), (), False, "f32", _mix),
    ("mix_lo_hi", "v_fma_mix_f32", 3, (("op_sel", (0, 1, 0)), ("op_sel_hi", (1, 1, 0))),
     (), False, "f32", _mix),
    ("mix_hi", "v_fma_mix_f32", 3, (("op_sel", (1, 1, 1)), ("op_sel_hi", (1, 1, 1))),
     (), False, "f32", _mix),
    ("mix_some", "v_fma_mix_f32", 3, (("op_sel", (0, 0, 1)), ("op_sel_hi", (0, 1, 1))),
     (), False, "f32", _mix),
    ("mul_f64", "v_mul_f64", 2, (), (), True, "f64", _f64s),
    ("fma_f64", "v_fma_f64", 3, (), (), True, "f64", _f64s),
]  # fmt: skip


# The u24 forms the compiler refuses on operands of unknown bound (passes'
# 24-bit range check): compiled as a stand-in and edited to the op, so the
# GPU sees high bytes.
STAND_IN = {"v_mul_u32_u24": "v_mul_lo_u32", "v_mad_u32_u24": "v_mad_u32_u16"}


@pytest.mark.parametrize("case", VALU, ids=[c[0] for c in VALU])
def test_a_valu_op_runs_as_it_emulates(case):
    _, op, n, fields, mods, wide, kind, make = case
    stand = STAND_IN.get(op, op)
    k = ops.compiled(ops.valu_kernel, op=stand, n=n, fields=fields, mods=mods, wide=wide)
    if stand != op:
        k = ops.edited(k, lambda t: t.replace(stand + " ", op + " "), ":" + op)
    rng = np.random.default_rng(zlib.crc32(case[0].encode()))
    srcs = make(rng)
    srcs += [np.zeros(ops.N, srcs[0].dtype)] * (3 - len(srcs))
    dt = np.uint64 if wide else np.uint32
    old = ops.u32_values(rng, ops.N).astype(dt)
    args = dict(zip("xyz", srcs, strict=True), w=old, out=np.full(ops.N, ops.POISON, dt))
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"], kind)


@pytest.mark.parametrize("e64", [True, False], ids=["e64", "e32"])
def test_cmp_class_runs_as_it_emulates_and_leaves_the_next_sgpr(e64):
    k = ops.compiled(ops.class_kernel, e64=e64)
    rng = np.random.default_rng(7)
    x = ops.f32_values(rng, ops.N)
    y = rng.integers(0, 1 << 10, ops.N).astype(np.uint32)
    y[: 2 * 18] = np.repeat([0x3FF, 0], 18)
    y[36:46] = 1 << np.arange(10)
    x[36:46] = [0x7F800001, 0x7FC00000, 0xFF800000, 0xBF800000, 0x80000001, 0x80000000,
                0, 1, 0x3F800000, 0x7F800000]  # fmt: skip
    args = {"x": x, "y": y, "out": np.full(2 * ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])
    if e64:
        assert set(gpu["out"][1::2].tolist()) == {0x1234}


CMP = [
    "lt",
    "eq",
    "le",
    "gt",
    "lg",
    "ge",
    "o",
    "u",
    "nge",
    "nlg",
    "ngt",
    "nle",
    "neq",
    "nlt",
]


@pytest.mark.parametrize("kind", CMP)
def test_a_float_compare_runs_as_it_emulates(kind):
    """Every f32 compare kind over NaN-heavy inputs, ordered and unordered."""
    k = ops.compiled(ops.cmp_kernel, kind=kind)
    rng = np.random.default_rng(13)
    x = ops.f32_values(rng, ops.N)
    y = ops.shuffled(rng, ops.f32_values(rng, ops.N))
    y[::4] = x[::4]  # equal pairs, NaN with NaN included
    args = {"x": x, "y": y, "out": np.full(ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])


DPP = [
    ("row_shr:1", 0xF, 0xF, False),
    ("row_shr:3", 0xF, 0xF, True),
    ("row_shl:2", 0x5, 0xF, True),
    ("row_ror:5", 0xF, 0x9, False),
    ("row_share:3", 0xF, 0xF, False),
    ("row_xmask:5", 0xA, 0x6, True),
    ("quad_perm:[1,0,3,2]", 0xF, 0xF, False),
    ("row_mirror", 0x3, 0xC, True),
]


@pytest.mark.parametrize(("ctrl", "rows", "banks", "bound"), DPP, ids=[c[0] for c in DPP])
def test_a_dpp_move_runs_as_it_emulates(ctrl, rows, banks, bound):
    """DPP controls with row and bank masks, bound_ctrl and inactive source lanes."""
    k = ops.compiled(ops.dpp_kernel, ctrl=ctrl, row_mask=rows, bank_mask=banks, bound=bound)
    rng = np.random.default_rng(17)
    x, w = ops.u32_values(rng, ops.N), ops.u32_values(rng, ops.N)
    args = {"x": x, "w": w, "out": np.full(ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args, outs=("out",))
    _check(emu["out"], gpu["out"])


@pytest.mark.parametrize("op", ["v_permlane16_b32", "v_permlanex16_b32"])
@pytest.mark.parametrize("seed", [23, 29])
def test_a_permlane_runs_as_it_emulates(op, seed):
    """Random lane selects within a row or across the wave's two rows, with
    inactive source lanes."""
    k = ops.compiled(ops.permlane_kernel, op=op)
    rng = np.random.default_rng(seed)
    x, w = ops.u32_values(rng, ops.N), ops.u32_values(rng, ops.N)
    lo, hi = (int(v) for v in rng.integers(0, 1 << 32, 2, dtype=np.uint64))
    args = {"x": x, "w": w, "out": np.full(ops.N, ops.POISON, np.uint32), "lo": lo, "hi": hi}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])


@pytest.mark.parametrize("offset", [0, 8, 124])
def test_a_bpermute_runs_as_it_emulates(offset):
    """ds_bpermute_b32 with an offset, over any byte addresses and inactive lanes."""
    k = ops.compiled(ops.bpermute_kernel, offset=offset)
    rng = np.random.default_rng(19)
    x = ops.u32_values(rng, ops.N)
    w = rng.integers(0, 1 << 12, ops.N).astype(np.uint32)
    args = {"x": x, "w": w, "out": np.full(ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])


D16 = [
    (op, True)
    for op in (
        "buffer_load_d16_b16",
        "buffer_load_d16_hi_b16",
        "buffer_load_d16_u8",
        "buffer_load_d16_hi_u8",
        "buffer_load_d16_i8",
        "buffer_load_d16_hi_i8",
    )
]
D16 += [("global_load_d16_b16", False), ("global_load_d16_hi_u8", False)]  # fmt: skip


@pytest.mark.parametrize(("op", "buffer"), D16, ids=[c[0] for c in D16])
def test_a_d16_load_runs_as_it_emulates(op, buffer):
    k = ops.compiled(ops.d16_kernel, op=op, buffer=buffer)
    rng = np.random.default_rng(11)
    x, w = ops.u32_values(rng, ops.N), ops.u32_values(rng, ops.N)
    x[::3] |= 0x80808080  # sign bits for the i8 forms
    args = {"x": x, "w": w, "out": np.full(ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])
    inactive = np.arange(ops.N) % 32 % 3 == 1
    assert np.array_equal(gpu["out"][inactive], w[inactive])


@pytest.mark.parametrize("op", ["ds_store_b16_d16_hi", "ds_store_b8_d16_hi"])
def test_a_d16_hi_ds_store_runs_as_it_emulates(op):
    k = ops.compiled(ops.ds_hi_kernel, op=op)
    rng = np.random.default_rng(13)
    args = {"x": ops.u32_values(rng, ops.N), "w": ops.u32_values(rng, ops.N),
            "out": np.full(ops.N, ops.POISON, np.uint32)}  # fmt: skip
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])


def test_a_64_bit_exec_keeps_exec_hi_on_wave32():
    """s_mov_b64 exec keeps EXEC_HI, which s_mov_b64 and exec_hi read back."""
    k = ops.edited(ops.compiled(ops.exec_kernel, keep_dead=True), ops.exec_text, ":exec")
    assert "s_mov_b64 exec," in k.asm
    rng = np.random.default_rng(17)
    x, y = ops.u32_values(rng, ops.N), ops.u32_values(rng, ops.N)
    x[0], y[0] = 0x0000FFFF, 0x12345678
    args = {"x": x, "y": y, "out": np.full(4 * ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])
    assert gpu["out"][:16].tolist() == [7, 0xFFFF, 0x12345678, 0x12345678] * 4


def test_exec_hi_is_0_at_dispatch():
    """EXEC read as a pair at the start; the emulator's EXEC_HI starts 0."""

    def text(t):
        return ops.exec_text(t.replace("// WRITE", "// NOP"))

    k = ops.edited(ops.compiled(ops.exec_kernel, keep_dead=True), text, ":exec0")
    rng = np.random.default_rng(19)
    args = {"x": ops.u32_values(rng, ops.N), "y": ops.u32_values(rng, ops.N),
            "out": np.full(4 * ops.N, ops.POISON, np.uint32)}  # fmt: skip
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])
    assert set(gpu["out"][2::4].tolist()) == {0}


SAVE = [("s_or_saveexec_b32", False), ("s_and_saveexec_b32", False),
        ("s_or_saveexec_b64", True), ("s_and_saveexec_b64", True)]  # fmt: skip


@pytest.mark.parametrize(("op", "wide"), SAVE, ids=[c[0] for c in SAVE])
def test_a_saveexec_runs_as_it_emulates(op, wide):
    k = ops.compiled(ops.saveexec_kernel, op=op, wide=wide)
    rng = np.random.default_rng(23)
    x, y, z, w = (ops.u32_values(rng, ops.N) for _ in range(4))
    # Per workgroup (its lane 0's): EXEC_LO, the source, EXEC_HI, the
    # source's high half: zero and nonzero halves.
    for g, (lo, s, hi, shi) in enumerate([(0xF0, 0x0F00, 7, 0x30), (0, 0, 0, 3),
                                          (0xFFFF, 0xFF00, 0, 0), (0x5, 0, 9, 0)]):  # fmt: skip
        x[32 * g], y[32 * g], z[32 * g], w[32 * g] = lo, s, hi, shi
    args = {"x": x, "y": y, "z": z, "w": w, "out": np.full(4 * ops.N, ops.POISON, np.uint32)}
    emu, gpu = _both(k, args)
    _check(emu["out"], gpu["out"])


#: the kernels of tests/test_partial_writes.py, compiled with the
# dead-write check on, on the GPU as they emulate.
@pytest.mark.parametrize("op", ["v_max_num_f16", "v_max_num_f16_e64", "v_lshlrev_b16"])
def test_a_kept_half_runs_as_it_emulates(op):
    import thin
    from tests import test_partial_writes as pw

    k = thin.compile(pw._kept, op=op)
    rng = np.random.default_rng(29)
    args = {"x": ops.f16_values(rng, pw.LANES),
            "out": np.full(2 * pw.LANES, ops.POISON, np.uint32)}  # fmt: skip
    emu, gpu = (f(k, (1,), args, ["out"])["out"] for f in (_emulated, _gpu))
    _check(emu, gpu, "f16x2" if "max" in op else "bits")
    assert set((gpu[: pw.LANES] >> 16).tolist()) == {0x1234}


def test_two_d16_halves_run_as_they_emulate():
    import thin
    from tests import test_partial_writes as pw

    k = thin.compile(pw._halves)
    rng = np.random.default_rng(31)
    x, y = ops.u32_values(rng, pw.LANES), ops.u32_values(rng, pw.LANES)
    args = {"x": x, "y": y, "out": np.full(pw.LANES, ops.POISON, np.uint32)}
    emu, gpu = (f(k, (1,), args, ["out"])["out"] for f in (_emulated, _gpu))
    _check(emu, gpu)
    assert np.array_equal(gpu, (x & 0xFFFF) | (y << 16))


# ------------------------------------------------------------- NaN rules

# (id, mnemonic, the sources' kinds, the old destination's, fields, dest dwords)
NAN_CASES = [(m, m, ["f32", "f32"], "rand", (), 1) for m in
             ("v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_max_num_f32",
              "v_min_num_f32")]  # fmt: skip
NAN_CASES += [
    ("v_ldexp_f32", "v_ldexp_f32", ["f32", "int"], "rand", (), 1),
    ("v_fma_f32", "v_fma_f32", ["f32", "f32", "f32"], "rand", (), 1),
    ("v_fmac_f32", "v_fmac_f32", ["f32", "f32"], "f32", (), 1),
    ("v_dot2_f32_bf16", "v_dot2_f32_bf16", ["bf16x2", "bf16x2", "f32"], "rand", (), 1),
    ("v_cvt_f32_f16", "v_cvt_f32_f16", ["f16x2"], "rand", (), 1),
    ("v_cvt_pk_rtz_f16_f32", "v_cvt_pk_rtz_f16_f32", ["f32", "f32"], "rand", (), 1),
    ("v_pk_mul_f16", "v_pk_mul_f16", ["f16x2", "f16x2"], "rand", (), 1),
    ("v_max_num_f16", "v_max_num_f16", ["f16x2", "f16x2"], "rand", (), 1),
    ("v_pack_b32_f16", "v_pack_b32_f16", ["f16x2", "f16x2"], "rand", (), 1),
    ("mix_f32", "v_fma_mix_f32", ["f32", "f32", "f32"], "rand", (), 1),
    ("mix_f16", "v_fma_mix_f32", ["f16x2", "f16x2", "f16x2"], "rand",
     (("op_sel", (0, 1, 0)), ("op_sel_hi", (1, 1, 1))), 1),
    ("mix_some", "v_fma_mix_f32", ["f32", "f16x2", "f16x2"], "rand",
     (("op_sel", (0, 0, 1)), ("op_sel_hi", (0, 1, 1))), 1),
    ("v_mul_f64", "v_mul_f64", ["f64", "f64"], "rand2", (), 2),
    ("v_fma_f64", "v_fma_f64", ["f64", "f64", "f64"], "rand2", (), 2),
    ("v_cvt_f32_fp8", "v_cvt_f32_fp8", ["byte"], "rand", (), 1),
    ("v_cvt_f32_bf8", "v_cvt_f32_bf8", ["byte"], "rand", (), 1),
    ("v_cvt_pk_f32_fp8", "v_cvt_pk_f32_fp8", ["byte"], "rand2", (), 2),
    ("v_cvt_pk_fp8_f32", "v_cvt_pk_fp8_f32", ["f32", "f32"], "rand", (), 1),
    ("v_cvt_pk_bf8_f32", "v_cvt_pk_bf8_f32", ["f32", "f32"], "rand", (), 1),
]  # fmt: skip
NAN_CASES += [(m, m, ["f32"] * 3, "rand", (), 1) for m in
              ("v_max3_num_f32", "v_min3_num_f32", "v_med3_num_f32", "v_minimum3_f32",
               "v_maximum3_f32")]  # fmt: skip
UNARY = ("v_ceil_f32", "v_floor_f32", "v_rndne_f32", "v_trunc_f32", "v_frexp_mant_f32",
         "v_frexp_exp_i32_f32", "v_cvt_f16_f32", "v_cvt_i32_f32", "v_cvt_u32_f32")  # fmt: skip
NAN_CASES += [(m, m, ["f32"], "rand", (), 1) for m in UNARY]
# The transcendental ops, word for word like the rest.
TRANS = ("v_exp_f32", "v_log_f32", "v_rcp_f32", "v_sqrt_f32", "v_rsq_f32", "v_rcp_iflag_f32")
NAN_CASES += [(m, m, ["f32"], "rand", (), 1) for m in TRANS]


def _nan_kernel(case):
    _, m, kinds, _, fields, ddw = case
    sizes = [2 if k == "f64" else 1 for k in kinds] + [1] * (3 - len(kinds)) + [ddw]
    return ops.compiled(ops.valu_kernel, op=m, n=len(kinds), fields=fields, dwords=tuple(sizes))


def _nan_args(case, seed):
    name, _, kinds, wk, _, ddw = case
    rng = np.random.default_rng(zlib.crc32(name.encode()) + seed)
    a = {n: ops.special(k, rng) for n, k in zip("xyz", [*kinds, "rand", "rand"][:3], strict=True)}
    a["w"] = ops.special("f32" if wk == "f32" else "rand", rng, ops.N * ddw)
    a["out"] = np.full(ops.N * ddw, ops.POISON, np.uint32)
    return a


@pytest.mark.parametrize("case", NAN_CASES, ids=[c[0] for c in NAN_CASES])
def test_nans_and_specials_are_the_gpus(case):
    k = _nan_kernel(case)
    for seed in range(3):
        args = _nan_args(case, seed)
        emu, gpu = _both(k, args)
        _check(emu["out"], gpu["out"], case[0])


# ------------------------------------------------------- one rounding

FUSED = [
    ("fma", "v_fma_f32", (), (1, 1, 1, 1), ops.fma_adversarial),
    ("fmac", "v_fmac_f32", (), (1, 1, 1, 1), ops.fma_adversarial),
    ("mix", "v_fma_mix_f32", (("op_sel", (0, 0, 1)), ("op_sel_hi", (0, 1, 1))), (1, 1, 1, 1),
     ops.mix_adversarial),
    ("fma_f64", "v_fma_f64", (), (2, 2, 2, 2), ops.f64_adversarial),
    ("dot2", "v_dot2_f32_bf16", (), (1, 1, 1, 1), ops.dot2_random),
    ("dot2_cancel", "v_dot2_f32_bf16", (), (1, 1, 1, 1),
     lambda rng: ops.dot2_random(rng, "cancel")),
    ("dot2_dominant", "v_dot2_f32_bf16", (), (1, 1, 1, 1),
     lambda rng: ops.dot2_random(rng, "dominant")),
]  # fmt: skip


def _ki8(name, a):
    """binary64 fused form of a binary32 case's sources."""

    def f32(u):
        return u.view(np.float32)

    def f16(u):
        return (u & 0xFFFF).astype(np.uint16).view(np.float16)

    if name == "mix":
        return ops.binary64_path(f32(a["x"]), f16(a["y"]), f16(a["z"] >> 16))
    if name == "fmac":
        return ops.binary64_path(f32(a["x"]), f32(a["y"]), f32(a["w"]))
    return ops.binary64_path(f32(a["x"]), f32(a["y"]), f32(a["z"]))


@pytest.mark.parametrize("case", FUSED, ids=[c[0] for c in FUSED])
def test_a_fused_op_rounds_as_the_gpu_does(case):
    name, m, fields, dws, make = case
    nsrc = 2 if name == "fmac" else 3
    k = ops.compiled(ops.valu_kernel, op=m, n=nsrc, fields=fields, dwords=dws)
    ki8 = 0
    for seed in range(4):
        rng = np.random.default_rng(zlib.crc32(name.encode()) + seed)
        x, y, z = make(rng)
        args = {"x": x, "y": y, "z": z, "w": np.zeros_like(x)}
        if name == "fmac":  # the addend is the accumulator d
            args["z"], args["w"] = np.zeros_like(x), z
        args["out"] = np.full(len(x), ops.POISON, np.uint32)
        emu, gpu = _both(k, args)
        _check(emu["out"], gpu["out"], name)
        if name in ("fma", "fmac", "mix"):
            ki8 += int((_ki8(name, args) != gpu["out"]).sum())
    if name in ("fma", "fmac", "mix"):
        assert ki8 > 0, "the inputs do not break the binary64 form"


# ------------------------------------------------------------ WMMA NaNs

WMMA_FLOAT = ["f32_16x16x16_f16", "f32_16x16x16_bf16", "f16_16x16x16_f16", "bf16_16x16x16_bf16",
              "f32_16x16x16_fp8_fp8", "f32_16x16x16_bf8_bf8"]  # fmt: skip


def _nan_d(form, codes):
    d = _wmma.FORMS[form][3]
    c = codes.astype(np.uint32)
    if d == "f16":
        return c & 0x7FFF > 0x7C00
    if d == "bf16":
        return c & 0x7FFF > 0x7F80
    return c & 0x7FFFFFFF > 0x7F800000


@pytest.mark.parametrize("form", WMMA_FLOAT)
def test_a_wmmas_nans_are_the_gpus(form):
    """NaN inputs (with payloads), 0 · inf, inf - inf among the products and with C,
    and NaN C elements: every element the GPU makes a NaN, bit for bit (the
    finite elements are test_wmma.py's)."""
    elem, _, k, d = _wmma.FORMS[form]
    rng = np.random.default_rng(zlib.crc32(form.encode()))
    A = (rng.standard_normal((16, k)) * 0.5).astype(np.float32)  # noqa: N806
    B = (rng.standard_normal((k, 16)) * 0.5).astype(np.float32)  # noqa: N806
    C = rng.standard_normal((16, 16)).astype(np.float32)  # noqa: N806
    A[0, 3] = np.nan
    A[4, 0], B[0, 6] = np.inf, 0
    A[5, 1], A[5, 2], B[1, :], B[2, :] = np.inf, -np.inf, 1, 1
    A[7, 0], B[0, 9], C[7, 9] = np.inf, 1, -np.inf
    C[9, 9], C[3, 4] = (
        np.uint32(0xFFC54321).view(np.float32),
        np.uint32(0x7F812345).view(np.float32),
    )
    if elem == "f16":
        A, B = A.astype(np.float16), B.astype(np.float16)  # noqa: N806
        A[1, 5], B[7, 2] = np.uint16(0xFD01).view(np.float16), np.uint16(0x7E34).view(np.float16)
    if d == "f16":
        C = C.astype(np.float16)  # noqa: N806
        C[9, 9] = np.uint16(0xFE21).view(np.float16)
    emu, gpu = _wmma.run(form, A, B, C)
    nan = _nan_d(form, gpu) | _nan_d(form, emu)
    assert nan.sum() >= 16
    _check(emu[nan], gpu[nan], form)


# --------------------------------------------------------- float atomics

ATOMICS = [("global_atomic_add_f32", "f32"), ("ds_add_f32", "f32"),
           ("buffer_atomic_pk_add_bf16", "bf16x2")]  # fmt: skip


@pytest.mark.parametrize(("op", "kind"), ATOMICS, ids=[c[0] for c in ATOMICS])
def test_a_float_atomics_nans_are_the_gpus(op, kind):
    if op.startswith("global_"):
        k = ops.returning(ops.atomic_kernel, op=op)
    else:
        k = ops.compiled(ops.atomic_kernel, op=op)
    for seed in range(3):
        rng = np.random.default_rng(zlib.crc32(op.encode()) + seed)
        args = {"x": ops.special(kind, rng), "y": ops.special(kind, rng),
                "out": np.full(ops.N, ops.POISON, np.uint32)}  # fmt: skip
        emu, gpu = _both(k, args, ("y", "out"))
        _check(emu["y"], gpu["y"], op)
        _check(emu["out"], gpu["out"], op)


# ------------------------------------------------------- 16-bit halves

HALF_FORMS = ops.half_forms()


@pytest.mark.parametrize(("name", "arity", "half"), HALF_FORMS, ids=[f[0] for f in HALF_FORMS])
def test_a_16_bit_result_keeps_the_other_half(name, arity, half):
    """Every VALU form of a 16-bit destination keeps the half it does not
    write (measured: none zeroes it); where the emulator models the op, the
    whole word is the GPU's."""
    from thin.runtime.errors import NotEmulated

    if name in ops.UNSPELLED:
        from thin.ir import AsmError, Diag

        with pytest.raises((AsmError, Diag)):
            ops.compiled(ops.valu_kernel, op=name, n=arity - 1)
        return
    if name == "v_cndmask_b16":
        k = ops.compiled(ops.cndmask16_kernel)
    else:
        k = ops.compiled(ops.valu_kernel, op=name, n=arity - 1)
    rng = np.random.default_rng(zlib.crc32(name.encode()))

    def src():
        v = (rng.standard_normal(ops.N) * 2).astype(np.float16).view(np.uint16)
        return v.astype(np.uint32) | rng.integers(0, 1 << 16, ops.N).astype(np.uint32) << 16

    w = rng.integers(0, 1 << 32, ops.N, dtype=np.uint64).astype(np.uint32)
    args = {"x": src(), "y": src(), "z": src(), "w": w, "out": np.zeros(ops.N, np.uint32)}
    gpu = _gpu(k, (ops.GROUPS,), args, ["out"])["out"]
    kept = 0xFFFF if half == "Hi" else 0xFFFF0000
    _check(gpu & kept, w & kept, f"{name}'s kept half")
    try:
        emu = _emulated(k, (ops.GROUPS,), args, ["out"])["out"]
    except NotEmulated:
        return
    _check(emu, gpu, name)


# --------------------------------------------------- f16 transcendentals

TRANS16 = ["v_exp_f16", "v_log_f16", "v_rcp_f16", "v_sqrt_f16", "v_rsq_f16"]


@pytest.mark.parametrize("op", TRANS16)
def test_an_f16_transcendental_is_the_gpus(op):
    """Every one of the 65 536 binary16 inputs, word for word: the result's
    low half and the destination's kept high half."""
    k = ops.compiled(ops.valu_kernel, op=op, n=1)
    rng = np.random.default_rng(zlib.crc32(op.encode()))
    every = np.arange(1 << 16, dtype=np.uint32)
    for s in range(0, len(every), ops.N):
        x = every[s : s + ops.N] | rng.integers(0, 1 << 16, ops.N).astype(np.uint32) << 16
        w = rng.integers(0, 1 << 32, ops.N, dtype=np.uint64).astype(np.uint32)
        z = np.zeros(ops.N, np.uint32)
        args = {"x": x, "y": z, "z": z, "w": w, "out": z.copy()}
        gpu = _gpu(k, (ops.GROUPS,), args, ["out"])["out"]
        emu = _emulated(k, (ops.GROUPS,), args, ["out"])["out"]
        _check(emu, gpu, f"{op} inputs {s:#x}..")


# ---------------------------------------------------- f32 transcendentals


def _trans32_inputs(rng, part):
    """2048 patterns: 32 consecutive ones from each of 64 random starts
    (every exponent and sign shows over the parts); part 0 leads with edge
    patterns: 1 and its neighbours, every power of two of both signs, the
    normal range's ends, the edges of log's and exp's special intervals."""
    starts = rng.integers(0, 1 << 32, ops.N // 32, dtype=np.uint64).astype(np.uint32)
    x = (starts[:, None] + np.arange(32, dtype=np.uint32)[None, :]).ravel()
    if part == 0:
        edge = [0x3F800000 + d for d in range(-8, 9)] + [
            0x00800000,
            0x007FFFFF,
            0x7F7FFFFF,
            0x3F7C0000,
            0x3F7BFFFF,
            0x3F840000,
            0x3F83FFFF,
            0x33800000,
            0x337FFFFF,
            0x43000000,
            0x42FC0000,
            0x42FC0001,
            0xC2FC0000,
            0xC2FC0001,
            0xC3000000,
        ]
        edge += [e << 23 for e in range(1, 255)] + [1 << 31 | e << 23 for e in range(1, 255)]
        x[: len(edge)] = edge
    return x


@pytest.mark.parametrize("op", TRANS)
def test_an_f32_transcendental_is_the_gpus(op):
    """32 x 2048 inputs a run, word for word (the full 2^32 inputs of each
    were checked once against the model, outside the suite)."""
    k = ops.compiled(ops.valu_kernel, op=op, n=1)
    rng = np.random.default_rng(zlib.crc32(op.encode()))
    for part in range(32):
        x = _trans32_inputs(rng, part)
        z = np.zeros(ops.N, np.uint32)
        args = {"x": x, "y": z, "z": z, "w": z.copy(), "out": z.copy()}
        _check(_emulated(k, (ops.GROUPS,), args, ["out"])["out"],
               _gpu(k, (ops.GROUPS,), args, ["out"])["out"], f"{op} part {part}")  # fmt: skip
