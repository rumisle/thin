"""Partial writes end to end: a 16-bit VALU op and a D16 load write half
of their destination and keep the other, so the
compiler reads the destination as well as writing it. Compiled with the
dead-write check on and emulated."""

from __future__ import annotations

import numpy as np
import pytest

LANES = 32


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _halves():
    """out[i]: the low half from x[i], the high half from y[i], each a D16
    load into one register no write came before."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y, out = k.load()
        o = V.u32(k.tid * 4, name="o")
        d = V.u32(name="d")
        b.emit(O.op("global_load_d16_b16", d, o, x))
        b.wait(d, "the low half landed")
        b.emit(O.op("global_load_d16_hi_b16", d, o, y))
        b.wait(d, "the high half landed")
        b.emit(O.global_store(o, d, out))

    return KernelDef("halves", Args(x=ptr, y=ptr, out=ptr), body, wgsize=LANES, max_grid=(1,))


def test_two_d16_halves_build_one_value():
    import thin

    rng = np.random.default_rng(1)
    x, y = (rng.integers(0, 1 << 32, LANES, dtype=np.uint64).astype(np.uint32) for _ in "xy")
    out = np.zeros(LANES, np.uint32)
    thin.compile(_halves).bind((1,), {"x": x, "y": y, "out": out}).emulate()
    assert np.array_equal(out, (x & 0xFFFF) | (y << 16))


def _kept(op):
    """out[i]: d = 0x12340000, then (past a label, so its value is not held
    by the dead-write window) a temporary t is made and stored, then `op d,
    a, a` (a = x[i]) writes d's low half. out[32 + i]: t."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, out = k.load()
        o = V.u32(k.tid * 4, name="o")
        d = V.u32(0x12340000, name="d")
        b.put_label(".Lgap")
        a = V.u32(name="a")
        b.emit(O.global_load(a, o, x))
        b.wait(a, "x landed")
        t = V.u32(a + 5, name="t")
        b.emit(O.global_store(o, t, out, offset=4 * LANES))
        b.emit(O.op(op, d, a, a))
        b.emit(O.global_store(o, d, out))

    return KernelDef("kept", Args(x=ptr, out=ptr), body, wgsize=LANES, max_grid=(1,))


@pytest.mark.parametrize("op", ["v_max_num_f16", "v_max_num_f16_e64", "v_lshlrev_b16"])
def test_a_16_bit_op_keeps_a_live_registers_high_half(op):
    import thin

    a = np.full(LANES, 0x3C00, np.uint32)  # 1.0 in f16; as a shift, 0
    out = np.zeros(2 * LANES, np.uint32)
    k = thin.compile(_kept, op=op)
    k.bind((1,), {"x": a, "out": out}).emulate()
    assert np.array_equal(out[LANES:], a + 5)
    assert np.array_equal(out[:LANES] >> 16, np.full(LANES, 0x1234))
    assert np.array_equal(out[:LANES] & 0xFFFF, a)


def _overwritten(first):
    """`first` (a whole or a half write) into d, then a whole write of d,
    then a store of d. (A D16 load's case is in the Rust tests: the DSL's
    wait for it reads d.)"""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        (out,) = k.load()
        o = V.u32(k.tid * 4, name="o")
        d = V.u32(name="d")
        if first == "whole":
            b.emit(O.op("v_mov_b32", d, 1))
        else:
            b.emit(O.op("v_cvt_f16_f32", d, o))
        b.emit(O.op("v_mov_b32", d, 2))
        b.emit(O.global_store(o, d, out))

    return KernelDef("over", Args(out=ptr), body, wgsize=LANES, max_grid=(1,))


@pytest.mark.parametrize(
    ("first", "op"),
    [("whole", "v_mov_b32"), ("half", "v_cvt_f16_f32")],
)
def test_a_write_a_whole_write_follows_is_still_dead(first, op):
    import thin
    from thin.ir import ScheduleError

    with pytest.raises(ScheduleError, match="dead write") as e:
        thin.compile(_overwritten, first=first)
    assert str(e.value).startswith(f"`{op} ")
