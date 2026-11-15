"""Regression kernels for wide constants and signed zero."""

from __future__ import annotations

import numpy as np

CONSTANTS = [0, 1, -1, -3, 2**31, 2**32 - 1, 2**32, 2**33, 2**40, 2**63, 2**64 - 1, -(2**63)]
"""constants c of `u64 ± c`."""
PS = [0, 1, 2**32 - 1, 2**32, 2**64 - 1, 0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210]
"""Its p values (the last two random-looking)."""
FORMS = ("p + c", "c + p", "p - c")
SLOTS = len(CONSTANTS) * len(FORMS)
M64 = (1 << 64) - 1


def _wide(file, V, S, p, forms=FORMS):  # noqa: N803 - the DSL's spelling
    """Every (c, form) result of p, in slot order (constant-major)."""
    make = S.u64 if file == "s" else V.u64
    out = []
    for i, c in enumerate(CONSTANTS):
        for form in forms:
            r = p + c if form == "p + c" else c + p if form == "c + p" else p - c
            out.append(make(r, name=f"{form[0]}{form[-1]}{i}"))
    return out


def gen_wide_scalar():
    """out[slot] = p ± c with p a u64 argument (the SALU chain)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u64

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        p, out = k.load()
        zero = V.u32(0, name="zero")
        with b.lanes(k.tid == 0):
            for slot, r in enumerate(_wide("s", V, S, p)):
                v = V.u64(r, name=f"v{slot}")
                b.emit(O.op("global_store_b64", zero, v, out, offset=8 * slot or None))

    return KernelDef("wide_scalar", Args(p=u64, out=ptr), body, wgsize=32, max_grid=(1,))


def gen_wide_vector(forms: tuple = FORMS):
    """out[lane * slots + slot] = p[lane] ± c (the VALU chain)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        ps, out = k.load()
        lane = V.u32(k.tid, name="lane")
        p = V.u64(name="p")
        b.emit(O.op("global_load_b64", p, V.u32(lane * 8, name="po"), ps))
        b.wait(p, "p landed")
        base = V.u32(lane * (8 * len(CONSTANTS) * len(forms)), name="base")
        for slot, r in enumerate(_wide("v", V, S, p, forms)):
            b.emit(O.op("global_store_b64", base, r, out, offset=8 * slot or None))

    name = "wide_vector_" + "_".join("add" if "+" in f else "sub" for f in forms)
    return KernelDef(name, Args(ps=ptr, out=ptr), body, wgsize=32, max_grid=(1,))


def wide_want(p: int, forms=FORMS) -> list[int]:
    """The slots' expected u64 values for p."""
    f = {"p + c": lambda c: p + c, "c + p": lambda c: c + p, "p - c": lambda c: p - c}
    return [f[form](c) & M64 for c in CONSTANTS for form in forms]


ZERO_INPUTS = np.array([0.0, -0.0, 1.0, 2.5, -1.0, -3.0], np.float32)
"""f per lane: both zeros and some others."""
ZERO_FORMS = ("V.f32(-0.0)", "S.f32(-0.0)", "f * -0.0", "f + -0.0", "-0.0 * f", "-0.0 + f")


def gen_zero():
    """out[lane * 6 + j] = form j for f = in[lane]."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        src, out = k.load()
        lane = V.u32(k.tid, name="lane")
        f = V.f32(name="f")
        b.emit(O.global_load(f, V.u32(lane * 4, name="fo"), src))
        b.wait(f, "f landed")
        rs = [V.f32(-0.0, name="nz"), V.f32(S.f32(-0.0, name="snz"), name="vsnz"),
              V.f32(f * -0.0, name="m"), V.f32(f + -0.0, name="a"),
              V.f32(-0.0 * f, name="mr"), V.f32(-0.0 + f, name="ar")]  # fmt: skip
        base = V.u32(lane * 24, name="base")
        for j, r in enumerate(rs):
            b.emit(O.global_store(base, r, out, offset=4 * j or None))

    return KernelDef("zero", Args(src=ptr, out=ptr), body, wgsize=32, max_grid=(1,))


def zero_want() -> np.ndarray:
    """The expected u32 bits, lane by lane, under IEEE RNE."""
    nz = np.float32(-0.0)
    rows = []
    for f in ZERO_INPUTS:
        rows.append([nz, nz, f * nz, f + nz, nz * f, nz + f])
    return np.array(rows, np.float32).view(np.uint32).reshape(-1)


BORROW_OPS = ("v.sub.co.u32", "v.subrev.co.u32", "v.subb.co.u32")
ACTIVE = 20
"""Lanes active in the borrow kernel (a partial EXEC: inactive lanes' masks are 0)."""


def gen_borrow():
    """For lanes < ACTIVE: d[op][lane] of each borrow op on (x, y), and each op's
    borrow mask in masks[op] (subb's borrow-in: the mask of a first sub)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        xs, ys, d, masks = k.load()
        lane = V.u32(k.tid, name="lane")
        off = V.u32(lane * 4, name="off")
        x, y = V.u32(name="x"), V.u32(name="y")
        b.emit(O.global_load(x, off, xs))
        b.emit(O.global_load(y, off, ys))
        b.wait(x, y, "x, y landed")
        ms = [S.u32(name=f"m{i}") for i in range(3)]
        with b.lanes(lane < ACTIVE):
            for i, op in enumerate(BORROW_OPS):
                r = V.u32(name=f"r{i}")
                if op == "v.subb.co.u32":
                    t = V.u32(name="t")
                    b.emit(O.op("v.sub.co.u32", t, "vcc_lo", x, y))
                    b.emit(O.op(op, r, "vcc_lo", x, y, "vcc_lo"))
                else:
                    b.emit(O.op(op, r, "vcc_lo", x, y))
                b.emit(O.op("s_mov_b32", ms[i], "vcc_lo"))
                b.emit(O.global_store(off, r, d, offset=128 * i or None))
        zero = V.u32(0, name="zero")
        with b.lanes(lane == 0):
            for i in range(3):
                b.emit(
                    O.global_store(zero, V.u32(ms[i], name=f"vm{i}"), masks, offset=4 * i or None)
                )

    return KernelDef(
        "borrow", Args(xs=ptr, ys=ptr, d=ptr, masks=ptr), body, wgsize=32, max_grid=(1,)
    )


def borrow_inputs():
    """x, y a lane: edges, equal values, and the subb tie (y + 1 > x with x = y)."""
    xs = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 5, 5, 0, 0xFFFFFFFF, 7, 100]
    ys = [1, 0, 2, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 5, 6, 0xFFFFFFFF, 0, 7, 3]
    xs += [(i * 2654435761) & 0xFFFFFFFF for i in range(32 - len(xs))]
    ys += [(i * 40503 + 17) & 0xFFFFFFFF for i in range(32 - len(ys))]
    return np.array(xs, np.uint32), np.array(ys, np.uint32)


def borrow_want():
    """(d words, masks) of the ISA guide's pseudocode for borrow_inputs()."""
    xs, ys = (a.astype(np.int64) for a in borrow_inputs())
    d = np.zeros(96, np.uint32)
    masks = [0, 0, 0]
    for lane in range(ACTIVE):
        x, y = int(xs[lane]), int(ys[lane])
        first = y > x
        for i, (r, bo) in enumerate(
            [(x - y, y > x), (y - x, x > y), (x - y - first, y + first > x)]
        ):
            d[32 * i + lane] = r & 0xFFFFFFFF
            masks[i] |= int(bo) << lane
    return d, np.array(masks, np.uint32)
