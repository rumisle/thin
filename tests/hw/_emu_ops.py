"""Synthetic kernel generators for operation-level GPU/CPU comparisons."""

from __future__ import annotations

import re

import numpy as np

GROUPS = 64
N = 32 * GROUPS
POISON = 0x7FC17FC1


def compiled(make, keep_dead=False, **params):
    """thin.compile of a kernel maker; keep_dead turns the dead-write check
    off, for exec_kernel alone: its.s is edited to write EXEC (exec_text),
    so the IR it is compiled from sees a write of v over one it never reads."""
    import thin
    from thin._options import using

    with using(keep_dead=keep_dead):
        return thin.compile(make, **params)


def _src(x, mod):
    from thin.dsl import O

    return {"": x, "neg": O.neg(x), "abs": O.abs_(x), "negabs": O.neg(O.abs_(x))}[mod]


def valu_kernel(op: str, n: int, fields=(), mods=(), wide=False, dwords=None):
    """out[i]: `op d, x, y[, z]` in lane i, d starting as w[i]; `fields` the
    op's modifiers as (name, value) pairs, `mods` each source's "", "neg",
    "abs" or "negabs". `dwords` gives the dwords a lane of x, y, z and d
    (w and out as d); `wide` is (2, 2, 2, 2), the default (1, 1, 1, 1)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    sizes = dwords or ((2, 2, 2, 2) if wide else (1, 1, 1, 1))

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y, z, w, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        offs = {}
        vals = []
        for name, p, dw in zip("xyzw", (x, y, z, w), sizes, strict=True):
            if dw not in offs:
                offs[dw] = V.u32(i * (4 * dw), name=f"o{dw}")
            v = V.u32(n=dw, name=name) if dw > 1 else V.u32(name=name)
            b.emit(O.global_load(v, offs[dw], p))
            vals.append(v)
        b.wait(*vals, "the sources landed")
        srcs = [_src(v, m) for v, m in zip(vals[:n], mods or ("",) * n, strict=True)]
        d = vals[3]
        b.emit(O.op(op, d, *srcs, **dict(fields)))
        b.emit(O.global_store(offs[sizes[3]], d, out))

    args = Args(x=ptr, y=ptr, z=ptr, w=ptr, out=ptr)
    return KernelDef(f"valu_{op}", args, body, wgsize=32, max_grid=(GROUPS,))


def class_kernel(e64: bool):
    """out[2 i]: lane i's bit of `v_cmp_class_f32 m, x, y` (m an SGPR whose
    neighbour holds 0x1234 when e64, else vcc_lo); out[2 i + 1]: that
    neighbour after it."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, y, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, c = V.u32(name="a"), V.u32(name="c")
        b.emit(O.global_load(a, V.u32(i * 4, name="xo"), x))
        b.emit(O.global_load(c, V.u32(i * 4, name="yo"), y))
        b.wait(a, c, "the sources landed")
        o = V.u32(i * 8, name="o")
        if e64:
            m = S.u32(n=2, name="m")
            b.emit(O.op("s_mov_b32", m[1], 0x1234))
            b.emit(O.op("v_cmp_class_f32_e64", m[0], a, c))
            r = V.u32(name="r")
            b.emit(O.op("v_cndmask_b32_e64", r, 0, 1, m[0]))
            b.emit(O.global_store(o, r, out))
            b.emit(O.global_store(o, V.u32(m[1], name="nb"), out, offset=4))
        else:
            b.emit(O.op("v_cmp_class_f32", O.VCC_LO, a, c))
            r = V.u32(name="r")
            b.emit(O.op("v_cndmask_b32", r, 0, 1, O.VCC_LO))
            b.emit(O.global_store(o, r, out))

    args = Args(x=ptr, y=ptr, out=ptr)
    return KernelDef(f"class{int(e64)}", args, body, wgsize=32, max_grid=(GROUPS,))


def cmp_kernel(kind: str):
    """out[i]: lane i's bit of `v_cmp_<kind>_f32 vcc_lo, x, y`."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, c = V.u32(name="a"), V.u32(name="c")
        o = V.u32(i * 4, name="o")
        b.emit(O.global_load(a, o, x))
        b.emit(O.global_load(c, o, y))
        b.wait(a, c, "the sources landed")
        b.emit(O.op(f"v_cmp_{kind}_f32", O.VCC_LO, a, c))
        r = V.u32(name="r")
        b.emit(O.op("v_cndmask_b32", r, 0, 1, O.VCC_LO))
        b.emit(O.global_store(o, r, out))

    args = Args(x=ptr, y=ptr, out=ptr)
    return KernelDef(f"cmp_{kind}", args, body, wgsize=32, max_grid=(GROUPS,))


def dpp_kernel(ctrl: str, row_mask: int, bank_mask: int, bound: bool):
    """out[i]: lane i's d (starting as w[i]) after `v_mov_b32_dpp d, x[i]
    <ctrl>` with the masks, in lanes with tid % 5 != 2 (so some source lanes
    are inactive)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, w, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, d = V.u32(name="a"), V.u32(name="d")
        o = V.u32(i * 4, name="o")
        b.emit(O.global_load(a, o, x))
        b.emit(O.global_load(d, o, w))
        b.wait(a, d, "x, w landed")
        lane = V.u32(k.tid, name="lane")
        fields = {"dpp": ctrl, "row_mask": row_mask, "bank_mask": bank_mask}
        if bound:
            fields["bound_ctrl"] = True
        with b.lanes(lane % 5 != 2):
            b.emit(O.op("v_mov_b32_dpp", d, a, **fields))
        b.emit(O.global_store(o, d, out))

    name = f"dpp_{ctrl.replace(':', '').replace(',', '_').replace('[', '').replace(']', '')}"
    return KernelDef(name, Args(x=ptr, w=ptr, out=ptr), body, wgsize=32, max_grid=(GROUPS,))


def permlane_kernel(op: str):
    """out[i]: lane i's d (starting as w[i]) after `op d, x[i], lo, hi` with
    the lane selects lo, hi of the arguments, in lanes with tid % 5 != 2 (so
    some source lanes are inactive: their readers' d is left unwritten)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, w, out, lo, hi = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, d = V.u32(name="a"), V.u32(name="d")
        o = V.u32(i * 4, name="o")
        b.emit(O.global_load(a, o, x))
        b.emit(O.global_load(d, o, w))
        b.wait(a, d, "x, w landed")
        lane = V.u32(k.tid, name="lane")
        with b.lanes(lane % 5 != 2):
            b.emit(O.op(op, d, a, lo, hi))
        b.emit(O.global_store(o, d, out))

    args = Args(x=ptr, w=ptr, out=ptr, lo=u32, hi=u32)
    return KernelDef(op, args, body, wgsize=32, max_grid=(GROUPS,))


def bpermute_kernel(offset: int):
    """out[i]: `ds_bpermute_b32 d, a, x[i] offset:<offset>` with a = w[i] (a
    byte address), in lanes with tid % 5 != 2."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, w, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, c, d = V.u32(name="a"), V.u32(name="c"), V.u32(0, name="d")
        o = V.u32(i * 4, name="o")
        b.emit(O.global_load(c, o, x))
        b.emit(O.global_load(a, o, w))
        b.wait(a, c, "x, w landed")
        lane = V.u32(k.tid, name="lane")
        with b.lanes(lane % 5 != 2):
            b.emit(O.op("ds_bpermute_b32", d, a, c, offset=offset))
            b.wait(d, "the permute landed")
        b.emit(O.global_store(o, d, out))

    return KernelDef(
        f"bpermute{offset}", Args(x=ptr, w=ptr, out=ptr), body, wgsize=32, max_grid=(GROUPS,)
    )


def d16_kernel(op: str, buffer: bool):
    """out[i]: lane i's register after `op` from x into it (starting as w[i]),
    in lanes with tid % 3 != 1. The byte offset is 4 i plus tid % 4 (bytes) or
    2 (tid % 2) (shorts); a buffer load's num_records stops 64 bytes short, so
    the last lanes read out of range."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    short = op.endswith("b16")

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, w, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        d = V.u32(name="d")
        b.emit(O.global_load(d, V.u32(i * 4, name="wo"), w))
        b.wait(d, "w landed")
        lane = V.u32(k.tid, name="lane")
        at = V.u32(i * 4 + ((lane & 1) * 2 if short else lane & 3), name="at")
        with b.lanes(lane % 3 != 1):
            if buffer:
                desc = S.u32(n=4, name="desc")
                b.emit(O.op("s_mov_b32", desc[0], x.lo))
                b.emit(O.op("s_and_b32", desc[1], x.hi, 0xFFFF))
                b.emit(O.op("s_mov_b32", desc[2], 4 * N - 64))
                b.emit(O.op("s_mov_b32", desc[3], 0x30020000))  # raw, OOB_SELECT 3
                b.emit(O.op(op, d, at, desc, O.NULL, offen=True))
            else:
                b.emit(O.op(op, d, at, x))
        b.wait(d, "the d16 load landed")
        b.emit(O.global_store(V.u32(i * 4, name="o"), d, out))

    return KernelDef(f"d16_{op}", Args(x=ptr, w=ptr, out=ptr), body, wgsize=32, max_grid=(GROUPS,))


def ds_hi_kernel(op: str):
    """out[i]: LDS dword tid after it held w[i] and lane tid (tid % 3 != 1) ran
    `op` with x[i] at byte 4 tid plus tid % 4 (b8) or 2 (tid % 2) (b16)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    short = op.startswith("ds_store_b16")

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, w, out = k.load()
        t = b.lds().region("T", (34,), u32)  # the range pass sees tid < 34
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, c = V.u32(name="a"), V.u32(name="c")
        b.emit(O.global_load(a, V.u32(i * 4, name="xo"), x))
        b.emit(O.global_load(c, V.u32(i * 4, name="wo"), w))
        b.wait(a, c, "the sources landed")
        lane = V.u32(k.tid, name="lane")
        b.emit(O.ds_write(t.at(lane), c))
        b.wait(t)
        at = V.u32(lane * 4 + ((lane & 1) * 2 if short else lane & 3), name="at")
        with b.lanes(lane % 3 != 1):
            b.emit(O.op(op, t + at, a))
        b.wait(t)
        r = V.u32(name="r")
        b.emit(O.ds_read(r, t.at(lane)))
        b.wait(r)
        b.emit(O.global_store(V.u32(i * 4, name="o"), r, out))

    return KernelDef(f"ds_{op}", Args(x=ptr, w=ptr, out=ptr), body, wgsize=32, max_grid=(GROUPS,))


def exec_kernel():
    """A 64-bit EXEC on wave32: lane i of workgroup g stores (out[4 i..]):
    7 if it ran under EXEC = s[a:b] (s_a = x[32 g], s_b = y[32 g]), else its
    lane id; EXEC read back as a pair (low, high); exec_hi. Written as
    `s_mov_b64 s, s` and edited to name `exec` (see `exec_text`): the DSL
    spells only exec_lo on wave32."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, y, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, c = V.u32(name="a"), V.u32(name="c")
        b.emit(O.global_load(a, V.u32(i * 4, name="xo"), x))
        b.emit(O.global_load(c, V.u32(i * 4, name="yo"), y))
        b.wait(a, c, "the sources landed")
        s = S.u32(n=2, name="s")
        b.emit(O.op("v_readfirstlane_b32", s[0], a))
        b.emit(O.op("v_readfirstlane_b32", s[1], c))
        v = V.u32(k.tid, name="v")
        b.emit(O.op("s_mov_b64", s, s, comment="WRITE"))
        r = S.u32(n=2, name="r")
        b.emit(O.op("s_mov_b64", r, s, comment="READ"))
        h = S.u32(name="h")
        b.emit(O.op("s_mov_b32", h, O.EXEC_HI))
        b.emit(O.op("v_mov_b32", v, 7))
        b.emit(O.op("s_mov_b32", O.EXEC_LO, -1))
        o = V.u32(i * 16, name="o")
        b.emit(O.global_store(o, v, out))
        b.emit(O.global_store(o, V.u32(r[0], name="rl"), out, offset=4))
        b.emit(O.global_store(o, V.u32(r[1], name="rh"), out, offset=8))
        b.emit(O.global_store(o, V.u32(h, name="hv"), out, offset=12))

    return KernelDef("exec64", Args(x=ptr, y=ptr, out=ptr), body, wgsize=32, max_grid=(GROUPS,))


def exec_text(text: str) -> str:
    """exec_kernel's.s with its marked moves naming exec."""

    def edit(line):
        if "// READ" in line:
            return re.sub(r"(s_mov_b64\s+s\[\d+:\d+\]), s\[\d+:\d+\]", r"\1, exec", line)
        if "// WRITE" in line:
            return re.sub(r"s_mov_b64\s+s\[\d+:\d+\], ", "s_mov_b64 exec, ", line)
        return line

    return "\n".join(edit(line) for line in text.splitlines()) + "\n"


def edited(k, edit, suffix):
    """Kernel k with its.s edited and assembled again."""
    from thin import codegen
    from thin.runtime.kernel import Kernel

    text = edit(k.asm)
    co = codegen.code_object(text)
    return Kernel(k.name, text, k.manifest, co, k.params, k.address + suffix, k.definition)


def saveexec_kernel(op: str, wide: bool):
    """`op d, s` after EXEC_LO = x[32 g], EXEC_HI = z[32 g] (wave32), s =
    y[32 g] (and w[32 g] above it when wide): lane i of workgroup g
    stores (out[4 i..]) d's low dword, EXEC_LO after, SCC after, EXEC_HI
    after."""
    from thin import KernelDef
    from thin.dsl import Args, O, first_lane, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, y, z, w, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        a, c, e, f = V.u32(name="a"), V.u32(name="c"), V.u32(name="e"), V.u32(name="f")
        for v, p in ((a, x), (c, y), (e, z), (f, w)):
            b.emit(O.global_load(v, V.u32(i * 4, name="io"), p))
        b.wait(a, c, e, f, "the sources landed")
        s = S.u32(n=2, name="s") if wide else S.u32(name="s")
        if wide:
            b.emit(O.op("v_readfirstlane_b32", s[0], c))
            b.emit(O.op("v_readfirstlane_b32", s[1], f))
        else:
            b.emit(O.op("v_readfirstlane_b32", s, c))
        hi = S.u32(first_lane(e), name="hi")
        lo = S.u32(first_lane(a), name="lo")
        d = S.u32(n=2, name="d") if wide else S.u32(name="d")
        b.emit(O.op("s_mov_b32", O.EXEC_HI, hi))
        b.emit(O.op("s_mov_b32", O.EXEC_LO, lo))
        b.emit(O.op(op, d, s))
        el, eh = S.u32(name="el"), S.u32(name="eh")
        b.emit(O.op("s_mov_b32", el, O.EXEC_LO))
        b.emit(O.op("s_mov_b32", eh, O.EXEC_HI))
        scc = S.u32(0, name="scc")
        b.emit(O.op("s_cbranch_scc0", ".Lscc0"))  # SCC out, by a branch
        scc[...] = 1
        b.put_label(".Lscc0")
        b.emit(O.op("s_mov_b32", O.EXEC_LO, -1))
        o = V.u32(i * 16, name="o")
        b.emit(O.global_store(o, V.u32(d[0] if wide else d, name="dv"), out))
        b.emit(O.global_store(o, V.u32(el, name="ev"), out, offset=4))
        b.emit(O.global_store(o, V.u32(scc, name="sv"), out, offset=8))
        b.emit(O.global_store(o, V.u32(eh, name="hv"), out, offset=12))

    args = Args(x=ptr, y=ptr, z=ptr, w=ptr, out=ptr)
    return KernelDef(f"save_{op}", args, body, wgsize=32, max_grid=(GROUPS,))


def atomic_kernel(op: str):
    """Lane i applies the float atomic `op` with x[i] to its dword: y[i]
    (global_atomic_add_f32, returning, its old value into out[i];
    buffer_atomic_pk_add_bf16) or LDS dword tid, holding y[i] first
    (ds_add_f32; out[i] its value after). Global atomics return only with
    th:TH_ATOMIC_RETURN, which the DSL cannot set: `atomic_text` adds it."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, y, out = k.load()
        i = V.u32(k.wgid * 32 + k.tid, name="i")
        o = V.u32(i * 4, name="o")
        a0 = V.u32(name="a0")
        b.emit(O.global_load(a0, o, x))
        b.wait(a0, "x landed")
        a = V.u32(name="a")  # untagged: the GEMM-less checker refuses tagged atomic data
        b.emit(O.op("v_xor_b32", a, 0, a0))
        if op.startswith("ds_"):
            t = b.lds().region("T", (34,), u32)
            lane = V.u32(k.tid, name="lane")
            c = V.u32(name="c")
            b.emit(O.global_load(c, o, y))
            b.wait(c, "y landed")
            b.emit(O.ds_write(t.at(lane), c))
            b.wait(t)
            b.emit(O.op(op, t + V.u32(lane * 4, name="at"), a))
            b.wait(t)
            r = V.u32(name="r")
            b.emit(O.ds_read(r, t.at(lane)))
            b.wait(r)
            b.emit(O.global_store(o, r, out))
        elif op.startswith("buffer_"):
            desc = S.u32(n=4, name="desc")
            b.emit(O.op("s_mov_b32", desc[0], y.lo))
            b.emit(O.op("s_and_b32", desc[1], y.hi, 0xFFFF))
            b.emit(O.op("s_mov_b32", desc[2], 4 * N))
            b.emit(O.op("s_mov_b32", desc[3], 0x30020000))
            b.emit(O.op(op, a, o, desc, O.NULL, offen=True))
        else:
            old = V.u32(name="old")
            b.emit(O.op(op, old, o, a, y))
            b.wait(old, "the old value")
            b.emit(O.global_store(o, old, out))

    return KernelDef(f"atomic_{op}", Args(x=ptr, y=ptr, out=ptr), body, wgsize=32,
                     max_grid=(GROUPS,))  # fmt: skip


def atomic_text(text: str) -> str:
    """atomic_kernel's.s with its global atomic returning."""
    return re.sub(r"^(\tglobal_atomic\S+ [^/\n]*?)(\s*(//.*)?)$", r"\1 th:TH_ATOMIC_RETURN\2",
                  text, flags=re.M)  # fmt: skip


def returning(make, **params):
    """A Kernel of a maker whose global atomic returns: its.s, which the DSL
    renders without th:TH_ATOMIC_RETURN (and so does not assemble), patched."""
    from thin import codegen
    from thin.runtime.kernel import Kernel

    asm = atomic_text(make(**params).compile().text)
    _, manifest = codegen.split_manifest(asm)
    co = codegen.code_object(asm)
    return Kernel(manifest.symbol, asm, manifest, co, {}, f"hw:{manifest.symbol}+rtn", make)


# The VALU forms of a 16-bit destination (Form.dst_half) thin cannot write:
# v_mov_b16 and v_swap_b16 only in LLVM's true16 spelling; v_fmaak_f16 and
# v_fmamk_f16 take a literal of their own class, which the encoder refuses.
UNSPELLED = ("v_mov_b16", "v_mov_b16_e64", "v_swap_b16", "v_fmaak_f16", "v_fmamk_f16")


def half_forms():
    """(name, arity, dst_half) of every non-DPP VALU form whose destination
    is a 16-bit half of a VGPR."""
    from thin.target import target

    forms = [f for f in target().forms() if f.dst_half and str(f.kind) == "valu"]
    return sorted({(f.name, f.arity, f.dst_half) for f in forms if not f.name.endswith("_dpp")})


def cndmask16_kernel():
    """valu_kernel's shape for v_cndmask_b16 d, x, y, vcc_lo (vcc_lo: tid > 15)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y, _z, w, out = k.load()
        o = V.u32((k.wgid * 32 + k.tid) * 4, name="o")
        a, c, d = V.u32(name="a"), V.u32(name="c"), V.u32(name="d")
        for v, p in ((a, x), (c, y), (d, w)):
            b.emit(O.global_load(v, o, p))
        b.wait(a, c, d, "the sources landed")
        b.emit(O.op("v_cmp_lt_u32", O.VCC_LO, 15, k.tid))
        b.emit(O.op("v_cndmask_b16", d, a, c, O.VCC_LO))
        b.emit(O.global_store(o, d, out))

    args = Args(x=ptr, y=ptr, z=ptr, w=ptr, out=ptr)
    return KernelDef("cndmask16", args, body, wgsize=32, max_grid=(GROUPS,))


# ------------------------------------------------------------------ inputs


def f16_values(rng, n):
    """f16 patterns: random finite ones and every class's edges."""
    edges = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF,
             0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x7E05, 0xFC07, 0x3555]  # fmt: skip
    v = rng.integers(0, 1 << 16, n, dtype=np.uint32)
    v[rng.random(n) < 0.5] &= 0xBFFF  # mostly finite, near 1
    v[: len(edges)] = edges
    return v


def f32_values(rng, n):
    """f32 patterns: random ones near 1, random ones, and every class's edges."""
    edges = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000,
             0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
             0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FA00001, 0xFFBFFFFF, 0x7FC00005]  # fmt: skip
    v = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    near = rng.random(n) < 0.6
    v[near] = (rng.standard_normal(near.sum()) * 4).astype(np.float32).view(np.uint32)
    v[: len(edges)] = edges
    return v


def f64_values(rng, n):
    """f64 patterns: random values, denormals, every class's edges."""
    edges = [0, 1 << 63, 1, 3, (1 << 63) | 1, 0x000FFFFFFFFFFFFF, 0x0010000000000000,
             0x3FF0000000000000, 0x3FE0000000000000, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000,
             0xFFF0000000000000, 0x7FF8000000000000, 0x7FF4000000000001,
             0xFFF8000000000007, 0x3FF0000000000001, 0x3CB0000000000000]  # fmt: skip
    v = (rng.standard_normal(n) * 8).view(np.uint64).copy()
    sub = rng.random(n) < 0.15
    v[sub] = rng.integers(0, 1 << 52, sub.sum(), dtype=np.uint64)
    v[: len(edges)] = np.array(edges, dtype=np.uint64)
    return v


def u32_values(rng, n):
    edges = [0, 1, 31, 32, 33, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x00800000, 0x007FFFFF,
             0x00FFFFFF, 0xFF800000, 16, 15, 8, 24]  # fmt: skip
    v = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    v[: len(edges)] = edges
    return v


def shuffled(rng, v):
    """v in another random order, so edges meet edges."""
    return v[rng.permutation(len(v))]


# ---------------------------------------------------- NaN-heavy and adversarial

# Every class of each format, NaNs with payloads of both signs first.
F32_SPECIALS = [0x7FC12345, 0xFFC54321, 0x7F801234, 0xFF80ABCD, 0x3F800000, 0xC0200000,
                0x7F800000, 0xFF800000, 0, 0x80000000, 0x00000003, 0x40400000, 0x7FC00000,
                0xFFC00000, 0x7F800001]  # fmt: skip
F16_SPECIALS = [0x7E12, 0xFE34, 0x7C56, 0xFD01, 0x3C00, 0xC100, 0x7C00, 0xFC00, 0, 0x8000,
                0x0003, 0x4200, 0x7E00, 0x7C01]  # fmt: skip
BF16_SPECIALS = [0x7FC1, 0xFFC5, 0x7F81, 0xFF8A, 0x3F80, 0xC020, 0x7F80, 0xFF80, 0, 0x8000,
                 0x0003, 0x4040]  # fmt: skip
F64_SPECIALS = [0x7FF8000000012345, 0xFFF8000000054321, 0x7FF0000000001234,
                0xFFF000000000ABCD, 0x3FF0000000000000, 0xC004000000000000,
                0x7FF0000000000000, 0xFFF0000000000000, 0, 1 << 63, 3,
                0x4008000000000000]  # fmt: skip


def _pick(rng, vals, n, bits=32):
    """Each element a special (3 in 4) or a random pattern."""
    v = rng.choice(np.array(vals, dtype=np.uint64), n)
    if bits == 64:
        r = rng.integers(0, 1 << 63, n, dtype=np.uint64) * np.uint64(2)
        r += rng.integers(0, 2, n, dtype=np.uint64)
    else:
        r = rng.integers(0, 1 << bits, n, dtype=np.uint64)
    return np.where(rng.random(n) < 0.75, v, r)


def special(kind: str, rng, n=None):
    """One source of words, lane-major, for a NaN-heavy case: f32, f16x2,
    bf16x2, f64 (two words a lane), int (an ldexp exponent), byte (any word)."""
    n = n or N
    if kind == "f32":
        return _pick(rng, F32_SPECIALS, n).astype(np.uint32)
    if kind == "f16x2":
        lo, hi = _pick(rng, F16_SPECIALS, n, 16), _pick(rng, F16_SPECIALS, n, 16)
        return (lo | hi << np.uint64(16)).astype(np.uint32)
    if kind == "bf16x2":
        lo, hi = _pick(rng, BF16_SPECIALS, n, 16), _pick(rng, BF16_SPECIALS, n, 16)
        return (lo | hi << np.uint64(16)).astype(np.uint32)
    if kind == "f64":
        return _pick(rng, F64_SPECIALS, n, 64).astype(np.uint64).view(np.uint32)
    if kind == "int":
        return rng.integers(-140, 140, n).astype(np.int32).view(np.uint32)
    if kind == "mix":
        return np.where(rng.random(n) < 0.5, special("f32", rng, n), special("f16x2", rng, n))
    return rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)


def _f32(a):
    return np.asarray(a, np.float32).view(np.uint32)


def halfway(rng, n, small_bits):
    """x (24-bit significands) and y (small_bits-bit ones) whose exact product
    lies on a binary32 midpoint: its bits past the 24th are 100...0."""
    xm = rng.integers(1 << 23, 1 << 24, n, dtype=np.int64) | 1
    ym = np.arange(1 << (small_bits - 1), 1 << small_bits, dtype=np.int64)
    ys = np.empty(n, np.int64)
    for i in range(n):
        p = xm[i] * ym
        drop = np.floor(np.log2(p.astype(np.float64))).astype(np.int64) + 1 - 24
        ok = np.flatnonzero(p & ((np.int64(1) << drop) - 1) == np.int64(1) << (drop - 1))
        ys[i] = ym[rng.choice(ok)] if len(ok) else ym[0]
    sx, sy = rng.choice([-1.0, 1.0], n), rng.choice([-1.0, 1.0], n)
    return xm.astype(np.float64) * sx, ys.astype(np.float64) * sy


def fma_adversarial(rng):
    """f32 (x, y, z), a third each: halfway products plus a z of either sign
    far below them (2^-54 to 2^-75: rounding the sum in binary64 loses it);
    cancellation, z = -fl(x y); products and z denormal."""
    n3 = N // 3
    xm, ym = halfway(rng, n3, 11)
    x = (xm * 2.0 ** rng.integers(-43, -3, n3)).astype(np.float32)
    y = (ym * 2.0 ** rng.integers(-30, 10, n3)).astype(np.float32)
    p = x.astype(np.float64) * y.astype(np.float64)
    z = (p * rng.choice([-1, 1], n3) * 2.0 ** -rng.integers(54, 76, n3)).astype(np.float32)
    x2 = (rng.standard_normal(n3) * 2.0 ** rng.integers(-30, 30, n3)).astype(np.float32)
    y2 = (rng.standard_normal(n3) * 2.0 ** rng.integers(-30, 30, n3)).astype(np.float32)
    z2 = -(x2 * y2)
    m = N - 2 * n3
    x3 = (rng.standard_normal(m) * 2.0**-63).astype(np.float32)
    y3 = (rng.standard_normal(m) * 2.0**-63).astype(np.float32)
    z3 = _f32(rng.integers(0, 1 << 23, m)).view(np.float32) * rng.choice([-1, 1], m)
    cat = np.concatenate
    return [_f32(cat(v)) for v in ((x, x2, x3), (y, y2, y3), (z, z2, z3.astype(np.float32)))]


def mix_adversarial(rng):
    """v_fma_mix_f32 with op_sel:[0,0,1] op_sel_hi:[0,1,1]: x an f32 near
    2^22, y an f16 (low half) such that x y is a binary32 midpoint near 2^32,
    z an f16 (high half) of 2^-24 or 2^-23, either sign: far below it."""
    xm, ym = halfway(rng, N, 11)
    x = (xm * 2.0 ** rng.integers(-1, 1, N)).astype(np.float32)
    y = (ym * 2.0 ** rng.integers(0, 5, N)).astype(np.float16)
    z = (rng.choice([-1, 1], N) * 2.0 ** -rng.integers(23, 25, N)).astype(np.float16)
    hi_junk = rng.integers(0, 1 << 16, N).astype(np.uint32) << 16
    return [_f32(x), y.view(np.uint16).astype(np.uint32) | hi_junk,
            z.view(np.uint16).astype(np.uint32) << 16]  # fmt: skip


def _bf16(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint32)


def dot2_random(rng, z_kind="random"):
    """v_dot2_f32_bf16's sources: bf16 pairs across 16 binades; z random,
    cancelling the products (-sum, nudged), or above them by 2^1 to 2^12."""

    def v():
        return _bf16(rng.standard_normal(N) * 2.0 ** rng.integers(-8, 8, N))

    x, y = v() | v() << 16, v() | v() << 16

    def f(h):
        return (h << 16).view(np.float32).astype(np.float64)

    s = f(x & 0xFFFF) * f(y & 0xFFFF) + f(x >> 16) * f(y >> 16)
    if z_kind == "cancel":
        z = -s * (1 + rng.choice([0, 1, -1], N) * 2.0 ** -rng.integers(8, 30, N))
    elif z_kind == "dominant":
        a = np.abs(f(x & 0xFFFF) * f(y & 0xFFFF)) + np.abs(f(x >> 16) * f(y >> 16))
        z = a * rng.choice([-1, 1], N) * 2.0 ** rng.uniform(1, 12, N)
    else:
        z = rng.standard_normal(N) * 2.0 ** rng.integers(-8, 8, N)
    return [x, y, _f32(z)]


def f64_adversarial(rng):
    """f64 (x, y, z): cancellation (z = -x y) and z far below x y, a fifth
    of the lanes scaled into the denormals."""
    x = rng.standard_normal(N) * 2.0 ** rng.integers(-40, 40, N)
    y = rng.standard_normal(N) * 2.0 ** rng.integers(-40, 40, N)
    p = x * y
    z = np.where(rng.random(N) < 0.5, -p, p * 2.0 ** -rng.integers(50, 60, N))
    d = rng.random(N) < 0.2
    x[d] *= 2.0**-520
    y[d] *= 2.0**-520
    return [np.asarray(v, np.float64).view(np.uint32) for v in (x, y, z)]


def binary64_path(x, y, z):
    """fused form, x y + z in binary64 then binary32 (float arrays)."""
    with np.errstate(all="ignore"):
        return (
            (x.astype(np.float64) * y.astype(np.float64) + z.astype(np.float64))
            .astype(np.float32)
            .view(np.uint32)
        )


def differences(emu, gpu) -> list[int]:
    """The words where emulator and GPU differ: every word, NaN or not."""
    return np.flatnonzero(emu != gpu).tolist()
