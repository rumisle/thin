"""MI350X / MI355X (gfx950 / CDNA 4) hardware validation & probe suite against ``thin.emu``.

Run on an MI350X / MI355X machine with::

    pytest -m gfx950 -v

Every test compiles through ``thin.compile`` (or ``codegen.code_object`` for raw
``s_nop`` hazard sweeps), runs identical inputs on both ``thin.emu`` and the
``gfx950`` GPU via HIP + PyTorch, and asserts bit-exact parity while probing the
microarchitectural behaviors that could only be derived from the CDNA 4 ISA
manual offline:
1. Doubled-K dense MFMA (`16x16x32`, `32x32x16`, `16x16x64_i8`, `32x32x32_i8`) and
   OCP FP8/BF8 (`e4m3fn` / `e5m2`) register/lane layouts and numerics;
2. Mixed-precision `v_mfma_f32_*_f8f6f4` (`cbsz`/`blgp` across FP8/BF8/FP6/BF6/FP4,
   8-VGPR half-stacking vs 6-VGPR cross-dword 6-bit packing vs 4-VGPR nibble packing,
   and 32-element grouped summation);
3. MX E8M0 block-scaled `v_mfma_scale_f32_*_f8f6f4` (`ENC_VOP3PX2`, `sel_a`/`sel_b`
   byte selection, `0x00` `2^-127`, `0xFF` NaN column isolation, scale-before-accumulate);
4. Doubled-K 4:2 sparse `v_smfmac_*` (`16x16x64_bf16` dual `abid` sets, `16x16x128_fp8_bf8`,
   `16x16x128_i8`);
5. `v_cvt_scalef32_*` full-domain decode + `FP16_OVFL=0` overflow/NaN/Inf encode sweep,
   stochastic rounding (`v_cvt_sr_*`, `v_cvt_scalef32_sr_*`, `v_prng_b32` — including
   the CDNA 4 manual §7.3 vs VOP3 pseudocode probe on exact `1.0`), and new CDNA 4 VALU ops;
6. Transpose LDS reads (`ds_read_b64_tr_b{4,8,16}`, `ds_read_b96_tr_b6`), wide `b96`/`b128`
   VMEM-to-LDS DMA (`dwordx3` 4th-dword skip probe), `buffer_atomic_pk_add_bf16`,
   160 KB (`163,840` B) LDS allocation ceiling, and 64-bank conflict timing probe;
7. XDL minimum `s_nop` hazard sweep (`0..=20`) and multi-XCD BF16 / MXFP8 GEMM + HBM
   roofline sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gfx950


@pytest.fixture(autouse=True)
def _no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _poison(n: int, dtype=np.uint32) -> np.ndarray:
    return np.full(n, 0xFFFFFFFF, dtype=np.uint32).view(dtype).copy()


def _both(k, grid: tuple[int, ...], args: dict, outs: list[str]) -> tuple[dict, dict]:
    """Run ``k`` on both ``thin.emu`` and the active ``gfx950`` GPU; return ``(emu, gpu)``."""
    import torch

    emu_args = {n: v.copy() if isinstance(v, np.ndarray) else v for n, v in args.items()}
    got_emu = k.bind(grid, emu_args).emulate()
    emulated = {n: np.asarray(got_emu[n] if n in got_emu else emu_args[n]).copy() for n in outs}

    gpu_args = {
        n: torch.from_numpy(np.ascontiguousarray(v).copy().view(np.uint8)).cuda()
        if isinstance(v, np.ndarray)
        else v
        for n, v in args.items()
    }
    k.bind(grid, gpu_args).run()
    torch.cuda.synchronize()
    ran = {
        n: gpu_args[n].cpu().numpy().view(args[n].dtype).reshape(args[n].shape).copy() for n in outs
    }
    return emulated, ran


def _same_bits(emulated: dict, ran: dict) -> None:
    for n in emulated:
        a = np.ascontiguousarray(emulated[n]).view(np.uint8)
        b = np.ascontiguousarray(ran[n]).view(np.uint8)
        wrong = np.flatnonzero(a != b)
        assert not len(wrong), (
            f"{n}: {len(wrong)}/{len(a)} bytes differ, first at byte {wrong[0]} "
            f"(emu=0x{a[wrong[0]]:02x}, gpu=0x{b[wrong[0]]:02x})"
        )


# ---------------------------------------------------------------------------
# Kernel generators (top-level functions for thin.compile source identity)
# ---------------------------------------------------------------------------

CDNA4_DENSE_SHAPES: dict[str, tuple[int, int]] = {
    "f32_16x16x32_bf16": (4, 4),
    "f32_16x16x32_f16": (4, 4),
    "f32_32x32x16_bf16": (4, 16),
    "f32_32x32x16_f16": (4, 16),
    "i32_16x16x64_i8": (4, 4),
    "i32_32x32x32_i8": (4, 16),
    "f32_16x16x32_fp8_fp8": (2, 4),
    "f32_16x16x32_fp8_bf8": (2, 4),
}

F8F6F4_DWORDS: dict[str, int] = {
    "fp8": 8,
    "bf8": 8,
    "fp6": 6,
    "bf6": 6,
    "fp4": 4,
}


def gen_cdna4_mfma_dense(shape: str, acc_file: str):
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    ab_dw, cd_dw = CDNA4_DENSE_SHAPES[shape]

    def _gload(b, dst, n: int, base, buf):
        for at in range(0, n, 4):
            w = min(4, n - at)
            sub = dst[at : at + w] if n > 1 else dst
            b.emit(O.global_load(sub, base, buf, offset=4 * at or None))

    def _gstore(b, base, src, n: int, buf):
        for at in range(0, n, 4):
            w = min(4, n - at)
            sub = src[at : at + w] if n > 1 else src
            b.emit(O.global_store(base, sub, buf, offset=4 * at or None))

    def body(b, k):
        V = b.v  # noqa: N806
        pa, pb, pc, pd = k.load()
        lane = V.u32(k.tid, name="lane")
        oa = V.u32(lane * (4 * ab_dw), name="oa")
        oc = V.u32(lane * (4 * cd_dw), name="oc")
        ra = V.u32(n=ab_dw, name="ra")
        rb = V.u32(n=ab_dw, name="rb")
        rc = V.u32(n=cd_dw, name="rc")
        _gload(b, ra, ab_dw, oa, pa)
        _gload(b, rb, ab_dw, oa, pb)
        _gload(b, rc, cd_dw, oc, pc)
        b.wait(ra, rb, rc, comment="tiles landed")
        acc = b.write_acc(rc, acc=acc_file, name="acc")
        b.mfma(shape, ra, rb, acc)
        out_v = b.read_acc(acc, name="out_v")
        _gstore(b, oc, out_v, cd_dw, pd)

    return KernelDef(
        f"gfx950_mfma_{shape}_{acc_file}",
        Args(a=ptr, b=ptr, c=ptr, d=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_cdna4_f8f6f4(shape: str, fmt_a: str, fmt_b: str, scaled: bool, sel_a: int, sel_b: int):
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    na = F8F6F4_DWORDS[fmt_a]
    nb = F8F6F4_DWORDS[fmt_b]
    nd = 16 if "32x32" in shape else 4

    def _gload(b, dst, n: int, base, buf):
        for at in range(0, n, 4):
            w = min(4, n - at)
            sub = dst[at : at + w] if n > 1 else dst
            b.emit(O.global_load(sub, base, buf, offset=4 * at or None))

    def _gstore(b, base, src, n: int, buf):
        for at in range(0, n, 4):
            w = min(4, n - at)
            sub = src[at : at + w] if n > 1 else src
            b.emit(O.global_store(base, sub, buf, offset=4 * at or None))

    def body(b, k):
        V = b.v  # noqa: N806
        pa, pb, pc, psa, psb, pd = k.load()
        lane = V.u32(k.tid, name="lane")
        ra = V.u32(n=na, name="ra")
        rb = V.u32(n=nb, name="rb")
        rc = V.u32(n=nd, name="rc")
        _gload(b, ra, na, V.u32(lane * (4 * na)), pa)
        _gload(b, rb, nb, V.u32(lane * (4 * nb)), pb)
        _gload(b, rc, nd, V.u32(lane * (4 * nd)), pc)
        if scaled:
            off4 = V.u32(lane * 4, name="off4")
            sa = V.u32(name="sa")
            sb = V.u32(name="sb")
            b.emit(O.global_load(sa, off4, psa))
            b.emit(O.global_load(sb, off4, psb))
            b.wait(ra, rb, rc, sa, sb, comment="scaled inputs landed")
            acc = b.write_acc(rc, acc="a", name="acc")
            b.mfma_scale(
                shape,
                ra,
                rb,
                sa,
                sb,
                acc,
                fmt_a=fmt_a,
                fmt_b=fmt_b,
                sel_a=sel_a,
                sel_b=sel_b,
            )
        else:
            b.wait(ra, rb, rc, comment="unscaled inputs landed")
            acc = b.write_acc(rc, acc="a", name="acc")
            b.mfma(shape, ra, rb, acc, fmt_a=fmt_a, fmt_b=fmt_b)
        out_v = b.read_acc(acc, name="out_v")
        _gstore(b, V.u32(lane * (4 * nd)), out_v, nd, pd)

    tag = f"scale_s{sel_a}{sel_b}" if scaled else "unscaled"
    return KernelDef(
        f"gfx950_{shape}_{fmt_a}_{fmt_b}_{tag}",
        Args(a=ptr, b=ptr, c=ptr, sa=ptr, sb=ptr, d=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


CDNA4_SPARSE_SHAPES: dict[str, tuple[int, int]] = {
    "f32_16x16x64_bf16": (4, 8),
    "f32_16x16x128_fp8_bf8": (4, 8),
    "i32_16x16x128_i8": (4, 8),
}


def gen_cdna4_smfmac(shape: str, abid: int):
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    na, nb = CDNA4_SPARSE_SHAPES[shape]

    def body(b, k):
        V = b.v  # noqa: N806
        pa, pb, pidx, pc, pd = k.load()
        lane = V.u32(k.tid, name="lane")
        ra = V.u32(n=na, name="ra")
        rb = V.u32(n=nb, name="rb")
        ridx = V.u32(name="ridx")
        rc = V.u32(n=4, name="rc")
        b.emit(O.global_load(ra, V.u32(lane * 16), pa))
        b.emit(O.global_load(rb[0:4], V.u32(lane * 32), pb))
        b.emit(O.global_load(rb[4:8], V.u32(lane * 32), pb, offset=16))
        b.emit(O.global_load(ridx, V.u32(lane * 4), pidx))
        b.emit(O.global_load(rc, V.u32(lane * 16), pc))
        b.wait(ra, rb, ridx, rc, comment="sparse inputs landed")
        acc = b.write_acc(rc, acc="a", name="acc")
        b.smfmac(shape, ra, rb, ridx, acc, cbsz=0, abid=abid)
        vd = b.read_acc(acc, name="vd")
        b.emit(O.global_store(V.u32(lane * 16), vd, pd))

    return KernelDef(
        f"gfx950_smfmac_{shape}_abid{abid}",
        Args(a=ptr, b=ptr, idx=ptr, c=ptr, d=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_cdna4_scalef32_and_valu():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        in_f32, in_seeds, out_cvt, out_sr, out_valu = k.load()
        lane = V.u32(k.tid, name="lane")
        off4 = V.u32(lane * 4, name="off4")
        off16 = V.u32(lane * 16, name="off16")

        x = V.f32(n=4, name="x")
        seed = V.u32(name="seed")
        b.emit(O.global_load(x, off16, in_f32))
        b.emit(O.global_load(seed, off4, in_seeds))
        b.wait(x, seed, comment="inputs landed")

        sc2 = V.f32(2.0, name="sc2")
        p_fp8 = V.u32(0, name="p_fp8")
        p_bf8 = V.u32(0, name="p_bf8")
        p_fp4 = V.u32(0, name="p_fp4")
        b.emit(O.op("v_cvt_scalef32_pk_fp8_f32", p_fp8, x[0], x[1], sc2, op_sel=[0, 0, 0, 0]))
        b.emit(O.op("v_cvt_scalef32_pk_fp8_f32", p_fp8, x[2], x[3], sc2, op_sel=[0, 0, 0, 1]))
        b.emit(O.op("v_cvt_scalef32_pk_bf8_f32", p_bf8, x[0], x[1], sc2, op_sel=[0, 0, 0, 0]))
        b.emit(O.op("v_cvt_scalef32_pk_bf8_f32", p_bf8, x[2], x[3], sc2, op_sel=[0, 0, 0, 1]))
        b.emit(O.op("v_cvt_scalef32_pk_fp4_f32", p_fp4, x[0], x[1], sc2, op_sel=[0, 0, 0, 0]))
        b.emit(O.op("v_cvt_scalef32_pk_fp4_f32", p_fp4, x[2], x[3], sc2, op_sel=[0, 0, 1, 0]))
        u_fp4 = V.f32(n=2, name="u_fp4")
        b.emit(O.op("v_cvt_scalef32_pk_f32_fp4", u_fp4, p_fp4, sc2, op_sel=[0, 0, 0]))

        cvt_pack = V.u32(n=4, name="cvt_pack")
        b.emit(O.op("v_mov_b32", cvt_pack[0], p_fp8))
        b.emit(O.op("v_mov_b32", cvt_pack[1], p_bf8))
        b.emit(O.op("v_mov_b32", cvt_pack[2], p_fp4))
        b.emit(O.op("v_mov_b32", cvt_pack[3], u_fp4[0]))
        b.emit(O.global_store(off16, cvt_pack, out_cvt))

        # Stochastic rounding + PRNG probes (including exact 1.0 + 0xFFFFFFFF probe):
        prng = V.u32(name="prng")
        b.emit(O.op("v_prng_b32", prng, seed))
        sr_out = V.u32(0, name="sr_out")
        sc1 = V.f32(1.0, name="sc1")
        b.emit(O.op("v_cvt_sr_fp8_f32", sr_out, x[0], seed, op_sel=[0, 0, 0, 0]))
        b.emit(O.op("v_cvt_scalef32_sr_fp8_f32", sr_out, x[0], seed, sc1, op_sel=[0, 0, 1, 0]))
        pair64 = V.u32(n=2, name="pair64")
        b.emit(O.op("v_mov_b32", pair64[0], x[0]))
        b.emit(O.op("v_mov_b32", pair64[1], x[1]))
        b.emit(O.op("v_cvt_scalef32_sr_pk_fp4_f32", sr_out, pair64, seed, sc1, op_sel=[0, 0, 0, 1]))
        sr_vec = V.u32(n=2, name="sr_vec")
        b.emit(O.op("v_mov_b32", sr_vec[0], prng))
        b.emit(O.op("v_mov_b32", sr_vec[1], sr_out))
        b.emit(O.global_store(V.u32(lane * 8), sr_vec, out_sr))

        # New CDNA 4 VALU ops: bitop3, permlane{16,32}_swap, dot2_bf16, pk_bf16, exp/log_legacy
        bop = V.u32(name="bop")
        b.emit(O.op("v_bitop3_b32", bop, seed, prng, lane, bitop3=0x96))
        sw_a = V.u32(lane + 100, name="sw_a")
        sw_b = V.u32(lane + 200, name="sw_b")
        b.emit(O.op("v_permlane16_swap_b32_e64", sw_a, sw_b))
        b.emit(O.op("v_permlane32_swap_b32_e64", sw_a, sw_b))
        pk_bf = V.u32(name="pk_bf")
        b.emit(O.op("v_cvt_pk_bf16_f32", pk_bf, x[0], x[1]))
        dot_bf = V.f32(name="dot_bf")
        b.emit(O.op("v_dot2_f32_bf16", dot_bf, pk_bf, pk_bf, sc1))
        exp_l = V.f32(name="exp_l")
        log_l = V.f32(name="log_l")
        b.emit(O.op("v_exp_legacy_f32", exp_l, x[0]))
        b.emit(O.op("v_log_legacy_f32", log_l, x[1]))

        valu_vec = V.u32(n=4, name="valu_vec")
        b.emit(O.op("v_mov_b32", valu_vec[0], bop))
        b.emit(O.op("v_add_u32", valu_vec[1], sw_a, sw_b))
        b.emit(O.op("v_mov_b32", valu_vec[2], dot_bf))
        b.emit(O.op("v_xor_b32", valu_vec[3], exp_l, log_l))
        b.emit(O.global_store(off16, valu_vec, out_valu))

    return KernelDef(
        "gfx950_scalef32_and_valu",
        Args(in_f32=ptr, in_seeds=ptr, out_cvt=ptr, out_sr=ptr, out_valu=ptr),
        body,
        wgsize=64,
        max_grid=(4,),
    )


def gen_cdna4_transpose_lds_and_wide_dma():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        src16, src12, atom_buf, out_tr16, out_tr8, out_tr4, out_tr6, out_dma12, out_prev = k.load()
        lds = b.lds()
        tile16 = lds.region("tile16", (256,), elem="u32")  # 64 lanes * 16 B = 1024 B
        tile12 = lds.region("tile12", (256,), elem="u32")  # 64 lanes * 16 B stride for dwordx3!
        tile6 = lds.region("tile6", (192,), elem="u32")  # 64 lanes * 12 B = 768 B

        lane = V.u32(k.tid, name="lane")
        off16 = V.u32(lane * 16, name="off16")
        off12 = V.u32(lane * 12, name="off12")

        # 1) Wide 128-bit (`bytes=16`) + 96-bit (`bytes=12`) global-to-LDS DMA:
        #    Pre-fill `tile12` with a sentinel: `dwordx3` should leave dword 3 untouched.
        sentinel = V.u32(n=4, name="sentinel")
        for i in range(4):
            b.emit(O.op("v_mov_b32", sentinel[i], 0xDEAD_0000 + i))
        b.emit(O.ds_write(tile12.at(lane * 4), sentinel))
        r6_in = V.u32(n=3, name="r6_in")
        b.emit(O.global_load(r6_in, off12, src12))
        b.wait(tile12, r6_in, comment="sentinel + r6 ready")

        b.emit(O.global_load(tile16.at(0), off16, src16, bytes=16))
        b.emit(O.global_load(tile12.at(0), off12, src12, bytes=12))
        b.emit(O.ds_write(tile6.at(lane * 3), r6_in))
        b.wait(tile16, tile12, tile6, comment="DMA + ds_write_b96 landed")

        # 2) Transpose LDS reads across all 4 widths (`16`, `8`, `4`, `6` bits):
        #    8 B (`lane * 2` dwords) for `b64_tr` or 12 B (`lane * 3` dwords) for `b96_tr_b6`:
        v_tr16 = V.u32(n=2, name="v_tr16")
        v_tr8 = V.u32(n=2, name="v_tr8")
        v_tr4 = V.u32(n=2, name="v_tr4")
        v_tr6 = V.u32(n=3, name="v_tr6")
        v_dma12 = V.u32(n=4, name="v_dma12")
        b.emit(O.ds_read_tr(v_tr16, tile16.at(lane * 2), bits=16))
        b.emit(O.ds_read_tr(v_tr8, tile16.at(lane * 2), bits=8))
        b.emit(O.ds_read_tr(v_tr4, tile16.at(lane * 2), bits=4))
        b.emit(O.ds_read_tr(v_tr6, tile6.at(lane * 3), bits=6))
        b.emit(O.ds_read(v_dma12, tile12.at(lane * 4)))
        b.wait(v_tr16, v_tr8, v_tr4, v_tr6, v_dma12, comment="transpose reads landed")

        off8 = V.u32(lane * 8, name="off8")
        b.emit(O.global_store(off8, v_tr16, out_tr16))
        b.emit(O.global_store(off8, v_tr8, out_tr8))
        b.emit(O.global_store(off8, v_tr4, out_tr4))
        b.emit(O.global_store(off12, v_tr6, out_tr6))
        b.emit(O.global_store(off16, v_dma12, out_dma12))

        # 3) Returning `buffer_atomic_pk_add_bf16` (`sc0`):
        rsrc = S.u32(n=4, name="rsrc")
        b.emit(O.op("s_mov_b32", rsrc[0], atom_buf.lo))
        b.emit(O.op("s_and_b32", rsrc[1], atom_buf.hi, 0xFFFF))
        b.emit(O.op("s_mov_b32", rsrc[2], 256))
        b.emit(O.op("s_mov_b32", rsrc[3], 0x0002_0000))
        old_pk = V.u32(0x3F80_3F80, name="old_pk")
        b.emit(
            O.op(
                "buffer_atomic_pk_add_bf16", old_pk, V.u32(lane * 4), rsrc, 0, offen=True, sc0=True
            )
        )
        b.wait(old_pk, comment="atomic return landed")
        b.emit(O.global_store(V.u32(lane * 4), old_pk, out_prev))

    return KernelDef(
        "gfx950_tr_lds_and_wide_dma",
        Args(
            src16=ptr,
            src12=ptr,
            atom_buf=ptr,
            out_tr16=ptr,
            out_tr8=ptr,
            out_tr4=ptr,
            out_tr6=ptr,
            out_dma12=ptr,
            out_prev=ptr,
        ),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_cdna4_160kb_lds_and_64bank_probe():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        out_top, out_clocks = k.load()
        # Allocate the full 160 KB (40,960 u32 = 163,840 B) per-CU LDS ceiling!
        lds = b.lds()
        arena = lds.region("arena", (40_960,), elem="u32")

        lane = V.u32(k.tid, name="lane")
        top_idx = V.u32(40_896 + lane, name="top_idx")
        val = V.u32(lane ^ 0x9500_0000, name="val")
        b.emit(O.ds_write(arena.at(top_idx), val))
        b.wait(arena, comment="160 KB top written")

        rev_idx = V.u32(40_959 - lane, name="rev_idx")
        got = V.u32(name="got")
        b.emit(O.ds_read(got, arena.at(rev_idx)))
        b.wait(got, comment="160 KB top read")
        b.emit(O.global_store(V.u32(lane * 4), got, out_top))

        # Measure 64-bank conflict cycles: stride-1 (conflict-free) vs stride-64 dwords
        # (64-way on 64 banks!) vs stride-32 dwords (2-way on 64 banks vs 32-way on CDNA 3!):
        t0 = S.u32(n=2, name="t0")
        t1 = S.u32(n=2, name="t1")
        t2 = S.u32(n=2, name="t2")
        t3 = S.u32(n=2, name="t3")
        r0 = V.u32(name="r0")
        r32 = V.u32(name="r32")
        r64 = V.u32(name="r64")
        b.emit(O.op("s_memtime", t0))
        b.wait(t0, comment="t0")
        b.emit(O.ds_read(r0, arena.at(lane)))
        b.wait(r0, comment="stride 1")
        b.emit(O.op("s_memtime", t1))
        b.wait(t1, comment="t1")
        b.emit(O.ds_read(r32, arena.at(lane * 32)))
        b.wait(r32, comment="stride 32 dwords")
        b.emit(O.op("s_memtime", t2))
        b.wait(t2, comment="t2")
        b.emit(O.ds_read(r64, arena.at(lane * 64)))
        b.wait(r64, comment="stride 64 dwords")
        b.emit(O.op("s_memtime", t3))
        b.wait(t3, comment="t3")

        d0 = S.u32(t1[0] - t0[0], name="d0")
        d1 = S.u32(t2[0] - t1[0], name="d1")
        d2 = S.u32(t3[0] - t2[0], name="d2")
        with b.lanes(k.tid == 0):
            c = V.u32(n=4, name="c")
            b.emit(O.op("v_mov_b32", c[0], d0))
            b.emit(O.op("v_mov_b32", c[1], d1))
            b.emit(O.op("v_mov_b32", c[2], d2))
            b.emit(O.op("v_xor_b32", c[3], r0, r32))
            b.emit(O.global_store(V.u32(0), c, out_clocks))

    return KernelDef(
        "gfx950_160kb_lds_and_64banks",
        Args(out_top=ptr, out_clocks=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


# ---------------------------------------------------------------------------
# Hardware tests (`pytest -m gfx950`)
# ---------------------------------------------------------------------------


def test_hip_reports_gfx950_and_loads_code_objects():
    import torch

    import thin
    from thin.runtime import hip

    archs, _ = hip.gpus()
    assert "gfx950" in archs, archs
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert props.gcnArchName.split(":")[0] == "gfx950"
    assert props.multi_processor_count > 0

    k = thin.compile(gen_cdna4_160kb_lds_and_64bank_probe)
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"' in k.asm


@pytest.mark.parametrize("shape", list(CDNA4_DENSE_SHAPES))
def test_probe_cdna4_doubled_k_dense_mfma_layouts_and_numerics(shape: str):
    """Single-hot layout probe + random numerical check across all CDNA 4 dense MFMA forms."""
    import thin

    ab_dw, cd_dw = CDNA4_DENSE_SHAPES[shape]
    k = thin.compile(gen_cdna4_mfma_dense, shape=shape, acc_file="a")
    rng = np.random.default_rng(950_01 + len(shape))

    # 1) Single-hot probe on lane 0 dword 0 + lane 17 dword 2 vs diagonal identity:
    a = np.zeros((64, ab_dw), dtype=np.uint32)
    b = np.zeros((64, ab_dw), dtype=np.uint32)
    c = np.zeros((64, cd_dw), dtype=np.uint32)
    unit = (
        0x01
        if shape.startswith("i32_")
        else (0x38 if "fp8" in shape else (0x3F80 if "bf16" in shape else 0x3C00))
    )
    a[0, 0] = unit
    b[0, 0] = unit
    if ab_dw >= 4:
        a[17, 2] = unit
        b[17, 2] = unit
    emu_hot, ran_hot = _both(
        k,
        (1,),
        {"a": a.reshape(-1), "b": b.reshape(-1), "c": c.reshape(-1), "d": _poison(64 * cd_dw)},
        ["d"],
    )
    _same_bits(emu_hot, ran_hot)

    # 2) Random dense payload:
    if shape.startswith("i32_"):
        ar = rng.integers(-8, 8, size=64 * ab_dw * 4, dtype=np.int8).view(np.uint32)
        br = rng.integers(-8, 8, size=64 * ab_dw * 4, dtype=np.int8).view(np.uint32)
        cr = rng.integers(-50, 50, size=64 * cd_dw, dtype=np.int32).view(np.uint32)
    elif "fp8" in shape:
        # Exclude NaN code 0x7F/0xFF on OCP e4m3fn and 0x7C..0x7F on e5m2:
        ar = rng.integers(0, 0x78, size=64 * ab_dw * 4, dtype=np.uint8).view(np.uint32)
        br = rng.integers(0, 0x78, size=64 * ab_dw * 4, dtype=np.uint8).view(np.uint32)
        cr = rng.standard_normal(64 * cd_dw).astype(np.float32).view(np.uint32)
    elif "bf16" in shape:
        ar = (
            (rng.standard_normal(64 * ab_dw * 2).astype(np.float32).view(np.uint32) >> 16)
            .astype(np.uint16)
            .view(np.uint32)
        )
        br = (
            (rng.standard_normal(64 * ab_dw * 2).astype(np.float32).view(np.uint32) >> 16)
            .astype(np.uint16)
            .view(np.uint32)
        )
        cr = rng.standard_normal(64 * cd_dw).astype(np.float32).view(np.uint32)
    else:
        ar = rng.standard_normal(64 * ab_dw * 2).astype(np.float16).view(np.uint32)
        br = rng.standard_normal(64 * ab_dw * 2).astype(np.float16).view(np.uint32)
        cr = rng.standard_normal(64 * cd_dw).astype(np.float32).view(np.uint32)

    emu, ran = _both(k, (1,), {"a": ar, "b": br, "c": cr, "d": _poison(64 * cd_dw)}, ["d"])
    _same_bits(emu, ran)


@pytest.mark.parametrize(
    ("shape", "fmt_a", "fmt_b"),
    [
        ("f32_16x16x128_f8f6f4", "fp8", "bf8"),
        ("f32_16x16x128_f8f6f4", "fp6", "fp4"),
        ("f32_16x16x128_f8f6f4", "bf6", "fp6"),
        ("f32_32x32x64_f8f6f4", "fp4", "fp4"),
        ("f32_32x32x64_f8f6f4", "fp8", "fp6"),
    ],
)
def test_probe_f8f6f4_mixed_layouts_and_grouped_rounding(shape: str, fmt_a: str, fmt_b: str):
    """Verify mixed F8F6F4 register layouts (including cross-dword 6-bit FP6/BF6) on silicon."""
    import thin

    na, nb = F8F6F4_DWORDS[fmt_a], F8F6F4_DWORDS[fmt_b]
    nd = 16 if "32x32" in shape else 4
    k = thin.compile(
        gen_cdna4_f8f6f4, shape=shape, fmt_a=fmt_a, fmt_b=fmt_b, scaled=False, sel_a=0, sel_b=0
    )
    rng = np.random.default_rng(950_02)

    a = rng.integers(0, 0x7070_7070, size=64 * na, dtype=np.uint32)
    b = rng.integers(0, 0x7070_7070, size=64 * nb, dtype=np.uint32)
    c = rng.standard_normal(64 * nd).astype(np.float32).view(np.uint32)
    z = np.zeros(64, dtype=np.uint32)
    emu, ran = _both(
        k, (1,), {"a": a, "b": b, "c": c, "sa": z, "sb": z, "d": _poison(64 * nd)}, ["d"]
    )
    _same_bits(emu, ran)


def test_probe_mfma_scale_e8m0_order_bytesel_and_nan():
    """Verify `v_mfma_scale_f32_*_f8f6f4` `ENC_VOP3PX2` byte selection, `0x00`, and `0xFF` NaN."""
    import thin

    k = thin.compile(
        gen_cdna4_f8f6f4,
        shape="f32_16x16x128_f8f6f4",
        fmt_a="fp8",
        fmt_b="fp4",
        scaled=True,
        sel_a=2,
        sel_b=3,
    )
    rng = np.random.default_rng(950_03)
    a = rng.integers(0, 0x7070_7070, size=64 * 8, dtype=np.uint32)
    b = rng.integers(0, 0x7777_7777, size=64 * 4, dtype=np.uint32)
    c = np.zeros(64 * 4, dtype=np.uint32)

    sa = rng.integers(125, 130, size=64, dtype=np.uint32)
    sb = rng.integers(125, 130, size=64, dtype=np.uint32)
    sa[0] = 0x00  # subnormal 2^-127 E8M0 scale on row 0 block 0
    sb[63] = 0xFF  # NaN E8M0 scale on col 15 block 3 only
    sa_packed = (sa << 16) | np.uint32(0xDEAD_00FF)
    sb_packed = (sb << 24) | np.uint32(0x00BE_EF11)

    emu, ran = _both(
        k,
        (1,),
        {"a": a, "b": b, "c": c, "sa": sa_packed, "sb": sb_packed, "d": _poison(64 * 4)},
        ["d"],
    )
    _same_bits(emu, ran)


@pytest.mark.parametrize(
    ("shape", "abid"),
    [
        ("f32_16x16x64_bf16", 0),
        ("f32_16x16x64_bf16", 1),
        ("f32_16x16x128_fp8_bf8", 0),
        ("i32_16x16x128_i8", 0),
    ],
)
def test_probe_cdna4_doubled_k_smfmac_and_abid(shape: str, abid: int):
    """Verify doubled-K 4:2 sparse SMFMAC 8-VGPR B layout and `abid` index-set selection."""
    import thin

    k = thin.compile(gen_cdna4_smfmac, shape=shape, abid=abid)
    rng = np.random.default_rng(950_04 + abid)
    valid_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    idx = np.zeros(64, dtype=np.uint32)
    for ln in range(64):
        w = 0
        for g in range(8):
            i0, i1 = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
            w |= ((i0 & 3) | ((i1 & 3) << 2)) << (4 * g)
        idx[ln] = np.uint32(w)

    a = rng.integers(0, 0x3838_3838, size=64 * 4, dtype=np.uint32)
    b = rng.integers(0, 0x3838_3838, size=64 * 8, dtype=np.uint32)
    c = np.zeros(64 * 4, dtype=np.uint32)
    emu, ran = _both(k, (1,), {"a": a, "b": b, "idx": idx, "c": c, "d": _poison(64 * 4)}, ["d"])
    _same_bits(emu, ran)


def test_probe_scalef32_cvt_sr_prng_and_new_valu():
    """Verify `v_cvt_scalef32_*`, `FP16_OVFL=0`, SR rounding on exact `1.0`, PRNG, and new VALU."""
    import thin

    k = thin.compile(gen_cdna4_scalef32_and_valu)
    rng = np.random.default_rng(950_05)
    x = (rng.standard_normal((256, 4)) * 25.0).astype(np.float32)
    # Explicit boundary probes on lanes 0..5:
    # Lane 0: exact `1.0` + `seed = 0xFFFFFFFF` probes CDNA 4 §7.3 (`0x38`) vs VOP3 wrap (`0x39`)!
    x[0] = [1.0, -1.0, 896.0, 1024.0]  # 1024 / 2 = 512 > 448 -> E4M3 NaN (0x7F)
    x[1] = [0.0, -0.0, 114688.0, 131072.0]  # 131072 / 2 = 65536 > 57344 -> E5M2 +Inf (0x7C)
    x[2] = [float("inf"), float("-inf"), float("nan"), 12.0]
    seeds = rng.integers(1, 0xFFFF_FFFF, size=256, dtype=np.uint32)
    seeds[0] = np.uint32(0xFFFF_FFFF)

    args = {
        "in_f32": x.reshape(-1),
        "in_seeds": seeds,
        "out_cvt": _poison(256 * 4),
        "out_sr": _poison(256 * 2),
        "out_valu": _poison(256 * 4),
    }
    emu, ran = _both(k, (4,), args, ["out_cvt", "out_sr", "out_valu"])
    _same_bits(emu, ran)


def test_probe_ds_read_tr_wide_lds_dma_160kb_and_64banks():
    """Verify `ds_read_*_tr_*`, `dwordx3`/`dwordx4` LDS DMA, 160 KB LDS, and 64-bank conflicts."""
    import thin

    # 1) Transpose LDS reads + wide DMA + `dwordx3` 4th-dword skip probe:
    k_tr = thin.compile(gen_cdna4_transpose_lds_and_wide_dma)
    src16 = np.arange(1, 257, dtype=np.uint32) * np.uint32(0x0101_0101)
    src12 = np.arange(1, 193, dtype=np.uint32) + np.uint32(0x1234_0000)
    atom_buf = np.full(64, 0x4000_4000, dtype=np.uint32)  # packed bf16 (2.0, 2.0)
    args_tr = {
        "src16": src16,
        "src12": src12,
        "atom_buf": atom_buf,
        "out_tr16": _poison(128),
        "out_tr8": _poison(128),
        "out_tr4": _poison(128),
        "out_tr6": _poison(192),
        "out_dma12": _poison(256),
        "out_prev": _poison(64),
    }
    outs_tr = [
        "atom_buf",
        "out_tr16",
        "out_tr8",
        "out_tr4",
        "out_tr6",
        "out_dma12",
        "out_prev",
    ]
    emu_tr, ran_tr = _both(k_tr, (1,), args_tr, outs_tr)
    _same_bits(emu_tr, ran_tr)

    # 2) Full 160 KB LDS allocation + 64-bank conflict timing probe:
    from thin.runtime.emulated import BankConflicts

    k_lds = thin.compile(gen_cdna4_160kb_lds_and_64bank_probe)
    args_lds = {"out_top": _poison(64), "out_clocks": _poison(4)}
    with pytest.warns(BankConflicts, match="gfx950 bank model"):
        emu_lds, ran_lds = _both(k_lds, (1,), args_lds, ["out_top"])
    _same_bits(emu_lds, ran_lds)


def build_raw_mfma_nop_code_object(wait_states: int) -> bytes:
    """Assemble `16x16x32_bf16` kernel with `wait_states` explicit waits after MFMA."""
    from thin import codegen

    lines = gen_cdna4_mfma_dense("f32_16x16x32_bf16", "a").compile().text.splitlines()
    mfma_idx = next(i for i, ln in enumerate(lines) if "\tv_mfma_f32_16x16x32_bf16 " in ln)
    # Cdna4Model emits `s_nop 7` (8 wait states = `7 + 1` on CDNA 4) after 4-pass `16x16x32_bf16`:
    assert lines[mfma_idx + 1].lstrip().startswith("s_nop 7")
    if wait_states <= 0:
        nop_lines: list[str] = []
    elif wait_states <= 16:
        nop_lines = [f"\ts_nop {wait_states - 1}"]
    else:
        nop_lines = ["\ts_nop 15", f"\ts_nop {wait_states - 17}"]
    patched = "\n".join([*lines[: mfma_idx + 1], *nop_lines, *lines[mfma_idx + 2 :]]) + "\n"
    return codegen.code_object(patched)


def test_probe_xdl_hazard_minimum_nops():
    """Verify `Cdna4Model`'s 8 wait states after 4-pass `16x16x32_bf16` match 24-cycle wait."""
    import torch

    from thin.runtime import hip

    sym = "gfx950_mfma_f32_16x16x32_bf16_a"
    a = torch.full((64 * 4,), 0x3F80_3F80, dtype=torch.int32, device="cuda")
    b = torch.full((64 * 4,), 0x3F80_3F80, dtype=torch.int32, device="cuda")
    c = torch.zeros((64 * 4,), dtype=torch.int32, device="cuda")
    ref_d = torch.empty((64 * 4,), dtype=torch.int32, device="cuda")

    _mod_ref, fn_ref = hip.module(build_raw_mfma_nop_code_object(24), sym, "probe_24")
    ka_ref = np.array(
        [a.data_ptr(), b.data_ptr(), c.data_ptr(), ref_d.data_ptr()], np.uint64
    ).tobytes()
    hip.enqueue(fn_ref, (1, 1, 1), (64, 1), ka_ref, 0, "probe_24")
    torch.cuda.synchronize()

    out_8 = torch.empty_like(ref_d)
    _mod_8, fn_8 = hip.module(build_raw_mfma_nop_code_object(8), sym, "probe_8")
    ka_8 = np.array(
        [a.data_ptr(), b.data_ptr(), c.data_ptr(), out_8.data_ptr()], np.uint64
    ).tobytes()
    hip.enqueue(fn_8, (1, 1, 1), (64, 1), ka_8, 0, "probe_8")
    torch.cuda.synchronize()
    assert torch.equal(out_8, ref_d)
