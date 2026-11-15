"""MI300X (gfx942 / CDNA 3) hardware validation suite against ``thin.emu``.

Run on an MI300X machine with::

    pytest -m gfx942 -v

Every test compiles through ``thin.compile`` (exercising DSL -> IR -> RA ->
Hazards -> Render -> Encode -> Code Object), runs the same inputs on both
``thin.emu`` and the MI300X GPU via HIP + PyTorch, and asserts bit-exact
parity.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gfx942


@pytest.fixture(autouse=True)
def _no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _poison(n: int, dtype=np.uint32) -> np.ndarray:
    return np.full(n, 0xFFFFFFFF, dtype=np.uint32).view(dtype).copy()


def _both(k, grid: tuple[int, ...], args: dict, outs: list[str]) -> tuple[dict, dict]:
    """Run ``k`` on both ``thin.emu`` and the active ``gfx942`` GPU; return ``(emu, gpu)``."""
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


def gen_collectives_and_clocks():
    from thin import KernelDef
    from thin.dsl import Args, O, across, ptr, scan

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        src, out_red, out_max, out_scan, clocks = k.load()
        off = V.u32(k.tid * 4, name="off")
        x = V.f32(name="x")
        b.emit(O.global_load(x, off, src))
        b.wait(x, comment="x landed")

        t0 = S.u32(n=2, name="t0")
        rt0 = S.u32(n=2, name="rt0")
        b.emit(O.op("s_memtime", t0))
        b.emit(O.op("s_memrealtime", rt0))
        b.wait(t0, rt0, comment="clocks before")

        red = V.f32(across(x, "add", 64), name="red")
        mx = V.f32(across(x, "max", 64), name="mx")
        sc = V.f32(scan(x, "add", 64), name="sc")
        b.emit(O.global_store(off, red, out_red))
        b.emit(O.global_store(off, mx, out_max))
        b.emit(O.global_store(off, sc, out_scan))

        t1 = S.u32(n=2, name="t1")
        rt1 = S.u32(n=2, name="rt1")
        b.emit(O.op("s_memtime", t1))
        b.emit(O.op("s_memrealtime", rt1))
        b.wait(t1, rt1, comment="clocks after")

        with b.lanes(k.tid == 0):
            z = V.u32(0, name="z")
            c = V.u32(n=4, name="c")
            b.emit(O.op("v_mov_b32", c[0], t0[0]))
            b.emit(O.op("v_mov_b32", c[1], t1[0]))
            b.emit(O.op("v_mov_b32", c[2], rt0[0]))
            b.emit(O.op("v_mov_b32", c[3], rt1[0]))
            b.emit(O.global_store(z, c, clocks))

    return KernelDef(
        "gfx942_collectives_clocks",
        Args(src=ptr, out_red=ptr, out_max=ptr, out_scan=ptr, clocks=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


DENSE_SHAPES: dict[str, tuple[int, int]] = {
    "f32_32x32x8_f16": (2, 16),
    "f32_16x16x16_f16": (2, 4),
    "f32_16x16x16_bf16": (2, 4),
    "f32_16x16x4_f32": (1, 4),
    "f32_16x16x8_xf32": (2, 4),
    "i32_16x16x32_i8": (2, 4),
    "f32_16x16x32_fp8_fp8": (2, 4),
    "f32_16x16x32_fp8_bf8": (2, 4),
    "f32_16x16x32_bf8_bf8": (2, 4),
    "f32_4x4x4_16b_f16": (2, 4),
}


def gen_mfma_dense(shape: str, acc_file: str):
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    ab_dw, cd_dw = DENSE_SHAPES[shape]

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
        f"mfma_{shape}_{acc_file}",
        Args(a=ptr, b=ptr, c=ptr, d=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_mfma_bcast_f64_and_hazards():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, A = b.v, b.a  # noqa: N806
        pa, pb, pf64, out_bcast, out_f64, out_chain = k.load()
        lane = V.u32(k.tid, name="lane")
        off8 = V.u32(lane * 8, name="off8")
        off16 = V.u32(lane * 16, name="off16")
        off32 = V.u32(lane * 32, name="off32")

        ra = V.u32(n=2, name="ra")
        rb = V.u32(n=2, name="rb")
        da = V.u32(n=2, name="da")
        db = V.u32(n=2, name="db")
        dc = V.u32(n=8, name="dc")
        b.emit(O.global_load(ra, off8, pa))
        b.emit(O.global_load(rb, off8, pb))
        b.emit(O.global_load(da, off8, pf64))
        b.emit(O.global_load(db, off8, pf64, offset=512))
        b.emit(O.global_load(dc[0:4], off32, pf64, offset=1024))
        b.emit(O.global_load(dc[4:8], off32, pf64, offset=1040))
        b.wait(ra, rb, da, db, dc, comment="inputs landed")

        # 1) Broadcast modifiers cbsz:1 abid:1 blgp:2 on multi-block 4x4x4_16b_f16:
        acc_bc = b.mfma("f32_4x4x4_16b_f16", ra, rb, cbsz=1, abid=1, blgp=2, name="acc_bc")
        v_bc = b.read_acc(acc_bc, name="v_bc")
        b.emit(O.global_store(off16, v_bc, out_bcast))

        # 2) Double-precision v_mfma_f64_16x16x4_f64 with neg:[1,0,1]:
        acc_d = A.u32(n=8, name="acc_d")
        for i in range(8):
            b.emit(O.op("v_accvgpr_write_b32", acc_d[i], dc[i]))
        b.emit(O.mfma("f64_16x16x4_f64", acc_d, da, db, acc_d, neg=[1, 0, 1]))
        vd = b.read_acc(acc_d, name="vd")
        b.emit(O.global_store(off32, vd[0:4], out_f64))
        b.emit(O.global_store(off32, vd[4:8], out_f64, offset=16))

        # 3) Back-to-back multi-pass MFMA -> AGPR read -> VALU -> AGPR write -> MFMA hazard chain:
        c0 = b.mfma("f32_16x16x16_f16", ra, rb, name="c0")
        b.mfma("f32_16x16x16_f16", rb, ra, c0)
        vc = b.read_acc(c0, name="vc")
        vc0 = V.f32(vc[0] + 1.0, name="vc0")
        b.emit(O.op("v_accvgpr_write_b32", c0[0], vc0))
        b.mfma("f32_16x16x16_f16", ra, rb, c0)
        v_done = b.read_acc(c0, name="v_done")
        b.emit(O.global_store(off16, v_done, out_chain))

    return KernelDef(
        "gfx942_mfma_bcast_f64_hazards",
        Args(pa=ptr, pb=ptr, pf64=ptr, out_bcast=ptr, out_f64=ptr, out_chain=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


SPARSE_SHAPES = (
    "f32_16x16x32_f16",
    "f32_16x16x32_bf16",
    "i32_16x16x64_i8",
    "f32_16x16x64_fp8_fp8",
)


def gen_smfmac(shape: str):
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        pa, pb, pidx, pc, pd = k.load()
        lane = V.u32(k.tid, name="lane")
        ra = V.u32(n=2, name="ra")
        rb = V.u32(n=4, name="rb")
        ridx = V.u32(name="ridx")
        rc = V.u32(n=4, name="rc")
        b.emit(O.global_load(ra, V.u32(lane * 8), pa))
        b.emit(O.global_load(rb, V.u32(lane * 16), pb))
        b.emit(O.global_load(ridx, V.u32(lane * 4), pidx))
        b.emit(O.global_load(rc, V.u32(lane * 16), pc))
        b.wait(ra, rb, ridx, rc, comment="sparse inputs landed")
        acc = b.write_acc(rc, acc="a", name="acc")
        b.smfmac(shape, ra, rb, ridx, acc)
        vd = b.read_acc(acc, name="vd")
        b.emit(O.global_store(V.u32(lane * 16), vd, pd))

    return KernelDef(
        f"smfmac_{shape}",
        Args(a=ptr, b=ptr, idx=ptr, c=ptr, d=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_fnuz_fp8_bf8_cvt():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        x0, x1, x2, x3, out_fp8, out_bf8 = k.load()
        off = V.u32(k.tid * 4, name="off")
        a0, a1, a2, a3 = V.f32(), V.f32(), V.f32(), V.f32()
        b.emit(O.global_load(a0, off, x0))
        b.emit(O.global_load(a1, off, x1))
        b.emit(O.global_load(a2, off, x2))
        b.emit(O.global_load(a3, off, x3))
        b.wait(a0, a1, a2, a3, comment="f32s landed")

        p_fp8 = V.u32(0xDEAD_BEEF, name="p_fp8")
        b.emit(O.op("v_cvt_pk_fp8_f32", p_fp8, a0, a1))
        b.emit(O.op("v_cvt_pk_fp8_f32", p_fp8, a2, a3, op_sel=[0, 0, 1]))

        p_bf8 = V.u32(0xCAFE_F00D, name="p_bf8")
        b.emit(O.op("v_cvt_pk_bf8_f32", p_bf8, a0, a1))
        b.emit(O.op("v_cvt_pk_bf8_f32", p_bf8, a2, a3, op_sel=[0, 0, 1]))

        b.emit(O.global_store(off, p_fp8, out_fp8))
        b.emit(O.global_store(off, p_bf8, out_bf8))

    return KernelDef(
        "gfx942_fnuz_cvt",
        Args(x0=ptr, x1=ptr, x2=ptr, x3=ptr, out_fp8=ptr, out_bf8=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_vmem_to_lds_and_ds2():
    from thin import KernelDef
    from thin.dsl import LGKM, Args, O, ptr
    from thin.ir import LdsAccess

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806
        src_g, src_b, out_g, out_b, out_ds2 = k.load()
        lds = b.lds()
        pad = lds.region("pad", (64,), elem="u32")
        tile_g = lds.region("tile_g", (64,), elem="u32")
        tile_b = lds.region("tile_b", (64,), elem="u32")
        tile_2 = lds.region("tile_2", (128,), elem="u32")

        lane = V.u32(k.tid, name="lane")
        off4 = V.u32(lane * 4, name="off4")
        off16 = V.u32(lane * 16, name="off16")

        # Touch pad so tile_g / tile_b sit at non-zero LDS base addresses:
        b.emit(O.ds_write(pad.at(lane), lane))
        b.wait(pad, comment="pad init")

        # 1) Global -> LDS DMA (`global_load_lds_dword`):
        b.emit(O.global_load(tile_g.at(0), off4, src_g))

        # 2) Buffer -> LDS DMA (`buffer_load_dword ... offen lds`) with CDNA3 SRD (0x00020000):
        rsrc = S.u32(n=4, name="rsrc")
        b.emit(O.op("s_mov_b32", rsrc[0], src_b.lo))
        b.emit(O.op("s_and_b32", rsrc[1], src_b.hi, 0xFFFF))
        b.emit(O.op("s_mov_b32", rsrc[2], 256))
        b.emit(O.op("s_mov_b32", rsrc[3], 0x0002_0000))
        b.emit(O.buffer_load(tile_b.at(0), off4, rsrc))

        # Both VMEM->LDS DMAs retire on `vmcnt(0)`:
        b.wait(tile_g, tile_b, comment="DMA tiles landed in LDS")

        # Read reversed lane from each DMA tile + exercise ds_write2_b32 / ds_read2_b32:
        rev = V.u32(63 - lane, name="rev")
        vg, vb = V.u32(name="vg"), V.u32(name="vb")
        b.emit(O.ds_read(vg, tile_g.at(rev)))
        b.emit(O.ds_read(vb, tile_b.at(rev)))
        b.wait(vg, vb, comment="LDS reads landed")
        b.emit(O.global_store(off4, vg, out_g))
        b.emit(O.global_store(off4, vb, out_b))

        addr2 = V.u32(tile_2.base + lane * 4, name="addr2")
        w2 = O.op("ds_write2_b32", addr2, vg, vb, offset0=0, offset1=64)
        w2.put(LdsAccess(tile_2, "tile_2"))
        b.emit(w2)
        b.wait(LGKM, comment="ds_write2")
        pair = V.u32(n=2, name="pair")
        r2 = O.op("ds_read2_b32", pair, addr2, offset0=64, offset1=0)
        r2.put(LdsAccess(tile_2, "tile_2"))
        b.emit(r2)
        b.wait(pair, comment="ds_read2")

        # Swizzled 16-byte `ds_read_b128` across tile_g (64 dwords = 16x4 dwords):
        grp = V.u32(lane & 15, name="grp")
        b128 = V.u32(n=4, name="b128")
        b.emit(O.ds_read(b128, tile_g.at(grp * 4)))
        b.wait(b128, comment="b128 landed")
        merged = V.u32(n=4, name="merged")
        b.emit(O.op("v_add_u32", merged[0], pair[0], b128[0]))
        b.emit(O.op("v_add_u32", merged[1], pair[1], b128[1]))
        b.emit(O.op("v_mov_b32", merged[2], b128[2]))
        b.emit(O.op("v_mov_b32", merged[3], b128[3]))
        b.emit(O.global_store(off16, merged, out_ds2))

    return KernelDef(
        "gfx942_vmem_lds_ds2",
        Args(src_g=ptr, src_b=ptr, out_g=ptr, out_b=ptr, out_ds2=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_dpp_and_packed_f32():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        src_p, out_dpp, out_pk = k.load()
        lane = V.u32(k.tid, name="lane")
        off8 = V.u32(lane * 8, name="off8")
        off32 = V.u32(lane * 32, name="off32")

        xy = V.u32(n=2, name="xy")
        zw = V.u32(n=2, name="zw")
        b.emit(O.global_load(xy, off8, src_p))
        b.emit(O.global_load(zw, off8, src_p, offset=512))
        b.wait(xy, zw, comment="inputs landed")

        # 1) GFX9 DPP controls:
        #    [0]: 64-lane inclusive prefix sum via row_shr + row_bcast:15 + row_bcast:31
        #    [1]: row_bcast:15 with row_mask:0xf & bound_ctrl:0 (OOB row 0 kept, row 2->31)
        #    [2]: row_bcast:31 with row_mask:0xf & bound_ctrl:1 (OOB rows 0..1->0, 2..3->31)
        #    [3]: wave_ror:1 -> wave_shl:1 (bound_ctrl:1) -> row_newbcast:7
        d_out = V.u32(n=4, name="d_out")
        b.emit(O.op("v_mov_b32", d_out[0], xy[0]))
        for s in (1, 2, 4, 8):
            b.emit(
                O.op(
                    "v_add_f32_dpp",
                    d_out[0],
                    d_out[0],
                    d_out[0],
                    dpp=f"row_shr:{s}",
                    row_mask=0xF,
                    bank_mask=0xF,
                    bound_ctrl=True,
                )
            )
        b.emit(
            O.op(
                "v_add_f32_dpp",
                d_out[0],
                d_out[0],
                d_out[0],
                dpp="row_bcast:15",
                row_mask=0xA,
                bank_mask=0xF,
            )
        )
        b.emit(
            O.op(
                "v_add_f32_dpp",
                d_out[0],
                d_out[0],
                d_out[0],
                dpp="row_bcast:31",
                row_mask=0xC,
                bank_mask=0xF,
            )
        )

        b.emit(O.op("v_mov_b32", d_out[1], 0xC2C6_0000))  # -99.0f
        b.emit(
            O.op("v_mov_b32_dpp", d_out[1], xy[0], dpp="row_bcast:15", row_mask=0xF, bank_mask=0xF)
        )

        b.emit(
            O.op(
                "v_mov_b32_dpp",
                d_out[2],
                xy[0],
                dpp="row_bcast:31",
                row_mask=0xF,
                bank_mask=0xF,
                bound_ctrl=True,
            )
        )

        tmp = V.u32(name="tmp")
        b.emit(
            O.op(
                "v_mov_b32_dpp",
                tmp,
                xy[0],
                dpp="wave_ror:1",
                row_mask=0xF,
                bank_mask=0xF,
                bound_ctrl=True,
            )
        )
        b.emit(
            O.op(
                "v_add_f32_dpp",
                d_out[3],
                tmp,
                xy[1],
                dpp="wave_shl:1",
                row_mask=0xF,
                bank_mask=0xF,
                bound_ctrl=True,
            )
        )
        b.emit(
            O.op(
                "v_add_f32_dpp",
                d_out[3],
                d_out[3],
                d_out[3],
                dpp="row_newbcast:7",
                row_mask=0xF,
                bank_mask=0xF,
            )
        )
        b.emit(O.global_store(V.u32(lane * 16), d_out, out_dpp))

        # 2) Packed f32 / b32 VOP3P ops:
        p0 = V.u32(n=2, name="p0")
        p1 = V.u32(n=2, name="p1")
        p2 = V.u32(n=2, name="p2")
        p3 = V.u32(n=2, name="p3")
        b.emit(
            O.op(
                "v_pk_fma_f32",
                p0,
                xy,
                zw,
                xy,
                op_sel=[1, 0, 1],
                op_sel_hi=[0, 1, 0],
                neg_lo=[1, 0, 0],
                neg_hi=[0, 1, 1],
                clamp=True,
            )
        )
        b.emit(O.op("v_pk_mul_f32", p1, xy, zw, op_sel=[0, 1], op_sel_hi=[1, 0], neg_hi=[0, 1]))
        b.emit(O.op("v_pk_add_f32", p2, xy, 1.0, op_sel_hi=[1, 0], neg_lo=[1, 0], clamp=True))
        # Copy then in-place swap via v_pk_mov_b32 (`op_sel_hi` ignored on hardware!):
        b.emit(O.op("v_pk_mov_b32", p3, xy, zw, op_sel=[0, 1], op_sel_hi=[0, 0]))
        b.emit(O.op("v_pk_mov_b32", p3, p3, p3, op_sel=[1, 0]))

        b.emit(O.global_store(off32, p0, out_pk, offset=0))
        b.emit(O.global_store(off32, p1, out_pk, offset=8))
        b.emit(O.global_store(off32, p2, out_pk, offset=16))
        b.emit(O.global_store(off32, p3, out_pk, offset=24))

    return KernelDef(
        "gfx942_dpp_and_pk_f32",
        Args(src_p=ptr, out_dpp=ptr, out_pk=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def gen_multi_xcd_atomics():
    from thin import KernelDef
    from thin.dsl import VM, Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        counts, per_wg = k.load()
        bucket = V.u32((k.tid & 15) * 4, name="bucket")
        val = V.u32((k.wgid + 1), name="val")
        b.emit(O.op("global_atomic_add", bucket, val, counts, sc1=True))
        b.wait(VM, comment="atomic completed")
        b.emit(O.op("buffer_wbl2", sc1=True))
        b.wait(VM, comment="wbl2 completed")
        with b.lanes(k.tid == 0):
            woff = V.u32(k.wgid * 4, name="woff")
            b.emit(O.global_store(woff, V.u32(k.wgid + 1000), per_wg))

    return KernelDef(
        "gfx942_multi_xcd_atomics",
        Args(counts=ptr, per_wg=ptr),
        body,
        wgsize=256,
        max_grid=(64,),
    )


# ---------------------------------------------------------------------------
# Hardware tests (run with `pytest -m gfx942`)
# ---------------------------------------------------------------------------


def test_hip_reports_gfx942_and_loads_code_objects():
    import torch

    import thin
    from examples.vadd import build_vadd
    from thin.runtime import hip

    archs, version = hip.gpus()
    assert "gfx942" in archs, archs
    assert version >= 60000000, version
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    assert props.gcnArchName.split(":")[0] == "gfx942"
    assert props.warp_size == 64

    k_vgpr = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    mod0, fn0 = hip.module(k_vgpr.co, k_vgpr.name, k_vgpr.name)
    assert mod0.value and fn0.value

    k_agpr = thin.compile(gen_mfma_dense, shape="f32_32x32x8_f16", acc_file="a")
    assert k_agpr.manifest.agpr >= 16
    mod1, fn1 = hip.module(k_agpr.co, k_agpr.name, k_agpr.name)
    assert mod1.value and fn1.value


@pytest.mark.parametrize("n", [1, 63, 64, 65, 256, 1029, 1 << 16])
def test_wave64_vadd_and_predication(n: int):
    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    rng = np.random.default_rng(94200 + n)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    edge = np.array([1e-40, -0.0, np.inf, np.nan], dtype=np.float32)
    x[: min(n, len(edge))] = edge[: min(n, len(edge))]
    args = {"x": x, "y": y, "z": _poison(n, np.float32), "n": n}
    emu, ran = _both(k, (-(-n // 256),), args, ["z"])
    _same_bits(emu, ran)


def test_wave64_collectives_and_clocks():
    import thin

    k = thin.compile(gen_collectives_and_clocks)
    x = (np.arange(64, dtype=np.float32) - 20.5) * 0.75
    args = {
        "src": x,
        "out_red": _poison(64, np.float32),
        "out_max": _poison(64, np.float32),
        "out_scan": _poison(64, np.float32),
        "clocks": np.zeros(4, dtype=np.uint32),
    }
    emu, ran = _both(k, (1,), args, ["out_red", "out_max", "out_scan", "clocks"])
    _same_bits(
        {m: emu[m] for m in ("out_red", "out_max", "out_scan")},
        {m: ran[m] for m in ("out_red", "out_max", "out_scan")},
    )
    # Hardware clock counters must be non-zero and advance monotonically (mod 2^32):
    c_gpu = ran["clocks"]
    assert ((int(c_gpu[1]) - int(c_gpu[0])) & 0xFFFFFFFF) > 0
    assert ((int(c_gpu[3]) - int(c_gpu[2])) & 0xFFFFFFFF) >= 0


@pytest.mark.parametrize(
    ("shape", "acc_file"),
    [
        ("f32_32x32x8_f16", "a"),
        ("f32_16x16x16_f16", "a"),
        ("f32_16x16x16_f16", "v"),
        ("f32_16x16x16_bf16", "a"),
        ("f32_16x16x4_f32", "a"),
        ("f32_16x16x8_xf32", "a"),
        ("i32_16x16x32_i8", "a"),
        ("f32_16x16x32_fp8_fp8", "a"),
        ("f32_16x16x32_fp8_bf8", "a"),
        ("f32_16x16x32_bf8_bf8", "v"),
        ("f32_4x4x4_16b_f16", "a"),
    ],
)
def test_mfma_dense_shapes_agpr_and_vgpr(shape: str, acc_file: str):
    import thin
    from thin import formats

    ab_dw, cd_dw = DENSE_SHAPES[shape]
    rng = np.random.default_rng(abs(hash((shape, acc_file))) % (1 << 31))

    if shape.endswith("_f16"):
        a = rng.uniform(-2.0, 2.0, size=64 * ab_dw * 2).astype(np.float16).view(np.uint32)
        b = rng.uniform(-2.0, 2.0, size=64 * ab_dw * 2).astype(np.float16).view(np.uint32)
    elif shape.endswith("_bf16"):
        fa = rng.uniform(-2.0, 2.0, size=64 * ab_dw * 2).astype(np.float32)
        fb = rng.uniform(-2.0, 2.0, size=64 * ab_dw * 2).astype(np.float32)
        a = ((fa.view(np.uint32) >> 16) & 0xFFFF).astype(np.uint16).view(np.uint32)
        b = ((fb.view(np.uint32) >> 16) & 0xFFFF).astype(np.uint16).view(np.uint32)
    elif shape.endswith("_i8"):
        a = rng.integers(-32, 32, size=64 * ab_dw * 4, dtype=np.int8).view(np.uint32)
        b = rng.integers(-32, 32, size=64 * ab_dw * 4, dtype=np.int8).view(np.uint32)
    elif "fp8" in shape or "bf8" in shape:
        dt_a = formats.DTYPES["e4m3fnuz" if "_fp8_" in shape else "e5m2fnuz"]
        dt_b = formats.DTYPES["e4m3fnuz" if shape.endswith("_fp8") else "e5m2fnuz"]
        fa = rng.uniform(-4.0, 4.0, size=64 * ab_dw * 4).astype(np.float32)
        fb = rng.uniform(-4.0, 4.0, size=64 * ab_dw * 4).astype(np.float32)
        a = np.frombuffer(formats.encode(fa, dt_a, saturate=True).tobytes(), dtype=np.uint32).copy()
        b = np.frombuffer(formats.encode(fb, dt_b, saturate=True).tobytes(), dtype=np.uint32).copy()
    else:
        a = rng.uniform(-2.0, 2.0, size=64 * ab_dw).astype(np.float32).view(np.uint32)
        b = rng.uniform(-2.0, 2.0, size=64 * ab_dw).astype(np.float32).view(np.uint32)
        if shape.endswith("_xf32"):
            # Probe bit-12 tie (`1.0 + 2^-11 = 0x3F801000`) to discriminate RTZ truncation vs RNE:
            a[0] = np.uint32(0x3F801000)
            b[0] = np.uint32(0x3F800000)

    if shape.startswith("i32_"):
        c = rng.integers(-100, 100, size=64 * cd_dw, dtype=np.int32).view(np.uint32)
    else:
        c = rng.uniform(-1.0, 1.0, size=64 * cd_dw).astype(np.float32).view(np.uint32)

    k = thin.compile(gen_mfma_dense, shape=shape, acc_file=acc_file)
    args = {"a": a, "b": b, "c": c, "d": _poison(64 * cd_dw)}
    emu, ran = _both(k, (1,), args, ["d"])
    _same_bits(emu, ran)


def test_mfma_broadcast_modifiers_f64_and_hazard_chains():
    import thin

    rng = np.random.default_rng(94205)
    pa = rng.uniform(-2.0, 2.0, size=64 * 4).astype(np.float16).view(np.uint32)
    pb = rng.uniform(-2.0, 2.0, size=64 * 4).astype(np.float16).view(np.uint32)
    da = rng.uniform(-3.0, 3.0, size=64).astype(np.float64)
    db = rng.uniform(-3.0, 3.0, size=64).astype(np.float64)
    dc = rng.uniform(-5.0, 5.0, size=64 * 4).astype(np.float64)
    pf64 = np.concatenate([da, db, dc]).view(np.uint32).copy()

    k = thin.compile(gen_mfma_bcast_f64_and_hazards)
    args = {
        "pa": pa,
        "pb": pb,
        "pf64": pf64,
        "out_bcast": _poison(64 * 4),
        "out_f64": _poison(64 * 8),
        "out_chain": _poison(64 * 4),
    }
    emu, ran = _both(k, (1,), args, ["out_bcast", "out_f64", "out_chain"])
    _same_bits(emu, ran)


@pytest.mark.parametrize("shape", SPARSE_SHAPES)
def test_smfmac_sparse_shapes(shape: str):
    import thin
    from thin import formats

    rng = np.random.default_rng(abs(hash(shape)) % (1 << 31))
    if shape.endswith("_f16"):
        a = rng.uniform(-2.0, 2.0, size=64 * 4).astype(np.float16).view(np.uint32)
        b = rng.uniform(-2.0, 2.0, size=64 * 8).astype(np.float16).view(np.uint32)
    elif shape.endswith("_bf16"):
        fa = rng.uniform(-2.0, 2.0, size=64 * 4).astype(np.float32)
        fb = rng.uniform(-2.0, 2.0, size=64 * 8).astype(np.float32)
        a = ((fa.view(np.uint32) >> 16) & 0xFFFF).astype(np.uint16).view(np.uint32)
        b = ((fb.view(np.uint32) >> 16) & 0xFFFF).astype(np.uint16).view(np.uint32)
    elif shape.endswith("_i8"):
        a = rng.integers(-16, 16, size=64 * 8, dtype=np.int8).view(np.uint32)
        b = rng.integers(-16, 16, size=64 * 16, dtype=np.int8).view(np.uint32)
    else:
        dt_fp8 = formats.DTYPES["e4m3fnuz"]
        fa = rng.uniform(-3.0, 3.0, size=64 * 8).astype(np.float32)
        fb = rng.uniform(-3.0, 3.0, size=64 * 16).astype(np.float32)
        a = np.frombuffer(
            formats.encode(fa, dt_fp8, saturate=True).tobytes(), dtype=np.uint32
        ).copy()
        b = np.frombuffer(
            formats.encode(fb, dt_fp8, saturate=True).tobytes(), dtype=np.uint32
        ).copy()

    # Valid 4:2 selector nibbles (< 2 bits -> 2 indices i < j in 0..3 per 4-bit group):
    valid_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    idx = np.zeros(64, dtype=np.uint32)
    for lane in range(64):
        word = 0
        for g in range(8):
            i0, i1 = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
            word |= ((i0 & 3) | ((i1 & 3) << 2)) << (4 * g)
        idx[lane] = np.uint32(word)

    c = (
        rng.integers(-50, 50, size=64 * 4, dtype=np.int32).view(np.uint32)
        if shape.startswith("i32_")
        else rng.uniform(-1.0, 1.0, size=64 * 4).astype(np.float32).view(np.uint32)
    )
    k = thin.compile(gen_smfmac, shape=shape)
    args = {"a": a, "b": b, "idx": idx, "c": c, "d": _poison(64 * 4)}
    emu, ran = _both(k, (1,), args, ["d"])
    _same_bits(emu, ran)


def test_fnuz_fp8_bf8_conversions():
    import thin

    specials = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            240.0,
            -240.0,
            300.0,
            -300.0,
            57344.0,
            -57344.0,
            1e6,
            -1e6,
            2.0**-10,
            -(2.0**-10),
            2.0**-16,
            -(2.0**-16),
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(94207)
    inputs = []
    for i in range(4):
        v = rng.uniform(-300.0, 300.0, size=64).astype(np.float32)
        v[: len(specials)] = np.roll(specials, i * 5)
        inputs.append(v)

    k = thin.compile(gen_fnuz_fp8_bf8_cvt)
    args = {
        "x0": inputs[0],
        "x1": inputs[1],
        "x2": inputs[2],
        "x3": inputs[3],
        "out_fp8": _poison(64),
        "out_bf8": _poison(64),
    }
    emu, ran = _both(k, (1,), args, ["out_fp8", "out_bf8"])
    _same_bits(emu, ran)


def test_vmem_to_lds_dma_and_ds_ops():
    import thin

    src_g = (np.arange(64, dtype=np.uint32) + 1000) * 3
    src_b = (np.arange(64, dtype=np.uint32) + 2000) * 5
    k = thin.compile(gen_vmem_to_lds_and_ds2)
    args = {
        "src_g": src_g,
        "src_b": src_b,
        "out_g": _poison(64),
        "out_b": _poison(64),
        "out_ds2": _poison(64 * 4),
    }
    emu, ran = _both(k, (1,), args, ["out_g", "out_b", "out_ds2"])
    _same_bits(emu, ran)


def test_gfx9_dpp_and_packed_f32_ops():
    import thin

    rng = np.random.default_rng(94209)
    xy = rng.uniform(-2.0, 2.0, size=(64, 2)).astype(np.float32)
    zw = rng.uniform(-2.0, 2.0, size=(64, 2)).astype(np.float32)
    xy[0, 0] = np.float32(-0.0)
    zw[0, 0] = np.float32(0.0)
    xy[1, 1] = np.float32("nan")
    src_p = np.concatenate([xy.reshape(-1), zw.reshape(-1)]).view(np.uint32).copy()

    k = thin.compile(gen_dpp_and_packed_f32)
    args = {
        "src_p": src_p,
        "out_dpp": _poison(64 * 4),
        "out_pk": _poison(64 * 8),
    }
    emu, ran = _both(k, (1,), args, ["out_dpp", "out_pk"])
    _same_bits(emu, ran)


def test_multi_xcd_sc1_atomics_and_coherence():
    import thin

    k = thin.compile(gen_multi_xcd_atomics)
    args = {
        "counts": np.zeros(16, dtype=np.uint32),
        "per_wg": _poison(64),
    }
    emu, ran = _both(k, (64,), args, ["counts", "per_wg"])
    _same_bits(emu, ran)
    # 64 WGs * 16 threads/bucket * mean(1..64)=32.5 -> each bucket == 16 * 64 * 65 / 2 = 33280
    assert ran["counts"].tolist() == [16 * 64 * 65 // 2] * 16

