"""Tiled bf16 matrix multiply (`C = A @ B`) for 256-CU CDNA 4 (`gfx950`).

Demonstrates the CDNA 4 (`gfx950` / Instinct MI350X & MI355X) instruction set
and 160 KB / 64-bank LDS hierarchy end to end:
- Doubled-K `v_mfma_f32_16x16x32_bf16` (`K = 32` per MFMA, 2 substeps per
  `BK = 64` stage);
- Wide 128-bit `O.buffer_load(..., bytes=16)` (`buffer_load_dwordx4 ... lds`)
  direct HBM-to-LDS DMA for both `A` and `B`, eliminating staging VGPRs and
  `ds_write` instructions;
- Hardware transpose LDS reads `O.ds_read_tr(..., bits=16)`
  (`ds_read_b64_tr_b16`) on row-major `B` tiles, eliminating `v_perm_b32`
  in-register transposes;
- Double-buffered `(A, B)` staging up to 128 KB inside CDNA 4's 160 KB
  (`163,840` B), 64-bank LDS with XOR bank swizzling for zero 64-bank conflicts;
- Hardware RNE `v_cvt_pk_bf16_f32` epilogue conversion and 256-CU (`8 x 32`)
  XCD workgroup remapping.

Note: verified functionally and for zero modeled 64-bank conflicts on
`thin.emu`; tile schedule tuning awaits Instinct MI350X/MI355X silicon.
"""

from __future__ import annotations

import math

from thin import KernelDef, cdiv
from thin.dsl import Args, O, ptr

NWAVES = 4
WGSIZE = 64 * NWAVES
BK = 64

_CONFIGS = (
    (1024, {"BM": 64, "BN": 128, "GROUP_M": 8}),
    (2048, {"BM": 128, "BN": 128, "GROUP_M": 8}),
    (4096, {"BM": 128, "BN": 256, "GROUP_M": 8}),
    (None, {"BM": 256, "BN": 256, "GROUP_M": 8}),
)


def default_gemm_config(m: int) -> dict[str, int]:
    """Per-size tile and L2 grouping parameters for a 256-CU (`8 x 32`) CDNA 4 GPU."""
    for limit, cfg in _CONFIGS:
        if limit is None or m <= limit:
            return dict(cfg)
    raise AssertionError("unreachable")


def _srd(b, base_ptr, num_bytes: int, name: str):
    """128-bit buffer descriptor (`SRD`); hardware clamps accesses past `num_bytes`."""
    desc = b.s.u32(n=4, name=name)
    b.emit(O.op("s_mov_b32", desc[0], base_ptr.lo))
    b.emit(O.op("s_and_b32", desc[1], base_ptr.hi, 0xFFFF))
    b.emit(O.op("s_mov_b32", desc[2], min(num_bytes, 0xFFFF_FFFF)))
    b.emit(O.op("s_mov_b32", desc[3], 0x0002_0000))
    return desc


def _udivmod_const(b, x_reg, d: int, x_max: int, tag: str):
    """Exact unsigned `(x // d, x % d)` on `0 <= x <= x_max` without scalar `s_udiv`."""
    S = b.s  # noqa: N806 - the DSL's spelling
    if d == 1:
        return x_reg, S.u32(0, name=f"{tag}_r")
    if (d & (d - 1)) == 0:
        sh = int(math.log2(d))
        return S.u32(x_reg >> sh, name=f"{tag}_q"), S.u32(x_reg & (d - 1), name=f"{tag}_r")
    mul = (1 << 32) // d + 1
    assert all(((v * mul) >> 32) == (v // d) for v in range(x_max + 1)), (d, x_max)
    q = S.u32(name=f"{tag}_q")
    b.emit(O.op("s_mul_hi_u32", q, x_reg, S.u32(mul, name=f"{tag}_mul")))
    return q, S.u32(x_reg - q * d, name=f"{tag}_r")


def _remap_xcd(b, wgid, num_tiles: int, xcds: int, cus_per_xcd: int):
    """Chunks round-robin hardware `wgid`s per 256-CU wave so each XCD gets adjacent `pid`s."""
    S = b.s  # noqa: N806
    if xcds <= 1 or num_tiles < xcds or num_tiles % xcds != 0:
        return S.u32(wgid, name="pid")
    log_xcds = int(math.log2(xcds))
    wave_cus = xcds * cus_per_xcd
    if num_tiles <= wave_cus:
        xcd = S.u32(wgid & (xcds - 1), name="xcd")
        return S.u32(xcd * (num_tiles // xcds) + (wgid >> log_xcds), name="pid")
    full_wgids = (num_tiles // wave_cus) * wave_cus
    rem = num_tiles - full_wgids
    if rem == 0:
        wq, wr = _udivmod_const(b, wgid, wave_cus, num_tiles, "xccg")
        xcd = S.u32(wr & (xcds - 1), name="xcd")
        return S.u32(wq * wave_cus + xcd * cus_per_xcd + (wr >> log_xcds), name="pid")
    pid = S.u32(name="pid")
    with b.when(wgid < full_wgids, name="xccg_full"):
        wq, wr = _udivmod_const(b, wgid, wave_cus, full_wgids, "xccg")
        xcd_f = S.u32(wr & (xcds - 1), name="xcd_f")
        pid_f = S.u32(wq * wave_cus + xcd_f * cus_per_xcd + (wr >> log_xcds), name="pid_f")
        b.emit(O.op("s_mov_b32", pid, pid_f))
    with b.otherwise():
        wr_t = S.u32(wgid - full_wgids, name="wr_t")
        xcd_t = S.u32(wr_t & (xcds - 1), name="xcd_t")
        pid_t = S.u32(full_wgids + xcd_t * (rem // xcds) + (wr_t >> log_xcds), name="pid_t")
        b.emit(O.op("s_mov_b32", pid, pid_t))
    return pid


def _tile_coords(b, pid, grid_m: int, grid_n: int, group_m: int):
    """Maps linear `pid` to `(tile_m, tile_n)` in `group_m`-tall slabs for L2 reuse."""
    S = b.s  # noqa: N806
    log_gn = int(math.log2(grid_n))
    if group_m <= 1:
        return S.u32(pid >> log_gn, name="tile_m"), S.u32(pid & (grid_n - 1), name="tile_n")
    s3 = S.u32(pid >> log_gn, name="s3")
    s2 = S.u32(pid & (grid_n - 1), name="s2")
    slab_m, rem_s3 = _udivmod_const(b, s3, group_m, grid_m, "wgm_s3")
    idx = S.u32(rem_s3 * grid_n + s2, name="wgm_idx")
    full_groups = grid_m // group_m
    rem_m = grid_m - full_groups * group_m
    if rem_m == 0:
        tn, in_m = _udivmod_const(b, idx, group_m, group_m * grid_n, "wgm_f")
        return S.u32(slab_m * group_m + in_m, name="tile_m"), tn
    tm, tn = S.u32(name="tile_m"), S.u32(name="tile_n")
    with b.when(slab_m < full_groups, name="wgm_full"):
        tn_f, in_f = _udivmod_const(b, idx, group_m, group_m * grid_n, "wgm_f")
        b.emit(O.op("s_mov_b32", tm, S.u32(slab_m * group_m + in_f, name="tm_f")))
        b.emit(O.op("s_mov_b32", tn, tn_f))
    with b.otherwise():
        tn_r, in_r = _udivmod_const(b, idx, rem_m, rem_m * grid_n, "wgm_t")
        b.emit(O.op("s_mov_b32", tm, S.u32(full_groups * group_m + in_r, name="tm_r")))
        b.emit(O.op("s_mov_b32", tn, tn_r))
    return tm, tn


def build_gemm_bf16(
    M: int,  # noqa: N803 - build parameters
    N: int | None = None,  # noqa: N803
    K: int | None = None,  # noqa: N803
    *,
    BM: int | None = None,  # noqa: N803
    BN: int | None = None,  # noqa: N803
    GROUP_M: int | None = None,  # noqa: N803
    XCDS: int = 8,  # noqa: N803
    CUS_PER_XCD: int = 32,  # noqa: N803
) -> KernelDef:
    """Row-major `C[M, N] = A[M, K] @ B[K, N]` with bf16 inputs/output and f32 accumulation.

    Design for a 256-CU (`8 x 32`) CDNA 4 GPU (`gfx950`, 160 KB / 64-bank LDS):
    - Four waves form a `2 x 2` warp tile (`WM = BM // 2`, `WN = BN // 2`), accumulating up to
      `8 x 8 = 64` `16x16` tiles purely in AGPRs (`a[0:255]`) via `v_mfma_f32_16x16x32_bf16`.
    - Both `A[BM, 64]` and `B[64, BN]` stream directly from HBM into double-buffered 160 KB LDS
      (`48..128` KB total) via 128-bit `O.buffer_load(..., bytes=16)` (`dwordx4 ... lds`),
      using 64-bank XOR swizzling on the global `voffset`s so linear `M0 + lane * 16` DMA writes
      land conflict-free in LDS.
    - `A` fragments read via `ds_read_b128` and row-major `B` fragments transpose directly into
      MFMA operand order via `O.ds_read_tr(..., bits=16)` (`ds_read_b64_tr_b16`) with zero
      64-bank conflicts and zero `v_perm_b32` instructions.
    - Epilogue packs `f32` accumulators with `v_cvt_pk_bf16_f32` and uses SRD byte bounds to
      clamp arbitrary `M` tails cleanly without predication branches.
    """
    N = M if N is None else N  # noqa: N806
    K = M if K is None else K  # noqa: N806
    defaults = default_gemm_config(M)
    BM = defaults["BM"] if BM is None else BM  # noqa: N806
    BN = defaults["BN"] if BN is None else BN  # noqa: N806
    GROUP_M = defaults["GROUP_M"] if GROUP_M is None else GROUP_M  # noqa: N806

    WM = BM // 2  # noqa: N806
    WN = BN // 2  # noqa: N806
    MTILES = WM // 16  # noqa: N806
    NTILES = WN // 16  # noqa: N806

    assert BM in (64, 128, 192, 256), BM
    assert BN in (128, 256) and N % BN == 0, (N, BN)
    assert K % BK == 0 and K >= BK, K
    assert MTILES * NTILES <= 64, (MTILES, NTILES)
    assert 2 * (BM + BN) * (BK * 2) <= 163_840, (BM, BN)
    assert (XCDS & (XCDS - 1)) == 0, XCDS

    a_stage_bytes = BM * (BK * 2)
    b_stage_bytes = BN * (BK * 2)
    a_wave_rows = BM // NWAVES  # multiple of 16
    a_wave_bytes = a_stage_bytes // NWAVES
    a_loads = a_wave_bytes // 1024
    b_wave_ntiles = (BN // 16) // NWAVES
    b_wave_bytes = b_stage_bytes // NWAVES

    grid_m = cdiv(M, BM)
    grid_n = N // BN
    num_tiles = grid_m * grid_n
    assert (grid_n & (grid_n - 1)) == 0, grid_n
    eff_group_m = max(1, min(GROUP_M, grid_m))

    stride_a = K * 2
    stride_b = N * 2
    stride_c = N * 2
    n_stages = K // BK

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        a_ptr, b_ptr, c_ptr = k.load("a", "b", "c")
        rsrc_a = _srd(b, a_ptr, M * K * 2, "rsrc_a")
        rsrc_b = _srd(b, b_ptr, K * N * 2, "rsrc_b")
        rsrc_c = _srd(b, c_ptr, M * N * 2, "rsrc_c")

        wave = k.wave
        b.wave_index(wave, NWAVES)
        wave_m = S.u32(wave >> 1, name="wave_m")
        wave_n = S.u32(wave & 1, name="wave_n")

        lane = V.u32(k.lane, name="lane")
        i_lane = V.u32(lane & 15, name="i_lane")
        g_lane = V.u32(lane >> 4, name="g_lane")

        lds = b.lds()
        lds_a0 = lds.region("lds_a0", nbytes=a_stage_bytes, align=16)
        lds_b0 = lds.region("lds_b0", nbytes=b_stage_bytes, align=16)
        lds_a1 = lds.region("lds_a1", nbytes=a_stage_bytes, align=16)
        lds_b1 = lds.region("lds_b1", nbytes=b_stage_bytes, align=16)

        # 64-bank conflict-free HBM -> LDS DMA `voffset`s for `A` (`1024` B = `8 x 128` B/load):
        u_a = V.u32(lane & 7, name="u_a")
        r8_a = V.u32(lane >> 3, name="r8_a")
        k_even_a = V.u32(u_a ^ g_lane, name="k_even_a")
        k_odd_a = V.u32(k_even_a ^ 4, name="k_odd_a")
        voff_a = [
            V.u32(
                (r8_a + ai * 8) * stride_a + ((k_odd_a if (ai & 1) else k_even_a) << 4),
                name=f"voff_a{ai}",
            )
            for ai in range(a_loads)
        ]

        # 64-bank conflict-free HBM -> LDS DMA `voffset`s for `B` (`1024` B = `32 x 32` B/load):
        k_lds_b = V.u32(lane >> 1, name="k_lds_b")
        k_swz_b = V.u32(k_lds_b ^ ((k_lds_b >> 1) & 4), name="k_swz_b")
        col_half_b = V.u32((lane & 1) << 4, name="col_half_b")
        voff_b_lo = V.u32(k_swz_b * stride_b + col_half_b, name="voff_b_lo")
        voff_b_hi = V.u32((k_swz_b + 32) * stride_b + col_half_b, name="voff_b_hi")

        # Per-lane LDS read offsets for `ds_read_b128` (`A`) and `ds_read_b64_tr_b16` (`B`):
        i_half = V.u32(i_lane >> 1, name="i_half")
        rd_a_local = [
            V.u32((i_lane << 7) + (((4 * s_half + g_lane) ^ i_half) << 4), name=f"rda_l{s_half}")
            for s_half in range(2)
        ]
        for s_half in range(2):
            b.assume(rd_a_local[s_half], lo=0, hi=2032, why="16x64 bf16 A tile byte offset")
        rd_a = [
            V.u32(rd_a_local[s_half] + wave_m * (MTILES * 2048), name=f"rd_a{s_half}")
            for s_half in range(2)
        ]

        g_odd_bit2 = V.u32((g_lane & 1) << 2, name="g_odd_bit2")
        i_q = V.u32(i_lane >> 2, name="i_q")
        i_c8 = V.u32((i_lane & 3) << 3, name="i_c8")
        rd_b_local = []
        for reg_half in range(2):
            k_in = V.u32((g_lane << 3) + 4 * reg_half + i_q, name=f"kb_in{reg_half}")
            k_tr = V.u32(k_in ^ g_odd_bit2, name=f"kb_tr{reg_half}")
            rbl = V.u32((k_tr << 5) + i_c8, name=f"rdb_l{reg_half}")
            b.assume(rbl, lo=0, hi=1016, why="32x16 bf16 B half-tile byte offset")
            rd_b_local.append(rbl)
        rd_b = [
            V.u32(rd_b_local[rh] + wave_n * (NTILES * 2048), name=f"rd_b{rh}") for rh in range(2)
        ]

        pid = _remap_xcd(b, k.wgid, num_tiles, XCDS, CUS_PER_XCD)
        tile_m, tile_n = _tile_coords(b, pid, grid_m, grid_n, eff_group_m)
        soff_a = S.u32((tile_m * BM + wave * a_wave_rows) * stride_a, name="soff_a")
        soff_b = S.u32(tile_n * (BN * 2) + wave * (b_wave_ntiles * 32), name="soff_b")

        accs = [
            [
                b.mfma_acc("f32_16x16x32_bf16", init=0, acc="a", name=f"acc_{mt}_{nt}")
                for nt in range(NTILES)
            ]
            for mt in range(MTILES)
        ]
        fa0 = [V.u32(n=4, name=f"fa0_{mt}") for mt in range(MTILES)]
        fa1 = [V.u32(n=4, name=f"fa1_{mt}") for mt in range(MTILES)]
        fb0 = [V.u32(n=4, name=f"fb0_{nt}") for nt in range(NTILES)]
        fb1 = [V.u32(n=4, name=f"fb1_{nt}") for nt in range(NTILES)]

        def issue_stage_dma(dst_a, dst_b) -> None:
            for ai in range(a_loads):
                b.emit(
                    O.buffer_load(
                        dst_a + wave * a_wave_bytes + ai * 1024,
                        voff_a[ai],
                        rsrc_a,
                        soff_a,
                        bytes=16,
                    )
                )
            b.emit(O.op("s_add_u32", soff_a, soff_a, BK * 2))
            for bi in range(b_wave_ntiles):
                b.emit(
                    O.buffer_load(
                        dst_b + wave * b_wave_bytes + bi * 2048,
                        voff_b_lo,
                        rsrc_b,
                        soff_b,
                        offset=bi * 32,
                        bytes=16,
                    )
                )
                b.emit(
                    O.buffer_load(
                        dst_b + wave * b_wave_bytes + bi * 2048 + 1024,
                        voff_b_hi,
                        rsrc_b,
                        soff_b,
                        offset=bi * 32,
                        bytes=16,
                    )
                )
            b.emit(O.op("s_add_u32", soff_b, soff_b, BK * stride_b))

        def issue_substep_reads(src_a, src_b, dst_fa, dst_fb, s_half: int) -> None:
            for mt in range(MTILES):
                b.emit(O.ds_read(dst_fa[mt], src_a + rd_a[s_half], offset=mt * 2048))
            for nt in range(NTILES):
                off_b = nt * 2048 + s_half * 1024
                b.emit(O.ds_read_tr(dst_fb[nt][0:2], src_b + rd_b[0], offset=off_b, bits=16))
                b.emit(O.ds_read_tr(dst_fb[nt][2:4], src_b + rd_b[1], offset=off_b, bits=16))

        def emit_substep_mfmas(src_fa, src_fb) -> None:
            for mt in range(MTILES):
                for nt in range(NTILES):
                    b.emit(
                        O.mfma(
                            "f32_16x16x32_bf16",
                            accs[mt][nt],
                            src_fa[mt],
                            src_fb[nt],
                            accs[mt][nt],
                        )
                    )

        # Prologue: load stage 0 into `(lds_a0, lds_b0)`, sync, prefetch stage 1 if `n_stages >= 2`,
        # and issue substep-0 DS reads from `(lds_a0, lds_b0)`:
        issue_stage_dma(lds_a0, lds_b0)
        b.wait(lds_a0, lds_b0, comment="stage0 DMA")
        b.barrier("stage0 sync")
        if n_stages >= 2:
            issue_stage_dma(lds_a1, lds_b1)
        issue_substep_reads(lds_a0, lds_b0, fa0, fb0, 0)

        def run_stage(a_curr, b_curr, a_next, b_next, *, has_next: bool, has_next2: bool, tag: str):
            b.wait(*fa0, *fb0, comment=f"{tag} sub0 fragments")
            issue_substep_reads(a_curr, b_curr, fa1, fb1, 1)
            emit_substep_mfmas(fa0, fb0)

            b.wait(*fa1, *fb1, comment=f"{tag} sub1 fragments")
            emit_substep_mfmas(fa1, fb1)
            if not has_next:
                return

            b.wait(a_next, b_next, comment=f"{tag} next DMA")
            b.barrier(f"{tag} sync")
            if has_next2:
                issue_stage_dma(a_curr, b_curr)
            issue_substep_reads(a_next, b_next, fa0, fb0, 0)

        if n_stages == 1:
            run_stage(lds_a0, lds_b0, None, None, has_next=False, has_next2=False, tag="only")
        else:
            n_pairs = (n_stages // 2) - 1
            if n_pairs > 0:
                for _ in b.loop(S.u32(n_pairs, name="trips"), name="k_mainloop"):
                    run_stage(
                        lds_a0, lds_b0, lds_a1, lds_b1, has_next=True, has_next2=True, tag="ping"
                    )
                    run_stage(
                        lds_a1, lds_b1, lds_a0, lds_b0, has_next=True, has_next2=True, tag="pong"
                    )
            if n_stages % 2 == 0:
                run_stage(
                    lds_a0, lds_b0, lds_a1, lds_b1, has_next=True, has_next2=False, tag="penult"
                )
                run_stage(lds_a1, lds_b1, None, None, has_next=False, has_next2=False, tag="final")
            else:
                run_stage(
                    lds_a0, lds_b0, lds_a1, lds_b1, has_next=True, has_next2=True, tag="odd_p0"
                )
                run_stage(
                    lds_a1, lds_b1, lds_a0, lds_b0, has_next=True, has_next2=False, tag="odd_p1"
                )
                run_stage(lds_a0, lds_b0, None, None, has_next=False, has_next2=False, tag="final")

        # Epilogue: CDNA 4 hardware `v_cvt_pk_bf16_f32` RNE conversion + SRD-clamped stores.
        c_row_base = V.u32(g_lane << 2, name="c_row_base")
        voff_c = [
            V.u32((c_row_base + r) * stride_c + (i_lane << 1), name=f"voff_c{r}") for r in range(4)
        ]
        soff_c_wave = S.u32(
            (tile_m * BM + wave_m * WM) * stride_c + (tile_n * BN + wave_n * WN) * 2,
            name="soff_c_wave",
        )
        pk01 = V.u32(name="pk01")
        pk23 = V.u32(name="pk23")
        hi1 = V.u32(name="hi1")
        hi3 = V.u32(name="hi3")
        for mt in range(MTILES):
            soff_c_mt = S.u32(soff_c_wave + mt * 16 * stride_c, name=f"sc_mt{mt}")
            for nt in range(NTILES):
                vc = b.read_acc(accs[mt][nt], name=f"vc_{mt}_{nt}")
                b.emit(O.op("v_cvt_pk_bf16_f32", pk01, vc[0], vc[1]))
                b.emit(O.op("v_cvt_pk_bf16_f32", pk23, vc[2], vc[3]))
                b.emit(O.op("v_lshrrev_b32", hi1, 16, pk01))
                b.emit(O.op("v_lshrrev_b32", hi3, 16, pk23))
                off_nt = nt * 32
                b.emit(
                    O.buffer_store(
                        pk01, voff_c[0], rsrc_c, soff_c_mt, offset=off_nt, bytes=2, nt=True
                    )
                )
                b.emit(
                    O.buffer_store(
                        hi1, voff_c[1], rsrc_c, soff_c_mt, offset=off_nt, bytes=2, nt=True
                    )
                )
                b.emit(
                    O.buffer_store(
                        pk23, voff_c[2], rsrc_c, soff_c_mt, offset=off_nt, bytes=2, nt=True
                    )
                )
                b.emit(
                    O.buffer_store(
                        hi3, voff_c[3], rsrc_c, soff_c_mt, offset=off_nt, bytes=2, nt=True
                    )
                )

    def host(bind, a, b, c):
        return bind(grid=(num_tiles,), args={"a": a, "b": b, "c": c})

    name = f"gemm_bf16_gfx950_{M}x{N}x{K}_t{BM}x{BN}x{BK}_g{eff_group_m}"
    return KernelDef(
        name,
        Args(a=ptr, b=ptr, c=ptr),
        body,
        wgsize=WGSIZE,
        max_grid=(num_tiles,),
        host=host,
    )
