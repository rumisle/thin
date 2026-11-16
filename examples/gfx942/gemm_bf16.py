"""Tiled bf16 matrix multiply (`C = A @ B`) for 304-CU CDNA 3 (`gfx942`)."""

from __future__ import annotations

import math

from thin import KernelDef, cdiv
from thin.dsl import Args, O, ptr

NWAVES = 4
WGSIZE = 64 * NWAVES
BK = 64

_CONFIGS = (
    (2048, {"BM": 64, "BN": 256, "GROUP_M": 8}),
    (4096, {"BM": 112, "BN": 512, "GROUP_M": 6}),
    (8192, {"BM": 144, "BN": 512, "GROUP_M": 6}),
    (None, {"BM": 160, "BN": 512, "GROUP_M": 12}),
)


def default_gemm_config(m: int) -> dict[str, int]:
    """Per-size tile and L2 grouping parameters tuned for a 304-CU MI300X."""
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


def _interleave(mfmas: list, side: list) -> None:
    """Spreads `side` thunks evenly across `mfmas` to hide non-MFMA issue slots."""
    n_m, n_s = len(mfmas), len(side)
    if n_m == 0:
        for fn in side:
            fn()
        return
    s_idx = 0
    for m_idx, m_fn in enumerate(mfmas):
        target = ((m_idx + 1) * n_s + n_m - 1) // n_m
        m_fn()
        while s_idx < target:
            side[s_idx]()
            s_idx += 1


def _remap_xcd(b, wgid, num_tiles: int, xcds: int, cus_per_xcd: int):
    """Chunks round-robin hardware `wgid`s per 304-CU wave so each XCD gets adjacent `pid`s."""
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
    CUS_PER_XCD: int = 38,  # noqa: N803
) -> KernelDef:
    """Row-major `C[M, N] = A[M, K] @ B[K, N]` with bf16 inputs/output and f32 accumulation.

    Design for a 304-CU (`8 x 38`) CDNA 3 GPU:
    - Non-power-of-two `BM` (`112, 144, 160`) matches the 304-CU grid; SRD byte bounds clamp
      `M`-tail loads to zero and skip out-of-bounds `C` rows without inner-loop predication.
    - Four waves split `BN` (`1 x 4`, `WN in {64, 128}`). Each wave streams its disjoint `B`
      columns directly into a sliding VGPR ring (`b_slots in {8, 16}`) and transposes `4 x 8`
      bf16 tiles in-lane with `v_perm_b32`, while only `A` uses double-buffered XOR-swizzled
      LDS (`BK = 64`, dual-substep `ds_read_b128`, zero bank conflicts, one barrier per stage).
    - Accumulators spill past `a[0:255]` (`64` tiles) into arch VGPRs (`72` or `80` tiles/wave),
      and pack to RNE bf16 for coalesced non-temporal (`nt=True`) `dwordx2`/`dwordx4` stores.
    """
    N = M if N is None else N  # noqa: N806
    K = M if K is None else K  # noqa: N806
    defaults = default_gemm_config(M)
    BM = defaults["BM"] if BM is None else BM  # noqa: N806
    BN = defaults["BN"] if BN is None else BN  # noqa: N806
    GROUP_M = defaults["GROUP_M"] if GROUP_M is None else GROUP_M  # noqa: N806

    WN = BN // NWAVES  # noqa: N806
    MTILES = BM // 16  # noqa: N806
    NTILES = WN // 16  # noqa: N806
    n_sub_mfmas = MTILES * NTILES

    assert BM % 16 == 0 and 4 <= MTILES <= 16, BM
    assert BN in (256, 512) and N % BN == 0, (N, BN)
    assert K % (2 * BK) == 0 and K >= 2 * BK, K
    assert n_sub_mfmas <= 80, (MTILES, NTILES)
    assert (XCDS & (XCDS - 1)) == 0, XCDS

    a_alloc_rows = cdiv(BM, 32) * 32
    a_stage_bytes = a_alloc_rows * (BK * 2)
    a_wave_bytes = a_stage_bytes // NWAVES
    a_wave_rows = a_alloc_rows // NWAVES
    a_loads = a_wave_bytes // 1024

    grid_m = cdiv(M, BM)
    grid_n = N // BN
    num_tiles = grid_m * grid_n
    assert (grid_n & (grid_n - 1)) == 0, grid_n
    eff_group_m = max(1, min(GROUP_M, grid_m))

    stride_a = K * 2
    stride_b = N * 2
    stride_c = N * 2
    m_tail_valid_mt = cdiv(M - (grid_m - 1) * BM, 16)
    b_dwords = NTILES // 2
    lane_shift = 3 if b_dwords == 2 else 4
    b_slots = 8 if n_sub_mfmas > 72 else 16

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        a_ptr, b_ptr, c_ptr = k.load("a", "b", "c")
        rsrc_a = _srd(b, a_ptr, M * K * 2, "rsrc_a")
        rsrc_b = _srd(b, b_ptr, K * N * 2, "rsrc_b")
        rsrc_c = _srd(b, c_ptr, M * N * 2, "rsrc_c")

        wave = k.wave
        b.wave_index(wave, NWAVES)
        lane = V.u32(k.lane, name="lane")
        i_lane = V.u32(lane & 15, name="i_lane")
        g_lane = V.u32(lane >> 4, name="g_lane")

        lds = b.lds()
        lds_a0 = lds.region("lds_a0", nbytes=a_stage_bytes, align=16)
        lds_a1 = lds.region("lds_a1", nbytes=a_stage_bytes, align=16)
        if 2 * a_stage_bytes <= 32768 and n_sub_mfmas <= 32:
            lds.region("lds_occ_pad", nbytes=33024 - 2 * a_stage_bytes, align=16)

        u_a = V.u32(lane & 7, name="u_a")
        r8_a = V.u32(lane >> 3, name="r8_a")
        voff_a0 = V.u32(r8_a * stride_a + (u_a << 4), name="voff_a0")
        voff_a = [voff_a0] + [
            V.u32(voff_a0 + i * 8 * stride_a, name=f"voff_a{i}") for i in range(1, a_loads)
        ]

        wr_a_loc = V.u32((r8_a << 7) + ((u_a ^ r8_a) << 4), name="wr_a_loc")
        wr_a = V.u32(wr_a_loc + wave * a_wave_bytes, name="wr_a")
        b.assume(wr_a, lo=0, hi=a_stage_bytes - 16, why="wave slice + 1 KiB block within LDS_A")

        i_low3 = V.u32(i_lane & 7, name="i_low3")
        rd_a = [
            V.u32((i_lane << 7) + (((4 * s_half + g_lane) ^ i_low3) << 4), name=f"rd_a{s_half}")
            for s_half in range(2)
        ]

        voff_b0 = V.u32((g_lane * 8) * stride_b + (i_lane << lane_shift), name="voff_b0")
        voff_b = [voff_b0] + [V.u32(voff_b0 + j * stride_b, name=f"voff_b{j}") for j in range(1, 4)]

        pid = _remap_xcd(b, k.wgid, num_tiles, XCDS, CUS_PER_XCD)
        tile_m, tile_n = _tile_coords(b, pid, grid_m, grid_n, eff_group_m)
        soff_a = S.u32((tile_m * BM + wave * a_wave_rows) * stride_a, name="soff_a")
        soff_b = S.u32(tile_n * (BN * 2) + wave * (WN * 2), name="soff_b")

        accs = [
            [
                b.mfma_acc(
                    "f32_16x16x16_bf16",
                    init=0,
                    acc="a" if (mt * NTILES + nt) < 64 else "v",
                    name=f"acc_{mt}_{nt}",
                )
                for nt in range(NTILES)
            ]
            for mt in range(MTILES)
        ]
        ga = [V.u32(n=4, name=f"ga_{i}") for i in range(a_loads)]
        gb = [V.u32(n=b_dwords, name=f"gb_{i}") for i in range(b_slots)]
        s_sel_lo = S.u32(0x0504_0100, name="s_sel_lo")
        s_sel_hi = S.u32(0x0706_0302, name="s_sel_hi")

        def issue_a(dst, idx: int) -> None:
            b.emit(O.buffer_load(dst, voff_a[idx], rsrc_a, soff_a))
            if idx == a_loads - 1:
                b.emit(O.op("s_add_u32", soff_a, soff_a, BK * 2))

        def issue_b(idx: int) -> None:
            b.emit(O.buffer_load(gb[idx & (b_slots - 1)], voff_b[idx & 3], rsrc_b, soff_b))
            if idx in (3, 11):
                b.emit(O.op("s_add_u32", soff_b, soff_b, 4 * stride_b))
            elif idx in (7, 15):
                b.emit(O.op("s_add_u32", soff_b, soff_b, 28 * stride_b))

        def transpose_b_thunks(dst_fb, src_gb) -> list:
            thunks = []
            for blk in range(NTILES // 4):
                for sub, (d, sel) in enumerate(
                    (
                        (2 * blk, s_sel_lo),
                        (2 * blk, s_sel_hi),
                        (2 * blk + 1, s_sel_lo),
                        (2 * blk + 1, s_sel_hi),
                    )
                ):
                    f_idx = 4 * blk + sub
                    thunks.append(
                        lambda fb=dst_fb, sr=src_gb, _f=f_idx, _d=d, _s=sel: (
                            b.emit(O.op("v_perm_b32", fb[_f][0], sr[1][_d], sr[0][_d], _s)),
                            b.emit(O.op("v_perm_b32", fb[_f][1], sr[3][_d], sr[2][_d], _s)),
                        )
                    )
            return thunks

        def ds_read_a_thunks(lds_src, dst_fa, s_half: int) -> list:
            return [
                lambda m_i=mt: b.emit(
                    O.ds_read(dst_fa[m_i], lds_src + rd_a[s_half], offset=m_i * 2048)
                )
                for mt in range(MTILES)
            ]

        def a_fifo_thunks(lds_dst, *, reload_a: bool) -> list:
            thunks = []
            for ai in range(a_loads):

                def _step(a_i=ai):
                    b.wait(ga[a_i], comment=f"ga {a_i}")
                    b.emit(O.ds_write(lds_dst + wr_a, ga[a_i], offset=a_i * 1024))
                    if reload_a:
                        issue_a(ga[a_i], a_i)

                thunks.append(_step)
            return thunks

        # Prologue: overlap stage-0 `A` loads (`ga_pro`) with the steady-state `PGR=2` FIFO.
        ga_pro = [V.u32(n=4, name=f"gap_{i}") for i in range(a_loads)]
        for i in range(a_loads):
            issue_a(ga_pro[i], i)
        if b_slots == 8:
            for i in range(a_loads):
                issue_a(ga[i], i)
            for i in range(8):
                issue_b(i)
        else:
            half_a = a_loads // 2
            for i in range(4):
                issue_b(i)
            for i in range(half_a):
                issue_a(ga[i], i)
            for i in range(4, 8):
                issue_b(i)
            for i in range(half_a, a_loads):
                issue_a(ga[i], i)
            for i in range(8, 16):
                issue_b(i)

        b.wait(*ga_pro, comment="stage0 A")
        for i in range(a_loads):
            b.emit(O.ds_write(lds_a0 + wr_a, ga_pro[i], offset=i * 1024))
        del ga_pro

        # Allocate fragment registers after `ga_pro` dies so `ga_pro` adds zero peak VGPRs.
        fa0 = [V.u32(n=4, name=f"fa0_{mt}") for mt in range(MTILES)]
        fa1 = [V.u32(n=4, name=f"fa1_{mt}") for mt in range(MTILES)]
        fb_even = [V.u32(n=2, name=f"fbe_{nt}") for nt in range(NTILES)]
        fb_odd = [V.u32(n=2, name=f"fbo_{nt}") for nt in range(NTILES)]

        def substep_mfmas(src_fa, hi_half: int, src_fb) -> list:
            sl = slice(2, 4) if hi_half else slice(0, 2)
            return [
                lambda m_i=mt, n_i=nt, fb=src_fb: b.emit(
                    O.mfma(
                        "f32_16x16x16_bf16",
                        accs[m_i][n_i],
                        src_fa[m_i][sl],
                        fb[n_i],
                        accs[m_i][n_i],
                    )
                )
                for mt in range(MTILES)
                for nt in range(NTILES)
            ]

        b.wait(lds_a0, comment="pro lds_a0")
        b.barrier("pro sync")
        for fn in ds_read_a_thunks(lds_a0, fa0, 0):
            fn()
        b.wait(*gb[0:4], comment="pro k0 B")
        for fn in transpose_b_thunks(fb_even, gb[0:4]):
            fn()

        perm_tail = min(12, max(2, (n_sub_mfmas - 2 * NTILES) // 2))

        def run_stage(lds_curr, lds_next, *, has_next: bool, has_next2_a: bool, tag: str) -> None:
            gb_k2 = gb[0:4] if b_slots == 8 else gb[8:12]
            gb_k3 = gb[4:8] if b_slots == 8 else gb[12:16]
            re_k01 = True if b_slots == 8 else has_next
            idx_k01 = 8 if b_slots == 8 else 0
            idx_k23 = 0 if b_slots == 8 else 8

            b.wait(fa0[0], comment=f"{tag} fa0_0")
            mfmas_0 = substep_mfmas(fa0, 0, fb_even)
            if re_k01:
                _interleave(
                    mfmas_0[:NTILES], [lambda idx=idx_k01 + i: issue_b(idx) for i in range(4)]
                )
            else:
                for fn in mfmas_0[:NTILES]:
                    fn()
            b.wait(fa0[1], comment=f"{tag} fa0_1")
            for fn in mfmas_0[NTILES : 2 * NTILES]:
                fn()
            b.wait(*fa0[2:], comment=f"{tag} fa0 rest")

            a_fifo = a_fifo_thunks(lds_next, reload_a=has_next2_a) if has_next else []
            half_fifo = len(a_fifo) // 2
            _interleave(
                mfmas_0[2 * NTILES : len(mfmas_0) - perm_tail],
                ds_read_a_thunks(lds_curr, fa1, 1) + a_fifo[:half_fifo],
            )
            b.wait(*gb[4:8], comment=f"{tag} k1 B")
            _interleave(mfmas_0[len(mfmas_0) - perm_tail :], transpose_b_thunks(fb_odd, gb[4:8]))

            mfmas_1 = substep_mfmas(fa0, 1, fb_odd)
            side_1 = (
                [lambda idx=idx_k01 + 4 + i: issue_b(idx) for i in range(4)] if re_k01 else []
            ) + a_fifo[half_fifo:]
            _interleave(mfmas_1[: len(mfmas_1) - perm_tail], side_1)
            b.wait(*gb_k2, comment=f"{tag} k2 B")
            _interleave(mfmas_1[len(mfmas_1) - perm_tail :], transpose_b_thunks(fb_even, gb_k2))

            b.wait(*fa1, comment=f"{tag} fa1")
            mfmas_2 = substep_mfmas(fa1, 0, fb_even)
            side_2 = [lambda idx=idx_k23 + i: issue_b(idx) for i in range(4)] if has_next else []
            _interleave(mfmas_2[: len(mfmas_2) - perm_tail], side_2)
            b.wait(*gb_k3, comment=f"{tag} k3 B")
            _interleave(mfmas_2[len(mfmas_2) - perm_tail :], transpose_b_thunks(fb_odd, gb_k3))

            mfmas_3 = substep_mfmas(fa1, 1, fb_odd)
            if not has_next:
                for fn in mfmas_3:
                    fn()
                return
            head_3 = min(8, n_sub_mfmas // 2)
            _interleave(
                mfmas_3[:head_3], [lambda idx=idx_k23 + 4 + i: issue_b(idx) for i in range(4)]
            )
            b.wait(lds_next, comment=f"{tag} next A")
            b.barrier(f"{tag} sync")
            rem_3 = mfmas_3[head_3:]
            tail_3 = min(12, max(2, len(rem_3) - MTILES))
            _interleave(rem_3[: len(rem_3) - tail_3], ds_read_a_thunks(lds_next, fa0, 0))
            b.wait(*gb[0:4], comment=f"{tag} next k0 B")
            _interleave(rem_3[len(rem_3) - tail_3 :], transpose_b_thunks(fb_even, gb[0:4]))

        n_pairs = (K // (2 * BK)) - 1
        if n_pairs > 0:
            for _ in b.loop(S.u32(n_pairs, name="trips"), name="k_mainloop"):
                run_stage(lds_a0, lds_a1, has_next=True, has_next2_a=True, tag="ping")
                run_stage(lds_a1, lds_a0, has_next=True, has_next2_a=True, tag="pong")
        run_stage(lds_a0, lds_a1, has_next=True, has_next2_a=False, tag="penult")
        run_stage(lds_a1, None, has_next=False, has_next2_a=False, tag="final")

        # Epilogue: RNE f32 -> bf16 pack and coalesced non-temporal (`nt=True`) stores.
        rbias = V.u32(0x7FFF, name="rbias")
        v_sel_hi = V.u32(0x0706_0302, name="v_sel_hi")
        lsb0, lsb1 = V.u32(name="lsb0"), V.u32(name="lsb1")
        rnd0, rnd1 = V.u32(name="rnd0"), V.u32(name="rnd1")
        out_vec = V.u32(n=b_dwords, name="out_vec")
        c_row_base = V.u32(g_lane << 2, name="c_row_base")
        voff_c = [
            V.u32((c_row_base + r) * stride_c + (i_lane << lane_shift), name=f"voff_c{r}")
            for r in range(4)
        ]
        soff_c_wave = S.u32(
            (tile_m * BM) * stride_c + (tile_n * BN + wave * WN) * 2,
            name="soff_c_wave",
        )

        def store_mt_row(mt_idx: int) -> None:
            soff_c_mt = S.u32(soff_c_wave + mt_idx * 16 * stride_c, name=f"sc_mt{mt_idx}")
            vcs = [b.read_acc(accs[mt_idx][nt], name=f"vc_{mt_idx}_{nt}") for nt in range(NTILES)]
            for r in range(4):
                for w_i in range(b_dwords):
                    b.emit(O.op("v_bfe_u32", lsb0, vcs[2 * w_i][r], 16, 1))
                    b.emit(O.op("v_add3_u32", rnd0, vcs[2 * w_i][r], rbias, lsb0))
                    b.emit(O.op("v_bfe_u32", lsb1, vcs[2 * w_i + 1][r], 16, 1))
                    b.emit(O.op("v_add3_u32", rnd1, vcs[2 * w_i + 1][r], rbias, lsb1))
                    b.emit(O.op("v_perm_b32", out_vec[w_i], rnd1, rnd0, v_sel_hi))
                b.emit(O.buffer_store(out_vec, voff_c[r], rsrc_c, soff_c_mt, nt=True))

        for mt in range(m_tail_valid_mt):
            store_mt_row(mt)
        if m_tail_valid_mt < MTILES:
            with b.when(tile_m < grid_m - 1, name="c_mt_interior"):
                for mt in range(m_tail_valid_mt, MTILES):
                    store_mt_row(mt)

    def host(bind, a, b, c):
        return bind(grid=(num_tiles,), args={"a": a, "b": b, "c": c})

    name = f"gemm_bf16_{M}x{N}x{K}_t{BM}x{BN}x{BK}_g{eff_group_m}"
    return KernelDef(
        name,
        Args(a=ptr, b=ptr, c=ptr),
        body,
        wgsize=WGSIZE,
        max_grid=(num_tiles,),
        host=host,
    )
