"""Offline cross-validation of `gfx942` (CDNA 3) matrix layouts, FNUZ FP8/BF8, and MFMA numerics.

Reference data in `crates/thin-emu/tests/data/emu/cdna3_matrix_ref.json.gz` is captured from:
1. AMD's official `matrix_calculator.py` (`v1.3.2`, `--architecture cdna3`) for all 46 CDNA 3
   `v_mfma_*` / `v_smfmac_*` instructions, including `cbsz`/`abid`/`blgp` and sparse `K[]` sets.
2. PyTorch (`torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz`) and `ml_dtypes` (`v0.6.0`) for all
   256 byte decodes and a 16,384-point `f32` boundary/tie/subnormal/saturation/NaN/Inf sweep.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import ml_dtypes
import mpmath
import numpy as np
import pytest

from thin import formats
from thin.emu import Cluster, Memory
from thin.target import target, targeting

REF_PATH = Path(__file__).parents[1] / "crates/thin-emu/tests/data/emu/cdna3_matrix_ref.json.gz"


def _load_ref() -> dict:
    with gzip.open(REF_PATH, "rb") as f:
        return json.load(f)


REF = _load_ref()


def _pack_from_mc_map(mat: np.ndarray, loc_map: list, gprs: int, *, bits: int) -> np.ndarray:
    """Pack logical matrix entries into `[64, gprs]` uint32 using `matrix_calculator` map."""
    regs = np.zeros((64, gprs), dtype=np.uint32)
    mask = (1 << bits) - 1
    blocks, rows, cols = mat.shape
    for b in range(blocks):
        for r in range(rows):
            for c in range(cols):
                dw, bit_lo, bit_hi, lane = loc_map[b][r][c]
                val = int(mat[b, r, c]) & mask
                if bits == 64:
                    assert bit_lo == 0 and bit_hi == 63
                    regs[lane, dw] = val & 0xFFFFFFFF
                    regs[lane, dw + 1] = (val >> 32) & 0xFFFFFFFF
                else:
                    assert bit_hi - bit_lo + 1 == bits
                    regs[lane, dw] |= np.uint32((val << bit_lo) & 0xFFFFFFFF)
    return regs


def _pack_sparse_a_and_k(
    a_vals: np.ndarray,
    k_nibs: np.ndarray,
    a_map: list,
    k_map: list,
    a_gprs: int,
    *,
    in_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack 4:2 sparse `A` (`[blocks, m, k//2]`) and nibbles (`[blocks, m, k//4]`) via MC maps."""
    regs_a = np.zeros((64, a_gprs), dtype=np.uint32)
    regs_k = np.zeros((64, 1), dtype=np.uint32)
    mask = (1 << in_bits) - 1
    blocks, m, k_half = a_vals.shape
    for b in range(blocks):
        for i in range(m):
            for g in range(k_half // 2):
                # In matrix_calculator, sparse A maps group `4*g..4*g+3` to a `2*in_bits` slot:
                dw_a, lo_a, hi_a, lane_a = a_map[b][i][4 * g]
                assert hi_a - lo_a + 1 == 2 * in_bits
                v0 = int(a_vals[b, i, 2 * g]) & mask
                v1 = int(a_vals[b, i, 2 * g + 1]) & mask
                regs_a[lane_a, dw_a] |= np.uint32((v0 << lo_a) | (v1 << (lo_a + in_bits)))

                dw_k, lo_k, hi_k, lane_k = k_map[b][i][4 * g]
                assert dw_k == 0 and hi_k - lo_k == 3
                nib = int(k_nibs[b, i, g]) & 0xF
                regs_k[lane_k, 0] |= np.uint32(nib << lo_k)
    return regs_a, regs_k


def _unpack_from_mc_map(regs: np.ndarray, d_map: list, *, out_bits: int) -> np.ndarray:
    """Unpack `[64, gprs]` uint32 into `[blocks, m, n]` raw uint32/uint64 via MC `D` map."""
    blocks = len(d_map)
    m = len(d_map[0])
    n = len(d_map[0][0])
    out = np.zeros((blocks, m, n), dtype=np.uint64 if out_bits == 64 else np.uint32)
    for b in range(blocks):
        for i in range(m):
            for j in range(n):
                dw, bit_lo, bit_hi, lane = d_map[b][i][j]
                if out_bits == 64:
                    assert bit_lo == 0 and bit_hi == 63
                    out[b, i, j] = int(regs[lane, dw]) | (int(regs[lane, dw + 1]) << 32)
                else:
                    assert bit_lo == 0 and bit_hi == 31
                    out[b, i, j] = regs[lane, dw]
    return out


def _run_mfma_asm(
    mnem: str,
    a_regs: np.ndarray,
    b_regs: np.ndarray,
    c_or_d_regs: np.ndarray,
    *,
    sparse: bool = False,
    k_regs: np.ndarray | None = None,
    mods: str = "",
) -> np.ndarray:
    """Execute one `v_mfma_*` or `v_smfmac_*` instruction in `thin.emu` and return `[64, gd]`."""
    ga = a_regs.shape[1]
    gb = b_regs.shape[1]
    gd = c_or_d_regs.shape[1]

    mem = Memory()
    buf_a = mem.alloc("a", a_regs.nbytes, tagged=False)
    buf_b = mem.alloc("b", b_regs.nbytes, tagged=False)
    buf_c = mem.alloc("c", c_or_d_regs.nbytes, tagged=False)
    buf_d = mem.alloc("d", c_or_d_regs.nbytes, tagged=False)
    buf_a.write(0, np.ascontiguousarray(a_regs).tobytes())
    buf_b.write(0, np.ascontiguousarray(b_regs).tobytes())
    buf_c.write(0, np.ascontiguousarray(c_or_d_regs).tobytes())

    ptrs = [buf_a.base, buf_b.base, buf_c.base, buf_d.base]
    if sparse:
        assert k_regs is not None
        buf_k = mem.alloc("k", k_regs.nbytes, tagged=False)
        buf_k.write(0, np.ascontiguousarray(k_regs).tobytes())
        ptrs.append(buf_k.base)
    ka = mem.alloc("ka", 8 * len(ptrs), tagged=False)
    ka.write(0, np.array(ptrs, dtype=np.uint64).tobytes())

    va = 4
    vb = (va + ga + 3) & ~3
    vk = vb + gb
    lines = [
        "\ts_load_dwordx8 s[4:11], s[0:1], 0x0",
    ]
    if sparse:
        lines.append("\ts_load_dwordx2 s[12:13], s[0:1], 0x20")
    lines.append("\ts_waitcnt lgkmcnt(0)")

    def load_vec(v_base: int, n_dw: int, s_lo: int) -> None:
        lines.append(f"\tv_mul_lo_u32 v1, v0, {n_dw * 4}")
        for d in range(0, n_dw, 4):
            w = min(4, n_dw - d)
            reg = f"v{v_base + d}" if w == 1 else f"v[{v_base + d}:{v_base + d + w - 1}]"
            op = "dword" if w == 1 else f"dwordx{w}"
            lines.append(f"\tglobal_load_{op} {reg}, v1, s[{s_lo}:{s_lo + 1}] offset:{d * 4}")

    load_vec(va, ga, 4)
    load_vec(vb, gb, 6)
    vc = (vk + 4) & ~3
    load_vec(vc, gd, 8)
    if sparse:
        load_vec(vk, 1, 12)
    lines.append("\ts_waitcnt vmcnt(0)")
    for d in range(gd):
        lines.append(f"\tv_accvgpr_write_b32 a{d}, v{vc + d}")

    op_d = "a0" if gd == 1 else f"a[0:{gd - 1}]"
    op_a = f"v{va}" if ga == 1 else f"v[{va}:{va + ga - 1}]"
    op_b = f"v{vb}" if gb == 1 else f"v[{vb}:{vb + gb - 1}]"
    mod_s = f" {mods}" if mods else ""
    if sparse:
        lines.append(f"\t{mnem} {op_d}, {op_a}, {op_b}, v{vk}{mod_s}")
    else:
        lines.append(f"\t{mnem} {op_d}, {op_a}, {op_b}, {op_d}{mod_s}")

    for d in range(gd):
        lines.append(f"\tv_accvgpr_read_b32 v{vc + d}, a{d}")
    lines.append(f"\tv_mul_lo_u32 v1, v0, {gd * 4}")
    for d in range(0, gd, 4):
        w = min(4, gd - d)
        reg = f"v{vc + d}" if w == 1 else f"v[{vc + d}:{vc + d + w - 1}]"
        op = "dword" if w == 1 else f"dwordx{w}"
        lines.append(f"\tglobal_store_{op} v1, {reg}, s[10:11] offset:{d * 4}")
    lines.append("\ts_endpgm")

    asm = "\n".join(lines) + "\n"
    vgpr_cnt = ((vc + gd + 3) & ~3) + 4
    agpr_cnt = (gd + 3) & ~3
    with targeting("gfx942"):
        Cluster(
            asm,
            {"wgsize": 64, "vgpr": vgpr_cnt, "agpr": agpr_cnt, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        ).run()
    return np.frombuffer(bytes(buf_d.data), dtype=np.uint32).reshape(64, gd).copy()


def _exact_codes_and_values(kind: str, count: int, rng: np.random.Generator):
    """Generate small dyadic / integer codes whose products and sums are exact in f32/f64/i32."""
    if kind == "int8":
        vals = rng.integers(-7, 8, size=count, dtype=np.int64)
        codes = (vals & 0xFF).astype(np.uint64)
        return codes, vals
    if kind == "fp16":
        choices = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float16)
        picked = choices[rng.integers(0, len(choices), size=count)]
        return picked.view(np.uint16).astype(np.uint64), picked.astype(np.float64)
    if kind == "bf16":
        choices = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        picked = choices[rng.integers(0, len(choices), size=count)]
        codes = (picked.view(np.uint32) >> 16).astype(np.uint64)
        return codes, picked.astype(np.float64)
    if kind in ("amd_fp8", "amd_bf8"):
        dt = formats.DTYPES["e4m3fnuz" if kind == "amd_fp8" else "e5m2fnuz"]
        choices = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        picked = choices[rng.integers(0, len(choices), size=count)]
        codes = formats.encode(picked, dt, saturate=False).astype(np.uint64)
        return codes, picked.astype(np.float64)
    if kind in ("fp32", "xf32"):
        choices = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0], np.float32)
        picked = choices[rng.integers(0, len(choices), size=count)]
        return picked.view(np.uint32).astype(np.uint64), picked.astype(np.float64)
    if kind == "fp64":
        choices = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0], np.float64)
        picked = choices[rng.integers(0, len(choices), size=count)]
        return picked.view(np.uint64), picked
    raise ValueError(f"unknown kind {kind}")


def test_cdna3_matrix_metadata_matches_official_calculator():
    """All 46 CDNA 3 matrix forms match AMD `matrix_calculator.py` metadata 100%."""
    tgt = target("gfx942")
    ref_insts = REF["instructions"]
    thin_mnems = sorted(
        {f.mnemonic for f in tgt.forms() if tgt.matrix_form(f.mnemonic) is not None}
    )
    assert thin_mnems == sorted(ref_insts), "CDNA3 matrix mnemonic inventory mismatch"
    assert len(thin_mnems) == 46

    for mnem, info in ref_insts.items():
        mf = tgt.matrix_form(mnem)
        assert mf is not None
        assert (mf.m, mf.n, mf.k, mf.blocks) == (
            info["m"],
            info["n"],
            info["k"],
            info["blocks"],
        ), mnem
        assert mf.wave == 64
        assert mf.passes == info["cycles"] // 4, mnem
        assert mf.sparse is info["sparse"], mnem
        assert mf.cbsz_abid is info["cbsz_abid"], mnem
        assert mf.blgp is info["blgp"], mnem
        assert mf.provisional is False, mnem
        want_c = 1 if info["sparse"] else info["gprs"]["c"]
        assert dict(mf.dwords) == {
            "a": info["gprs"]["a"],
            "b": info["gprs"]["b"],
            "c": want_c,
            "d": info["gprs"]["d"],
        }, mnem


@pytest.mark.parametrize("mnem", sorted(REF["instructions"]))
def test_every_cdna3_mfma_and_smfmac_layout_via_emulator(mnem: str):
    """Execute every one of the 46 CDNA 3 MFMA/SMFMAC ops packed/unpacked by `matrix_calculator`."""
    info = REF["instructions"][mnem]
    cls = REF["layout_classes"][info["layout_class"]]
    m, n, k, blocks = info["m"], info["n"], info["k"], info["blocks"]
    in_bits = cls["in_bits"]
    out_bits = cls["out_bits"]
    ga, gb, gd = info["gprs"]["a"], info["gprs"]["b"], info["gprs"]["d"]

    rng = np.random.default_rng(abs(hash(mnem)) & 0xFFFFFFFF)
    c_codes, c_vals = _exact_codes_and_values(
        "int8" if info["out_type"] == "int32" else ("fp64" if out_bits == 64 else "fp32"),
        blocks * m * n,
        rng,
    )
    if info["out_type"] == "int32":
        c_vals = rng.integers(-500, 500, size=(blocks, m, n), dtype=np.int64)
        c_codes = (c_vals & 0xFFFFFFFF).astype(np.uint64)
    else:
        c_codes = c_codes.reshape(blocks, m, n)
        c_vals = c_vals.reshape(blocks, m, n)
    c_regs = _pack_from_mc_map(c_codes, cls["d"], gd, bits=out_bits)

    b_codes, b_vals = _exact_codes_and_values(info["in_type_src1"], blocks * k * n, rng)
    b_codes = b_codes.reshape(blocks, k, n)
    b_vals = b_vals.reshape(blocks, k, n)
    b_regs = _pack_from_mc_map(b_codes, cls["b"]["0"], gb, bits=in_bits)

    if info["sparse"]:
        a_codes, a_vals = _exact_codes_and_values(info["in_type"], blocks * m * (k // 2), rng)
        a_codes = a_codes.reshape(blocks, m, k // 2)
        a_vals = a_vals.reshape(blocks, m, k // 2)
        valid_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        picks = rng.integers(0, len(valid_pairs), size=(blocks, m, k // 4))
        k_nibs = np.zeros((blocks, m, k // 4), dtype=np.uint64)
        a_dense = np.zeros((blocks, m, k), dtype=a_vals.dtype)
        for blk in range(blocks):
            for i in range(m):
                for g in range(k // 4):
                    i0, i1 = valid_pairs[int(picks[blk, i, g])]
                    k_nibs[blk, i, g] = i0 | (i1 << 2)
                    a_dense[blk, i, 4 * g + i0] = a_vals[blk, i, 2 * g]
                    a_dense[blk, i, 4 * g + i1] = a_vals[blk, i, 2 * g + 1]
        a_regs, k_regs = _pack_sparse_a_and_k(
            a_codes, k_nibs, cls["a"]["0:0"], cls["k_idx"]["0:0"], ga, in_bits=in_bits
        )
        got_regs = _run_mfma_asm(mnem, a_regs, b_regs, c_regs, sparse=True, k_regs=k_regs)
    else:
        a_codes, a_vals = _exact_codes_and_values(info["in_type"], blocks * m * k, rng)
        a_codes = a_codes.reshape(blocks, m, k)
        a_dense = a_vals.reshape(blocks, m, k)
        a_regs = _pack_from_mc_map(a_codes, cls["a"]["0:0"], ga, bits=in_bits)
        got_regs = _run_mfma_asm(mnem, a_regs, b_regs, c_regs)

    got = _unpack_from_mc_map(got_regs, cls["d"], out_bits=out_bits)
    want_val = np.einsum("bik,bkj->bij", a_dense, b_vals) + c_vals
    if info["out_type"] == "int32":
        want_bits = (want_val.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    elif out_bits == 64:
        want_bits = want_val.astype(np.float64).view(np.uint64)
    else:
        want_bits = want_val.astype(np.float32).view(np.uint32)
    np.testing.assert_array_equal(got, want_bits, err_msg=f"layout/value mismatch on {mnem}")


def test_cdna3_cbsz_abid_blgp_and_sparse_sets_against_calculator():
    """Verify every `cbsz`/`abid`, `blgp` (0..7), and sparse `K[]` index set in `thin.emu`."""
    rng = np.random.default_rng(942_2026)

    # 1) Dense multi-block `cbsz`/`abid` across 2-block, 4-block, and 16-block layouts:
    for mnem in (
        "v_mfma_f32_32x32x4_2b_f16",
        "v_mfma_f32_16x16x4_4b_bf16",
        "v_mfma_i32_4x4x4_16b_i8",
    ):
        info = REF["instructions"][mnem]
        cls = REF["layout_classes"][info["layout_class"]]
        m, n, k, blocks = info["m"], info["n"], info["k"], info["blocks"]
        in_bits, out_bits = cls["in_bits"], cls["out_bits"]
        ga, gb, gd = info["gprs"]["a"], info["gprs"]["b"], info["gprs"]["d"]

        a_codes, a_vals = _exact_codes_and_values(info["in_type"], blocks * m * k, rng)
        b_codes, b_vals = _exact_codes_and_values(info["in_type_src1"], blocks * k * n, rng)
        a_codes, a_vals = a_codes.reshape(blocks, m, k), a_vals.reshape(blocks, m, k)
        b_codes, b_vals = b_codes.reshape(blocks, k, n), b_vals.reshape(blocks, k, n)
        a_regs = _pack_from_mc_map(a_codes, cls["a"]["0:0"], ga, bits=in_bits)
        b_regs = _pack_from_mc_map(b_codes, cls["b"]["0"], gb, bits=in_bits)
        c_regs = np.zeros((64, gd), dtype=np.uint32)

        # Build reverse lookup `(dw, bit_lo, lane) -> (b_src, i_src, k_src)` from base `0:0` map:
        a_rev = {}
        for b0 in range(blocks):
            for i0 in range(m):
                for k0 in range(k):
                    dw, lo, _, lane = cls["a"]["0:0"][b0][i0][k0]
                    a_rev[(dw, lo, lane)] = a_vals[b0, i0, k0]

        for key, a_mod_map in cls["a"].items():
            if key == "0:0":
                continue
            cbsz_s, abid_s = key.split(":")
            got_regs = _run_mfma_asm(
                mnem, a_regs, b_regs, c_regs, mods=f"cbsz:{cbsz_s} abid:{abid_s}"
            )
            got = _unpack_from_mc_map(got_regs, cls["d"], out_bits=out_bits)
            a_eff = np.zeros_like(a_vals)
            for b0 in range(blocks):
                for i0 in range(m):
                    for k0 in range(k):
                        dw, lo, _, lane = a_mod_map[b0][i0][k0]
                        a_eff[b0, i0, k0] = a_rev[(dw, lo, lane)]
            want = np.einsum("bik,bkj->bij", a_eff, b_vals)
            want_bits = (
                (want.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
                if info["out_type"] == "int32"
                else want.astype(np.float32).view(np.uint32)
            )
            np.testing.assert_array_equal(got, want_bits, err_msg=f"{mnem} {key}")

    # 2) All 8 `blgp` values (`0..7`) against `matrix_calculator.py` B-lane map:
    for mnem in ("v_mfma_f32_16x16x4_f32", "v_mfma_f32_4x4x4_16b_f16"):
        info = REF["instructions"][mnem]
        cls = REF["layout_classes"][info["layout_class"]]
        m, n, k, blocks = info["m"], info["n"], info["k"], info["blocks"]
        in_bits, out_bits = cls["in_bits"], cls["out_bits"]
        ga, gb, gd = info["gprs"]["a"], info["gprs"]["b"], info["gprs"]["d"]

        a_codes, a_vals = _exact_codes_and_values(info["in_type"], blocks * m * k, rng)
        b_codes, b_vals = _exact_codes_and_values(info["in_type_src1"], blocks * k * n, rng)
        a_codes, a_vals = a_codes.reshape(blocks, m, k), a_vals.reshape(blocks, m, k)
        b_codes, b_vals = b_codes.reshape(blocks, k, n), b_vals.reshape(blocks, k, n)
        a_regs = _pack_from_mc_map(a_codes, cls["a"]["0:0"], ga, bits=in_bits)
        b_regs = _pack_from_mc_map(b_codes, cls["b"]["0"], gb, bits=in_bits)
        c_regs = np.zeros((64, gd), dtype=np.uint32)

        b_rev = {}
        for b0 in range(blocks):
            for k0 in range(k):
                for j0 in range(n):
                    dw, lo, _, lane = cls["b"]["0"][b0][k0][j0]
                    b_rev[(dw, lo, lane)] = b_vals[b0, k0, j0]

        for blgp in range(1, 8):
            b_mod_map = cls["b"][str(blgp)]
            got_regs = _run_mfma_asm(mnem, a_regs, b_regs, c_regs, mods=f"blgp:{blgp}")
            got = _unpack_from_mc_map(got_regs, cls["d"], out_bits=out_bits)
            b_eff = np.zeros_like(b_vals)
            for b0 in range(blocks):
                for k0 in range(k):
                    for j0 in range(n):
                        dw, lo, _, lane = b_mod_map[b0][k0][j0]
                        b_eff[b0, k0, j0] = b_rev[(dw, lo, lane)]
            want_bits = np.einsum("bik,bkj->bij", a_vals, b_eff).astype(np.float32).view(np.uint32)
            np.testing.assert_array_equal(got, want_bits, err_msg=f"{mnem} blgp:{blgp}")

    # 3) Sparse `K[]` index sets (`cbsz=0, abid=0..max_sets-1` and `cbsz=1..3 -> set 0`):
    for mnem in ("v_smfmac_f32_16x16x32_f16", "v_smfmac_i32_16x16x64_i8"):
        info = REF["instructions"][mnem]
        cls = REF["layout_classes"][info["layout_class"]]
        m, n, k, blocks = info["m"], info["n"], info["k"], info["blocks"]
        in_bits, out_bits = cls["in_bits"], cls["out_bits"]
        ga, gb, gd = info["gprs"]["a"], info["gprs"]["b"], info["gprs"]["d"]
        nsets = info["compression_sets"]

        a_codes, a_vals = _exact_codes_and_values(info["in_type"], blocks * m * (k // 2), rng)
        b_codes, b_vals = _exact_codes_and_values(info["in_type_src1"], blocks * k * n, rng)
        a_codes, a_vals = a_codes.reshape(blocks, m, k // 2), a_vals.reshape(blocks, m, k // 2)
        b_codes, b_vals = b_codes.reshape(blocks, k, n), b_vals.reshape(blocks, k, n)
        b_regs = _pack_from_mc_map(b_codes, cls["b"]["0"], gb, bits=in_bits)
        c_regs = np.zeros((64, gd), dtype=np.uint32)

        # Populate distinct index patterns into EVERY compression set of `k_regs`:
        valid_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        k_regs = np.zeros((64, 1), dtype=np.uint32)
        want_by_set = {}
        a_regs = None
        for s_id in range(nsets):
            picks = rng.integers(0, len(valid_pairs), size=(blocks, m, k // 4))
            k_nibs = np.zeros((blocks, m, k // 4), dtype=np.uint64)
            a_dense = np.zeros((blocks, m, k), dtype=a_vals.dtype)
            for i in range(m):
                for g in range(k // 4):
                    i0, i1 = valid_pairs[int(picks[0, i, g])]
                    k_nibs[0, i, g] = i0 | (i1 << 2)
                    a_dense[0, i, 4 * g + i0] = a_vals[0, i, 2 * g]
                    a_dense[0, i, 4 * g + i1] = a_vals[0, i, 2 * g + 1]
            ra, rk = _pack_sparse_a_and_k(
                a_codes, k_nibs, cls["a"]["0:0"], cls["k_idx"][f"0:{s_id}"], ga, in_bits=in_bits
            )
            if a_regs is None:
                a_regs = ra
            k_regs |= rk
            prod = np.einsum("bik,bkj->bij", a_dense, b_vals)
            want_by_set[s_id] = (
                (prod.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
                if info["out_type"] == "int32"
                else prod.astype(np.float32).view(np.uint32)
            )

        assert a_regs is not None
        for key in cls["k_idx"]:
            cbsz_i, abid_i = (int(x) for x in key.split(":"))
            got_regs = _run_mfma_asm(
                mnem,
                a_regs,
                b_regs,
                c_regs,
                sparse=True,
                k_regs=k_regs,
                mods=f"cbsz:{cbsz_i} abid:{abid_i}",
            )
            got = _unpack_from_mc_map(got_regs, cls["d"], out_bits=out_bits)
            eff_set = abid_i if cbsz_i == 0 else 0
            np.testing.assert_array_equal(got, want_by_set[eff_set], err_msg=f"{mnem} {key}")


def test_fnuz_exhaustive_256_decode_and_16384_f32_sweep():
    """Verify all 256 FNUZ decodes and 16,384 `f32` encodes across `thin.formats` and `thin.emu`."""
    fnuz = REF["fnuz"]
    raw_u8 = np.arange(256, dtype=np.uint8)
    f8_dt = formats.DTYPES["e4m3fnuz"]
    b8_dt = formats.DTYPES["e5m2fnuz"]

    # A further live cross-check when torch is installed in this Python env:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None

    # 1) Exhaustive 256-code decode in `thin.formats`:
    for dt, ref_bits_list, t_attr, m_attr in [
        (f8_dt, fnuz["decode_e4m3fnuz_u32"], "float8_e4m3fnuz", "float8_e4m3fnuz"),
        (b8_dt, fnuz["decode_e5m2fnuz_u32"], "float8_e5m2fnuz", "float8_e5m2fnuz"),
    ]:
        ref_u32 = np.array(ref_bits_list, dtype=np.uint32)
        got_f32 = formats.decode(raw_u8, dt)
        assert np.flatnonzero(np.isnan(got_f32)).tolist() == [0x80]
        fin = raw_u8 != 0x80
        np.testing.assert_array_equal(got_f32.view(np.uint32)[fin], ref_u32[fin])
        if torch is not None:
            t_dec = torch.from_numpy(raw_u8.copy()).view(getattr(torch, t_attr)).to(torch.float32)
            assert np.flatnonzero(np.isnan(t_dec.numpy())).tolist() == [0x80]
            np.testing.assert_array_equal(
                got_f32.view(np.uint32)[fin], t_dec.numpy().view(np.uint32)[fin]
            )
        m_dec = raw_u8.view(getattr(ml_dtypes, m_attr)).astype(np.float32)
        assert np.flatnonzero(np.isnan(m_dec)).tolist() == [0x80]
        np.testing.assert_array_equal(got_f32.view(np.uint32)[fin], m_dec.view(np.uint32)[fin])

    # 2) Exhaustive 256-code decode in `thin.emu` (`v_cvt_f32_fp8/bf8` and `v_cvt_pk_f32_fp8/bf8`):
    asm_dec = """\
\ts_load_dwordx8 s[4:11], s[0:1], 0x0
\ts_load_dwordx2 s[12:13], s[0:1], 0x20
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 6, s2
\tv_add_u32 v1, v1, v0
\tv_lshlrev_b32 v2, 2, v1
\tglobal_load_dword v3, v2, s[4:5]
\ts_waitcnt vmcnt(0)
\tv_cvt_f32_fp8 v4, v3
\tv_cvt_f32_bf8 v5, v3
\tv_cvt_pk_f32_fp8 v[6:7], v3
\tv_cvt_pk_f32_bf8 v[8:9], v3
\tglobal_store_dword v2, v4, s[6:7]
\tglobal_store_dword v2, v5, s[8:9]
\tv_lshlrev_b32 v10, 3, v1
\tglobal_store_dwordx2 v10, v[6:7], s[10:11]
\tglobal_store_dwordx2 v10, v[8:9], s[12:13]
\ts_endpgm
"""
    # Feed 256 dwords (4 WGs of 64 lanes): low byte = i, byte 1 = 255 - i
    in_dwords = raw_u8.astype(np.uint32) | ((255 - raw_u8).astype(np.uint32) << 8)
    mem_d = Memory()
    b_in_d = mem_d.alloc("in", in_dwords.nbytes, tagged=False)
    b_in_d.write(0, in_dwords.tobytes())
    b_s_fp8 = mem_d.alloc("s_fp8", 256 * 4, tagged=False)
    b_s_bf8 = mem_d.alloc("s_bf8", 256 * 4, tagged=False)
    b_p_fp8 = mem_d.alloc("p_fp8", 256 * 8, tagged=False)
    b_p_bf8 = mem_d.alloc("p_bf8", 256 * 8, tagged=False)
    ka_d = mem_d.alloc("ka", 40, tagged=False)
    ka_d.write(
        0,
        np.array(
            [b_in_d.base, b_s_fp8.base, b_s_bf8.base, b_p_fp8.base, b_p_bf8.base],
            dtype=np.uint64,
        ).tobytes(),
    )
    with targeting("gfx942"):
        Cluster(
            asm_dec,
            {"wgsize": 64, "vgpr": 12, "agpr": 0, "user_sgprs": 2},
            mem_d,
            [ka_d],
            grid=(4,),
        ).run()

    ref_e4 = np.array(fnuz["decode_e4m3fnuz_u32"], dtype=np.uint32)
    ref_e5 = np.array(fnuz["decode_e5m2fnuz_u32"], dtype=np.uint32)
    np.testing.assert_array_equal(np.frombuffer(bytes(b_s_fp8.data), dtype=np.uint32), ref_e4)
    np.testing.assert_array_equal(np.frombuffer(bytes(b_s_bf8.data), dtype=np.uint32), ref_e5)
    pk_e4 = np.frombuffer(bytes(b_p_fp8.data), dtype=np.uint32).reshape(256, 2)
    pk_e5 = np.frombuffer(bytes(b_p_bf8.data), dtype=np.uint32).reshape(256, 2)
    np.testing.assert_array_equal(pk_e4[:, 0], ref_e4)
    np.testing.assert_array_equal(pk_e4[:, 1], ref_e4[::-1])
    np.testing.assert_array_equal(pk_e5[:, 0], ref_e5)
    np.testing.assert_array_equal(pk_e5[:, 1], ref_e5[::-1])

    # 3) Dense 16,384-value `f32` encode sweep in BOTH `thin.formats` and `thin.emu`:
    sweep_u32 = np.array(fnuz["encode_f32_u32"], dtype=np.uint32)
    sweep_f32 = sweep_u32.view(np.float32)
    want_e4 = np.array(fnuz["encode_e4m3fnuz_u8"], dtype=np.uint8)
    want_e5 = np.array(fnuz["encode_e5m2fnuz_u8"], dtype=np.uint8)

    np.testing.assert_array_equal(formats.encode(sweep_f32, f8_dt, saturate=False), want_e4)
    np.testing.assert_array_equal(formats.encode(sweep_f32, b8_dt, saturate=False), want_e5)

    # Saturate=True clamps finite overflows and +-Inf to 0x7F/0xFF while keeping NaN at 0x80:
    for dt, max_v, want_ns in [(f8_dt, 240.0, want_e4), (b8_dt, 57344.0, want_e5)]:
        got_sat = formats.encode(sweep_f32, dt, saturate=True)
        is_nan = np.isnan(sweep_f32)
        np.testing.assert_array_equal(got_sat[is_nan], 0x80)
        in_range = (~is_nan) & (np.abs(sweep_f32) <= max_v)
        np.testing.assert_array_equal(got_sat[in_range], want_ns[in_range])
        pos_ovf = (~is_nan) & (sweep_f32 > max_v) & (want_ns == 0x80)
        neg_ovf = (~is_nan) & (sweep_f32 < -max_v) & (want_ns == 0x80)
        np.testing.assert_array_equal(got_sat[pos_ovf], 0x7F)
        np.testing.assert_array_equal(got_sat[neg_ovf], 0xFF)

    asm_enc = """\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_load_dwordx2 s[8:9], s[0:1], 0x10
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 6, s2
\tv_add_u32 v1, v1, v0
\tv_lshlrev_b32 v2, 4, v1
\tglobal_load_dwordx4 v[4:7], v2, s[4:5]
\ts_waitcnt vmcnt(0)
\tv_cvt_pk_fp8_f32 v8, v4, v5
\tv_cvt_pk_fp8_f32 v8, v6, v7 op_sel:[0,0,1]
\tv_cvt_pk_bf8_f32 v9, v4, v5
\tv_cvt_pk_bf8_f32 v9, v6, v7 op_sel:[0,0,1]
\tv_lshlrev_b32 v3, 2, v1
\tglobal_store_dword v3, v8, s[6:7]
\tglobal_store_dword v3, v9, s[8:9]
\ts_endpgm
"""
    n_wg = len(sweep_f32) // 256
    mem_e = Memory()
    b_in_e = mem_e.alloc("in", sweep_f32.nbytes, tagged=False)
    b_in_e.write(0, sweep_f32.tobytes())
    b_out_e4 = mem_e.alloc("e4", len(sweep_f32), tagged=False)
    b_out_e5 = mem_e.alloc("e5", len(sweep_f32), tagged=False)
    ka_e = mem_e.alloc("ka", 24, tagged=False)
    ka_e.write(0, np.array([b_in_e.base, b_out_e4.base, b_out_e5.base], dtype=np.uint64).tobytes())
    with targeting("gfx942"):
        Cluster(
            asm_enc,
            {"wgsize": 64, "vgpr": 12, "agpr": 0, "user_sgprs": 2},
            mem_e,
            [ka_e],
            grid=(n_wg,),
        ).run()

    np.testing.assert_array_equal(np.frombuffer(bytes(b_out_e4.data), dtype=np.uint8), want_e4)
    np.testing.assert_array_equal(np.frombuffer(bytes(b_out_e5.data), dtype=np.uint8), want_e5)


def _rne_bits(val: mpmath.mpf, exp_bits: int, mant_bits: int) -> int:
    """Round an exact finite `mpmath.mpf` to IEEE binary32/binary64 RNE bit pattern."""
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    emax = bias
    total_bits = 1 + exp_bits + mant_bits
    if val == 0:
        return 0
    sign = 1 if val < 0 else 0
    x = abs(val)
    e = int(mpmath.floor(mpmath.log(x, 2)))
    if e < emin:
        # Subnormal step 2^(emin - mant_bits)
        step = mpmath.ldexp(1, emin - mant_bits)
        q_exact = x / step
        q_floor = int(mpmath.floor(q_exact))
        rem = q_exact - q_floor
        if rem > 0.5 or (rem == 0.5 and (q_floor & 1) == 1):
            q_floor += 1
        if q_floor >= (1 << mant_bits):
            return (sign << (total_bits - 1)) | (1 << mant_bits)
        return (sign << (total_bits - 1)) | q_floor
    step = mpmath.ldexp(1, e - mant_bits)
    q_exact = x / step
    q_floor = int(mpmath.floor(q_exact))
    rem = q_exact - q_floor
    if rem > 0.5 or (rem == 0.5 and (q_floor & 1) == 1):
        q_floor += 1
    if q_floor >= (1 << (mant_bits + 1)):
        e += 1
        q_floor >>= 1
    if e > emax:
        return (sign << (total_bits - 1)) | (((1 << exp_bits) - 1) << mant_bits)
    return (
        (sign << (total_bits - 1)) | ((e + bias) << mant_bits) | (q_floor & ((1 << mant_bits) - 1))
    )


def _f32_to_mpf(bits: int) -> mpmath.mpf:
    sign = -1 if (bits >> 31) & 1 else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0:
        return sign * mpmath.ldexp(frac, -149)
    return sign * mpmath.ldexp((1 << 23) | frac, exp - 150)


def _f64_to_mpf(bits: int) -> mpmath.mpf:
    sign = -1 if (bits >> 63) & 1 else 1
    exp = (bits >> 52) & 0x7FF
    frac = bits & ((1 << 52) - 1)
    if exp == 0:
        return sign * mpmath.ldexp(frac, -1074)
    return sign * mpmath.ldexp((1 << 52) | frac, exp - 1075)


def test_cdna3_mfma_numerics_against_independent_mpmath_and_cpu_references():
    """Cross-check SGEMM, DGEMM, I8, XF32, F16, BF16, and FP8 MFMA numerics against references."""
    mpmath.mp.prec = 256

    # 1) Adversarial single-rounded IEEE binary32 FMA (`v_mfma_f32_4x4x1_16b_f32`):
    #    a = 2^15 * (1 + 2^-23) (`0x47000001`), b = 2^15 * (1 - 2^-23) (`0x46FFFFFE`),
    #    c = 2^54 + 2^31 (`0x5A800001`, odd mantissa, half-ULP_f32 = 2^30, ULP_f64 = 4).
    #    Exact a*b = 2^30 - 2^-16 < 2^30 (strictly below the f32 midpoint!), so single-rounded
    #    IEEE `fma_f32` rounds down to `0x5A800001`, whereas `(f64(a)*f64(b) + f64(c)) as f32`
    #    double-rounds up to `2^54 + 2^30` in f64 and then ties-to-even up to `0x5A800002`!
    info_f32 = REF["instructions"]["v_mfma_f32_4x4x1_16b_f32"]
    cls_f32 = REF["layout_classes"][info_f32["layout_class"]]
    a_adv = np.full((16, 4, 1), 0x47000001, dtype=np.uint64)
    b_adv = np.full((16, 1, 4), 0x46FFFFFE, dtype=np.uint64)
    c_adv = np.full((16, 4, 4), 0x5A800001, dtype=np.uint64)
    # Also test C = signaling NaN with payload (0x7F800042) when k=0 has valid vs 0*Inf!
    c_adv[1, 0, 0] = 0x7F800042
    c_adv[2, 0, 0] = 0x7F800042
    a_adv[2, 0, 0] = 0x00000000
    b_adv[2, 0, 0] = 0x7F800000  # 0 * +Inf -> must produce DNAN32 (0xFFC00000), NOT quieted C!
    a_adv[3, 0, 0] = 0x7F800013  # A sNaN payload -> quieted to 0x7FC00013 on MI300X SGEMM!
    b_adv[3, 0, 0] = 0x3F800000
    c_adv[3, 0, 0] = 0x00000000

    exact_prod_sum = _f32_to_mpf(0x47000001) * _f32_to_mpf(0x46FFFFFE) + _f32_to_mpf(0x5A800001)
    want_single_rne = _rne_bits(exact_prod_sum, 8, 23)
    assert want_single_rne == 0x5A800001  # strictly below midpoint -> stays 0x5A800001!

    got_f32_regs = _run_mfma_asm(
        "v_mfma_f32_4x4x1_16b_f32",
        _pack_from_mc_map(a_adv, cls_f32["a"]["0:0"], 1, bits=32),
        _pack_from_mc_map(b_adv, cls_f32["b"]["0"], 1, bits=32),
        _pack_from_mc_map(c_adv, cls_f32["d"], 4, bits=32),
    )
    got_f32 = _unpack_from_mc_map(got_f32_regs, cls_f32["d"], out_bits=32)
    assert int(got_f32[0, 0, 0]) == 0x5A800001
    assert int(got_f32[1, 0, 0]) == (0x7F800042 | 0x00400000)
    assert int(got_f32[2, 0, 0]) == 0xFFC00000
    assert int(got_f32[3, 0, 0]) == 0x7FC00013

    # 2) DGEMM `v_mfma_f64_4x4x4_4b_f64` against sequential exact 53-bit RNE `mpmath` FMA
    #    PLUS `C = NaN` at k=0 followed by `0 * Inf` at k=2!
    info_f64 = REF["instructions"]["v_mfma_f64_4x4x4_4b_f64"]
    cls_f64 = REF["layout_classes"][info_f64["layout_class"]]
    rng = np.random.default_rng(942_64)
    a_f64 = rng.standard_normal((4, 4, 4)).astype(np.float64).view(np.uint64).copy()
    b_f64 = rng.standard_normal((4, 4, 4)).astype(np.float64).view(np.uint64).copy()
    c_f64 = rng.standard_normal((4, 4, 4)).astype(np.float64).view(np.uint64).copy()
    # Block 3, (0, 0): C is payload NaN, k=0..1 are finite, k=2 is 0.0 * Inf -> must return DNAN64!
    c_f64[3, 0, 0] = 0x7FF0_0000_0000_0099
    a_f64[3, 0, 2] = 0x0000_0000_0000_0000
    b_f64[3, 2, 0] = 0x7FF0_0000_0000_0000
    # Block 3, (1, 1): C is payload NaN, all k=0..3 finite -> must return quieted C!
    c_f64[3, 1, 1] = 0x7FF0_0000_0000_0077

    got_f64_regs = _run_mfma_asm(
        "v_mfma_f64_4x4x4_4b_f64",
        _pack_from_mc_map(a_f64, cls_f64["a"]["0:0"], 2, bits=64),
        _pack_from_mc_map(b_f64, cls_f64["b"]["0"], 2, bits=64),
        _pack_from_mc_map(c_f64, cls_f64["d"], 2, bits=64),
    )
    got_f64 = _unpack_from_mc_map(got_f64_regs, cls_f64["d"], out_bits=64)
    for blk in range(3):
        for i in range(4):
            for j in range(4):
                acc_bits = int(c_f64[blk, i, j])
                for kk in range(4):
                    prod_sum = _f64_to_mpf(int(a_f64[blk, i, kk])) * _f64_to_mpf(
                        int(b_f64[blk, kk, j])
                    ) + _f64_to_mpf(acc_bits)
                    acc_bits = _rne_bits(prod_sum, 11, 52)
                assert int(got_f64[blk, i, j]) == acc_bits
    assert int(got_f64[3, 0, 0]) == 0xFFF8_0000_0000_0000
    assert int(got_f64[3, 1, 1]) == (0x7FF0_0000_0000_0077 | 0x0008_0000_0000_0000)

    # 3) XDLOP (`f16`, `bf16`, `xf32`, `fp8_bf8`): cross-check `thin.emu` bit-for-bit against an
    #    independent `mpmath` implementation of the MI300X-measured 128-bit grouped XDLOP dot law,
    #    and verify agreement within `1e-5` against infinite-precision `mpmath`:
    def _mp_xdlop_dot(a_row, b_col, c_u32: int, grp: int) -> int:
        if (c_u32 & 0x7FFF_FFFF) > 0x7F80_0000:
            return 0xFFC0_0000
        acc_u = c_u32
        for s0 in range(0, len(a_row), grp):
            prods = []
            esums = []
            for kk in range(s0, min(s0 + grp, len(a_row))):
                xa = float(a_row[kk])
                yb = float(b_col[kk])
                if np.isnan(xa) or np.isnan(yb):
                    return 0xFFC0_0000
                p = mpmath.mpf(xa) * mpmath.mpf(yb)
                prods.append(p)
                if p != 0:
                    ea = int(mpmath.floor(mpmath.log(abs(xa), 2)))
                    eb = int(mpmath.floor(mpmath.log(abs(yb), 2)))
                    esums.append(ea + eb)
            if not esums:
                continue
            qe = max(esums) - 24
            q_step = mpmath.power(2, qe)
            s_int = sum(int(mpmath.mpf(p) / q_step) for p in prods if p != 0)
            c_mp = _f32_to_mpf(acc_u)
            if c_mp == 0:
                tot = mpmath.mpf(s_int) * q_step
            else:
                scaled_c = c_mp / q_step
                if scaled_c == mpmath.floor(scaled_c):
                    tot = c_mp + mpmath.mpf(s_int) * q_step
                else:
                    tot = (mpmath.floor(scaled_c) + mpmath.mpf(s_int)) * q_step
            acc_u = _rne_bits(tot, 8, 23)
        return acc_u

    for mnem, xdl_grp in [
        ("v_mfma_f32_16x16x16_f16", 8),
        ("v_mfma_f32_16x16x16_bf16", 8),
        ("v_mfma_f32_16x16x8_xf32", 4),
        ("v_mfma_f32_16x16x32_fp8_bf8", 16),
    ]:
        info = REF["instructions"][mnem]
        cls = REF["layout_classes"][info["layout_class"]]
        m, n, k = info["m"], info["n"], info["k"]
        in_bits = cls["in_bits"]
        ga, gb, gd = info["gprs"]["a"], info["gprs"]["b"], info["gprs"]["d"]

        if info["in_type"] == "fp16":
            af = rng.standard_normal((1, m, k)).astype(np.float16)
            bf = rng.standard_normal((1, k, n)).astype(np.float16)
            ac, bc = af.view(np.uint16).astype(np.uint64), bf.view(np.uint16).astype(np.uint64)
            av, bv = af.astype(np.float64), bf.astype(np.float64)
        elif info["in_type"] == "bf16":
            af = rng.standard_normal((1, m, k)).astype(np.float32)
            bf = rng.standard_normal((1, k, n)).astype(np.float32)
            ac = (af.view(np.uint32) >> 16).astype(np.uint64)
            bc = (bf.view(np.uint32) >> 16).astype(np.uint64)
            av = (ac.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            bv = (bc.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        elif info["in_type"] == "xf32":
            af = rng.standard_normal((1, m, k)).astype(np.float32)
            bf = rng.standard_normal((1, k, n)).astype(np.float32)
            ac, bc = af.view(np.uint32).astype(np.uint64), bf.view(np.uint32).astype(np.uint64)
            av = (af.view(np.uint32) & 0xFFFFE000).view(np.float32).astype(np.float64)
            bv = (bf.view(np.uint32) & 0xFFFFE000).view(np.float32).astype(np.float64)
        else:
            af8 = formats.encode(
                rng.standard_normal((1, m, k)).astype(np.float32),
                formats.DTYPES["e4m3fnuz"],
                saturate=True,
            )
            bf8 = formats.encode(
                rng.standard_normal((1, k, n)).astype(np.float32),
                formats.DTYPES["e5m2fnuz"],
                saturate=True,
            )
            ac, bc = af8.astype(np.uint64), bf8.astype(np.uint64)
            av = formats.decode(af8, formats.DTYPES["e4m3fnuz"]).astype(np.float64)
            bv = formats.decode(bf8, formats.DTYPES["e5m2fnuz"]).astype(np.float64)

        cf = rng.standard_normal((1, m, n)).astype(np.float32)
        cc = cf.view(np.uint32).astype(np.uint64)
        # Also verify XDLOP `C = payload NaN` returns canonical `0xFFC00000` (measured on MI300X):
        cc[0, 0, 0] = 0x7F80_0037

        got_regs = _run_mfma_asm(
            mnem,
            _pack_from_mc_map(ac, cls["a"]["0:0"], ga, bits=in_bits),
            _pack_from_mc_map(bc, cls["b"]["0"], gb, bits=in_bits),
            _pack_from_mc_map(cc, cls["d"], gd, bits=32),
        )
        got = _unpack_from_mc_map(got_regs, cls["d"], out_bits=32)[0]

        xdlop_ref = np.zeros((m, n), dtype=np.uint32)
        exact_bits = np.zeros((m, n), dtype=np.uint32)
        for i in range(m):
            for j in range(n):
                xdlop_ref[i, j] = _mp_xdlop_dot(av[0, i, :], bv[0, :, j], int(cc[0, i, j]), xdl_grp)
                if i == 0 and j == 0:
                    exact_bits[i, j] = 0xFFC0_0000
                else:
                    tot = _f32_to_mpf(int(cc[0, i, j]))
                    for kk in range(k):
                        tot += mpmath.mpf(float(av[0, i, kk])) * mpmath.mpf(float(bv[0, kk, j]))
                    exact_bits[i, j] = _rne_bits(tot, 8, 23)

        assert np.array_equal(got, xdlop_ref), mnem
        assert int(got[0, 0]) == 0xFFC0_0000
        finite = np.ones((m, n), dtype=bool)
        finite[0, 0] = False
        np.testing.assert_allclose(
            got[finite].view(np.float32), exact_bits[finite].view(np.float32), rtol=1e-5, atol=1e-5
        )
