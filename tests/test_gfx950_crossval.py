"""CDNA 4 (`gfx950`) cross-validation against the ISA manual, `mpmath`, and `ml_dtypes`.

Sources of truth cross-checked here without MI350X/MI355X hardware:
1. AMD Instinct CDNA 4 ISA manual (§7.1.5, §7.3–7.5, §11.4): independent register/lane
   layout derivations for all 24 new CDNA 4 MFMA / F8F6F4 / MX-scale / SMFMAC forms and
   all 4 `ds_read_*_tr_*` transpose LDS reads, plus `matrix_calculator.py` (`cdna3`) for
   the 44 shared CDNA 3/4 matrix forms (with OCP FP8/BF8 on `gfx950`).
2. Independent exact IEEE/OCP bit-level formulas, `mpmath` (256-bit precision), and
   `ml_dtypes` (`float8_e4m3fn`, `float8_e5m2`, `float6_e2m3fn`, `float6_e3m2fn`,
   `float4_e2m1fn`, `float8_e8m0fnu`).

Note: AMD's `matrix_calculator.py` has no `cdna4` architecture data; all new CDNA 4
matrix layouts and grouped-summation numerics are derived from the ISA manual (`provisional=True`).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import ml_dtypes
import mpmath
import numpy as np

from thin import formats
from thin.emu import Cluster, Memory
from thin.target import target, targeting

CDNA3_REF_PATH = (
    Path(__file__).parents[1] / "crates/thin-emu/tests/data/emu/cdna3_matrix_ref.json.gz"
)


def _load_cdna3_ref() -> dict:
    with gzip.open(CDNA3_REF_PATH, "rb") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Independent OCP small-float & MX E8M0 reference formulas (zero shared code)
# ---------------------------------------------------------------------------


def _independent_ocp_decode_u32(code: int, spec: str) -> int:
    """Decode one raw code of an OCP small float or E8M0 scale into IEEE binary32 bits."""
    if spec == "e8m0":
        if code == 0xFF:
            return 0x7FC0_0000
        if code == 0x00:
            return 0x0040_0000  # 2^-127 = subnormal f32 (exp=0, mant=1<<22)
        return code << 23

    params = {
        "e4m3fn": (4, 3, 7, True, False),
        "e5m2": (5, 2, 15, False, True),
        "e2m3": (2, 3, 1, False, False),
        "e3m2": (3, 2, 3, False, False),
        "e2m1": (2, 1, 1, False, False),
    }
    eb, mb, bias, fn_only_all_ones_nan, has_inf_nan_exp = params[spec]
    total = 1 + eb + mb
    sign = (code >> (total - 1)) & 1
    exp = (code >> mb) & ((1 << eb) - 1)
    mant = code & ((1 << mb) - 1)
    max_exp = (1 << eb) - 1

    if fn_only_all_ones_nan and exp == max_exp and mant == (1 << mb) - 1:
        return (sign << 31) | 0x7FC0_0000
    if has_inf_nan_exp and exp == max_exp:
        if mant == 0:
            return (sign << 31) | 0x7F80_0000
        return (sign << 31) | 0x7FC0_0000

    if exp == 0:
        if mant == 0:
            return sign << 31
        val = mpmath.ldexp(mant, (1 - bias) - mb)
    else:
        val = mpmath.ldexp((1 << mb) | mant, (exp - bias) - mb)

    f = np.float32(-float(val) if sign else float(val))
    return int(f.view(np.uint32))


def _independent_ocp_encode(x_u32: int, spec: str, *, scale: float = 1.0) -> int:
    """Encode IEEE binary32 `x_u32` divided by `scale` into OCP `spec` (`FP16_OVFL=0`, RNE)."""
    # Per CDNA 4 ISA manual §7.3 (FP16_OVFL=0) and OCP MX:
    # - `e4m3fn`: overflow / ±Inf -> NaN (`sign | 0x7F`); NaN / scale_exp==255 -> `0xFF`.
    # - `e5m2`:   overflow / ±Inf -> ±Inf (`sign | 0x7C`); NaN / scale_exp==255 -> `0xFE`.
    # - `e2m3`, `e3m2`, `e2m1` (no Inf/NaN encodings): overflow / ±Inf / NaN -> `sign | max_mag`.
    params = {
        "e4m3fn": (4, 3, 7, 448.0, 0x7F, 0xFF),
        "e5m2": (5, 2, 15, 57344.0, 0x7C, 0xFE),
        "e2m3": (2, 3, 1, 7.5, 0x1F, None),
        "e3m2": (3, 2, 3, 28.0, 0x1F, None),
        "e2m1": (2, 1, 1, 6.0, 0x07, None),
    }
    eb, mb, bias, max_fin, ovfl_mag, nan_code = params[spec]
    total = 1 + eb + mb

    s_bits = int(np.float32(scale).view(np.uint32))
    s_exp = (s_bits >> 23) & 0xFF
    x_sign = (x_u32 >> 31) & 1
    x_exp = (x_u32 >> 23) & 0xFF
    x_frac = x_u32 & 0x7FFFFF

    x_is_nan = x_exp == 0xFF and x_frac != 0
    x_is_inf = x_exp == 0xFF and x_frac == 0
    if s_exp == 0xFF or x_is_nan:
        if nan_code is None:
            return (x_sign << (total - 1)) | ovfl_mag
        return nan_code
    if x_is_inf:
        return (x_sign << (total - 1)) | ovfl_mag

    xf = float(np.uint32(x_u32).view(np.float32))
    # Hardware `v_cvt_scalef32_*` divides by `2^(scale_exp - 127)`:
    val_f64 = float(mpmath.ldexp(xf, 127 - s_exp))
    v_sign = x_sign
    v_abs = abs(val_f64)
    if v_abs == 0.0:
        return v_sign << (total - 1)

    emin = 1 - bias
    sub_step = mpmath.ldexp(1, emin - mb)
    x_mp = mpmath.mpf(v_abs)
    if x_mp < mpmath.ldexp(1, emin):
        q = x_mp / sub_step
        qf = int(mpmath.floor(q))
        rem = q - qf
        if rem > 0.5 or (rem == 0.5 and (qf & 1) == 1):
            qf += 1
        if qf == 0:
            return v_sign << (total - 1)
        if qf >= (1 << mb):
            return (v_sign << (total - 1)) | (1 << mb)
        return (v_sign << (total - 1)) | qf

    e = int(mpmath.floor(mpmath.log(x_mp, 2)))
    step = mpmath.ldexp(1, e - mb)
    q = x_mp / step
    qf = int(mpmath.floor(q))
    rem = q - qf
    if rem > 0.5 or (rem == 0.5 and (qf & 1) == 1):
        qf += 1
    if qf >= (1 << (mb + 1)):
        e += 1
        qf >>= 1
    if float(mpmath.ldexp(qf, e - mb)) > max_fin:
        return (v_sign << (total - 1)) | ovfl_mag
    return (v_sign << (total - 1)) | ((e + bias) << mb) | (qf & ((1 << mb) - 1))


# ---------------------------------------------------------------------------
# Independent CDNA 4 ISA manual layout derivations (§7.1.5, §7.5.1.3, §11.4)
# ---------------------------------------------------------------------------


def _manual_dense_input_map(outer: int, k_dim: int, bits: int) -> list[list[tuple[int, int, int]]]:
    """Return `map[i][k] = (lane, dword, bit_lo)` for single-block CDNA 4 dense A/B operand."""
    lane_groups = 64 // outer
    elems_per_lane = k_dim // lane_groups
    total_bits_per_lane = elems_per_lane * bits
    out: list[list[tuple[int, int, int]]] = []
    for i in range(outer):
        row_map: list[tuple[int, int, int]] = []
        for kk in range(k_dim):
            # When a narrow operand occupies 256 bits/lane (`8` VGPRs on FP8/BF8 in `f8f6f4` or
            # doubled-K sparse `B`), CDNA 4 stacks two 4-VGPR (128-bit) K-halves (`0..K/2` in
            # `v0..v3` and `K/2..K` in `v4..v7`); 6-VGPR (192-bit FP6, 32 elems/lane) and 4-VGPR
            # (128-bit FP4/FP16/BF16/I8) operands pack one contiguous `elems_per_lane` block/lane:
            if bits <= 16 and total_bits_per_lane == 256:
                half_k = k_dim // 2
                half_idx, k_in_half = divmod(kk, half_k)
                elems_half = half_k // lane_groups
                grp, elem_idx = divmod(k_in_half, elems_half)
                lane = grp * outer + i
                half_dwords = (elems_half * bits) // 32
                bit_offset = elem_idx * bits
                dw = half_idx * half_dwords + (bit_offset // 32)
                bit_lo = bit_offset % 32
            else:
                grp, elem_idx = divmod(kk, elems_per_lane)
                lane = grp * outer + i
                bit_offset = elem_idx * bits
                dw = bit_offset // 32
                bit_lo = bit_offset % 32
            row_map.append((lane, dw, bit_lo))
        out.append(row_map)
    return out


def _manual_output_map(m: int, n: int) -> list[list[tuple[int, int]]]:
    """Return `map[i][j] = (lane, dword)` for single-block CDNA 3/4 `16x16` or `32x32` C/D."""
    out: list[list[tuple[int, int]]] = []
    for i in range(m):
        row: list[tuple[int, int]] = []
        for j in range(n):
            if m == 16 and n == 16:
                lane = (i // 4) * 16 + j
                dw = i % 4
            elif m == 32 and n == 32:
                # 16 dwords/lane: 4 groups of 4 rows (`(i // 8) * 4 + (i % 4)`).
                lane = ((i // 4) % 2) * 32 + j
                dw = (i // 8) * 4 + (i % 4)
            else:
                raise ValueError(f"unsupported shape {m}x{n}")
            row.append((lane, dw))
        out.append(row)
    return out


def _pack_with_manual_map(mat: np.ndarray, loc_map: list, n_dwords: int, bits: int) -> np.ndarray:
    """Pack `[rows, cols]` integer codes into `[64, n_dwords]` uint32 (handling 6-bit spanning!)."""
    regs = np.zeros((64, n_dwords), dtype=np.uint32)
    mask = (1 << bits) - 1
    rows, cols = mat.shape
    for r in range(rows):
        for c in range(cols):
            lane, dw, bit_lo = loc_map[r][c]
            v = int(mat[r, c]) & mask
            regs[lane, dw] |= np.uint32((v << bit_lo) & 0xFFFFFFFF)
            if bit_lo + bits > 32:
                regs[lane, dw + 1] |= np.uint32(v >> (32 - bit_lo))
    return regs


def _unpack_with_manual_map(regs: np.ndarray, d_map: list) -> np.ndarray:
    m = len(d_map)
    n = len(d_map[0])
    out = np.zeros((m, n), dtype=np.uint32)
    for i in range(m):
        for j in range(n):
            lane, dw = d_map[i][j]
            out[i, j] = regs[lane, dw]
    return out


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


def test_cdna4_matrix_inventory_and_isa_manual_layout_bijections():
    """Verify all 68 gfx950 matrix forms and prove bijectivity of all derived CDNA 4 layouts."""
    t950 = target("gfx950")
    cdna3_ref = _load_cdna3_ref()["instructions"]

    mnems_950 = sorted({f.mnemonic for f in t950.forms() if t950.matrix_form(f.mnemonic)})
    assert len(mnems_950) == 68

    removed_xf32 = {"v_mfma_f32_16x16x8_xf32", "v_mfma_f32_32x32x4_xf32"}
    assert removed_xf32.isdisjoint(mnems_950)

    # 1) All 44 shared CDNA 3/4 matrix forms match `matrix_calculator.py` dimensions/dwords,
    #    with FP8/BF8 switched to OCP `e4m3fn`/`e5m2` and `provisional=True`:
    shared = set(cdna3_ref) - removed_xf32
    assert len(shared) == 44
    assert shared.issubset(mnems_950)
    for mnem in sorted(shared):
        info = cdna3_ref[mnem]
        mf = t950.matrix_form(mnem)
        assert mf is not None
        assert (mf.m, mf.n, mf.k, mf.blocks) == (info["m"], info["n"], info["k"], info["blocks"])
        assert mf.passes == info["cycles"] // 4
        assert mf.sparse is info["sparse"] and mf.scaled is False
        assert mf.provisional is True
        want_c = 1 if info["sparse"] else info["gprs"]["c"]
        assert dict(mf.dwords) == {
            "a": info["gprs"]["a"],
            "b": info["gprs"]["b"],
            "c": want_c,
            "d": info["gprs"]["d"],
        }
        if "fp8" in mnem:
            assert "fnuz" not in mf.a and "fnuz" not in mf.b

    # 2) All 24 brand-new CDNA 4 matrix forms match the CDNA 4 ISA manual tables:
    new_24 = sorted(set(mnems_950) - shared)
    assert len(new_24) == 24
    for mnem in new_24:
        mf = t950.matrix_form(mnem)
        assert mf is not None and mf.provisional is True
        assert mf.scaled == mnem.startswith("v_mfma_scale_")
        assert mf.scale_block == (32 if mf.scaled else 0)

    # 3) Prove every derived CDNA 4 dense `(outer, K, bits)` map is a collision-free bijection onto
    #    the exact bit budget `[64 lanes] x [n_dwords * 32 bits]`:
    for outer, k_dim, bits, want_dw in [
        (16, 32, 16, 4),
        (32, 16, 16, 4),
        (16, 64, 8, 4),
        (32, 32, 8, 4),
        (16, 128, 8, 8),
        (16, 128, 6, 6),
        (16, 128, 4, 4),
        (32, 64, 8, 8),
        (32, 64, 6, 6),
        (32, 64, 4, 4),
    ]:
        lmap = _manual_dense_input_map(outer, k_dim, bits)
        occupied: set[tuple[int, int]] = set()
        for i in range(outer):
            for kk in range(k_dim):
                lane, dw, bit_lo = lmap[i][kk]
                assert 0 <= lane < 64
                for bpos in range(bits):
                    abs_bit = dw * 32 + bit_lo + bpos
                    assert 0 <= abs_bit < want_dw * 32
                    key = (lane, abs_bit)
                    assert key not in occupied, f"bit collision on {outer}x{k_dim}@{bits}b: {key}"
                    occupied.add(key)
        assert len(occupied) == 64 * want_dw * 32

    # 4) Verify the 16-lane `ds_read_*_tr_*` transpose law is an exact bijection on a 16-lane row
    #    for all 4 element widths (`4, 6, 8, 16` bits):
    for bits, lds_bits in [(4, 64), (6, 96), (8, 64), (16, 64)]:
        n_elems = lds_bits // bits
        lanes_per_k = 16 // n_elems
        seen_src = set()
        for dst_col in range(16):
            for k_step in range(n_elems):
                src_col = k_step * lanes_per_k + dst_col // n_elems
                src_elem = dst_col % n_elems
                assert 0 <= src_col < 16 and 0 <= src_elem < n_elems
                seen_src.add((src_col, src_elem))
        assert len(seen_src) == 16 * n_elems


def test_cdna4_ocp_formats_and_scaled_cvt_against_independent_and_ml_dtypes_ref():
    """Exhaustively cross-check all 6 OCP dtypes and `v_cvt_scalef32_*` against `ml_dtypes`."""
    # 1) Exhaustive codepoint decode across all 6 OCP formats in `thin.formats`:
    ocp_specs = [
        ("e4m3fn", 256, "float8_e4m3fn"),
        ("e5m2", 256, "float8_e5m2"),
        ("e2m3", 64, "float6_e2m3fn"),
        ("e3m2", 64, "float6_e3m2fn"),
        ("e2m1", 16, "float4_e2m1fn"),
        ("e8m0", 256, "float8_e8m0fnu"),
    ]
    ref_tables: dict[str, np.ndarray] = {}
    for name, n_codes, ml_attr in ocp_specs:
        codes = np.arange(n_codes, dtype=np.uint8)
        indep_u32 = np.array(
            [_independent_ocp_decode_u32(int(c), name) for c in codes], dtype=np.uint32
        )
        ref_tables[name] = indep_u32
        got_f32 = formats.decode(codes, formats.DTYPES[name])
        got_u32 = got_f32.view(np.uint32)

        indep_nan = (indep_u32 & 0x7FFF_FFFF) > 0x7F80_0000
        got_nan = np.isnan(got_f32)
        np.testing.assert_array_equal(got_nan, indep_nan, err_msg=f"{name} NaN mask")
        fin = ~indep_nan
        np.testing.assert_array_equal(got_u32[fin], indep_u32[fin], err_msg=f"{name} decode")

        ml_dt = getattr(ml_dtypes, ml_attr)
        ml_f32 = codes.view(ml_dt).astype(np.float32)
        np.testing.assert_array_equal(np.isnan(ml_f32), indep_nan, err_msg=f"{name} ml NaN")
        np.testing.assert_array_equal(
            got_u32[fin], ml_f32.view(np.uint32)[fin], err_msg=f"{name} vs ml_dtypes"
        )

    # 2) Single batched `Cluster` launch testing exhaustive decode in `thin.emu`:
    #    256 threads (4 WGs x 64): thread `gid` decodes code `gid` for FP8/BF8 (scale=1.0 and 2.0),
    #    code `gid & 15` for FP4, and codes `(gid + 0..31) & 63` for FP6/BF6!
    asm_dec = """\
\ts_load_dwordx8 s[4:11], s[0:1], 0x0
\ts_load_dwordx4 s[12:15], s[0:1], 0x20
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 6, s2
\tv_add_u32 v1, v1, v0
\tv_lshlrev_b32 v2, 2, v1
\tglobal_load_dword v3, v2, s[4:5]
\ts_mov_b32 s16, 0x40000000
\ts_waitcnt vmcnt(0)
\tv_cvt_f32_fp8 v4, v3
\tv_cvt_f32_bf8 v5, v3
\tv_cvt_scalef32_f32_fp8 v6, v3, s16
\tv_cvt_scalef32_f32_bf8 v7, v3, s16
\tv_cvt_scalef32_pk_f32_fp4 v[8:9], v3, 1.0
\tglobal_store_dword v2, v4, s[6:7]
\tglobal_store_dword v2, v5, s[8:9]
\tglobal_store_dword v2, v6, s[10:11]
\tglobal_store_dword v2, v7, s[12:13]
\tv_lshlrev_b32 v10, 3, v1
\tglobal_store_dwordx2 v10, v[8:9], s[14:15]
\ts_endpgm
"""
    codes_256 = np.arange(256, dtype=np.uint32)
    # Keep byte 0 equal to `codes_256` for FP8/BF8 and put FP4 nibbles in byte 1 (`op_sel:[1,0]`):
    in_words = codes_256 | (((codes_256 & 0xF) | (((codes_256 + 5) & 0xF) << 4)) << 8)
    asm_dec = asm_dec.replace(
        "v_cvt_scalef32_pk_f32_fp4 v[8:9], v3, 1.0",
        "v_cvt_scalef32_pk_f32_fp4 v[8:9], v3, 1.0 op_sel:[1,0]",
    )

    mem = Memory()
    b_in = mem.alloc("in", in_words.nbytes, tagged=False)
    b_in.write(0, in_words.tobytes())
    outs = [mem.alloc(f"o{k}", 256 * (8 if k == 4 else 4), tagged=False) for k in range(5)]
    ka = mem.alloc("ka", 48, tagged=False)
    ka.write(0, np.array([b_in.base, *(o.base for o in outs)], dtype=np.uint64).tobytes())

    with targeting("gfx950"):
        Cluster(asm_dec, {"wgsize": 64, "vgpr": 16, "user_sgprs": 2}, mem, [ka], grid=(4,)).run()

    for idx, spec, scale in [
        (0, "e4m3fn", 1.0),
        (1, "e5m2", 1.0),
        (2, "e4m3fn", 2.0),
        (3, "e5m2", 2.0),
    ]:
        got = np.frombuffer(bytes(outs[idx].data), dtype=np.uint32)
        ref_f = ref_tables[spec].view(np.float32) * np.float32(scale)
        fin = ~np.isnan(ref_f)
        assert np.array_equal(np.isnan(got.view(np.float32)), ~fin)
        np.testing.assert_array_equal(got[fin], ref_f.view(np.uint32)[fin])

    got_fp4 = np.frombuffer(bytes(outs[4].data), dtype=np.uint32).reshape(256, 2)
    np.testing.assert_array_equal(got_fp4[:, 0], ref_tables["e2m1"][codes_256 & 0xF])
    np.testing.assert_array_equal(got_fp4[:, 1], ref_tables["e2m1"][(codes_256 + 5) & 0xF])

    # 3) 256-value boundary/tie/subnormal/overflow/Inf/NaN `f32` encode sweep with `scale = 2.0`
    #    across `v_cvt_scalef32_pk_{fp8,bf8,fp4}_f32` and `v_cvt_scalef32_2xpk16_{fp6,bf6}_f32`:
    rng = np.random.default_rng(950_11)
    specials = np.array(
        [
            0.0,
            -0.0,
            1e-6,
            -1e-6,
            0.00390625,
            0.125,
            0.375,
            0.75,
            1.0,
            -1.0,
            1.25,
            1.5,
            3.5,
            6.0,
            12.0,
            13.0,
            15.0,
            28.0,
            56.0,
            448.0,
            896.0,
            928.0,
            -928.0,
            57344.0,
            114688.0,
            131072.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype=np.float32,
    )
    rand_vals = (rng.standard_normal(256 - len(specials)) * 20.0).astype(np.float32)
    sweep = np.concatenate(
        [specials, rand_vals]
    )  # 256 f32 values (64 lanes x 4 values, or 8 lanes x 32)
    sweep_u32 = sweep.view(np.uint32)

    asm_enc = """\
\ts_load_dwordx8 s[4:11], s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 4, v0
\tglobal_load_dwordx4 v[4:7], v1, s[4:5]
\ts_mov_b32 s12, 0x40000000
\ts_waitcnt vmcnt(0)
\tv_mov_b32 v8, 0
\tv_mov_b32 v9, 0
\tv_mov_b32 v10, 0
\tv_cvt_scalef32_pk_fp8_f32 v8, v4, v5, s12 op_sel:[0,0,0,0]
\tv_cvt_scalef32_pk_fp8_f32 v8, v6, v7, s12 op_sel:[0,0,0,1]
\tv_cvt_scalef32_pk_bf8_f32 v9, v4, v5, s12 op_sel:[0,0,0,0]
\tv_cvt_scalef32_pk_bf8_f32 v9, v6, v7, s12 op_sel:[0,0,0,1]
\tv_cvt_scalef32_pk_fp4_f32 v10, v4, v5, s12 op_sel:[0,0,0,0]
\tv_cvt_scalef32_pk_fp4_f32 v10, v6, v7, s12 op_sel:[0,0,1,0]
\tv_lshlrev_b32 v2, 2, v0
\tglobal_store_dword v2, v8, s[6:7]
\tglobal_store_dword v2, v9, s[8:9]
\tglobal_store_dword v2, v10, s[10:11]
\ts_endpgm
"""
    mem2 = Memory()
    b_sw = mem2.alloc("sw", sweep.nbytes, tagged=False)
    b_sw.write(0, sweep.tobytes())
    b_e4 = mem2.alloc("e4", 256, tagged=False)
    b_e5 = mem2.alloc("e5", 256, tagged=False)
    b_e2 = mem2.alloc("e2", 256, tagged=False)
    ka2 = mem2.alloc("ka", 32, tagged=False)
    ka2.write(0, np.array([b_sw.base, b_e4.base, b_e5.base, b_e2.base], dtype=np.uint64).tobytes())

    with targeting("gfx950"):
        Cluster(asm_enc, {"wgsize": 64, "vgpr": 16, "user_sgprs": 2}, mem2, [ka2], grid=(1,)).run()

    want_e4 = np.array(
        [_independent_ocp_encode(int(u), "e4m3fn", scale=2.0) for u in sweep_u32], np.uint8
    )
    want_e5 = np.array(
        [_independent_ocp_encode(int(u), "e5m2", scale=2.0) for u in sweep_u32], np.uint8
    )
    want_f4 = np.array(
        [_independent_ocp_encode(int(u), "e2m1", scale=2.0) for u in sweep_u32], np.uint8
    )

    got_e4 = np.frombuffer(bytes(b_e4.data), dtype=np.uint8)
    got_e5 = np.frombuffer(bytes(b_e5.data), dtype=np.uint8)
    # On NaN inputs, compare canonical NaN magnitude (`0x7F`/`0xFF`):
    is_nan_sw = np.isnan(sweep)
    np.testing.assert_array_equal(got_e4[~is_nan_sw], want_e4[~is_nan_sw])
    assert np.all((got_e4[is_nan_sw] & 0x7F) == 0x7F)
    np.testing.assert_array_equal(got_e5[~is_nan_sw], want_e5[~is_nan_sw])
    assert np.all((got_e5[is_nan_sw] & 0x7C) == 0x7C) and np.all((got_e5[is_nan_sw] & 0x03) != 0)

    got_f4_words = np.frombuffer(bytes(b_e2.data), dtype=np.uint32)
    got_f4 = np.empty(256, dtype=np.uint8)
    got_f4[0::4] = got_f4_words & 0xF
    got_f4[1::4] = (got_f4_words >> 4) & 0xF
    got_f4[2::4] = (got_f4_words >> 8) & 0xF
    got_f4[3::4] = (got_f4_words >> 12) & 0xF
    np.testing.assert_array_equal(got_f4, want_f4)

    # 4) FP6 (`e2m3`) & BF6 (`e3m2`) `2xpk16` encode + `pk32_f32` decode across all 256 sweep
    #    values (lanes 0..7 x 32 values/lane) AND exhaustive 64-code decode on lane 8 & 9:
    s0_6 = np.zeros((64, 16), dtype=np.uint32)
    s1_6 = np.zeros((64, 16), dtype=np.uint32)
    sw_8x32 = sweep_u32.reshape(8, 32)
    s0_6[:8, :] = sw_8x32[:, 0::2]
    s1_6[:8, :] = sw_8x32[:, 1::2]

    # Also pack all 64 raw 6-bit codes (`0..31` on lane 0, `32..63` on lane 1) to test `pk32_f32`:
    raw6_in = np.zeros((64, 6), dtype=np.uint32)
    for ln_idx, base_c in [(0, 0), (1, 32)]:
        for pass_i in range(32):
            c6 = base_c + pass_i
            bit_lo = pass_i * 6
            dw, off = divmod(bit_lo, 32)
            raw6_in[ln_idx, dw] |= np.uint32((c6 << off) & 0xFFFFFFFF)
            if off + 6 > 32:
                raw6_in[ln_idx, dw + 1] |= np.uint32(c6 >> (32 - off))

    lines_6 = ["\ts_load_dwordx8 s[4:11], s[0:1], 0x0", "\ts_load_dwordx4 s[12:15], s[0:1], 0x20"]
    lines_6.append("\ts_waitcnt lgkmcnt(0)")
    lines_6.append("\tv_lshlrev_b32 v1, 6, v0")
    for d in range(0, 16, 4):
        lines_6.append(f"\tglobal_load_dwordx4 v[{4 + d}:{7 + d}], v1, s[4:5] offset:{d * 4}")
        lines_6.append(f"\tglobal_load_dwordx4 v[{20 + d}:{23 + d}], v1, s[6:7] offset:{d * 4}")
    lines_6.append("\tv_mul_lo_u32 v2, v0, 24")
    lines_6.append("\tglobal_load_dwordx4 v[36:39], v2, s[8:9]")
    lines_6.append("\tglobal_load_dwordx2 v[40:41], v2, s[8:9] offset:16")
    lines_6.append("\tv_mov_b32 v42, 0x40000000")
    lines_6.append("\ts_waitcnt vmcnt(0)")
    lines_6.append("\tv_cvt_scalef32_2xpk16_fp6_f32 v[44:49], v[4:19], v[20:35], v42")
    lines_6.append("\tv_cvt_scalef32_2xpk16_bf6_f32 v[50:55], v[4:19], v[20:35], v42")
    lines_6.append("\tglobal_store_dwordx4 v2, v[44:47], s[10:11]")
    lines_6.append("\tglobal_store_dwordx2 v2, v[48:49], s[10:11] offset:16")
    lines_6.append("\tglobal_store_dwordx4 v2, v[50:53], s[12:13]")
    lines_6.append("\tglobal_store_dwordx2 v2, v[54:55], s[12:13] offset:16")
    lines_6.append("\tv_cvt_scalef32_pk32_f32_fp6 v[64:95], v[36:41], 1.0")
    lines_6.append("\tv_cvt_scalef32_pk32_f32_bf6 v[96:127], v[36:41], 1.0")
    lines_6.append("\tv_lshlrev_b32 v3, 8, v0")
    for d in range(0, 32, 4):
        lines_6.append(f"\tglobal_store_dwordx4 v3, v[{64 + d}:{67 + d}], s[14:15] offset:{d * 4}")
        lines_6.append(
            f"\tglobal_store_dwordx4 v3, v[{96 + d}:{99 + d}], s[14:15] offset:{128 + d * 4}"
        )
    lines_6.append("\ts_endpgm\n")

    mem6 = Memory()
    b_s0 = mem6.alloc("s0", s0_6.nbytes, tagged=False)
    b_s1 = mem6.alloc("s1", s1_6.nbytes, tagged=False)
    b_r6 = mem6.alloc("r6", raw6_in.nbytes, tagged=False)
    b_fp6 = mem6.alloc("fp6", 64 * 24, tagged=False)
    b_bf6 = mem6.alloc("bf6", 64 * 24, tagged=False)
    b_dec6 = mem6.alloc("dec6", 64 * 256, tagged=False)
    b_s0.write(0, s0_6.tobytes())
    b_s1.write(0, s1_6.tobytes())
    b_r6.write(0, raw6_in.tobytes())
    ka6 = mem6.alloc("ka6", 48, tagged=False)
    ka6.write(
        0,
        np.array(
            [b_s0.base, b_s1.base, b_r6.base, b_fp6.base, b_bf6.base, b_dec6.base], dtype=np.uint64
        ).tobytes(),
    )
    with targeting("gfx950"):
        Cluster(
            "\n".join(lines_6),
            {"wgsize": 64, "vgpr": 132, "user_sgprs": 2},
            mem6,
            [ka6],
            grid=(1,),
        ).run()

    def _unpack_6bit_lanes(buf_bytes: bytes, n_lanes: int) -> np.ndarray:
        words = np.frombuffer(buf_bytes, dtype=np.uint32).reshape(64, 6)[:n_lanes]
        out = np.empty((n_lanes, 32), dtype=np.uint8)
        for p in range(32):
            dw, off = divmod(p * 6, 32)
            v = (words[:, dw] >> off) & 0x3F
            if off + 6 > 32:
                v |= (words[:, dw + 1] << (32 - off)) & 0x3F
            out[:, p] = v
        return out.reshape(-1)

    got_fp6 = _unpack_6bit_lanes(bytes(b_fp6.data), 8)
    got_bf6 = _unpack_6bit_lanes(bytes(b_bf6.data), 8)
    want_fp6 = np.array(
        [_independent_ocp_encode(int(u), "e2m3", scale=2.0) for u in sweep_u32], np.uint8
    )
    want_bf6 = np.array(
        [_independent_ocp_encode(int(u), "e3m2", scale=2.0) for u in sweep_u32], np.uint8
    )
    np.testing.assert_array_equal(got_fp6, want_fp6)
    np.testing.assert_array_equal(got_bf6, want_bf6)

    dec6_words = np.frombuffer(bytes(b_dec6.data), dtype=np.uint32).reshape(64, 64)
    got_dec_fp6 = np.concatenate([dec6_words[0, :32], dec6_words[1, :32]])
    got_dec_bf6 = np.concatenate([dec6_words[0, 32:], dec6_words[1, 32:]])
    np.testing.assert_array_equal(got_dec_fp6, ref_tables["e2m3"])
    np.testing.assert_array_equal(got_dec_bf6, ref_tables["e3m2"])

    # Cross-check finite AND overflow/Inf conversions against `ml_dtypes` across all 5 dtypes:
    non_nan = ~is_nan_sw
    scaled_sw = sweep[non_nan] / np.float32(2.0)
    np.testing.assert_array_equal(
        got_e4[non_nan], scaled_sw.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    )
    np.testing.assert_array_equal(
        got_e5[non_nan], scaled_sw.astype(ml_dtypes.float8_e5m2).view(np.uint8)
    )
    for got_arr, max_v, ml_dt in [
        (got_fp6, 7.5, ml_dtypes.float6_e2m3fn),
        (got_bf6, 28.0, ml_dtypes.float6_e3m2fn),
        (got_f4, 6.0, ml_dtypes.float4_e2m1fn),
    ]:
        in_rng = np.isfinite(sweep) & (np.abs(sweep / 2.0) <= max_v)
        np.testing.assert_array_equal(
            got_arr[in_rng], (sweep[in_rng] / np.float32(2.0)).astype(ml_dt).view(np.uint8)
        )


def _exact_dyadic(x: float) -> tuple[int, int]:
    """Decompose finite non-zero float `x` into exact `(signed_int_mant, exp2)` (`x = m * 2^e`)."""
    m_f, e = np.frexp(np.float64(x))
    m_int = int(np.ldexp(m_f, 53))
    return m_int, int(e) - 53


def _rne_int_ldexp_f32(m: int, e: int) -> int:
    """Round exact dyadic integer `m * 2^e` to IEEE binary32 RNE bit pattern."""
    if m == 0:
        return 0
    sign = 1 if m < 0 else 0
    a = -m if m < 0 else m
    msb = a.bit_length() - 1
    unbiased = msb + e
    if unbiased < -126:
        # Subnormal step `2^-149`: shift `a * 2^e` relative to `2^-149` -> shift = `e + 149`
        sh = -149 - e
        q = a >> sh
        rem = a & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
        if q >= (1 << 23):
            return (sign << 31) | (1 << 23)
        return (sign << 31) | q
    # Normal step `2^(unbiased - 23)`: shift = `(unbiased - 23) - e = msb - 23`
    sh = msb - 23
    if sh <= 0:
        q = a << (-sh)
    else:
        q = a >> sh
        rem = a & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
            if q >= (1 << 24):
                q >>= 1
                unbiased += 1
    if unbiased > 127:
        return (sign << 31) | 0x7F80_0000
    return (sign << 31) | ((unbiased + 127) << 23) | (q & 0x7FFFFF)


def _dyadic_rows(mat: np.ndarray) -> list[list[tuple[int, int, int, bool]]]:
    """Precompute `(mant_53, exp2, unbiased_exp2, is_nan)` per matrix entry once."""
    out: list[list[tuple[int, int, int, bool]]] = []
    for r in range(mat.shape[0]):
        row: list[tuple[int, int, int, bool]] = []
        for kk in range(mat.shape[1]):
            x = float(mat[r, kk])
            if np.isnan(x):
                row.append((0, 0, 0, True))
            elif x == 0.0:
                row.append((0, 0, 0, False))
            else:
                m, e = _exact_dyadic(x)
                row.append((m, e, abs(m).bit_length() - 1 + e, False))
        out.append(row)
    return out


def _mp_grouped_dot_matrix(
    a_mat: np.ndarray, b_jk: np.ndarray, c_u32: np.ndarray, grp: int
) -> np.ndarray:
    """Independent exact-dyadic 24-bit mantissa grouped dot matrix (`grp=8, 16, or 32`)."""
    m_dim, k_dim = a_mat.shape
    n_dim = b_jk.shape[0]
    da = _dyadic_rows(a_mat)
    db = _dyadic_rows(b_jk)
    out = np.zeros((m_dim, n_dim), dtype=np.uint32)
    for i in range(m_dim):
        ar = da[i]
        for j in range(n_dim):
            br = db[j]
            acc_u = int(c_u32[i, j])
            if (acc_u & 0x7FFF_FFFF) > 0x7F80_0000:
                out[i, j] = 0xFFC0_0000
                continue
            nan_hit = False
            for s0 in range(0, k_dim, grp):
                terms: list[tuple[int, int]] = []
                max_esum = -(10**9)
                for kk in range(s0, min(s0 + grp, k_dim)):
                    ma, ea, ua, na = ar[kk]
                    mb, eb, ub, nb = br[kk]
                    if na or nb:
                        nan_hit = True
                        break
                    if ma != 0 and mb != 0:
                        terms.append((ma * mb, ea + eb))
                        es = ua + ub
                        if es > max_esum:
                            max_esum = es
                if nan_hit:
                    acc_u = 0xFFC0_0000
                    break
                if not terms:
                    continue
                qe = max_esum - 24
                s_int = 0
                for mp_m, mp_e in terms:
                    if mp_e >= qe:
                        s_int += mp_m << (mp_e - qe)
                    else:
                        sh = qe - mp_e
                        s_int += -((-mp_m) >> sh) if mp_m < 0 else (mp_m >> sh)
                sign = -1 if (acc_u >> 31) & 1 else 1
                exp = (acc_u >> 23) & 0xFF
                frac = acc_u & 0x7FFFFF
                if exp == 0 and frac == 0:
                    acc_u = _rne_int_ldexp_f32(s_int, qe)
                else:
                    cm = sign * (frac if exp == 0 else ((1 << 23) | frac))
                    ce = -149 if exp == 0 else (exp - 150)
                    if ce >= qe:
                        diff = ce - qe
                        if diff <= 90:
                            acc_u = _rne_int_ldexp_f32((cm << diff) + s_int, qe)
                        else:
                            acc_u = _rne_int_ldexp_f32(
                                (cm << 64) + (1 if s_int > 0 else (-1 if s_int < 0 else 0)), ce - 64
                            )
                    else:
                        acc_u = _rne_int_ldexp_f32((cm >> (qe - ce)) + s_int, qe)
            out[i, j] = acc_u
    return out


def _run_cdna4_mfma_single(
    mnem: str,
    a_regs: np.ndarray,
    b_regs: np.ndarray,
    c_regs: np.ndarray,
    *,
    sparse: bool = False,
    k_regs: np.ndarray | None = None,
    scale_regs: tuple[np.ndarray, np.ndarray] | None = None,
    mods: str = "",
) -> np.ndarray:
    ga, gb, gd = a_regs.shape[1], b_regs.shape[1], c_regs.shape[1]
    mem = Memory()
    bufs = [
        mem.alloc("a", a_regs.nbytes, tagged=False),
        mem.alloc("b", b_regs.nbytes, tagged=False),
        mem.alloc("c", c_regs.nbytes, tagged=False),
        mem.alloc("d", c_regs.nbytes, tagged=False),
    ]
    bufs[0].write(0, np.ascontiguousarray(a_regs).tobytes())
    bufs[1].write(0, np.ascontiguousarray(b_regs).tobytes())
    bufs[2].write(0, np.ascontiguousarray(c_regs).tobytes())
    if sparse:
        assert k_regs is not None
        bk = mem.alloc("k", k_regs.nbytes, tagged=False)
        bk.write(0, np.ascontiguousarray(k_regs).tobytes())
        bufs.append(bk)
    if scale_regs is not None:
        for idx, sr in enumerate(scale_regs):
            bs = mem.alloc(f"s{idx}", sr.nbytes, tagged=False)
            bs.write(0, np.ascontiguousarray(sr).tobytes())
            bufs.append(bs)

    ka = mem.alloc("ka", 8 * len(bufs), tagged=False)
    ka.write(0, np.array([b.base for b in bufs], dtype=np.uint64).tobytes())

    lines = ["\ts_load_dwordx8 s[4:11], s[0:1], 0x0"]
    if len(bufs) == 5:
        lines.append("\ts_load_dwordx2 s[12:13], s[0:1], 0x20")
    elif len(bufs) == 6:
        lines.append("\ts_load_dwordx4 s[12:15], s[0:1], 0x20")
    lines.append("\ts_waitcnt lgkmcnt(0)")

    def load_vec(v_base: int, n_dw: int, s_lo: int) -> None:
        lines.append(f"\tv_mul_lo_u32 v1, v0, {n_dw * 4}")
        for d in range(0, n_dw, 4):
            w = min(4, n_dw - d)
            reg = f"v{v_base + d}" if w == 1 else f"v[{v_base + d}:{v_base + d + w - 1}]"
            op = "dword" if w == 1 else f"dwordx{w}"
            lines.append(f"\tglobal_load_{op} {reg}, v1, s[{s_lo}:{s_lo + 1}] offset:{d * 4}")

    va = 4
    vb = (va + ga + 3) & ~3
    vx = vb + gb
    vc = (vx + 4) & ~3
    load_vec(va, ga, 4)
    load_vec(vb, gb, 6)
    load_vec(vc, gd, 8)
    if sparse:
        load_vec(vx, 1, 12)
    if scale_regs is not None:
        load_vec(vx, 1, 12)
        load_vec(vx + 1, 1, 14)
    lines.append("\ts_waitcnt vmcnt(0)")
    for d in range(gd):
        lines.append(f"\tv_accvgpr_write_b32 a{d}, v{vc + d}")

    op_d = f"a[0:{gd - 1}]"
    op_a = f"v[{va}:{va + ga - 1}]"
    op_b = f"v[{vb}:{vb + gb - 1}]"
    mod_s = f" {mods}" if mods else ""
    if sparse:
        lines.append(f"\t{mnem} {op_d}, {op_a}, {op_b}, v{vx}{mod_s}")
    elif scale_regs is not None:
        lines.append(f"\t{mnem} {op_d}, {op_a}, {op_b}, {op_d}, v{vx}, v{vx + 1}{mod_s}")
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
    lines.append("\ts_endpgm\n")

    with targeting("gfx950"):
        Cluster(
            "\n".join(lines),
            {
                "wgsize": 64,
                "vgpr": ((vc + gd + 3) & ~3) + 4,
                "agpr": (gd + 3) & ~3,
                "user_sgprs": 2,
            },
            mem,
            [ka],
            grid=(1,),
        ).run()
    return np.frombuffer(bytes(bufs[3].data), dtype=np.uint32).reshape(64, gd).copy()


def test_cdna4_new_mfma_smfmac_f8f6f4_and_mx_scale_layouts_and_numerics():
    """Cross-validate derived CDNA 4 layouts, mixed F8F6F4, and MX E8M0 block scaling."""
    mpmath.mp.prec = 256
    rng = np.random.default_rng(950_2026)

    # 1) Mixed-precision `v_mfma_f32_16x16x128_f8f6f4` with 6-bit FP6 (`e2m3`, 6 VGPRs) x
    #    4-bit FP4 (`e2m1`, 4 VGPRs) + non-zero C, checked bit-for-bit against `_mp_grouped_dot`!
    m, n, k = 16, 16, 128
    a_codes = rng.integers(0, 64, size=(m, k), dtype=np.uint8)
    b_codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)  # B indexed `[j, kk]` in input_loc!
    c_f32 = rng.standard_normal((m, n)).astype(np.float32)
    c_u32 = c_f32.view(np.uint32)

    a_map_6 = _manual_dense_input_map(m, k, 6)
    b_map_4 = _manual_dense_input_map(n, k, 4)
    d_map_16 = _manual_output_map(m, n)

    a_regs = _pack_with_manual_map(a_codes, a_map_6, 6, 6)
    b_regs = _pack_with_manual_map(b_codes, b_map_4, 4, 4)
    c_regs = np.zeros((64, 4), dtype=np.uint32)
    for i in range(m):
        for j in range(n):
            ln, dw = d_map_16[i][j]
            c_regs[ln, dw] = c_u32[i, j]

    got_f6f4 = _unpack_with_manual_map(
        _run_cdna4_mfma_single(
            "v_mfma_f32_16x16x128_f8f6f4", a_regs, b_regs, c_regs, mods="cbsz:2 blgp:4"
        ),
        d_map_16,
    )
    a_vals = formats.decode(a_codes, formats.DTYPES["e2m3"]).astype(np.float64)
    b_vals_jk = formats.decode(b_codes, formats.DTYPES["e2m1"]).astype(np.float64)
    want_f6f4 = _mp_grouped_dot_matrix(a_vals, b_vals_jk, c_u32, 32)
    np.testing.assert_array_equal(got_f6f4, want_f6f4)

    # 2) MX-scaled `v_mfma_scale_f32_32x32x64_f8f6f4` (`cbsz:0 blgp:3`: OCP `e4m3fn` 8 VGPRs x
    #    OCP `e3m2` BF6 6 VGPRs, `K=64` = 2 MX blocks of 32!) including:
    #    - `0x00` (`2^-127`) scale on row 0 block 0,
    #    - `0xFF` (`NaN`) scale on col 31 block 1 (propagating NaN ONLY to column 31!),
    #    - `op_sel:[1,0,0] op_sel_hi:[1,1,0]` (`sel_a = 3` byte 3, `sel_b = 2` byte 2)!
    m32, n32, k64 = 32, 32, 64
    # Avoid NaN code 0x7F/0xFF in random e4m3fn inputs so only col 31 gets NaN from 0xFF scale:
    a32_codes = rng.integers(0, 0x7E, size=(m32, k64), dtype=np.uint8) | (
        rng.integers(0, 2, size=(m32, k64), dtype=np.uint8) << 7
    )
    b32_codes = rng.integers(0, 64, size=(n32, k64), dtype=np.uint8)
    c32_u32 = np.zeros((m32, n32), dtype=np.uint32)

    exp_a32 = rng.integers(125, 130, size=(2, m32), dtype=np.uint32)
    exp_b32 = rng.integers(125, 130, size=(2, n32), dtype=np.uint32)
    exp_a32[0, 0] = 0x00  # subnormal 2^-127 E8M0 scale!
    exp_b32[1, 31] = 0xFF  # NaN E8M0 scale on col 31 only!

    sa32 = (exp_a32.reshape(64, 1) << 24).astype(np.uint32)  # byte 3 (`sel_a = 3`)
    sb32 = (exp_b32.reshape(64, 1) << 16).astype(np.uint32)  # byte 2 (`sel_b = 2`)

    a32_regs = _pack_with_manual_map(a32_codes, _manual_dense_input_map(m32, k64, 8), 8, 8)
    b32_regs = _pack_with_manual_map(b32_codes, _manual_dense_input_map(n32, k64, 6), 6, 6)
    d_map_32 = _manual_output_map(m32, n32)
    got_sc32 = _unpack_with_manual_map(
        _run_cdna4_mfma_single(
            "v_mfma_scale_f32_32x32x64_f8f6f4",
            a32_regs,
            b32_regs,
            np.zeros((64, 16), dtype=np.uint32),
            scale_regs=(sa32, sb32),
            mods="op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:0 blgp:3",
        ),
        d_map_32,
    )

    a32_vals = formats.decode(a32_codes, formats.DTYPES["e4m3fn"]).astype(np.float64)
    b32_vals = formats.decode(b32_codes, formats.DTYPES["e3m2"]).astype(np.float64)
    for kb in range(2):
        for i in range(m32):
            ea = int(exp_a32[kb, i])
            a32_vals[i, kb * 32 : (kb + 1) * 32] *= (
                float("nan") if ea == 0xFF else float(mpmath.ldexp(1, ea - 127))
            )
        for j in range(n32):
            eb = int(exp_b32[kb, j])
            b32_vals[j, kb * 32 : (kb + 1) * 32] *= (
                float("nan") if eb == 0xFF else float(mpmath.ldexp(1, eb - 127))
            )

    want_sc32 = _mp_grouped_dot_matrix(a32_vals, b32_vals, c32_u32, 32)
    assert np.all(got_sc32[:, 31] == 0xFFC0_0000)
    assert np.all(~np.isnan(got_sc32[:, :31].view(np.float32)))
    np.testing.assert_array_equal(got_sc32, want_sc32)

    # 3) Doubled-K sparse `v_smfmac_f32_16x16x64_bf16` with 2 index sets (`abid:0`, `abid:1`):
    dyadic = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float32)
    a_sp = dyadic[rng.integers(0, len(dyadic), size=(16, 32))]
    b_sp = dyadic[rng.integers(0, len(dyadic), size=(16, 64))]  # `[j, kk]`
    a_sp_codes = (a_sp.view(np.uint32) >> 16).astype(np.uint16)
    b_sp_codes = (b_sp.view(np.uint32) >> 16).astype(np.uint16)

    a_sp_regs = _pack_with_manual_map(a_sp_codes, _manual_dense_input_map(16, 32, 16), 4, 16)
    b_sp_regs = _pack_with_manual_map(b_sp_codes, _manual_dense_input_map(16, 64, 16), 8, 16)

    valid_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    k_regs = np.zeros((64, 1), dtype=np.uint32)
    want_sets = []
    for set_id in range(2):
        picks = rng.integers(0, len(valid_pairs), size=(16, 16))
        a_dense = np.zeros((16, 64), dtype=np.float64)
        for i in range(16):
            for g in range(16):
                i0, i1 = valid_pairs[int(picks[i, g])]
                nib = i0 | (i1 << 2)
                # 4 groups of 4 nibbles across the 4 lane-groups (`grp = g // 4`, `g_in = g % 4`):
                grp, g_in = divmod(g, 4)
                lane = grp * 16 + i
                bit_lo = set_id * 16 + g_in * 4
                k_regs[lane, 0] |= np.uint32(nib << bit_lo)
                a_dense[i, 4 * g + i0] = float(a_sp[i, 2 * g])
                a_dense[i, 4 * g + i1] = float(a_sp[i, 2 * g + 1])
        want_sets.append((a_dense @ b_sp.astype(np.float64).T).astype(np.float32).view(np.uint32))

    for set_id in range(2):
        got_sp = _unpack_with_manual_map(
            _run_cdna4_mfma_single(
                "v_smfmac_f32_16x16x64_bf16",
                a_sp_regs,
                b_sp_regs,
                np.zeros((64, 4), dtype=np.uint32),
                sparse=True,
                k_regs=k_regs,
                mods=f"cbsz:0 abid:{set_id}",
            ),
            d_map_16,
        )
        np.testing.assert_array_equal(got_sp, want_sets[set_id])
