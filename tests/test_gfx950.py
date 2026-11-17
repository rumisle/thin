"""End-to-end tests for AMD Instinct MI350X/MI355X (gfx950 / CDNA 4) target support."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.golden import data_path


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def test_gfx950_target_facts_and_160kb_lds_granules():
    from thin.target import Flags, target

    t = target("gfx950")
    assert t.name == "gfx950"
    assert t.wave == 64
    assert t.lds_max == 163840
    assert t.cu_lds == 163840
    assert t.barrier == [("s_barrier", [])]
    assert t.waits == [("s_waitcnt", ["vmcnt", "expcnt", "lgkmcnt"])]
    assert t.xcds == 8 and t.cus == 256 and t.vgprs_per_lane == 512
    assert t.sgpr_max == 102 and t.vgprs == 256

    assert t.allocated_vgprs(0, 0) == 0
    assert t.allocated_vgprs(1, 0) == 8
    assert t.allocated_vgprs(8, 4) == 16
    assert t.counters == {"vm": "vmcnt", "lgkm": "lgkmcnt"}
    assert t.returning("buffer_atomic_add", None) is False
    assert t.returning("buffer_atomic_add", "sc0") is True

    matrix_forms = [f for f in t.forms() if f.flags & Flags.MATRIX]
    assert len(matrix_forms) == 68
    modelled = [f.mnemonic for f in matrix_forms if t.matrix_form(f.mnemonic) is not None]
    assert len(modelled) == 68

    # Common FP8/BF8 MFMA forms on gfx950 use OCP e4m3fn/e5m2 (not FNUZ) and are marked provisional.
    fp8_mf = t.matrix_form("v_mfma_f32_16x16x32_fp8_bf8")
    assert fp8_mf is not None
    assert (fp8_mf.a, fp8_mf.b, fp8_mf.scaled) == ("e4m3fn", "e5m2", False)
    assert fp8_mf.provisional is True
    scale_mf = t.matrix_form("v_mfma_scale_f32_16x16x128_f8f6f4")
    assert scale_mf is not None
    assert (scale_mf.a, scale_mf.b, scale_mf.scaled) == ("f8f6f4", "f8f6f4", True)

    # Variable tuple widths on f8f6f4 MFMA and VOP3PX2 scaled MFMA forms.
    scale_form = t.form("v_mfma_scale_f32_16x16x128_f8f6f4", 6)
    assert scale_form.arity == 6
    assert scale_form.slots[1].widths == [4, 6, 8]
    assert scale_form.slots[2].widths == [4, 6, 8]
    assert [f.name for f in scale_form.fields] == ["op_sel", "op_sel_hi", "cbsz", "blgp"]


def test_gfx950_code_object_and_vadd_emulation():
    import thin
    from examples.vadd import build_vadd
    from thin.target import targeting

    with targeting("gfx950"):
        k = thin.compile(build_vadd)

    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"' in k.asm
    assert ".amdhsa_wavefront_size32" not in k.asm
    # ELF e_flags at offset 48..52 is 0x54F (EF_AMDGPU_MACH_AMDGCN_GFX950 = 0x4F).
    e_flags = int.from_bytes(k.co[48:52], "little")
    assert e_flags == 0x54F

    # Run emulate() outside targeting("gfx950") to verify code-object ELF 0x4F target inference.
    for n in (1, 64, 65, 257):
        x = np.arange(n, dtype=np.float32)
        y = 1.5 * x + 2.0
        out = k(x=x, y=y, z=np.zeros(n, np.float32), n=n).emulate()
        assert np.allclose(out["z"], x + y)


def _gen_lds_kernel(size: int):
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def body(b, k):
        b.lds().region("buf", nbytes=size)

    return KernelDef(f"lds_{size}", Args(p=ptr), body, wgsize=64)


def test_gfx950_vadd_with_laps_and_160kb_lds_limit():
    import thin
    from examples.vadd import build_vadd
    from thin.target import targeting

    with targeting("gfx950"):
        k = thin.compile(build_vadd, laps=True)

    assert "s_memtime" in k.asm
    assert "s_memrealtime" in k.asm
    n = 256
    x = np.arange(n, dtype=np.float32)
    y = np.full(n, 0.25, dtype=np.float32)
    z = np.zeros(n, np.float32)
    bound = k(x=x, y=y, z=z, n=n)
    bound.emulate()
    assert z.tolist() == (x + y).tolist()

    # 160 KiB (163840 bytes) LDS compiles cleanly on gfx950, whereas 163844 exceeds lds_max.
    with targeting("gfx950"):
        k160 = thin.compile(_gen_lds_kernel, size=163_840)
        assert k160.manifest.lds == 163_840
        assert "163840" in k160.asm
        with pytest.raises(ValueError, match="163840"):
            thin.compile(_gen_lds_kernel, size=163_844)


COMPARE = json.loads((data_path("metron", "kernels.json")).read_text())["compare"]


@pytest.mark.parametrize(
    "case", COMPARE, ids=[f"stride{c['stride']}x{c['copies']}" for c in COMPARE]
)
def test_gfx950_metron_differ_on_emulator(case):
    import thin
    from metron import maker
    from thin.target import targeting

    with targeting("gfx950"):
        k = thin.compile(maker.gen_differ)

    rng = np.random.default_rng(case["stride"] + case["copies"])
    ref = rng.integers(0, 256, case["stride"], dtype=np.uint8)
    base = np.tile(ref, case["copies"])
    for copy, byte in case["changes"]:
        base[copy * case["stride"] + byte] ^= 0x40
    flags = np.zeros(case["copies"], np.uint32)
    grid = (-(-case["stride"] // maker.CHUNK), case["copies"])
    k.bind(
        grid=grid, args={"base": base, "ref": ref, "flags": flags, "stride": case["stride"]}
    ).emulate()
    assert flags.tolist() == case["flags"]


def _pack_cdna4_input(mat: np.ndarray, outer: int, k_dim: int, bits: int) -> np.ndarray:
    """Packs `[outer, k_dim]` codes into `[64, dwords]` per CDNA 4 manual §7.1.5 / §7.5.1.3."""
    lane_groups = 64 // outer
    elems_per_lane = k_dim // lane_groups
    dwords = (elems_per_lane * bits) // 32
    out = np.zeros((64, dwords), dtype=np.uint32)
    mask = (1 << bits) - 1
    if bits <= 16 and dwords == 8:
        half_k = k_dim // 2
        elems_half = half_k // lane_groups
        for i in range(outer):
            for k in range(k_dim):
                half_idx = k // half_k
                k_in = k % half_k
                local = k_in % elems_half
                lane = (k_in // elems_half) * outer + i
                tot = local * bits
                out[lane, 4 * half_idx + tot // 32] |= (int(mat[i, k]) & mask) << (tot % 32)
        return out
    for i in range(outer):
        for k in range(k_dim):
            local = k % elems_per_lane
            lane = (k // elems_per_lane) * outer + i
            tot = local * bits
            dw, bit_lo = tot // 32, tot % 32
            val = int(mat[i, k]) & mask
            out[lane, dw] |= (val << bit_lo) & 0xFFFFFFFF
            if bit_lo + bits > 32:
                out[lane, dw + 1] |= val >> (32 - bit_lo)
    return out


def _unpack_cdna4_output(lanes: np.ndarray, m: int, n: int) -> np.ndarray:
    """Unpacks `[64, dwords]` uint32 into `[m, n]` uint32."""
    multirows_per_reg = 64 // n
    out = np.zeros((m, n), dtype=np.uint32)
    for i in range(m):
        for j in range(n):
            local = (i // (4 * multirows_per_reg)) * 4 + (i % 4)
            lane = ((i // 4) % multirows_per_reg) * n + j
            out[i, j] = lanes[lane, local]
    return out


def _run_cdna4_mfma(
    mnemonic: str,
    a_lanes: np.ndarray,
    b_lanes: np.ndarray,
    c_lanes: np.ndarray | str,
    d_dwords: int,
    *,
    mods: str = "",
    use_agpr: bool = True,
    sparse: bool = False,
    sparse_init: int = 0,
    scale_lanes: tuple[np.ndarray, np.ndarray] | None = None,
    target_name: str = "gfx950",
) -> np.ndarray:
    from thin.emu import Cluster, Memory
    from thin.target import targeting

    na, nb = a_lanes.shape[1], b_lanes.shape[1]
    lines = ["\ts_load_dwordx8 s[4:11], s[0:1], 0x0"]
    if scale_lanes is not None:
        lines.append("\ts_load_dwordx4 s[12:15], s[0:1], 0x20")
    lines.extend(
        [
            "\ts_waitcnt lgkmcnt(0)",
            "\tv_lshlrev_b32 v1, 2, v0",
            f"\tv_mul_lo_u32 v2, v0, {na * 4}",
        ]
    )
    for d in range(na):
        lines.append(f"\tglobal_load_dword v{16 + d}, v2, s[4:5] offset:{d * 4}")
    lines.append(f"\tv_mul_lo_u32 v3, v0, {nb * 4}")
    for d in range(nb):
        lines.append(f"\tglobal_load_dword v{48 + d}, v3, s[6:7] offset:{d * 4}")
    if scale_lanes is not None:
        lines.append("\tglobal_load_dword v76, v1, s[12:13]")
        lines.append("\tglobal_load_dword v77, v1, s[14:15]")

    c_is_buf = isinstance(c_lanes, np.ndarray)
    if sparse:
        assert c_is_buf
        lines.append("\tglobal_load_dword v79, v1, s[8:9]")
        lines.append("\ts_waitcnt vmcnt(0)")
        init_lit = hex(sparse_init & 0xFFFFFFFF)
        for d in range(d_dwords):
            if use_agpr:
                lines.append(f"\tv_accvgpr_write_b32 a{d}, {init_lit}")
            else:
                lines.append(f"\tv_mov_b32 v{80 + d}, {init_lit}")
        src2_op = "v79"
    elif c_is_buf:
        nc = c_lanes.shape[1]
        lines.append(f"\tv_mul_lo_u32 v4, v0, {nc * 4}")
        for d in range(nc):
            lines.append(f"\tglobal_load_dword v{80 + d}, v4, s[8:9] offset:{d * 4}")
        lines.append("\ts_waitcnt vmcnt(0)")
        if use_agpr:
            for d in range(nc):
                lines.append(f"\tv_accvgpr_write_b32 a{d}, v{80 + d}")
            src2_op = "a0" if nc == 1 else f"a[0:{nc - 1}]"
        else:
            src2_op = "v80" if nc == 1 else f"v[80:{80 + nc - 1}]"
    else:
        lines.append("\ts_waitcnt vmcnt(0)")
        src2_op = str(c_lanes)

    dst_op = (
        ("a0" if d_dwords == 1 else f"a[0:{d_dwords - 1}]")
        if use_agpr
        else ("v80" if d_dwords == 1 else f"v[80:{80 + d_dwords - 1}]")
    )
    a_op = "v16" if na == 1 else f"v[16:{16 + na - 1}]"
    b_op = "v48" if nb == 1 else f"v[48:{48 + nb - 1}]"
    scale_ops = ", v76, v77" if scale_lanes is not None else ""
    suffix = f" {mods}" if mods else ""
    lines.append(f"\t{mnemonic} {dst_op}, {a_op}, {b_op}, {src2_op}{scale_ops}{suffix}")
    lines.append(f"\tv_mul_lo_u32 v5, v0, {d_dwords * 4}")
    for d in range(d_dwords):
        if use_agpr:
            lines.append(f"\tv_accvgpr_read_b32 v6, a{d}")
            lines.append(f"\tglobal_store_dword v5, v6, s[10:11] offset:{d * 4}")
        else:
            lines.append(f"\tglobal_store_dword v5, v{80 + d}, s[10:11] offset:{d * 4}")
    lines.append("\ts_endpgm\n")
    asm = "\n".join(lines)

    mem = Memory()
    buf_a = mem.alloc("a", max(4, a_lanes.nbytes), tagged=False)
    buf_b = mem.alloc("b", max(4, b_lanes.nbytes), tagged=False)
    buf_c = mem.alloc("c", max(4, c_lanes.nbytes if c_is_buf else 256), tagged=False)
    buf_d = mem.alloc("d", 64 * d_dwords * 4, tagged=False)
    buf_a.write(0, np.ascontiguousarray(a_lanes, dtype=np.uint32).tobytes())
    buf_b.write(0, np.ascontiguousarray(b_lanes, dtype=np.uint32).tobytes())
    if c_is_buf:
        buf_c.write(0, np.ascontiguousarray(c_lanes, dtype=np.uint32).tobytes())
    ptrs = [buf_a.base, buf_b.base, buf_c.base, buf_d.base]
    if scale_lanes is not None:
        sa_arr, sb_arr = scale_lanes
        buf_sa = mem.alloc("sa", 256, tagged=False)
        buf_sb = mem.alloc("sb", 256, tagged=False)
        buf_sa.write(0, np.ascontiguousarray(sa_arr, dtype=np.uint32).tobytes())
        buf_sb.write(0, np.ascontiguousarray(sb_arr, dtype=np.uint32).tobytes())
        ptrs.extend([buf_sa.base, buf_sb.base])
    ka = mem.alloc("ka", 8 * len(ptrs), tagged=False)
    ka.write(0, np.array(ptrs, dtype=np.uint64).tobytes())
    with targeting(target_name):
        Cluster(
            asm,
            {"wgsize": 64, "vgpr": 128, "agpr": 32, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        ).run()[0]
    return np.frombuffer(buf_d.read(0, 64 * d_dwords * 4), dtype=np.uint32).reshape(64, d_dwords)


def test_gfx950_new_dense_mfma_shapes_and_ocp_fp8():
    from thin import formats

    rng = np.random.default_rng(950)

    # All 6 new CDNA 4 doubled-K dense MFMA shapes:
    dense_cases = [
        ("v_mfma_f32_16x16x32_bf16", 16, 16, 32, "bf16", False),
        ("v_mfma_f32_16x16x32_f16", 16, 16, 32, "f16", True),
        ("v_mfma_f32_32x32x16_bf16", 32, 32, 16, "bf16", True),
        ("v_mfma_f32_32x32x16_f16", 32, 32, 16, "f16", False),
    ]
    for mnem, m, n, k, dt_name, use_agpr in dense_cases:
        dt = formats.DTYPES[dt_name]
        a_f32 = (rng.integers(-4, 5, (m, k)) * 0.5).astype(np.float32)
        b_f32 = (rng.integers(-4, 5, (k, n)) * 0.5).astype(np.float32)
        a_codes = formats.encode(a_f32, dt, saturate=False)
        b_codes = formats.encode(b_f32, dt, saturate=False)
        a_lanes = _pack_cdna4_input(a_codes, m, k, 16)
        b_lanes = _pack_cdna4_input(b_codes.T, n, k, 16)
        d_dw = (m * n) // 64
        got_lanes = _run_cdna4_mfma(mnem, a_lanes, b_lanes, "0", d_dw, use_agpr=use_agpr)
        got = _unpack_cdna4_output(got_lanes, m, n).view(np.float32)
        want = formats.decode(a_codes, dt) @ formats.decode(b_codes, dt)
        assert np.array_equal(got, want), mnem

    for mnem, m, n, k, use_agpr in [
        ("v_mfma_i32_16x16x64_i8", 16, 16, 64, True),
        ("v_mfma_i32_32x32x32_i8", 32, 32, 32, False),
    ]:
        a_i8 = rng.integers(-16, 16, (m, k), dtype=np.int8)
        b_i8 = rng.integers(-16, 16, (k, n), dtype=np.int8)
        a_lanes = _pack_cdna4_input(a_i8.view(np.uint8), m, k, 8)
        b_lanes = _pack_cdna4_input(b_i8.T.view(np.uint8), n, k, 8)
        d_dw = (m * n) // 64
        got_lanes = _run_cdna4_mfma(mnem, a_lanes, b_lanes, "-5", d_dw, use_agpr=use_agpr)
        got = _unpack_cdna4_output(got_lanes, m, n).view(np.int32)
        want = a_i8.astype(np.int32) @ b_i8.astype(np.int32) - 5
        assert np.array_equal(got, want), mnem

    # OCP FP8 on gfx950 vs FNUZ FP8 on gfx942:
    # 0x80 is -0.0 in OCP e4m3fn (finite 0) vs NaN in FNUZ; 0x7E is 448.0 in OCP vs 224.0 in FNUZ;
    # 0x7F is NaN in OCP vs 240.0 in FNUZ.
    a_ocp = np.zeros((16, 32), dtype=np.uint8)
    b_ocp = np.zeros((32, 16), dtype=np.uint8)
    a_ocp[0, 0] = 0x7E  # 448.0 in OCP e4m3fn, 224.0 in FNUZ
    a_ocp[0, 1] = 0x80  # -0.0 in OCP e4m3fn, NaN in FNUZ
    a_ocp[1, 0] = 0x7F  # NaN in OCP e4m3fn, 240.0 in FNUZ
    b_ocp[0, 0] = 0x3C  # 1.0 in OCP e5m2, 0.5 in FNUZ e5m2fnuz
    b_ocp[1, 0] = 0x3C
    a_fp8_lanes = _pack_cdna4_input(a_ocp, 16, 32, 8)
    b_bf8_lanes = _pack_cdna4_input(b_ocp.T, 16, 32, 8)
    d950 = _unpack_cdna4_output(
        _run_cdna4_mfma(
            "v_mfma_f32_16x16x32_fp8_bf8", a_fp8_lanes, b_bf8_lanes, "0", 4, target_name="gfx950"
        ),
        16,
        16,
    )
    d942 = _unpack_cdna4_output(
        _run_cdna4_mfma(
            "v_mfma_f32_16x16x32_fp8_bf8", a_fp8_lanes, b_bf8_lanes, "0", 4, target_name="gfx942"
        ),
        16,
        16,
    )
    assert d950.view(np.float32)[0, 0] == 448.0
    assert d950[1, 0] == 0xFFC00000
    assert d942[0, 0] == 0xFFC00000  # 0x80 was NaN on gfx942
    assert d942.view(np.float32)[1, 0] == 120.0  # 240.0 * 0.5 on gfx942


def test_gfx950_new_sparse_smfmac_shapes():
    from thin import formats

    rng = np.random.default_rng(1950)

    def build_sparse(m: int, k: int, bits: int, set_idx: int, num_sets: int):
        """Constructs 4:2 sparse `a_full[m, k]`, `a_comp[m, k//2]`, and `idx_lanes[64, 1]`."""
        groups = k // 4
        a_comp = np.zeros((m, k // 2), dtype=np.int32)
        a_full = np.zeros((m, k), dtype=np.int32)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        lane_groups = 64 // m
        elems_contig = (k // 2) // lane_groups
        # Fill non-selected sets with distinct garbage indices so wrong-set selection fails!
        idx_lanes = np.full((64, 1), 0xE4E4E4E4, dtype=np.uint32)
        set_bits = 2 * elems_contig
        set_mask = ((1 << set_bits) - 1) if set_bits < 32 else 0xFFFFFFFF
        idx_lanes[:, 0] &= np.uint32(~(set_mask << (set_bits * set_idx)) & 0xFFFFFFFF)
        for i in range(m):
            for g in range(groups):
                p0, p1 = pairs[(i * 7 + g * 3) % len(pairs)]
                v0 = int(rng.integers(-3, 4))
                v1 = int(rng.integers(-3, 4))
                a_comp[i, 2 * g] = v0
                a_comp[i, 2 * g + 1] = v1
                a_full[i, 4 * g + p0] = v0
                a_full[i, 4 * g + p1] = v1
                k_eff = 2 * g
                lane = (k_eff // elems_contig) * m + i
                local = k_eff % elems_contig
                bit_lo = ((local + elems_contig * set_idx) // 2) * 4
                nib = p0 | (p1 << 2)
                idx_lanes[lane, 0] |= np.uint32(nib << bit_lo)
        return a_full, a_comp, idx_lanes

    # 1. 16-bit sparse SMFMAC: 16x16x64 (Set0 abid:0 and Set1 abid:1) & 32x32x32 (bf16 & f16)
    for mnem, m, n, k, dt_name, abid in [
        ("v_smfmac_f32_16x16x64_bf16", 16, 16, 64, "bf16", 0),
        ("v_smfmac_f32_16x16x64_f16", 16, 16, 64, "f16", 1),
        ("v_smfmac_f32_32x32x32_bf16", 32, 32, 32, "bf16", 1),
        ("v_smfmac_f32_32x32x32_f16", 32, 32, 32, "f16", 0),
    ]:
        dt = formats.DTYPES[dt_name]
        a_full, a_comp, idx_lanes = build_sparse(m, k, 16, abid, 2)
        b_vals = rng.integers(-3, 4, (k, n)).astype(np.float32)
        # Make second half of K distinct so two-half 8-VGPR B layout is strictly exercised.
        b_vals[k // 2 :] += 2.0
        a_codes = formats.encode(a_comp.astype(np.float32), dt, saturate=False)
        b_codes = formats.encode(b_vals, dt, saturate=False)
        a_lanes = _pack_cdna4_input(a_codes, m, k // 2, 16)
        b_lanes = _pack_cdna4_input(b_codes.T, n, k, 16)
        d_dw = (m * n) // 64
        mods = f"cbsz:0 abid:{abid}" if abid else ""
        init_f32 = np.float32(3.0).view(np.uint32).item()
        got_lanes = _run_cdna4_mfma(
            mnem,
            a_lanes,
            b_lanes,
            idx_lanes,
            d_dw,
            mods=mods,
            use_agpr=(abid == 0),
            sparse=True,
            sparse_init=int(init_f32),
        )
        got = _unpack_cdna4_output(got_lanes, m, n).view(np.float32)
        want = a_full.astype(np.float32) @ b_vals + 3.0
        assert np.array_equal(got, want), mnem

    # 2. 8-bit sparse SMFMAC: all 8 OCP FP8/BF8 forms + 2 I8 forms (16x16x128 & 32x32x64)
    for m, n, k in [(16, 16, 128), (32, 32, 64)]:
        a_full, a_comp, idx_lanes = build_sparse(m, k, 8, 0, 1)
        b_int = rng.integers(-3, 4, (k, n), dtype=np.int32)
        b_int[k // 2 :] += 1
        d_dw = (m * n) // 64

        for suffix, dt_a_name, dt_b_name in [
            ("bf8_bf8", "e5m2", "e5m2"),
            ("bf8_fp8", "e5m2", "e4m3fn"),
            ("fp8_bf8", "e4m3fn", "e5m2"),
            ("fp8_fp8", "e4m3fn", "e4m3fn"),
        ]:
            mnem = f"v_smfmac_f32_{m}x{n}x{k}_{suffix}"
            a_codes = formats.encode(
                a_comp.astype(np.float32), formats.DTYPES[dt_a_name], saturate=False
            )
            b_codes = formats.encode(
                b_int.astype(np.float32), formats.DTYPES[dt_b_name], saturate=False
            )
            a_lanes = _pack_cdna4_input(a_codes, m, k // 2, 8)
            b_lanes = _pack_cdna4_input(b_codes.T, n, k, 8)
            got_lanes = _run_cdna4_mfma(
                mnem, a_lanes, b_lanes, idx_lanes, d_dw, sparse=True, sparse_init=0
            )
            got = _unpack_cdna4_output(got_lanes, m, n).view(np.float32)
            want = a_full.astype(np.float32) @ b_int.astype(np.float32)
            assert np.array_equal(got, want), mnem

        mnem_i8 = f"v_smfmac_i32_{m}x{n}x{k}_i8"
        a_lanes_i8 = _pack_cdna4_input(a_comp.astype(np.int8).view(np.uint8), m, k // 2, 8)
        b_lanes_i8 = _pack_cdna4_input(b_int.T.astype(np.int8).view(np.uint8), n, k, 8)
        got_i8 = _unpack_cdna4_output(
            _run_cdna4_mfma(
                mnem_i8,
                a_lanes_i8,
                b_lanes_i8,
                idx_lanes,
                d_dw,
                use_agpr=False,
                sparse=True,
                sparse_init=7,
            ),
            m,
            n,
        ).view(np.int32)
        assert np.array_equal(got_i8, a_full @ b_int + 7), mnem_i8


def test_gfx950_f8f6f4_all_25_format_pairs_and_mx_scale_mfma():
    from thin import formats

    rng = np.random.default_rng(2950)
    f8f6f4_table = [
        (0, "e4m3fn", 8, {}),
        (1, "e5m2", 8, {}),
        (2, "e2m3", 6, {"nan": 0x1F}),
        (3, "e3m2", 6, {"nan": 0x1F}),
        (4, "e2m1", 4, {"nan": 0x07}),
    ]

    # Pre-pack all 5 formats for A[16, 128] and B[128, 16] using small exact half-integers
    # (-1.5..1.5 in 0.5 steps are exact in every one of e4m3fn, e5m2, e2m3, e3m2, e2m1!).
    a_val16 = (rng.integers(-3, 4, (16, 128)) * 0.5).astype(np.float32)
    b_val16 = (rng.integers(-3, 4, (128, 16)) * 0.5).astype(np.float32)
    want16 = a_val16 @ b_val16
    packed_a16 = {}
    packed_b16 = {}
    rm = formats.row_major()
    for code, dt_name, bits, kw in f8f6f4_table:
        dt = formats.DTYPES[dt_name]
        ac = formats.encode(a_val16, dt, saturate=True, **kw)
        bc = formats.encode(b_val16, dt, saturate=True, **kw)
        al = _pack_cdna4_input(ac, 16, 128, bits)
        bl = _pack_cdna4_input(bc.T, 16, 128, bits)
        # Verify 6-bit and 4-bit lane packing agrees bit-for-bit with thin-formats sub-byte packing:
        if bits < 8:
            lane0 = formats.to_bytes(ac[0:1, :32].reshape(1, 1, 32), rm, dt).view(np.uint32)
            assert np.array_equal(al[0], lane0), dt_name
        packed_a16[code] = al
        packed_b16[code] = bl

    # Exercise all 25 (cbsz, blgp) format combinations on v_mfma_f32_16x16x128_f8f6f4!
    for cbsz in range(5):
        for blgp in range(5):
            mods = f"cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
            got_lanes = _run_cdna4_mfma(
                "v_mfma_f32_16x16x128_f8f6f4",
                packed_a16[cbsz],
                packed_b16[blgp],
                "0",
                4,
                mods=mods,
                use_agpr=((cbsz + blgp) % 2 == 0),
            )
            got = _unpack_cdna4_output(got_lanes, 16, 16).view(np.float32)
            assert np.array_equal(got, want16), f"cbsz={cbsz} blgp={blgp}"

    # Also verify 32x32x64 unscaled mixed formats (FP6xFP4 and BF8xBF6):
    a_val32 = (rng.integers(-3, 4, (32, 64)) * 0.5).astype(np.float32)
    b_val32 = (rng.integers(-3, 4, (64, 32)) * 0.5).astype(np.float32)
    want32 = a_val32 @ b_val32
    for cbsz, blgp in [(2, 4), (1, 3)]:
        _, da_n, ba, kwa = f8f6f4_table[cbsz]
        _, db_n, bb, kwb = f8f6f4_table[blgp]
        al = _pack_cdna4_input(
            formats.encode(a_val32, formats.DTYPES[da_n], saturate=True, **kwa), 32, 64, ba
        )
        bl = _pack_cdna4_input(
            formats.encode(b_val32, formats.DTYPES[db_n], saturate=True, **kwb).T, 32, 64, bb
        )
        got32 = _unpack_cdna4_output(
            _run_cdna4_mfma(
                "v_mfma_f32_32x32x64_f8f6f4", al, bl, "0", 16, mods=f"cbsz:{cbsz} blgp:{blgp}"
            ),
            32,
            32,
        ).view(np.float32)
        assert np.array_equal(got32, want32), (cbsz, blgp)

    # MX block-scaled v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4 using formats.quantize
    # with non-zero byte selectors (sel_a = {op_sel_hi[0], op_sel[0]}).
    for mnem, m, n, k, cbsz, blgp, sel_a, sel_b in [
        ("v_mfma_scale_f32_16x16x128_f8f6f4", 16, 16, 128, 0, 4, 2, 3),  # e4m3fn x e2m1
        ("v_mfma_scale_f32_16x16x128_f8f6f4", 16, 16, 128, 2, 3, 1, 0),  # e2m3 x e3m2
        ("v_mfma_scale_f32_32x32x64_f8f6f4", 32, 32, 64, 1, 4, 3, 1),  # e5m2 x e2m1
    ]:
        dt_a = formats.DTYPES[f8f6f4_table[cbsz][1]]
        dt_b = formats.DTYPES[f8f6f4_table[blgp][1]]
        bits_a, bits_b = f8f6f4_table[cbsz][2], f8f6f4_table[blgp][2]
        # Construct exact MX-representable inputs with wide per-block dynamic range (2^-6 .. 2^8):
        raw_a = (
            rng.standard_normal((m, k))
            * np.repeat(2.0 ** rng.integers(-6, 9, (m, k // 32)), 32, axis=1)
        ).astype(np.float32)
        raw_b = (
            rng.standard_normal((n, k))
            * np.repeat(2.0 ** rng.integers(-6, 9, (n, k // 32)), 32, axis=1)
        ).astype(np.float32)
        ac, sa_bytes = formats.quantize(raw_a, dt_a)
        bc, sb_bytes = formats.quantize(raw_b, dt_b)
        al = _pack_cdna4_input(ac, m, k, bits_a)
        bl = _pack_cdna4_input(bc, n, k, bits_b)

        # Pack scale bytes into byte `sel_a` / `sel_b` of lane `(k_blk * m + i)`,
        # filling unused bytes with 0xFF (NaN) so reading the wrong byte immediately fails!
        sa_lanes = np.full(64, 0xFFFFFFFF, dtype=np.uint32)
        sb_lanes = np.full(64, 0xFFFFFFFF, dtype=np.uint32)
        for k_blk in range(k // 32):
            for i in range(m):
                lane = k_blk * m + i
                sa_lanes[lane] = (
                    sa_lanes[lane] & np.uint32(~(0xFF << (8 * sel_a)) & 0xFFFFFFFF)
                ) | (np.uint32(sa_bytes[i, k_blk]) << (8 * sel_a))
            for j in range(n):
                lane = k_blk * n + j
                sb_lanes[lane] = (
                    sb_lanes[lane] & np.uint32(~(0xFF << (8 * sel_b)) & 0xFFFFFFFF)
                ) | (np.uint32(sb_bytes[j, k_blk]) << (8 * sel_b))

        mods_parts = []
        if sel_a & 1 or sel_b & 1:
            mods_parts.append(f"op_sel:[{sel_a & 1},{sel_b & 1},0]")
        if (sel_a >> 1) or (sel_b >> 1):
            mods_parts.append(f"op_sel_hi:[{sel_a >> 1},{sel_b >> 1},0]")
        if cbsz:
            mods_parts.append(f"cbsz:{cbsz}")
        if blgp:
            mods_parts.append(f"blgp:{blgp}")
        d_dw = (m * n) // 64
        got_scaled = _unpack_cdna4_output(
            _run_cdna4_mfma(
                mnem,
                al,
                bl,
                "0",
                d_dw,
                mods=" ".join(mods_parts),
                scale_lanes=(sa_lanes, sb_lanes),
            ),
            m,
            n,
        ).view(np.float32)

        # Dequantize reference in float64 and compare:
        deq_a = formats.decode(ac, dt_a).astype(np.float64) * np.repeat(
            np.exp2(sa_bytes.astype(np.float64) - 127.0), 32, axis=1
        )
        deq_b = formats.decode(bc, dt_b).astype(np.float64) * np.repeat(
            np.exp2(sb_bytes.astype(np.float64) - 127.0), 32, axis=1
        )
        ref = (deq_a @ deq_b.T).astype(np.float32)
        assert np.allclose(got_scaled, ref, rtol=2e-6, atol=1e-5), mnem

    # Inject 0xFF (E8M0 NaN scale) into row 2, block 1 of A -> only row 2 becomes 0xFFC00000!
    sa_nan = np.full(64, 0x7F7F7F7F, dtype=np.uint32)
    sb_ok = np.full(64, 0x7F7F7F7F, dtype=np.uint32)
    sa_nan[1 * 16 + 2] = 0x7F7F7FFF  # byte 0 = 0xFF in lane 18 (k_blk=1, m=2)
    nan_out = _unpack_cdna4_output(
        _run_cdna4_mfma(
            "v_mfma_scale_f32_16x16x128_f8f6f4",
            packed_a16[4],
            packed_b16[4],
            "0",
            4,
            mods="cbsz:4 blgp:4",
            scale_lanes=(sa_nan, sb_ok),
        ),
        16,
        16,
    )
    assert np.all(nan_out[2, :] == 0xFFC00000)
    assert np.all(nan_out[:2, :] != 0xFFC00000) and np.all(nan_out[3:, :] != 0xFFC00000)


def _run_lanes(asm_body: str, n_out: int, target_name: str = "gfx950") -> np.ndarray:
    from thin.emu import Cluster, Memory
    from thin.target import targeting

    lines = [
        "\ts_load_dwordx2 s[4:5], s[0:1], 0x0",
        "\ts_waitcnt lgkmcnt(0)",
        asm_body,
        f"\tv_mul_lo_u32 v90, v0, {n_out * 4}",
    ]
    for d in range(n_out):
        lines.append(f"\tglobal_store_dword v90, v{80 + d}, s[4:5] offset:{d * 4}")
    lines.append("\ts_endpgm\n")
    mem = Memory()
    out_buf = mem.alloc("out", 64 * n_out * 4, tagged=False)
    ka = mem.alloc("ka", 8, tagged=False)
    ka.write(0, np.array([out_buf.base], dtype=np.uint64).tobytes())
    with targeting(target_name):
        Cluster(
            "\n".join(lines),
            {"wgsize": 64, "vgpr": 96, "agpr": 0, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        ).run()[0]
    return np.frombuffer(out_buf.read(0, 64 * n_out * 4), dtype=np.uint32).reshape(64, n_out)


def test_gfx950_cvt_scalef32_and_sr_ops():
    # Single wave64 kernel exercising OCP FP8, BF16/F16 pack/unpack, SR, and scalef32:
    asm = "\n".join(
        [
            # v80: OCP FP8 pack of (-0.0, 448.0, -240.0, +inf) via v_cvt_pk_fp8_f32
            "\tv_mov_b32 v1, 0x80000000",
            "\tv_mov_b32 v2, 0x43e00000",  # 448.0
            "\tv_mov_b32 v3, 0xc3700000",  # -240.0
            "\tv_mov_b32 v4, 0x7f800000",  # +inf
            "\tv_cvt_pk_fp8_f32 v80, v1, v2",
            "\tv_cvt_pk_fp8_f32 v80, v3, v4 op_sel:[0,0,1]",
            # v81: v_cvt_f32_bf16_e64 with -v (0x3fc0 = 1.5 -> -1.5 = 0xbfc00000)
            "\tv_mov_b32 v5, 0xdead3fc0",
            "\tv_cvt_f32_bf16_e64 v81, -v5",
            # v82: v_cvt_pk_bf16_f32 RNE midpoint (1.0 + 2^-8 ties to even -> 0x3f80;
            #      1.0 + 3*2^-8 -> 0x3f82; SNaN -> quieted 0x7fc5)
            "\tv_mov_b32 v6, 0x3f808000",
            "\tv_mov_b32 v7, 0x7f850001",
            "\tv_cvt_pk_bf16_f32 v82, v6, v7",
            # v83: SR midpoint test across 64 lanes: seed = (lane << 26)
            #      1.5 in BF16 + 0.5 ULP = 0x3fc08000 -> 0x3fc0 for lane<32, 0x3fc1 for lane>=32
            "\tv_lshlrev_b32 v8, 26, v0",
            "\tv_mov_b32 v9, 0x3fc08000",
            "\tv_mov_b32 v83, 0x12345678",
            "\tv_cvt_sr_bf16_f32 v83, v9, v8 op_sel:[0,0,1]",
            # v84: v_cvt_scalef32_sr_fp8_f32 at midpoint 1.0625 (between 1.0=0x38 & 1.125=0x39)
            #      with scale = 2.0 (so input is 2.125 = 0x40080000), writing byte 2 (OPSEL[3:2]=2):
            "\tv_mov_b32 v10, 0x40080000",
            "\tv_mov_b32 v84, 0xaabbccdd",
            "\tv_cvt_scalef32_sr_fp8_f32 v84, v10, v8, 2.0 op_sel:[0,0,0,1]",
            # v85..v86: FP4 pack (scale=2.0 on [-3.0, 12.0] -> E2M1 [-1.5=0xb, 6.0=0x7] = 0x7b)
            #           into byte 3, then unpack back with scale=2.0 -> [-3.0, 12.0]:
            "\tv_mov_b32 v11, 0xc0400000",  # -3.0
            "\tv_mov_b32 v12, 0x41400000",  # 12.0
            "\tv_mov_b32 v13, 0x11223344",
            "\tv_cvt_scalef32_pk_fp4_f32 v13, v11, v12, 2.0 op_sel:[0,0,1,1]",
            "\tv_cvt_scalef32_pk_f32_fp4 v[85:86], v13, 2.0 op_sel:[1,1,0]",
            # v87: NaN scale (e == 255, 0x7f800000) in v_cvt_scalef32_f32_fp8 -> 0xffc00000:
            "\tv_cvt_scalef32_f32_fp8 v87, v80, v4",
            # v88..v89: pk32 BF6 from BF16 -> unpack back to BF16 (scale = 0.5):
            "\tv_mov_b32 v16, 0xc0003f80",  # [1.0, -2.0] in BF16 -> scaled by 2: [2.0, -4.0]
            *(f"\tv_mov_b32 v{17 + d}, 0x3f00bf00" for d in range(15)),  # [-0.5, +0.5]
            "\tv_cvt_scalef32_pk32_bf6_bf16 v[32:37], v[16:31], 0.5",
            "\tv_cvt_scalef32_pk32_bf16_bf6 v[48:63], v[32:37], 0.5",
            "\tv_mov_b32 v88, v48",
            "\tv_mov_b32 v89, v53",
        ]
    )
    out = _run_lanes(asm, 10)
    # OCP E4M3: -0.0->0x80, 448.0->0x7e, -240.0->0xf7, +inf->0x7f
    assert np.all(out[:, 0] == 0x7FF77E80)
    assert np.all(out[:, 1] == 0xBFC00000)
    assert np.all(out[:, 2] == 0x7FC53F80)
    # SR BF16 preserves low half 0x5678; high half is 0x3fc0 for lanes 0..31 and 0x3fc1 for 32..63:
    assert np.all(out[:32, 3] == 0x3FC05678)
    assert np.all(out[32:, 3] == 0x3FC15678)
    # SR FP8 writes byte 2 (0x38 for lanes 0..31, 0x39 for lanes 32..63), preserving 0xaa..ccdd:
    assert np.all(out[:32, 4] == 0xAA38CCDD)
    assert np.all(out[32:, 4] == 0xAA39CCDD)
    # FP4 scaled round-trip:
    assert np.all(out[:, 5].view(np.float32) == -3.0)
    assert np.all(out[:, 6].view(np.float32) == 12.0)
    # e == 255 scale -> canonical NaN 0xffc00000:
    assert np.all(out[:, 7] == 0xFFC00000)
    # BF6 pk32 round-trip:
    assert np.all(out[:, 8] == 0xC0003F80)
    assert np.all(out[:, 9] == 0x3F00BF00)


def test_gfx950_new_valu_ops():
    # Single wave64 kernel exercising prng, bitop3, ashr_pk, permlane_swap, dot2/dot2c bf16,
    # pk_min/maximum3_f16, and exp/log_legacy_f32:
    asm = "\n".join(
        [
            # v80: v_prng_b32 (lane 0: 0 -> 0; lane 1: 0x80000001 -> 2 ^ 197 = 199)
            "\tv_lshlrev_b32 v1, 31, v0",
            "\tv_or_b32 v1, v1, v0",
            "\tv_prng_b32 v80, v1",
            # v81: v_bitop3_b32 0xea = (A & B) | C, and v82: v_bitop3_b16 0x96 = A ^ B ^ C
            "\tv_mov_b32 v2, 0xf0f0aa55",
            "\tv_mov_b32 v3, 0xff0055aa",
            "\tv_mov_b32 v4, 0x00ff0f0f",
            "\tv_bitop3_b32 v81, v2, v3, v4 bitop3:0xea",
            "\tv_mov_b32 v82, 0x1111dead",
            "\tv_bitop3_b16 v82, v2, v3, v4 op_sel:[1,0,1,1] bitop3:0x96",
            # v83: v_ashr_pk_i8_i32 & v_ashr_pk_u8_i32 into low/high halves:
            "\tv_mov_b32 v5, -600",
            "\tv_mov_b32 v6, 520",
            "\tv_mov_b32 v7, 2",
            "\tv_ashr_pk_i8_i32 v83, v5, v6, v7",  # [-150, 130] -> [-128=0x80, 127=0x7f] = 0x7f80
            "\tv_ashr_pk_u8_i32 v83, v5, v6, v7 op_sel:[0,0,0,1]",  # -> [0x00, 130=0x82] = 0x8200
            # v84, v85: v_permlane16_swap_b32 with masked EXEC + bound_ctrl:1:
            "\tv_add_u32 v10, 0x100, v0",
            "\tv_add_u32 v11, 0x200, v0",
            "\ts_mov_b64 s[6:7], exec",
            "\ts_mov_b32 exec_hi, 0x0000ffff",  # lanes 48..63 inactive!
            "\tv_permlane16_swap_b32_e64 v10, v11 bound_ctrl:1",
            "\ts_mov_b64 exec, s[6:7]",
            "\tv_mov_b32 v84, v10",
            "\tv_mov_b32 v85, v11",
            # v86: v_dot2_f32_bf16 (neg_lo/neg_hi), v87: v_dot2c_f32_bf16_e64 (-v, |v|=neg_hi):
            "\tv_mov_b32 v12, 0x40003f80",  # lo=1.0, hi=2.0
            "\tv_mov_b32 v13, 0x40404080",  # lo=4.0, hi=3.0
            "\tv_mov_b32 v14, 0x3f800000",  # 1.0
            "\tv_dot2_f32_bf16 v86, v12, v13, v14 neg_lo:[1,0,0] neg_hi:[0,1,0]",
            "\tv_mov_b32 v87, 0x41200000",  # 10.0
            "\tv_dot2c_f32_bf16_e64 v87, -v12, |v13|",  # 10 + (-1)*4 + 2*(-3) = 0.0
            # v88: v_pk_minimum3_f16 (-0.0 < +0.0, quieted SNaN) & v89: clamp -> +0.0:
            "\tv_mov_b32 v15, 0x7c050000",  # lo=+0.0, hi=SNaN(0x7c05)
            "\tv_mov_b32 v16, 0x3c008000",  # lo=-0.0, hi=1.0
            "\tv_mov_b32 v17, 0x40003800",  # lo=0.5,  hi=2.0
            "\tv_pk_minimum3_f16 v88, v15, v16, v17",
            "\tv_pk_maximum3_f16 v89, v15, v16, v17 clamp",
        ]
    )
    out = _run_lanes(asm, 10)
    assert out[0, 0] == 0 and out[1, 0] == ((0x80000001 << 1) & 0xFFFFFFFF) ^ 197
    assert np.all(out[:, 1] == ((0xF0F0AA55 & 0xFF0055AA) | 0x00FF0F0F))
    hi_xor3 = (0xF0F0 ^ 0x55AA ^ 0x00FF) & 0xFFFF
    assert np.all(out[:, 2] == ((hi_xor3 << 16) | 0xDEAD))
    assert np.all(out[:, 3] == 0x82007F80)
    # permlane16_swap with lanes 48..63 inactive and bound_ctrl:1:
    # - rows 0 & 1 (0..31) both active -> normal swap
    # - row 2 (32..47, active) reads v10[48..63] (inactive!) -> bound_ctrl zeroes v11[32..47]!
    # - row 3 (48..63, inactive) -> v10[48..63] untouched (0x130..0x13f)!
    assert np.array_equal(out[16:32, 4], 0x200 + np.arange(0, 16, dtype=np.uint32))
    assert np.array_equal(out[0:16, 5], 0x100 + np.arange(16, 32, dtype=np.uint32))
    assert np.all(out[32:48, 5] == 0)
    assert np.array_equal(out[48:64, 4], 0x100 + np.arange(48, 64, dtype=np.uint32))
    assert np.all(out[:, 6].view(np.float32) == -9.0)
    assert np.all(out[:, 7].view(np.float32) == 0.0)
    assert np.all(out[:, 8] == 0x7E058000)
    assert np.all(out[:, 9] == 0x00003800)

    # Also check v_exp_legacy_f32 and v_log_legacy_f32 (±0.0 -> -FLT_MAX = 0xff7fffff):
    leg = _run_lanes(
        "\tv_mov_b32 v1, 0x40400000\n"  # 3.0
        "\tv_exp_legacy_f32 v80, v1\n"
        "\tv_mov_b32 v2, 0x80000000\n"  # -0.0
        "\tv_log_legacy_f32 v81, v2\n"
        "\tv_log_legacy_f32 v82, v80",
        3,
    )
    assert np.all(leg[:, 0].view(np.float32) == 8.0)
    assert np.all(leg[:, 1] == 0xFF7FFFFF)
    assert np.all(leg[:, 2].view(np.float32) == 3.0)


def _pack_col_major_subbytes(mat: np.ndarray, bits: int) -> bytes:
    """Pack a column-major (M, K) code matrix (M contiguous) into raw LDS bytes."""
    flat = np.ascontiguousarray(mat.T).reshape(-1)  # (K, M) -> M fastest!
    if bits == 16:
        return flat.astype(np.uint16).tobytes()
    if bits == 8:
        return flat.astype(np.uint8).tobytes()
    if bits == 4:
        lo = flat[0::2].astype(np.uint8) & 0xF
        hi = (flat[1::2].astype(np.uint8) & 0xF) << 4
        return (lo | hi).tobytes()
    # bits == 6: pack every 4 elements (24 bits) into 3 bytes:
    c = flat.astype(np.uint32) & 0x3F
    w24 = c[0::4] | (c[1::4] << 6) | (c[2::4] << 12) | (c[3::4] << 18)
    out = np.empty(w24.size * 3, dtype=np.uint8)
    out[0::3] = w24 & 0xFF
    out[1::3] = (w24 >> 8) & 0xFF
    out[2::3] = (w24 >> 16) & 0xFF
    return out.tobytes()


def test_gfx950_transpose_lds_reads_feed_mfma_across_all_4_widths():
    """Verify DS_READ_B64_TR_B{16,8,4} and DS_READ_B96_TR_B6 feed directly into MFMA."""
    from thin import formats
    from thin.emu import Cluster, Memory
    from thin.target import targeting

    rng = np.random.default_rng(9508)

    cases = [
        # (tr_op, mfma_op, m, n, k, bits, dt_name, mods, a_reg0, a_reg1, a_mfma, second_off)
        (
            "ds_read_b64_tr_b16",
            "v_mfma_f32_32x32x16_bf16",
            32,
            32,
            16,
            16,
            "bf16",
            "",
            "v[16:17]",
            "v[18:19]",
            "v[16:19]",
            256,
        ),
        (
            "ds_read_b64_tr_b8",
            "v_mfma_i32_16x16x64_i8",
            16,
            16,
            64,
            8,
            None,
            "",
            "v[16:17]",
            "v[18:19]",
            "v[16:19]",
            128,
        ),
        (
            "ds_read_b64_tr_b4",
            "v_mfma_f32_32x32x64_f8f6f4",
            32,
            32,
            64,
            4,
            "e2m1",
            "cbsz:4 blgp:4",
            "v[16:17]",
            "v[18:19]",
            "v[16:19]",
            256,
        ),
        # Odd VGPR tuple v[17:19] + v[20:22] -> v[17:22] for 96-bit 6-bit transpose!
        (
            "ds_read_b96_tr_b6",
            "v_mfma_f32_32x32x64_f8f6f4",
            32,
            32,
            64,
            6,
            "e2m3",
            "cbsz:2 blgp:2",
            "v[17:19]",
            "v[20:22]",
            "v[17:22]",
            384,
        ),
    ]

    for tr_op, mfma_op, m, n, k, bits, dt_name, mods, r0, r1, a_op, off2 in cases:
        if dt_name is None:
            a_i8 = rng.integers(-8, 8, (m, k), dtype=np.int8)
            b_i8 = rng.integers(-8, 8, (n, k), dtype=np.int8)
            a_codes = a_i8.view(np.uint8)
            b_codes = b_i8.view(np.uint8)
            want = a_i8.astype(np.int32) @ b_i8.astype(np.int32).T
        else:
            dt = formats.DTYPES[dt_name]
            a_f32 = (rng.integers(-3, 4, (m, k)) * 0.5).astype(np.float32)
            b_f32 = (rng.integers(-3, 4, (n, k)) * 0.5).astype(np.float32)
            kw = {"nan": 0} if bits < 8 else {}
            a_codes = formats.encode(a_f32, dt, saturate=True, **kw)
            b_codes = formats.encode(b_f32, dt, saturate=True, **kw)
            want = (
                formats.decode(a_codes, dt).astype(np.float64)
                @ formats.decode(b_codes, dt).astype(np.float64).T
            ).astype(np.float32)

        lds_blob = _pack_col_major_subbytes(a_codes, bits)
        b_lanes = _pack_cdna4_input(b_codes, n, k, bits)
        nb = b_lanes.shape[1]
        d_dw = (m * n) // 64

        # Per-lane LDS address for `ds_read_*_tr_*`:
        col_bytes = (m * bits) // 8
        lds_read_bytes = 12 if bits == 6 else 8
        n_elems = (8 * lds_read_bytes) // bits
        lanes_per_k = 16 // n_elems
        addrs = np.zeros(64, dtype=np.uint32)
        for lane in range(64):
            row, col = divmod(lane, 16)
            k_local = col // lanes_per_k
            m_grp = col % lanes_per_k
            if m == 16:
                k_col = (k // 4) * row + k_local
                m_byte = m_grp * lds_read_bytes
            else:
                k_col = (k // 2) * (row // 2) + k_local
                m_byte = ((row % 2) * 16 + m_grp * n_elems) * bits // 8
            addrs[lane] = k_col * col_bytes + m_byte

        # Stage `lds_blob` into LDS via global_load_lds_dwordx4 (`64 * 16 = 1024` or 2x):
        padded = lds_blob + b"\x00" * ((-len(lds_blob)) % 1024)
        n_dma = len(padded) // 1024
        lines = [
            "\ts_load_dwordx8 s[4:11], s[0:1], 0x0",
            "\ts_waitcnt lgkmcnt(0)",
            "\tv_lshlrev_b32 v2, 4, v0",  # 16 * lane
        ]
        for step in range(n_dma):
            lines.append(f"\ts_mov_b32 m0, {step * 1024}")
            lines.append(f"\tglobal_load_lds_dwordx4 v2, s[4:5] offset:{step * 1024}")
        lines.extend(
            [
                "\tv_lshlrev_b32 v3, 2, v0",
                "\tglobal_load_dword v4, v3, s[6:7]",
                f"\tv_mul_lo_u32 v5, v0, {nb * 4}",
                *(f"\tglobal_load_dword v{48 + d}, v5, s[8:9] offset:{d * 4}" for d in range(nb)),
                "\ts_waitcnt vmcnt(0)",
                f"\t{tr_op} {r0}, v4",
                f"\t{tr_op} {r1}, v4 offset:{off2}",
                "\ts_waitcnt lgkmcnt(0)",
                f"\t{mfma_op} a[0:{d_dw - 1}], {a_op}, v[48:{48 + nb - 1}], 0"
                + (f" {mods}" if mods else ""),
                f"\tv_mul_lo_u32 v6, v0, {d_dw * 4}",
                *(
                    f"\tv_accvgpr_read_b32 v7, a{d}\n"
                    f"\tglobal_store_dword v6, v7, s[10:11] offset:{d * 4}"
                    for d in range(d_dw)
                ),
                "\ts_endpgm\n",
            ]
        )

        mem = Memory()
        buf_lds = mem.alloc("lds_src", len(padded), tagged=False)
        buf_addr = mem.alloc("addrs", 256, tagged=False)
        buf_b = mem.alloc("b", b_lanes.nbytes, tagged=False)
        buf_d = mem.alloc("d", 64 * d_dw * 4, tagged=False)
        buf_lds.write(0, padded)
        buf_addr.write(0, addrs.tobytes())
        buf_b.write(0, np.ascontiguousarray(b_lanes, dtype=np.uint32).tobytes())
        ka = mem.alloc("ka", 32, tagged=False)
        ka.write(
            0,
            np.array(
                [buf_lds.base, buf_addr.base, buf_b.base, buf_d.base], dtype=np.uint64
            ).tobytes(),
        )
        with targeting("gfx950"):
            cl = Cluster(
                "\n".join(lines),
                {
                    "wgsize": 64,
                    "vgpr": 80,
                    "agpr": 32,
                    "lds": 4096,
                    "user_sgprs": 2,
                },
                mem,
                [ka],
                grid=(1,),
            )
            rep = cl.run()[0]
        got_u32 = np.frombuffer(buf_d.read(0, 64 * d_dw * 4), dtype=np.uint32).reshape(64, d_dw)
        got = _unpack_cdna4_output(got_u32, m, n)
        if dt_name is None:
            assert np.array_equal(got.view(np.int32), want), tr_op
        else:
            assert np.array_equal(got.view(np.float32), want), tr_op
            # For 32x32 B16 and B4 without swizzle, 64-bank LDS achieves ZERO bank conflicts!
            if bits in (16, 4):
                assert all(v[2] == 0 for v in rep.banks.values()), (tr_op, rep.banks)


def test_gfx950_wide_lds_dma_160kb_lds_64_banks_and_buffer_atomic_bf16():
    """Verify b96/b128 LDS DMA (dwordx3 skips dword 3), 160 KB LDS, 64 banks, and pk_add_bf16."""
    from thin.emu import Cluster, Memory
    from thin.target import targeting

    # 160 KB LDS = 163_840 bytes; place b128 DMA at top 1 KB (162_816..163_840) and
    # b96 DMA over pre-filled 0xdeadbeef at 161_792..162_816: dwordx3 leaves dword 3 untouched.
    src_x4 = np.arange(1, 257, dtype=np.uint32)  # 64 * 4 dwords = 1024 B
    src_x3 = (0x1000 + np.arange(64 * 3, dtype=np.uint32)).reshape(64, 3)  # 64 * 12 B = 768 B
    buf_x3 = (0x2000 + np.arange(64 * 3, dtype=np.uint32)).reshape(64, 3)

    asm = "\n".join(
        [
            "\ts_load_dwordx8 s[4:11], s[0:1], 0x0",
            "\ts_load_dwordx2 s[12:13], s[0:1], 0x20",
            "\ts_waitcnt lgkmcnt(0)",
            # Pre-fill LDS[161792..162816] and LDS[0..1024] with 0xdeadbeef (dword 3 must survive):
            "\tv_lshlrev_b32 v1, 4, v0",
            "\tv_mov_b32 v2, 0xdeadbeef",
            "\tv_mov_b32 v3, 0xdeadbeef",
            "\tv_mov_b32 v4, 0xdeadbeef",
            "\tv_mov_b32 v5, 0xdeadbeef",
            "\tv_add_u32 v6, 161792, v1",
            "\tds_write_b128 v6, v[2:5]",
            "\tds_write_b128 v1, v[2:5]",
            "\ts_waitcnt lgkmcnt(0)",
            # 1) global_load_lds_dwordx4 into top 1 KB of 160 KB LDS (162816..163840):
            "\ts_mov_b32 m0, 162816",
            "\tglobal_load_lds_dwordx4 v1, s[4:5]",
            # 2) global_load_lds_dwordx3 into 161792..162816 (stride 12 B global, 16 B LDS):
            "\tv_mul_lo_u32 v7, v0, 12",
            "\ts_mov_b32 m0, 161792",
            "\tglobal_load_lds_dwordx3 v7, s[6:7]",
            # 3) buffer_load_dwordx3 ... lds into 0..1024 via MUBUF descriptor s[16:19]:
            "\ts_mov_b32 s16, s8",
            "\ts_mov_b32 s17, s9",
            "\ts_mov_b32 s18, 768",
            "\ts_mov_b32 s19, 0x00020000",
            "\ts_mov_b32 m0, 0",
            "\tbuffer_load_dwordx3 v7, s[16:19], 0 offen lds",
            "\ts_waitcnt vmcnt(0)",
            # Read back all 3 regions (and test 64-bank conflict-free 64-lane b32 read at 162816):
            "\tv_lshlrev_b32 v8, 2, v0",
            "\tv_add_u32 v9, 162816, v8",
            "\tds_read_b32 v20, v9",  # 64 distinct dwords -> 0 bank conflicts on 64-bank LDS!
            "\tds_read_b128 v[24:27], v6",
            "\tds_read_b128 v[28:31], v1",
            "\ts_waitcnt lgkmcnt(0)",
            # 4) buffer_atomic_pk_add_bf16 with sc0 (returning original value):
            "\ts_mov_b32 s20, s10",
            "\ts_mov_b32 s21, s11",
            "\ts_mov_b32 s22, 256",
            "\ts_mov_b32 s23, 0x00020000",
            "\tv_mov_b32 v32, 0x40003f80",  # add [1.0, 2.0] in BF16
            "\tbuffer_atomic_pk_add_bf16 v32, v8, s[20:23], 0 offen sc0",
            "\ts_waitcnt vmcnt(0)",
            # Store [v20, v24..27, v28..31, v32] (10 dwords/lane) to out buffer:
            "\tv_mul_lo_u32 v40, v0, 40",
            "\tglobal_store_dword v40, v20, s[12:13] offset:0",
            *(
                f"\tglobal_store_dword v40, v{24 + d}, s[12:13] offset:{4 + 4 * d}"
                for d in range(4)
            ),
            *(
                f"\tglobal_store_dword v40, v{28 + d}, s[12:13] offset:{20 + 4 * d}"
                for d in range(4)
            ),
            "\tglobal_store_dword v40, v32, s[12:13] offset:36",
            "\ts_endpgm\n",
        ]
    )

    mem = Memory()
    b_gx4 = mem.alloc("gx4", src_x4.nbytes, tagged=False)
    b_gx3 = mem.alloc("gx3", src_x3.nbytes, tagged=False)
    b_bx3 = mem.alloc("bx3", buf_x3.nbytes, tagged=False)
    b_atm = mem.alloc("atm", 256, tagged=False)
    b_out = mem.alloc("out", 64 * 40, tagged=False)
    b_gx4.write(0, src_x4.tobytes())
    b_gx3.write(0, src_x3.tobytes())
    b_bx3.write(0, buf_x3.tobytes())
    init_bf16 = np.full(64, 0x3F004040, dtype=np.uint32)  # [3.0, 0.5] in BF16
    b_atm.write(0, init_bf16.tobytes())
    ka = mem.alloc("ka", 40, tagged=False)
    ka.write(
        0,
        np.array(
            [b_gx4.base, b_gx3.base, b_bx3.base, b_atm.base, b_out.base], dtype=np.uint64
        ).tobytes(),
    )

    with targeting("gfx950"):
        rep = Cluster(
            asm,
            {"wgsize": 64, "vgpr": 48, "agpr": 0, "lds": 163_840, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        ).run()[0]

    got = np.frombuffer(b_out.read(0, 64 * 40), dtype=np.uint32).reshape(64, 10)
    # 1) Top of 160 KB LDS (162816..163072) read via 64-lane ds_read_b32:
    assert np.array_equal(got[:, 0], src_x4[:64])
    # 2) global_load_lds_dwordx3 copied 3 dwords and left the 4th dword untouched:
    assert np.array_equal(got[:, 1:4], src_x3)
    assert np.all(got[:, 4] == 0xDEADBEEF)
    # 3) buffer_load_dwordx3 ... lds copied 3 dwords and left the 4th dword untouched:
    assert np.array_equal(got[:, 5:8], buf_x3)
    assert np.all(got[:, 8] == 0xDEADBEEF)
    # 4) Returning buffer_atomic_pk_add_bf16 returned [3.0, 0.5] and wrote [4.0, 2.5]:
    assert np.all(got[:, 9] == 0x3F004040)
    atm_after = np.frombuffer(b_atm.read(0, 256), dtype=np.uint32)
    assert np.all(atm_after == 0x40204080)  # lo=4.0 (0x4080), hi=2.5 (0x4020)
    # 5) Zero bank conflicts across all 64-bank aligned LDS ops:
    assert all(v[2] == 0 for v in rep.banks.values()), rep.banks
    # Contrast 64-bank vs 32-bank on stride-32-dword (128 B) access:
    # On 64-bank gfx950, lane*128 hits banks 0 and 32 (32-way conflict -> 31 conflict cycles),
    # whereas on 32-bank gfx942, lane*128 hits bank 0 for all 64 lanes (63 conflict cycles)!
    stride_asm = (
        "\tv_lshlrev_b32 v1, 7, v0\n\tds_write_b32 v1, v0\n\ts_waitcnt lgkmcnt(0)\n\ts_endpgm\n"
    )
    for tname, want_entry in [("gfx950", [1, 32, 31]), ("gfx942", [1, 64, 62])]:
        m2 = Memory()
        ka2 = m2.alloc("ka", 8, tagged=False)
        with targeting(tname):
            r2 = Cluster(
                stride_asm, {"wgsize": 64, "vgpr": 4, "lds": 16384}, m2, [ka2], grid=(1,)
            ).run()[0]
        assert list(r2.banks.values()) == [want_entry]


def test_gfx950_cdna4_hazards_in_compiled_kernel():
    """Compiled gfx950 kernels emit exact CDNA 4 wait states matching LLVM GCNHazardRecognizer."""
    import thin
    from thin.target import targeting

    def make_cdna4_hazards():
        from thin import KernelDef
        from thin.dsl import Args, O, ptr
        from thin.target import PHYS

        vcc = PHYS["vcc"]
        exec_reg = PHYS["exec"]

        def body(b, k):
            V, S, A = b.v, b.s, b.a  # noqa: N806
            p_in, p_out = k.load()
            lane = V.u32(k.tid, name="lane")
            w4_a = V.u32(n=4)
            w4_b = V.u32(n=4)
            w8_a = V.u32(n=8)
            w8_b = V.u32(n=8)
            b.emit(O.global_load(w4_a, V.u32(lane * 16), p_in))
            b.emit(O.global_load(w4_b, V.u32(lane * 16), p_in, offset=1024))
            b.emit(O.global_load(w8_a[:4], V.u32(lane * 32), p_in, offset=2048))
            b.emit(O.global_load(w8_a[4:], V.u32(lane * 32), p_in, offset=2064))
            b.emit(O.global_load(w8_b[:4], V.u32(lane * 32), p_in, offset=3072))
            b.emit(O.global_load(w8_b[4:], V.u32(lane * 32), p_in, offset=3088))
            b.wait(w4_a, w4_b, w8_a, w8_b, comment="ready")

            # 1) 4-pass XDL (`16x16x32_bf16`): same-op SrcC = 0, read_acc = 8 states (`s_nop 7`)
            acc4 = A.f32(n=4)
            b.emit(O.op("v_mfma_f32_16x16x32_bf16", acc4, w4_a, w4_b, 0))
            b.emit(O.op("v_mfma_f32_16x16x32_bf16", acc4, w4_b, w4_a, acc4))
            r_bf16 = V.f32()
            b.emit(O.op("v_accvgpr_read_b32", r_bf16, acc4[0]))

            # 2) `16x16x128_f8f6f4` FP8 (`cbsz:0 blgp:0` -> 8 passes -> 12 states = `s_nop 11`)
            acc_f8 = A.f32(n=4)
            b.emit(O.op("v_mfma_f32_16x16x128_f8f6f4", acc_f8, w8_a, w8_b, 0, cbsz=0, blgp=0))
            r_f8 = V.f32()
            b.emit(O.op("v_accvgpr_read_b32", r_f8, acc_f8[0]))

            # 3) `16x16x128_f8f6f4` FP4 (`cbsz:4 blgp:4` -> 4 passes -> 8 states = `s_nop 7`)
            acc_f4 = A.f32(n=4)
            b.emit(O.op("v_mfma_f32_16x16x128_f8f6f4", acc_f4, w4_a, w4_b, 0, cbsz=4, blgp=4))
            r_f4 = V.f32()
            b.emit(O.op("v_accvgpr_read_b32", r_f4, acc_f4[0]))

            # 4) `32x32x64_f8f6f4` FP8 (`cbsz:0 blgp:0` -> 16 passes -> 20 states = 15 + 3)
            acc16 = A.f32(n=16)
            b.emit(
                O.op(
                    "v_mfma_scale_f32_32x32x64_f8f6f4",
                    acc16,
                    w8_a,
                    w8_b,
                    0,
                    w4_a[0],
                    w4_b[0],
                    cbsz=0,
                    blgp=0,
                )
            )
            r_16p = V.f32()
            b.emit(O.op("v_accvgpr_read_b32", r_16p, acc16[0]))

            # 5) `16x16x4_f64` DGEMM -> VALU read: 19 states (`s_nop 15; s_nop 2`)
            acc_d = A.u32(n=8)
            b.emit(O.op("v_mfma_f64_16x16x4_f64", acc_d, w4_a[:2], w4_b[:2], 0))
            r_d = V.u32()
            b.emit(O.op("v_accvgpr_read_b32", r_d, acc_d[0]))

            # 6) CDNA 4 PERMLANE hazards: VALU -> swap = `s_nop 1`; CMPX -> swap = `s_nop 3`
            p1 = V.u32(w4_b[0])
            p0 = V.u32(w4_a[0] + 1)
            b.emit(O.op("v_permlane16_swap_b32", p0, p1))
            saved = S.u64()
            with b.raw_exec("cmpx permlane hazard"):
                b.emit(O.op("s_mov_b64", saved, exec_reg))
                b.emit(O.op("v_cmpx_ne_u32", vcc, w4_a[0], -1))
                b.emit(O.op("v_permlane32_swap_b32", p0, p1))
                b.emit(O.op("s_mov_b64", exec_reg, saved))

            # 7) `v_cvt_scalef32_sr_pk_fp4_f32` (`HasFP4DstByteSel`): `s_nop 0` even at `op_sel` 0
            sr_dst = V.u32(0)
            b.emit(
                O.op(
                    "v_cvt_scalef32_sr_pk_fp4_f32",
                    sr_dst,
                    w4_a[:2],
                    w4_b[0],
                    1.0,
                    op_sel=[0, 0, 0, 0],
                )
            )
            sr_use = V.u32()
            b.emit(O.op("v_add_u32", sr_use, 1, sr_dst))

            tot = V.f32(r_bf16 + r_f8 + r_f4 + r_16p)
            tot_u = V.u32(r_d + p0 + p1 + sr_use)
            b.emit(O.global_store(V.u32(lane * 4), tot, p_out))
            b.emit(O.global_store(V.u32(lane * 4), tot_u, p_out, offset=256))

        return KernelDef(
            "cdna4_hazards_probe",
            Args(p_in=ptr, p_out=ptr),
            body,
            wgsize=64,
            max_grid=(1,),
        )

    with targeting("gfx950"):
        kern = thin.compile(make_cdna4_hazards)

    ins = [
        ln.strip().split("//")[0].split(";")[0].strip()
        for ln in kern.asm.splitlines()
        if ln.startswith("\t") and not ln.strip().startswith((".", ";", "//"))
    ]

    def between(prefix_a: str, prefix_b: str, start: int = 0) -> tuple[int, list[str]]:
        ia = next(i for i in range(start, len(ins)) if ins[i].startswith(prefix_a))
        ib = next(i for i in range(ia + 1, len(ins)) if ins[i].startswith(prefix_b))
        return ib, ins[ia + 1 : ib]

    pos, mid = between("v_mfma_f32_16x16x32_bf16", "v_mfma_f32_16x16x32_bf16", 0)
    assert mid == []
    pos, mid = between("v_mfma_f32_16x16x32_bf16", "v_accvgpr_read_b32", pos)
    assert mid == ["s_nop 7"]
    pos, mid = between("v_mfma_f32_16x16x128_f8f6f4", "v_accvgpr_read_b32", pos)
    assert mid == ["s_nop 11"]
    pos, mid = between("v_mfma_f32_16x16x128_f8f6f4", "v_accvgpr_read_b32", pos)
    assert mid == ["s_nop 7"]
    pos, mid = between("v_mfma_scale_f32_32x32x64_f8f6f4", "v_accvgpr_read_b32", pos)
    assert mid == ["s_nop 15", "s_nop 3"]
    pos, mid = between("v_mfma_f64_16x16x4_f64", "v_accvgpr_read_b32", pos)
    assert mid == ["s_nop 15", "s_nop 2"]
    pos, mid = between("v_add_", "v_permlane16_swap_b32", pos)
    assert mid == ["s_nop 1"]
    pos, mid = between("v_cmpx_ne_u32", "v_permlane32_swap_b32", pos)
    assert mid == ["s_nop 3"]
    pos, mid = between("v_cvt_scalef32_sr_pk_fp4_f32", "v_add_u32", pos)
    assert mid == ["s_nop 0"]


def test_gfx950_dsl_mfma_scale_transpose_read_and_wide_dma():
    """DSL builders support f8f6f4 formats, b.mfma_scale, O.ds_read_tr, and wide 12/16B LDS DMA."""
    import thin
    from thin import formats
    from thin.dsl import Builder, O
    from thin.ir import MemError
    from thin.target import target, targeting

    t950 = target("gfx950")
    mf_sc = t950.matrix_form("v_mfma_scale_f32_16x16x128_f8f6f4")
    mf_un = t950.matrix_form("v_mfma_f32_16x16x128_f8f6f4")
    assert mf_sc is not None and mf_un is not None
    assert mf_sc.scale_block == 32 and mf_un.scale_block == 0
    assert [mf_sc.operand_dwords("a", cbsz=c) for c in range(5)] == [8, 8, 6, 6, 4]
    assert [mf_sc.operand_dwords("b", blgp=c) for c in range(5)] == [8, 8, 6, 6, 4]
    assert mf_sc.operand_dwords("s") == 1

    # 1) End-to-end DSL MXFP4 tile kernel: wide 16B global-to-LDS DMA -> transpose `ds_read_tr` ->
    #    `b.mfma_scale` with `fmt_a="fp4", fmt_b="fp4", sel_a=1, sel_b=2`, verified on emulator!
    rng = np.random.default_rng(95010)
    a_f32 = rng.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0], size=(16, 128)).astype(
        np.float32
    )
    b_f32 = rng.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0], size=(128, 16)).astype(
        np.float32
    )
    # Per-block E8M0 scales: 4 blocks along K=128 (`k_blk in 0..4`).
    # Put A scales in byte 1 (`sel_a=1`) and B scales in byte 2 (`sel_b=2`) of each scale VGPR!
    exp_a = rng.integers(126, 130, size=(4, 16), dtype=np.uint32)  # 2^{-1} .. 2^2
    exp_b = rng.integers(126, 130, size=(4, 16), dtype=np.uint32)
    sa_lanes = (exp_a.reshape(64) << 8).astype(np.uint32)
    sb_lanes = (exp_b.reshape(64) << 16).astype(np.uint32)

    # Pack 16x128 -> col-major 128x16 so each K-column is 16 FP4 = 8 B (2 cols = 16 B/lane).
    # `ds_read_b64_tr_b4` transposes a 16x16 subtile; two reads fill K=[0..64) and K=[64..128).
    def pack_col_major_fp4(mat_16x128: np.ndarray) -> np.ndarray:
        # Store 16-K chunks in `[0, 2, 1, 3, 4, 6, 5, 7]` order so half-wave `row=0,1`
        # hits disjoint 128B halves (`banks 0..31` and `32..63`) -> ZERO bank conflicts!
        col_major = np.ascontiguousarray(mat_16x128.T.reshape(2, 2, 2, 16, 16).swapaxes(1, 2))
        codes = formats.encode(col_major.reshape(-1), formats.DTYPES["e2m1"], saturate=True, nan=0)
        lo = codes[0::2] & 0xF
        hi = (codes[1::2] & 0xF) << 4
        return (lo | hi).view(np.uint32)  # 256 dwords = 1024 bytes = 64 lanes * 16 bytes!

    a_packed = pack_col_major_fp4(a_f32)  # (256,) uint32 = 1024 B
    b_packed = pack_col_major_fp4(b_f32.T)  # B is (K=128, N=16) -> B.T is (N=16, K=128), 1024 B!

    # Reference MX-scaled result:
    a_scaled = a_f32.astype(np.float64).copy()
    b_scaled = b_f32.astype(np.float64).copy()
    for kb in range(4):
        s_a = np.ldexp(1.0, exp_a[kb].astype(np.int32) - 127)[:, None]  # (16, 1)
        s_b = np.ldexp(1.0, exp_b[kb].astype(np.int32) - 127)[None, :]  # (1, 16)
        a_scaled[:, kb * 32 : (kb + 1) * 32] *= s_a
        b_scaled[kb * 32 : (kb + 1) * 32, :] *= s_b
    ref_c = (a_scaled @ b_scaled).astype(np.float32)

    def make_mxfp4_dsl_kernel():
        from thin import KernelDef
        from thin.dsl import Args, O, ptr

        def body(b, k):
            V = b.v  # noqa: N806
            p_a, p_b, p_sa, p_sb, p_c = k.load()
            lds = b.lds()
            lds_a = lds.region("lds_a", nbytes=1024, align=16)
            lds_b = lds.region("lds_b", nbytes=1024, align=16)

            lane = V.u32(k.tid, name="lane")
            # Wide 16-byte global-to-LDS DMA: 64 lanes * 16 B = 1024 B per matrix!
            b.emit(O.global_load(lds_a, V.u32(lane * 16), p_a, bytes=16))
            b.emit(O.global_load(lds_b, V.u32(lane * 16), p_b, bytes=16))
            sa = V.u32()
            sb = V.u32()
            b.emit(O.global_load(sa, V.u32(lane * 4), p_sa))
            b.emit(O.global_load(sb, V.u32(lane * 4), p_sb))
            b.wait(lds_a, lds_b, sa, sb, comment="tiles + scales ready")

            # Transpose LDS reads (`ds_read_b64_tr_b4`): half-wave lanes 0..31 read contiguous
            # 0..256 B (64 distinct banks, zero conflicts!); +256 B reads the second 16-K half!
            tr_off = V.u32(((lane >> 5) * 64 + ((lane >> 4) & 1) * 16 + (lane & 15)) * 8)

            a_frag = V.u32(n=4)
            b_frag = V.u32(n=4)
            b.emit(O.ds_read_tr(a_frag[:2], lds_a + tr_off, "fp4"))
            b.emit(O.ds_read(a_frag[2:], lds_a + tr_off, offset=256, tr=4))
            b.emit(O.ds_read_tr(b_frag[:2], lds_b + tr_off, 4))
            b.emit(O.ds_read_tr(b_frag[2:], lds_b + tr_off, "b4", offset=256))
            b.wait(a_frag, b_frag, comment="transposed fragments ready")

            acc = b.mfma_acc("scale_f32_16x16x128_f8f6f4", 0)
            b.mfma_scale(
                "f32_16x16x128_f8f6f4",
                a_frag,
                b_frag,
                sa,
                sb,
                acc,
                dst=acc,
                fmt_a="fp4",
                fmt_b="e2m1",
                sel_a=1,
                sel_b=2,
            )
            c_out = b.read_acc(acc)
            # 16x16 f32 output: col = `lane & 15`, rows = `4*(lane>>4) + d`.
            col = V.u32(lane & 15)
            row0 = V.u32((lane >> 4) * 4)
            for d in range(4):
                elem_idx = V.u32((row0 + d) * 16 + col)
                b.emit(O.global_store(V.u32(elem_idx * 4), c_out[d], p_c))

        return KernelDef(
            "mxfp4_dsl_gemm16",
            Args(p_a=ptr, p_b=ptr, p_sa=ptr, p_sb=ptr, p_c=ptr),
            body,
            wgsize=64,
            max_grid=(1,),
        )

    with targeting("gfx950"):
        kern = thin.compile(make_mxfp4_dsl_kernel)
        assert "global_load_lds_dwordx4" in kern.asm
        assert "ds_read_b64_tr_b4" in kern.asm
        assert "v_mfma_scale_f32_16x16x128_f8f6f4" in kern.asm
        assert "op_sel:[1,0,0]" in kern.asm and "op_sel_hi:[0,1,0]" in kern.asm
        assert "cbsz:4" in kern.asm and "blgp:4" in kern.asm

    c_got = np.zeros((16, 16), dtype=np.float32)
    kern.bind(
        (1,),
        {
            "p_a": a_packed,
            "p_b": b_packed,
            "p_sa": sa_lanes,
            "p_sb": sb_lanes,
            "p_c": c_got.reshape(-1),
        },
    ).emulate()
    np.testing.assert_allclose(c_got, ref_c, rtol=0, atol=0)

    # 2) Multi-wave wide 16-byte DMA race check (`64 * 16 = 1024` B per wave -> stride 256 races!).
    def make_wide_dma_race():
        from thin import KernelDef
        from thin.dsl import Args, O, ptr

        def body(b, k):
            src, _ = k.load()
            tile = b.lds().region("tile", nbytes=2048, align=16)
            b.emit(O.global_load(tile + k.wave * 256, b.v.u32(k.tid * 16), src, bytes=16))
            b.wait(tile, "landed")
            b.barrier("done")

        return KernelDef("wide_dma_race", Args(src=ptr, out=ptr), body, wgsize=128, max_grid=(1,))

    with targeting("gfx950"), pytest.raises(MemError, match="may touch the same bytes"):
        thin.compile(make_wide_dma_race)

    # 3) Builder & Maker coverage for FP6 (`b96_tr_b6` + 12B buffer DMA) and clean refusals.
    with targeting("gfx950"):
        b = Builder()
        v6 = b.v.u32(n=6)
        v8 = b.v.u32(n=8)
        v3 = b.v.u32(n=3)
        v2 = b.v.u32(n=2)
        v1 = b.v.u32()
        a4 = b.a.f32(n=4)
        s4 = b.s.u32(n=4)
        s1 = b.s.u32()
        lds = b.lds()
        reg_b6 = lds.region("reg_b6", nbytes=1024, align=16)

        # 12-byte buffer DMA (`buffer_load_dwordx3 ... lds`) and `ds_read_b96_tr_b6` (3 dwords):
        b.emit(O.buffer_load(reg_b6, v1, s4, bytes=12))
        b.emit(O.ds_read_tr(v3, reg_b6 + v1, "fp6"))
        # Mixed FP6 x BF8 `b.mfma` (6 x 8 dwords -> `cbsz:2 blgp:1`) and `O.mfma_scale`:
        acc = b.mfma("f32_16x16x128_f8f6f4", v6, v8, 0, fmt_a="fp6", fmt_b="bf8")
        ins_sc = O.mfma_scale(
            "f32_16x16x128_f8f6f4",
            acc,
            v6,
            v6,
            v1,
            v1,
            fmt_a="e2m3",
            fmt_b="e3m2",
            sel_a=3,
            sel_b=0,
        )
        assert ins_sc.op == "v_mfma_scale_f32_16x16x128_f8f6f4"

        # Validation refusals on gfx950:
        with pytest.raises(ValueError, match="MX-scaled matrix op — use mfma_scale"):
            O.mfma("scale_f32_16x16x128_f8f6f4", a4, v8, v8, a4)
        with pytest.raises(ValueError, match="disagree"):
            O.mfma("f32_16x16x128_f8f6f4", a4, v6, v8, a4, fmt_a="fp6", cbsz=4)
        with pytest.raises(ValueError, match="does not take abid"):
            O.mfma("f32_16x16x128_f8f6f4", a4, v8, v8, a4, abid=1)
        with pytest.raises(ValueError, match="form takes 4"):
            O.mfma("f32_16x16x128_f8f6f4", a4, v8, v8, a4, fmt_a="fp4")
        with pytest.raises(ValueError, match="takes 'v' \\(VGPR\\)"):
            O.mfma_scale("f32_16x16x128_f8f6f4", a4, v8, v8, s1, v1, a4)
        with pytest.raises(ValueError, match="sel_a 4 out of range"):
            O.mfma_scale("f32_16x16x128_f8f6f4", a4, v8, v8, v1, v1, a4, sel_a=4)
        with pytest.raises(ValueError, match="6-bit transpose read takes 3 dwords"):
            O.ds_read_tr(v2, v1, 6)
        with pytest.raises(ValueError, match="bytes=8: 1, 2, 4, 12, or 16"):
            O.global_load(reg_b6, v1, bytes=8)

    # Target capability refusals on gfx942 (CDNA 3):
    with targeting("gfx942"):
        b942 = Builder()
        v2_942 = b942.v.u32(n=2)
        v8_942 = b942.v.u32(n=8)
        v1_942 = b942.v.u32()
        a4_942 = b942.a.f32(n=4)
        s4_942 = b942.s.u32(n=4)
        tile942 = b942.lds().region("t", nbytes=1024, align=16)
        with pytest.raises(ValueError, match="has no such MX-scaled MFMA form"):
            O.mfma_scale("f32_16x16x128_f8f6f4", a4_942, v8_942, v8_942, v1_942, v1_942, a4_942)
        with pytest.raises(ValueError, match="has no transpose LDS read"):
            O.ds_read_tr(v2_942, v1_942, 16)
        with pytest.raises(ValueError, match="has no global-to-LDS DMA"):
            O.global_load(tile942, v1_942, bytes=16)
        with pytest.raises(ValueError, match="has no buffer-to-LDS DMA"):
            O.buffer_load(tile942, v1_942, s4_942, bytes=16)
