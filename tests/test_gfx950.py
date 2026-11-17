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
        ).run()
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
