"""End-to-end tests for AMD Instinct MI300X (gfx942 / CDNA 3) target support."""

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


def _gen_across(lanes: int = 64):
    from thin import KernelDef
    from thin.dsl import Args, O, across, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        x, y = k.load("x", "y")
        off = V.u32(k.tid * 4, name="off")
        a = V.f32(name="a")
        b.emit(O.global_load(a, off, x))
        b.wait(a, "input loaded")
        r = V.f32(across(a, "add", lanes), name="r")
        b.emit(O.global_store(off, r, y))

    return KernelDef(f"across64_{lanes}", Args(x=ptr, y=ptr), body, wgsize=64, max_grid=(1,))


def _gen_atomics():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        acc, prev = k.load("acc", "prev")
        zero = V.u32(0, name="zero")
        val = V.u32(k.tid + 1, name="val")
        old = V.u32(name="old")
        b.emit(O.op("global_atomic_add", old, zero, val, acc, sc0=True))
        b.wait(old, "atomic return")
        out_off = V.u32((k.wgid * 64 + k.tid) * 4, name="out_off")
        b.emit(O.global_store(out_off, old, prev))
        # Non-returning atomic adds 1 per lane into acc[1].
        four = V.u32(4, name="four")
        one = V.u32(1, name="one")
        b.emit(O.op("global_atomic_add", four, one, acc))

    return KernelDef("gfx942_atomics", Args(acc=ptr, prev=ptr), body, wgsize=64, max_grid=(2,))


def _party_decl():
    from thin.lib import xwg

    return xwg.Gens(per="party", words=2, wgsize=128)


def _gen_party():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    decl = _party_decl()

    def body(b, k):
        (out,) = k.load("out")
        g = decl.begin(b, k, at=0)
        b.barrier("read word")
        with b.when(k.wave == 0):
            g.advance()
        o = b.v.u32(k.wave * 4, name="o")
        b.emit(O.global_store(o, b.v.u32(g.g1, name="g1"), out))

    return KernelDef("gfx942_party", Args(out=ptr, **decl.args()), body, wgsize=128, max_grid=(1,))


def _gen_mfma():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V, A, S = b.v, b.a, b.s  # noqa: N806
        (out,) = k.load("out")
        ma = V.u32(n=2, name="ma")
        mb = V.u32(n=2, name="mb")
        ma[0][...] = 0x3C003C00
        ma[1][...] = 0x3C003C00
        mb[0][...] = 0x3C003C00
        mb[1][...] = 0x3C003C00
        acc = A.f32(n=4, name="acc")
        for idx in range(4):
            acc[idx][...] = 0
        b.emit(O.op("v_mfma_f32_16x16x16_f16", acc, ma, mb, acc))
        vout = V.f32(acc[0], name="vout")
        scaled = V.f32(name="scaled")
        b.emit(O.op("v_fma_f32", scaled, vout, 1.0, vout, omod=1))
        # Exercise VALU -> VMEM SADDR hazard (v_readfirstlane_b32 writing SGPR).
        s_off = S.u32(name="s_off")
        b.emit(O.op("v_readfirstlane_b32", s_off, V.u32(0, name="z")))
        addr = S.u64(name="addr")
        b.emit(O.op("s_add_u32", addr.lo, out.lo, s_off))
        b.emit(O.op("s_addc_u32", addr.hi, out.hi, 0))
        b.emit(O.global_store(V.u32(k.tid * 4, name="off"), scaled, addr))

    return KernelDef("gfx942_mfma", Args(out=ptr), body, wgsize=64, max_grid=(1,))


def test_gfx942_target_facts_and_allocation_granules():
    from thin.target import target

    t = target("gfx942")
    assert t.name == "gfx942"
    assert t.wave == 64
    assert t.lds_max == 65536
    assert t.barrier == [("s_barrier", [])]
    assert t.waits == [("s_waitcnt", ["vmcnt", "expcnt", "lgkmcnt"])]
    assert t.xcds == 8 and t.cus == 304 and t.vgprs_per_lane == 512
    assert t.sgpr_max == 102 and t.vgprs == 256

    assert t.allocated_vgprs(0, 0) == 0
    assert t.allocated_vgprs(1, 0) == 8
    assert t.allocated_vgprs(8, 0) == 8
    assert t.allocated_vgprs(8, 4) == 16
    assert t.allocated_vgprs(1, 1) == 8

    assert t.counters == {"vm": "vmcnt", "lgkm": "lgkmcnt"}
    assert t.returning("buffer_atomic_add", None) is False
    assert t.returning("buffer_atomic_add", "sc0") is True


@pytest.mark.parametrize("n", [1, 63, 64, 65, 255, 256, 1000])
def test_gfx942_vadd_wave64_predicated_emulation(n):
    import thin
    from examples.vadd import build_vadd
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(build_vadd)

    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"' in k.asm
    assert ".amdhsa_wavefront_size32" not in k.asm
    assert "s_mov_b64 exec, vcc" in k.asm
    assert "s_mov_b64 exec, -1" in k.asm
    assert "s_waitcnt vmcnt(" in k.asm

    # Run emulate() outside targeting("gfx942") to verify code-object target inference.
    x = np.arange(n, dtype=np.float32)
    y = 2.5 * x - 1.0
    out = k(x=x, y=y, z=np.zeros(n, np.float32), n=n).emulate()
    assert np.allclose(out["z"], x + y)


def test_gfx942_vadd_with_laps_uses_memtime_and_memrealtime():
    import thin
    from examples.vadd import build_vadd
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(build_vadd, laps=True)

    assert "s_memtime" in k.asm
    assert "s_memrealtime" in k.asm
    n = 1000
    x = np.arange(n, dtype=np.float32)
    y = np.full(n, 0.5, dtype=np.float32)
    z = np.zeros(n, np.float32)
    bound = k(x=x, y=y, z=z, n=n)
    bound.emulate()
    assert z.tolist() == (x + y).tolist()
    timeline = k.definition.meta["laps"].decode(bound.owned["laps"], 4)
    # 256 lanes / wave64 = 4 waves per workgroup across 4 workgroups
    assert timeline.rounds == [[1] * 4] * 4
    phases = timeline.phases()
    assert len(phases["load"]) == len(phases["store"]) == 16
    assert all(p > 0 for p in phases["load"] + phases["store"])


COMPARE = json.loads((data_path("metron", "kernels.json")).read_text())["compare"]


@pytest.mark.parametrize(
    "case", COMPARE, ids=[f"stride{c['stride']}x{c['copies']}" for c in COMPARE]
)
def test_gfx942_metron_differ_on_emulator(case):
    import thin
    from metron import maker
    from thin.target import targeting

    with targeting("gfx942"):
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


def test_gfx942_metron_icache_inv_waits():
    import thin
    from metron import maker
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(maker.gen_icache_inv)
    body = [line.split("//")[0].strip() for line in k.asm.splitlines() if line.startswith("\t")]
    body = [line for line in body if line and not line.startswith(".")]
    assert body[-18:] == ["s_icache_inv", *["s_nop 0"] * 16, "s_endpgm"]


@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32, 64])
def test_gfx942_wave64_across_reduction(lanes):
    import thin
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(_gen_across, lanes=lanes)

    rng = np.random.default_rng(19 + lanes)
    x = rng.integers(-8, 9, 64).astype(np.float32)
    y = np.zeros(64, np.float32)
    k.bind((1,), {"x": x, "y": y}).emulate()
    want = np.repeat(np.sum(x.reshape(-1, lanes), axis=1), lanes)
    assert y.tolist() == want.tolist()


def test_gfx942_returning_and_non_returning_atomics_and_xwg_gens():
    import thin
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(_gen_atomics)

    acc = np.array([100, 0], dtype=np.uint32)
    prev = np.zeros(128, dtype=np.uint32)
    k.bind((2,), {"acc": acc, "prev": prev}).emulate()
    assert int(acc[0]) == 100 + 2 * (64 * 65 // 2)
    assert int(acc[1]) == 128
    assert len(set(prev.tolist())) == 128

    with targeting("gfx942"):
        decl = _party_decl()
        kp = thin.compile(_gen_party)
    assert ", 6" in kp.asm  # wave64: wave = first_lane(tid) >> 6
    out = np.zeros(2, dtype=np.uint32)
    bound = kp.bind(grid=1, args={"out": out, **decl.sizes()})
    bound.emulate()
    assert out[0] == out[1]


def test_gfx942_multi_xcd_l2_scope_sc1_enforcement():
    """On MI300X (8 XCDs, private L2 per XCD), cross-XCD visibility requires sc1."""
    from thin.emu import Cluster, Memory, ScopeModel
    from thin.target import targeting

    def run_pair(coherent: bool):
        sc = " sc1" if coherent else ""
        asm = f"""\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_mov_b32 v1, 0
\ts_cmp_eq_u32 s2, 0
\ts_cbranch_scc0 .Lreader
\tv_mov_b32 v2, 42
\tglobal_store_dword v1, v2, s[4:5]{sc}
\ts_waitcnt vmcnt(0)
\tv_mov_b32 v3, 1
\tglobal_store_dword v1, v3, s[6:7] sc1
\ts_waitcnt vmcnt(0)
\ts_branch .Ldone
.Lreader:
.Lspin:
\tglobal_load_dword v4, v1, s[6:7] sc1
\ts_waitcnt vmcnt(0)
\tv_cmp_eq_u32 vcc, v4, 1
\ts_cbranch_vccz .Lspin
\tglobal_load_dword v5, v1, s[4:5]{sc}
\ts_waitcnt vmcnt(0)
\tglobal_store_dword v1, v5, s[4:5] offset:64 sc1
.Ldone:
\ts_endpgm
"""
        m = Memory()
        data = m.alloc("data", 128)
        flag = m.alloc("flag", 64)
        k = m.alloc("k", 16, tagged=False)
        k.write(0, np.array([data.base, flag.base], dtype=np.uint64).tobytes())
        with targeting("gfx942"):
            Cluster(asm, {"wgsize": 64, "vgpr": 8}, m, [k], grid=(2,)).run()
        return int.from_bytes(data.read(64, 4), "little")

    with pytest.raises(ScopeModel):
        run_pair(coherent=False)
    assert run_pair(coherent=True) == 42


def test_gfx942_mfma_agpr_dsl_hazards_descriptor_and_device_mismatch():
    import thin
    from thin import formats
    from thin.runtime import program
    from thin.runtime.emulated import EmulatedDevice
    from thin.runtime.errors import LaunchError
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(_gen_mfma)

    assert "v_mfma_f32_16x16x16_f16" in k.asm
    assert "v_accvgpr_read_b32" in k.asm
    assert "mul:2" in k.asm
    assert ".amdhsa_accum_offset" in k.asm
    assert "s_nop" in k.asm

    obj = formats.read(k.co)
    assert obj.target == "gfx942"
    meta = obj.metadata["amdhsa.kernels"][0]
    assert meta[".agpr_count"] >= 4
    assert meta[".vgpr_count"] >= meta[".agpr_count"] + 4

    # Loading a gfx942 code object onto an explicitly gfx1201 EmulatedDevice is refused.
    dev = EmulatedDevice(target="gfx1201")
    try:
        with pytest.raises(
            LaunchError, match="a code object for gfx942, the emulated device is gfx1201"
        ):
            program(dev, k.co, k.asm, k.name)
    finally:
        dev.close()


def _gen_fnuz_cvt():
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806
        inp, packed_fp8, packed_bf8, up_fp8, up_bf8 = k.load(
            "inp", "packed_fp8", "packed_bf8", "up_fp8", "up_bf8"
        )
        off16 = V.u32(k.tid * 16, name="off16")
        off4 = V.u32(k.tid * 4, name="off4")
        off8 = V.u32(k.tid * 8, name="off8")
        x = V.f32(n=4, name="x")
        b.emit(O.global_load(x, off16, inp))
        b.wait(x, "f32 inputs")

        pf = V.u32(0, name="pf")
        b.emit(O.op("v_cvt_pk_fp8_f32", pf, x[0], x[1]))
        b.emit(O.op("v_cvt_pk_fp8_f32", pf, x[2], x[3], op_sel=[0, 0, 1]))
        b.emit(O.global_store(off4, pf, packed_fp8))

        pb = V.u32(0, name="pb")
        b.emit(O.op("v_cvt_pk_bf8_f32", pb, x[0], x[1]))
        b.emit(O.op("v_cvt_pk_bf8_f32", pb, x[2], x[3], op_sel=[0, 0, 1]))
        b.emit(O.global_store(off4, pb, packed_bf8))

        uf = V.f32(n=2, name="uf")
        b.emit(O.op("v_cvt_pk_f32_fp8", uf, pf))
        b.emit(O.global_store(off8, uf, up_fp8))

        ub = V.f32(name="ub")
        b.emit(O.op("v_cvt_f32_bf8", ub, pb))
        b.emit(O.global_store(off4, ub, up_bf8))

    return KernelDef(
        "gfx942_fnuz_cvt",
        Args(inp=ptr, packed_fp8=ptr, packed_bf8=ptr, up_fp8=ptr, up_bf8=ptr),
        body,
        wgsize=64,
        max_grid=(1,),
    )


def test_gfx942_fnuz_fp8_and_bf8_conversions_end_to_end():
    import thin
    from thin import formats
    from thin.target import targeting

    with targeting("gfx942"):
        k = thin.compile(_gen_fnuz_cvt)

    specials = np.array(
        [
            0.0,
            -0.0,
            -1e-30,
            2**-10,
            -(2**-10),
            2**-17,
            1.0,
            -1.5,
            240.0,
            -240.0,
            248.0,
            -248.0,
            57344.0,
            -57344.0,
            61440.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(942)
    rest = rng.uniform(-300.0, 300.0, 256 - len(specials)).astype(np.float32)
    inp = np.concatenate([specials, rest])
    packed_fp8 = np.zeros(64, dtype=np.uint32)
    packed_bf8 = np.zeros(64, dtype=np.uint32)
    up_fp8 = np.zeros(128, dtype=np.float32)
    up_bf8 = np.zeros(64, dtype=np.float32)

    k.bind(
        (1,),
        {
            "inp": inp,
            "packed_fp8": packed_fp8,
            "packed_bf8": packed_bf8,
            "up_fp8": up_fp8,
            "up_bf8": up_bf8,
        },
    ).emulate()

    f8 = formats.DTYPES["e4m3fnuz"]
    b8 = formats.DTYPES["e5m2fnuz"]
    want_fp8 = formats.encode(inp, f8, saturate=False).view(np.uint32)
    want_bf8 = formats.encode(inp, b8, saturate=False).view(np.uint32)
    assert packed_fp8.tolist() == want_fp8.tolist()
    assert packed_bf8.tolist() == want_bf8.tolist()

    # Check CDNA3 FNUZ specifics: -0.0 -> 0x00, 240.0 -> 0x7F, 248.0 / +-inf / nan -> 0x80.
    raw_fp8 = packed_fp8.view(np.uint8)
    assert raw_fp8[1] == 0x00
    assert raw_fp8[2] == 0x00
    assert raw_fp8[8] == 0x7F
    assert raw_fp8[9] == 0xFF
    assert raw_fp8[10] == 0x80
    assert raw_fp8[15] == 0x80
    assert raw_fp8[16] == 0x80
    assert raw_fp8[17] == 0x80

    # Unpack of low 2 bytes per lane matches formats.decode, with 0x80 -> DNAN32 (0xFFC00000).
    low2 = packed_fp8.view(np.uint8).reshape(64, 4)[:, :2].reshape(-1)
    want_up_fp8 = formats.decode(low2, f8)
    nan_mask = low2 == 0x80
    assert np.all(up_fp8.view(np.uint32)[nan_mask] == 0xFFC00000)
    assert np.array_equal(up_fp8[~nan_mask], want_up_fp8[~nan_mask])

    low1 = packed_bf8.view(np.uint8).reshape(64, 4)[:, 0]
    want_up_bf8 = formats.decode(low1, b8)
    bnan = low1 == 0x80
    assert np.all(up_bf8.view(np.uint32)[bnan] == 0xFFC00000)
    assert np.array_equal(up_bf8[~bnan], want_up_bf8[~bnan])


def _pack_cdna3_input(mat: np.ndarray, bits: int, *, sparse: bool = False) -> np.ndarray:
    """Packs `[blocks, outer, k_in]` raw element integers into `[64, dwords]` uint32 per lane."""
    blocks, outer, k_in = mat.shape
    elems_contig = k_in // (64 // (outer * blocks))
    dwords = max(1, (elems_contig * bits) // 32)
    out = np.zeros((64, dwords), dtype=np.uint32)
    mask = (1 << bits) - 1
    for b in range(blocks):
        for i in range(outer):
            for k in range(k_in):
                local = k % elems_contig
                lane = b * outer + (k // elems_contig) * (outer * blocks) + i
                tot_bits = local * bits
                dw = tot_bits // 32
                bit_lo = tot_bits % 32
                val = int(mat[b, i, k]) & mask
                if bits == 64:
                    out[lane, dw] = val & 0xFFFFFFFF
                    out[lane, dw + 1] = (val >> 32) & 0xFFFFFFFF
                else:
                    out[lane, dw] |= val << bit_lo
    return out


def _unpack_cdna3_output(lanes: np.ndarray, blocks: int, m: int, n: int, d_bits: int) -> np.ndarray:
    """Unpacks `[64, dwords]` uint32 per lane into `[blocks, m, n]` raw integer elements."""
    multirows_per_reg = 64 // n
    multirow_h = 1 if d_bits == 64 else 4
    chunk = (n * m) // multirow_h
    blocks_per_reg = (64 + chunk - 1) // chunk
    out = np.zeros((blocks, m, n), dtype=np.uint64 if d_bits == 64 else np.uint32)
    for b in range(blocks):
        for i in range(m):
            for j in range(n):
                local = (
                    b * ((m * n) // 64)
                    + (i // (multirow_h * multirows_per_reg)) * multirow_h
                    + (i % multirow_h)
                )
                lane = (
                    (b % blocks_per_reg) * n
                    + ((i // multirow_h) % multirows_per_reg) * blocks_per_reg * n
                    + j
                )
                if d_bits == 64:
                    out[b, i, j] = int(lanes[lane, 2 * local]) | (
                        int(lanes[lane, 2 * local + 1]) << 32
                    )
                else:
                    out[b, i, j] = lanes[lane, local]
    return out


def _run_mfma_asm(
    mnemonic: str,
    a_lanes: np.ndarray,
    b_lanes: np.ndarray,
    c_lanes: np.ndarray | str,
    d_dwords: int,
    *,
    mods: str = "",
    use_agpr: bool = True,
    sparse: bool = False,
) -> np.ndarray:
    """Executes one wave64 MFMA/SMFMAC instruction on gfx942; returns `[64, d_dwords]` uint32."""
    from thin.emu import Cluster, Memory
    from thin.target import targeting

    na = a_lanes.shape[1]
    nb = b_lanes.shape[1]
    lines = [
        "\ts_load_dwordx8 s[4:11], s[0:1], 0x0",
        "\ts_waitcnt lgkmcnt(0)",
    ]
    # Load A into v[16:16+na-1]
    lines.append("\tv_lshlrev_b32 v1, 2, v0")
    lines.append(f"\tv_mul_lo_u32 v2, v0, {na * 4}")
    for d in range(na):
        lines.append(f"\tglobal_load_dword v{16 + d}, v2, s[4:5] offset:{d * 4}")
    lines.append(f"\tv_mul_lo_u32 v3, v0, {nb * 4}")
    for d in range(nb):
        lines.append(f"\tglobal_load_dword v{48 + d}, v3, s[6:7] offset:{d * 4}")

    c_is_buf = isinstance(c_lanes, np.ndarray)
    if sparse:
        assert c_is_buf
        # Initialize D (in a[0:...] or v[80:...]) to 10.0 per element, and load sparse idx into v79
        lines.append("\tglobal_load_dword v79, v1, s[8:9]")
        lines.append("\ts_waitcnt vmcnt(0)")
        init_bits = hex(np.float32(10.0).view(np.uint32).item())
        for d in range(d_dwords):
            if use_agpr:
                lines.append(f"\tv_accvgpr_write_b32 a{d}, {init_bits}")
            else:
                lines.append(f"\tv_mov_b32 v{80 + d}, {init_bits}")
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
    mod_suffix = f" {mods}" if mods else ""
    lines.append(f"\t{mnemonic} {dst_op}, {a_op}, {b_op}, {src2_op}{mod_suffix}")

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
    ka = mem.alloc("ka", 32, tagged=False)
    ka.write(
        0,
        np.array([buf_a.base, buf_b.base, buf_c.base, buf_d.base], dtype=np.uint64).tobytes(),
    )
    with targeting("gfx942"):
        Cluster(
            asm,
            {"wgsize": 64, "vgpr": 128, "agpr": 32, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        ).run()
    return np.frombuffer(buf_d.read(0, 64 * d_dwords * 4), dtype=np.uint32).reshape(64, d_dwords)


def test_gfx942_mfma_dense_numerical_and_layout_parity():
    from thin import formats

    rng = np.random.default_rng(300)

    # 1. 32x32x8_f16 -> 16 dwords of f32 per lane
    a_f16 = rng.integers(-4, 5, (1, 32, 8)).astype(np.float16)
    b_f16 = rng.integers(-4, 5, (1, 8, 32)).astype(np.float16)
    c_f32 = rng.integers(-10, 11, (1, 32, 32)).astype(np.float32)
    a_lanes = _pack_cdna3_input(a_f16.view(np.uint16), 16)
    b_lanes = _pack_cdna3_input(np.swapaxes(b_f16, 1, 2).view(np.uint16), 16)
    c_lanes = _pack_cdna3_input(
        np.zeros((1, 32, 16), dtype=np.uint32), 32
    )  # build C directly via output layout inverse:
    # Pack C in output layout (row_groups = 64//32 = 2, sub_rows = 4):
    c_lanes = np.zeros((64, 16), dtype=np.uint32)
    c_u32 = c_f32.view(np.uint32)
    for i in range(32):
        for j in range(32):
            rg = (i // 4) % 2
            major = (i // 4) // 2
            lane = rg * 32 + j
            dw = major * 4 + (i % 4)
            c_lanes[lane, dw] = c_u32[0, i, j]
    d_lanes = _run_mfma_asm("v_mfma_f32_32x32x8_f16", a_lanes, b_lanes, c_lanes, 16, use_agpr=True)
    got = _unpack_cdna3_output(d_lanes, 1, 32, 32, 32).view(np.float32)[0]
    want = a_f16[0].astype(np.float32) @ b_f16[0].astype(np.float32) + c_f32[0]
    assert np.array_equal(got, want)

    # 2. 16x16x16_bf16 in VGPRs with inline literal C = 1.0
    a_bf = (rng.integers(-5, 6, (1, 16, 16)).astype(np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )
    b_bf = (rng.integers(-5, 6, (1, 16, 16)).astype(np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )
    d_lanes = _run_mfma_asm(
        "v_mfma_f32_16x16x16_bf16",
        _pack_cdna3_input(a_bf, 16),
        _pack_cdna3_input(np.swapaxes(b_bf, 1, 2), 16),
        "1.0",
        4,
        use_agpr=False,
    )
    got_bf = _unpack_cdna3_output(d_lanes, 1, 16, 16, 32).view(np.float32)[0]
    a_ref = (a_bf.astype(np.uint32) << 16).view(np.float32)[0]
    b_ref = (b_bf.astype(np.uint32) << 16).view(np.float32)[0]
    assert np.array_equal(got_bf, a_ref @ b_ref + 1.0)

    # 3. 16x16x32_fp8_bf8 (FNUZ e4m3fnuz * e5m2fnuz, plus 0x80 NaN -> 0xFFC00000)
    f8 = formats.DTYPES["e4m3fnuz"]
    b8 = formats.DTYPES["e5m2fnuz"]
    a_fp8 = formats.encode(
        rng.integers(-6, 7, (1, 16, 32)).astype(np.float32) * 0.5, f8, saturate=False
    )
    b_bf8 = formats.encode(
        rng.integers(-6, 7, (1, 32, 16)).astype(np.float32) * 0.5, b8, saturate=False
    )
    a_fp8[0, 3, 7] = 0x80  # inject FNUZ NaN into row 3
    d_lanes = _run_mfma_asm(
        "v_mfma_f32_16x16x32_fp8_bf8",
        _pack_cdna3_input(a_fp8, 8),
        _pack_cdna3_input(np.swapaxes(b_bf8, 1, 2), 8),
        "0",
        4,
    )
    got_u32 = _unpack_cdna3_output(d_lanes, 1, 16, 16, 32)[0]
    assert np.all(got_u32[3, :] == 0xFFC00000)
    want_fp8 = formats.decode(a_fp8[0], f8) @ formats.decode(b_bf8[0], b8)
    mask = np.ones((16, 16), dtype=bool)
    mask[3, :] = False
    assert np.array_equal(got_u32.view(np.float32)[mask], want_fp8[mask])

    # 4. 16x16x8_xf32: low 13 mantissa bits of non-NaN FP32 inputs are truncated (`& 0xFFFFE000`)
    a_raw = np.full((1, 16, 8), 0x3F801FFF, dtype=np.uint32)  # 1.0 + low-13-bit noise -> 1.0
    b_raw = np.full((1, 8, 16), 0x40001FFF, dtype=np.uint32)  # 2.0 + low-13-bit noise -> 2.0
    d_lanes = _run_mfma_asm(
        "v_mfma_f32_16x16x8_xf32",
        _pack_cdna3_input(a_raw, 32),
        _pack_cdna3_input(np.swapaxes(b_raw, 1, 2), 32),
        "0",
        4,
    )
    got_xf32 = _unpack_cdna3_output(d_lanes, 1, 16, 16, 32).view(np.float32)[0]
    assert np.all(got_xf32 == 16.0)

    # 5. 16x16x4_f64 with `neg:[1,0,1]` and `1.0` inline C: D = (-A) @ B + (-1.0)
    a_f64 = rng.integers(-5, 6, (1, 16, 4)).astype(np.float64) * 0.25
    b_f64 = rng.integers(-5, 6, (1, 4, 16)).astype(np.float64) * 0.5
    d_lanes = _run_mfma_asm(
        "v_mfma_f64_16x16x4_f64",
        _pack_cdna3_input(a_f64.view(np.uint64), 64),
        _pack_cdna3_input(np.swapaxes(b_f64, 1, 2).view(np.uint64), 64),
        "1.0",
        8,
        mods="neg:[1,0,1]",
    )
    got_f64 = _unpack_cdna3_output(d_lanes, 1, 16, 16, 64).view(np.float64)[0]
    assert np.array_equal(got_f64, (-a_f64[0]) @ b_f64[0] - 1.0)

    # 6. 16x16x32_i8: signed int8 -> int32 exact accumulation
    a_i8 = rng.integers(-128, 128, (1, 16, 32), dtype=np.int8)
    b_i8 = rng.integers(-128, 128, (1, 32, 16), dtype=np.int8)
    d_lanes = _run_mfma_asm(
        "v_mfma_i32_16x16x32_i8",
        _pack_cdna3_input(a_i8.view(np.uint8), 8),
        _pack_cdna3_input(np.swapaxes(b_i8, 1, 2).view(np.uint8), 8),
        "-7",
        4,
    )
    got_i32 = _unpack_cdna3_output(d_lanes, 1, 16, 16, 32).view(np.int32)[0]
    assert np.array_equal(got_i32, a_i8[0].astype(np.int32) @ b_i8[0].astype(np.int32) - 7)


def test_gfx942_mfma_broadcast_modifiers_and_smfmac_sparse():
    rng = np.random.default_rng(9420)

    # 1a. Multi-block `16x16x4_4b_f16` (16 dwords D) with `cbsz:2 abid:1`: all 4 blocks use A[1]
    a_4b = rng.integers(-3, 4, (4, 16, 4)).astype(np.float16)
    b_4b = rng.integers(-3, 4, (4, 4, 16)).astype(np.float16)
    d_lanes = _run_mfma_asm(
        "v_mfma_f32_16x16x4_4b_f16",
        _pack_cdna3_input(a_4b.view(np.uint16), 16),
        _pack_cdna3_input(np.swapaxes(b_4b, 1, 2).view(np.uint16), 16),
        "0",
        16,
        mods="cbsz:2 abid:1",
    )
    got_4b = _unpack_cdna3_output(d_lanes, 4, 16, 16, 32).view(np.float32)
    for blk in range(4):
        want_blk = a_4b[1].astype(np.float32) @ b_4b[blk].astype(np.float32)
        assert np.array_equal(got_4b[blk], want_blk)

    # 1b. Multi-block `4x4x4_16b_f16` (4 dwords D across 16 blocks) with `cbsz:4 abid:9`
    a_16b = rng.integers(-3, 4, (16, 4, 4)).astype(np.float16)
    b_16b = rng.integers(-3, 4, (16, 4, 4)).astype(np.float16)
    d_16b = _run_mfma_asm(
        "v_mfma_f32_4x4x4_16b_f16",
        _pack_cdna3_input(a_16b.view(np.uint16), 16),
        _pack_cdna3_input(np.swapaxes(b_16b, 1, 2).view(np.uint16), 16),
        "0",
        4,
        mods="cbsz:4 abid:9",
    )
    got_16b = _unpack_cdna3_output(d_16b, 16, 4, 4, 32).view(np.float32)
    for blk in range(16):
        want_blk = a_16b[9].astype(np.float32) @ b_16b[blk].astype(np.float32)
        assert np.array_equal(got_16b[blk], want_blk)

    # 2. `blgp:1..7` on `v_mfma_f32_16x16x4_f32`: permutes/broadcasts B lanes before GEMM
    a_f32 = rng.integers(-3, 4, (1, 16, 4)).astype(np.float32)
    b_f32 = rng.integers(-3, 4, (1, 4, 16)).astype(np.float32)
    a_pack = _pack_cdna3_input(a_f32.view(np.uint32), 32)
    b_pack = _pack_cdna3_input(np.swapaxes(b_f32, 1, 2).view(np.uint32), 32)
    for blgp in range(1, 8):
        d_lanes = _run_mfma_asm(
            "v_mfma_f32_16x16x4_f32", a_pack, b_pack, "0", 4, mods=f"blgp:{blgp}"
        )
        got = _unpack_cdna3_output(d_lanes, 1, 16, 16, 32).view(np.float32)[0]
        # Reconstruct effective B after blgp lane remap:
        src_lanes = [
            lane
            if blgp == 0
            else (lane % 32)
            if blgp == 1
            else (lane % 32) + 32
            if blgp == 2
            else (lane + 16) % 64
            if blgp == 3
            else (lane % 16) + 16 * (blgp - 4)
            for lane in range(64)
        ]
        b_swz = b_pack[src_lanes]
        b_eff = np.zeros((4, 16), dtype=np.float32)
        for j in range(16):
            for k in range(4):
                b_eff[k, j] = b_swz[k * 16 + j, :1].view(np.float32)[0]
        assert np.array_equal(got, a_f32[0] @ b_eff)

    # 3. 4:2 sparse `v_smfmac_f32_16x16x32_f16` with `cbsz:0 abid:0` and `cbsz:0 abid:2`
    a_comp = rng.integers(-4, 5, (1, 16, 16)).astype(np.float16)  # compressed K/2 = 16
    b_dense = rng.integers(-4, 5, (1, 32, 16)).astype(np.float16)
    a_sp_lanes = _pack_cdna3_input(a_comp.view(np.uint16), 16, sparse=True)
    b_sp_lanes = _pack_cdna3_input(np.swapaxes(b_dense, 1, 2).view(np.uint16), 16)

    # Random valid 4:2 index nibbles per (row i, group g in 0..8) for abid=2 (bits 16..31)
    idx_pairs = np.zeros((16, 8, 2), dtype=np.int32)
    idx_lanes = np.zeros((64, 1), dtype=np.uint32)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i in range(16):
        for g in range(8):
            p0, p1 = pairs[int(rng.integers(0, len(pairs)))]
            idx_pairs[i, g] = (p0, p1)
            nib = (p0 & 3) | ((p1 & 3) << 2)
            # 16x16x32_f16 has 4 lane groups -> 2 nibbles/lane; abid=2 shifts by 4 nibbles.
            lane = (g // 2) * 16 + i
            nib_pos = 4 + (g % 2)
            idx_lanes[lane, 0] |= nib << (nib_pos * 4)

    d_lanes = _run_mfma_asm(
        "v_smfmac_f32_16x16x32_f16",
        a_sp_lanes,
        b_sp_lanes,
        idx_lanes,
        4,
        mods="cbsz:0 abid:2",
        sparse=True,
    )
    got_sp = _unpack_cdna3_output(d_lanes, 1, 16, 16, 32).view(np.float32)[0]
    a_expanded = np.zeros((16, 32), dtype=np.float32)
    for i in range(16):
        for g in range(8):
            p0, p1 = idx_pairs[i, g]
            if p0 < p1:
                a_expanded[i, 4 * g + p0] = float(a_comp[0, i, 2 * g])
            a_expanded[i, 4 * g + p1] = float(a_comp[0, i, 2 * g + 1])
    want_sp = a_expanded @ b_dense[0].astype(np.float32) + 10.0
    assert np.array_equal(got_sp, want_sp)


def test_gfx942_target_matrix_form_model():
    from thin.target import Flags, target

    t = target("gfx942")
    matrix_forms = [f for f in t.forms() if f.flags & Flags.MATRIX]
    assert len(matrix_forms) == 46
    for f in matrix_forms:
        mf = t.matrix_form(f.mnemonic)
        assert mf is not None
        assert mf.wave == 64
        assert mf.sparse == f.mnemonic.startswith("v_smfmac_")
        assert mf.dwords == {
            "a": f.slots[1].dwords,
            "b": f.slots[2].dwords,
            "c": f.slots[3].dwords,
            "d": f.slots[0].dwords,
        }

    f16_32 = t.matrix_form("v_mfma_f32_32x32x8_f16")
    assert (f16_32.m, f16_32.n, f16_32.k, f16_32.blocks, f16_32.passes) == (32, 32, 8, 1, 8)
    assert (f16_32.a, f16_32.b, f16_32.d) == ("f16", "f16", "f32")
    assert not f16_32.cbsz_abid and not f16_32.blgp and not f16_32.sparse
    f16_2b = t.matrix_form("v_mfma_f32_32x32x4_2b_f16")
    assert f16_2b.cbsz_abid and f16_2b.blgp
    assert t.matrix_form("v_mfma_f32_16x16x4_f32").blgp

    fp8_16 = t.matrix_form("v_mfma_f32_16x16x32_fp8_bf8")
    assert (fp8_16.m, fp8_16.n, fp8_16.k, fp8_16.passes) == (16, 16, 32, 4)
    assert (fp8_16.a, fp8_16.b, fp8_16.d) == ("e4m3fnuz", "e5m2fnuz", "f32")
    assert not fp8_16.cbsz_abid and not fp8_16.blgp

    sp = t.matrix_form("v_smfmac_f32_16x16x32_f16")
    assert (sp.m, sp.n, sp.k, sp.passes, sp.sparse) == (16, 16, 32, 4, True)
    assert sp.dwords == {"a": 2, "b": 4, "c": 1, "d": 4}
    assert t.matrix_form("v_add_f32") is None


def test_gfx942_gemm_provenance_checker_agpr_vgpr_and_faults():
    from thin.emu import Cluster, Memory, gemm
    from thin.target import targeting

    # 1. Multi-K-block (K=64, two 16x16x32_fp8_bf8 steps):
    # Step 0 accumulates in VGPRs v[80:83], transfers via v_accvgpr_write_b32 -> a[0:3] (resume),
    # Step 1 accumulates into a[0:3], moves via v_accvgpr_mov_b32 -> a[4:7], reads & stores D.
    asm_k64 = """\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_load_dwordx2 s[8:9], s[0:1], 0x10
\ts_waitcnt lgkmcnt(0)
\t// lane = (q * 16) + row_or_col: q = v0 >> 4, idx = v0 & 15
\tv_and_b32 v1, 15, v0
\tv_lshrrev_b32 v2, 4, v0
\t// A[idx, q*8 .. q*8+7] at byte offset idx*64 + q*8
\tv_lshlrev_b32 v3, 6, v1
\tv_lshl_add_u32 v3, v2, 3, v3
\t// B is rowmajor("B", ld_bytes=64, elem_bytes=1): col=idx, k=q*8..q*8+7 -> offset idx*64 + q*8
\tglobal_load_dwordx2 v[16:17], v3, s[4:5]
\tglobal_load_dwordx2 v[20:21], v3, s[6:7]
\tglobal_load_dwordx2 v[24:25], v3, s[4:5] offset:32
\tglobal_load_dwordx2 v[28:29], v3, s[6:7] offset:32
\ts_waitcnt vmcnt(0)
\tv_mfma_f32_16x16x32_fp8_bf8 v[80:83], v[16:17], v[20:21], 0
\tv_accvgpr_write_b32 a0, v80
\tv_accvgpr_write_b32 a1, v81
\tv_accvgpr_write_b32 a2, v82
\tv_accvgpr_write_b32 a3, v83
\tv_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[24:25], v[28:29], a[0:3]
\tv_accvgpr_mov_b32 a4, a0
\tv_accvgpr_mov_b32 a5, a1
\tv_accvgpr_mov_b32 a6, a2
\tv_accvgpr_mov_b32 a7, a3
\t// D[row, col]: lane l has col = l & 15, row_base = (l >> 4) * 4; row stride = 16*4 = 64B
\tv_lshlrev_b32 v4, 8, v2
\tv_lshl_add_u32 v4, v1, 2, v4
\tv_accvgpr_read_b32 v32, a4
\tv_accvgpr_read_b32 v33, a5
\tv_accvgpr_read_b32 v34, a6
\tv_accvgpr_read_b32 v35, a7
\tglobal_store_dword v4, v32, s[8:9] offset:0
\tglobal_store_dword v4, v33, s[8:9] offset:64
\tglobal_store_dword v4, v34, s[8:9] offset:128
\tglobal_store_dword v4, v35, s[8:9] offset:192
\ts_endpgm
"""

    def run_gemm_check(asm_text: str, *, k_total: int = 64, m_dim: int = 16, elem_b: int = 1):
        mem = Memory()
        ld_ab = k_total * elem_b
        ld_d = m_dim * 4
        buf_a = mem.alloc("A", m_dim * ld_ab)
        buf_b = mem.alloc("B", m_dim * ld_ab)
        buf_d = mem.alloc("D", m_dim * ld_d)
        ka = mem.alloc("ka", 24, tagged=False)
        ka.write(0, np.array([buf_a.base, buf_b.base, buf_d.base], dtype=np.uint64).tobytes())
        prob = gemm.Problem(
            {
                "A": gemm.rowmajor("A", ld_ab, elem_bytes=elem_b),
                "B": gemm.rowmajor("B", ld_ab, elem_bytes=elem_b),
                "D": gemm.rowmajor_d(ld_d, elem_bytes=4),
            },
            K=k_total,
            d_rows=np.arange(m_dim),
            d_cols=np.arange(m_dim),
            fail_fast=False,
        )
        hooks = gemm.GemmHooks(prob, mem)
        with targeting("gfx942"):
            Cluster(
                asm_text,
                {"wgsize": 64, "vgpr": 96, "agpr": 32, "user_sgprs": 2},
                mem,
                [ka],
                grid=(1,),
                hooks=hooks,
            ).run()
        return prob

    prob_ok = run_gemm_check(asm_k64, k_total=64, m_dim=16, elem_b=1)
    assert prob_ok.errors == []
    assert len(prob_ok.stores) == 256

    # 2. Fault detection: adding k-block 0 twice reports duplicate k-block accumulation.
    asm_dup = asm_k64.replace(
        "v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[24:25], v[28:29], a[0:3]",
        "v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[16:17], v[20:21], a[0:3]",
    )
    prob_dup = run_gemm_check(asm_dup, k_total=64, m_dim=16, elem_b=1)
    assert any("adds k-blocks [0] a second time" in e for e in prob_dup.errors)

    # 3. 32x32x8_f16 (16 AGPR dwords, 2-byte elements, K=8) provenance check across all 1024 cells:
    asm_32 = (
        """\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_load_dwordx2 s[8:9], s[0:1], 0x10
\ts_waitcnt lgkmcnt(0)
\t// lane = q*32 + idx: q = v0 >> 5, idx = v0 & 31
\tv_and_b32 v1, 31, v0
\tv_lshrrev_b32 v2, 5, v0
\t// A/B ld_bytes = 8 * 2 = 16; each lane reads 4 f16s = 8B at idx*16 + q*8
\tv_lshlrev_b32 v3, 4, v1
\tv_lshl_add_u32 v3, v2, 3, v3
\tglobal_load_dwordx2 v[16:17], v3, s[4:5]
\tglobal_load_dwordx2 v[20:21], v3, s[6:7]
\ts_waitcnt vmcnt(0)
\tv_mfma_f32_32x32x8_f16 a[0:15], v[16:17], v[20:21], 0
\t// D ld_bytes = 32 * 4 = 128B.
\t// Output layout: row = ((dw // 4) * 2 + q) * 4 + (dw % 4), col = idx.
\tv_lshlrev_b32 v4, 9, v2
\tv_lshl_add_u32 v4, v1, 2, v4
"""
        + "".join(
            f"\tv_accvgpr_read_b32 v32, a{dw}\n"
            f"\tglobal_store_dword v4, v32, s[8:9] offset:{((dw // 4) * 8 + (dw % 4)) * 128}\n"
            for dw in range(16)
        )
        + "\ts_endpgm\n"
    )
    prob_32 = run_gemm_check(asm_32, k_total=8, m_dim=32, elem_b=2)
    assert prob_32.errors == []
    assert len(prob_32.stores) == 1024


def test_gfx942_dsl_high_level_mfma_agpr_and_vgpr(tmp_path):
    """High-level DSL `b.mfma`, `b.mfma_acc`, `b.smfmac`, `b.read_acc`, `b.write_acc`."""
    import numpy as np
    import pytest

    import thin
    from thin.dsl import Builder
    from thin.target import targeting

    rng = np.random.default_rng(20261116)

    # 1. Same 2-tile K=32 f16 GEMM kernel compiled with AGPR (`acc="a"`) vs VGPR (`acc="v"`):
    def make_gemm_def(acc_file: str):
        from thin import KernelDef
        from thin.dsl import Args, O, ptr

        def body(b, k):
            V = b.v  # noqa: N806
            a0_ptr, b0_ptr, a1_ptr, b1_ptr, out_ptr = k.load()
            lane = V.u32(k.tid, name="lane")
            in_off = V.u32(lane * 8, name="in_off")
            out_off = V.u32(lane * 16, name="out_off")

            a0, b0 = V.u32(n=2, name="a0"), V.u32(n=2, name="b0")
            a1, b1 = V.u32(n=2, name="a1"), V.u32(n=2, name="b1")
            b.emit(O.global_load(a0, in_off, a0_ptr))
            b.emit(O.global_load(b0, in_off, b0_ptr))
            b.emit(O.global_load(a1, in_off, a1_ptr))
            b.emit(O.global_load(b1, in_off, b1_ptr))
            b.wait(a0, b0, a1, b1, comment="tiles landed")

            # Tile 0 allocates & zeroes accumulator (`c=0`), Tile 1 accumulates in-place:
            acc = b.mfma("f32_16x16x16_f16", a0, b0, acc=acc_file, name="acc")
            b.mfma("f32_16x16x16_f16", a1, b1, acc)

            v_out = b.read_acc(acc)
            b.emit(O.global_store(out_off, v_out, out_ptr))

        return KernelDef(
            f"dsl_mfma_{acc_file}",
            Args(a0=ptr, b0=ptr, a1=ptr, b1=ptr, out=ptr),
            body,
            wgsize=64,
            max_grid=(1,),
        )

    a0_m = rng.integers(-3, 4, (1, 16, 16)).astype(np.float16)
    b0_m = rng.integers(-3, 4, (1, 16, 16)).astype(np.float16)
    a1_m = rng.integers(-3, 4, (1, 16, 16)).astype(np.float16)
    b1_m = rng.integers(-3, 4, (1, 16, 16)).astype(np.float16)
    expected = a0_m[0].astype(np.float32) @ b0_m[0].astype(np.float32) + a1_m[0].astype(
        np.float32
    ) @ b1_m[0].astype(np.float32)

    a0_buf = _pack_cdna3_input(a0_m.view(np.uint16), 16).reshape(-1)
    b0_buf = _pack_cdna3_input(np.swapaxes(b0_m, 1, 2).view(np.uint16), 16).reshape(-1)
    a1_buf = _pack_cdna3_input(a1_m.view(np.uint16), 16).reshape(-1)
    b1_buf = _pack_cdna3_input(np.swapaxes(b1_m, 1, 2).view(np.uint16), 16).reshape(-1)

    before_s = thin.use_store(None)
    try:
        with targeting("gfx942"):
            k_a = thin.compile(make_gemm_def, acc_file="a")
            k_v = thin.compile(make_gemm_def, acc_file="v")

            assert k_a.manifest.agpr == 4
            assert "v_accvgpr_read_b32" in k_a.asm
            assert k_v.manifest.agpr == 0
            assert "v_accvgpr_read_b32" not in k_v.asm

            for kern in (k_a, k_v):
                out = np.zeros(64 * 4, dtype=np.uint32)
                kern.bind(
                    (1,),
                    {"a0": a0_buf, "b0": b0_buf, "a1": a1_buf, "b1": b1_buf, "out": out},
                ).emulate()
                got = _unpack_cdna3_output(out.reshape(64, 4), 1, 16, 16, 32).view(np.float32)[0]
                assert np.array_equal(got, expected)

            # 2. Sparse 4:2 `b.smfmac` + `b.mfma_acc` + `b.write_acc` across AGPR and VGPR:
            def make_smfmac_def(use_agpr: bool):
                from thin import KernelDef
                from thin.dsl import Args, O, ptr

                def body(b, k):
                    V = b.v  # noqa: N806
                    a_ptr, b_ptr, idx_ptr, c_ptr, out_ptr = k.load()
                    lane = V.u32(k.tid, name="lane")
                    a_in = V.u32(n=2, name="a_in")
                    b_in = V.u32(n=4, name="b_in")
                    idx_in = V.u32(name="idx_in")
                    c_in = V.f32(n=4, name="c_in")
                    b.emit(O.global_load(a_in, V.u32(lane * 8), a_ptr))
                    b.emit(O.global_load(b_in, V.u32(lane * 16), b_ptr))
                    b.emit(O.global_load(idx_in, V.u32(lane * 4), idx_ptr))
                    b.emit(O.global_load(c_in, V.u32(lane * 16), c_ptr))
                    b.wait(a_in, b_in, idx_in, c_in, comment="inputs landed")

                    acc = b.mfma_acc(
                        "smfmac_f32_16x16x32_f16", init=None, agpr=use_agpr, name="acc"
                    )
                    b.write_acc(c_in, dst=acc)
                    b.smfmac("f32_16x16x32_f16", a_in, b_in, idx_in, acc, agpr=use_agpr)
                    b.emit(O.global_store(V.u32(lane * 16), b.read_acc(acc), out_ptr))

                return KernelDef(
                    f"dsl_smfmac_{int(use_agpr)}",
                    Args(a=ptr, b=ptr, idx=ptr, c=ptr, out=ptr),
                    body,
                    wgsize=64,
                    max_grid=(1,),
                )

            a_comp = rng.integers(-2, 3, (1, 16, 16)).astype(np.float16)
            b_dense = rng.integers(-2, 3, (1, 32, 16)).astype(np.float16)
            c_init = rng.integers(-5, 6, (1, 16, 16)).astype(np.float32)
            # Every 4-element group selects indices (0, 2) -> nibble 0x8 -> 0x8888 per 16-bit set:
            a_exp = np.zeros((16, 32), dtype=np.float32)
            for g in range(8):
                a_exp[:, 4 * g + 0] = a_comp[0, :, 2 * g + 0]
                a_exp[:, 4 * g + 2] = a_comp[0, :, 2 * g + 1]
            exp_sparse = a_exp @ b_dense[0].astype(np.float32) + c_init[0]

            a_sbuf = _pack_cdna3_input(a_comp.view(np.uint16), 16).reshape(-1)
            b_sbuf = _pack_cdna3_input(np.swapaxes(b_dense, 1, 2).view(np.uint16), 16).reshape(-1)
            idx_sbuf = np.full(64, 0x88888888, dtype=np.uint32)
            c_sbuf = np.zeros((64, 4), dtype=np.uint32)
            for lane in range(64):
                col, grp = lane % 16, lane // 16
                c_sbuf[lane] = c_init[0, 4 * grp : 4 * grp + 4, col].view(np.uint32)

            for use_agpr in (True, False):
                ks = thin.compile(make_smfmac_def, use_agpr=use_agpr)
                assert (ks.manifest.agpr > 0) == use_agpr
                out_s = np.zeros(64 * 4, dtype=np.uint32)
                ks.bind(
                    (1,),
                    {
                        "a": a_sbuf,
                        "b": b_sbuf,
                        "idx": idx_sbuf,
                        "c": c_sbuf.reshape(-1),
                        "out": out_s,
                    },
                ).emulate()
                got_s = _unpack_cdna3_output(out_s.reshape(64, 4), 1, 16, 16, 32).view(np.float32)[
                    0
                ]
                assert np.array_equal(got_s, exp_sparse)

            # 3. Binary64 `f64_16x16x4_f64` with `neg=(1, 0, 0)` and `O.mfma`:
            def make_f64_def():
                from thin import KernelDef
                from thin.dsl import Args, O, ptr

                def body(b, k):
                    V = b.v  # noqa: N806
                    a_ptr, b_ptr, out_ptr = k.load()
                    lane = V.u32(k.tid, name="lane")
                    ma, mb = V.u64(name="ma"), V.u64(name="mb")
                    b.emit(O.global_load(ma, V.u32(lane * 8), a_ptr))
                    b.emit(O.global_load(mb, V.u32(lane * 8), b_ptr))
                    b.wait(ma, mb, comment="f64 inputs")
                    acc = b.mfma_acc("f64_16x16x4_f64", init=None, acc="a", name="acc64")
                    b.emit(O.mfma("f64_16x16x4_f64", acc, ma, mb, 0, neg=(1, 0, 0)))
                    v_acc = b.read_acc(acc, name="v_acc64")
                    for elem in range(4):
                        b.emit(
                            O.global_store(
                                V.u32(lane * 32), v_acc[elem], out_ptr, offset=8 * elem or None
                            )
                        )

                return KernelDef(
                    "dsl_mfma_f64",
                    Args(a=ptr, b=ptr, out=ptr),
                    body,
                    wgsize=64,
                    max_grid=(1,),
                )

            kf64 = thin.compile(make_f64_def)
            a64 = rng.integers(-4, 5, (1, 16, 4)).astype(np.float64) * 0.5
            b64 = rng.integers(-4, 5, (1, 4, 16)).astype(np.float64) * 0.25
            out64 = np.zeros(64 * 8, dtype=np.uint32)
            kf64.bind(
                (1,),
                {
                    "a": _pack_cdna3_input(a64.view(np.uint64), 64).reshape(-1),
                    "b": _pack_cdna3_input(np.swapaxes(b64, 1, 2).view(np.uint64), 64).reshape(-1),
                    "out": out64,
                },
            ).emulate()
            got64 = _unpack_cdna3_output(out64.reshape(64, 8), 1, 16, 16, 64).view(np.float64)[0]
            assert np.array_equal(got64, (-a64[0]) @ b64[0])

            # 4. Validation refusals:
            b_err = Builder()
            va2 = b_err.v.u32(n=2, name="va2")
            va3 = b_err.v.u32(n=3, name="va3")
            with pytest.raises(ValueError, match="disagree"):
                b_err.mfma_acc("f32_16x16x16_f16", acc="a", agpr=False)
            with pytest.raises(ValueError, match="form takes 2"):
                b_err.mfma("f32_16x16x16_f16", va3, va2)
            with pytest.raises(ValueError, match="does not take blgp"):
                b_err.mfma("f32_16x16x16_f16", va2, va2, blgp=1)
            with pytest.raises(ValueError, match="does not take neg"):
                b_err.mfma("f32_16x16x16_f16", va2, va2, neg=(1, 0, 0))
            with pytest.raises(ValueError, match="use smfmac"):
                b_err.mfma("smfmac_f32_16x16x32_f16", va2, va2)

        # Target without AGPRs rejects `acc="a"` cleanly:
        with targeting("gfx1201"):
            b_rdna = Builder()
            with pytest.raises(ValueError, match="has no a registers"):
                b_rdna.mfma_acc("v_wmma_f32_16x16x16_f16", acc="a")
            acc_rdna = b_rdna.mfma_acc("v_wmma_f32_16x16x16_f16")
            assert acc_rdna.file == "v" and acc_rdna.width == 8
    finally:
        thin.use_store(before_s)


def test_gfx942_pass_aware_mfma_hazards() -> None:
    """Compiled gfx942 kernels insert exact pass-count-aware MFMA wait states matching llc."""
    import thin
    from thin.target import targeting

    before_s = thin.use_store(None)
    try:
        with targeting("gfx942"):

            def make_hazard_probe(form: str, src_words: int):
                from thin import KernelDef
                from thin.dsl import Args, O, ptr

                def body(b, k):
                    V = b.v  # noqa: N806
                    a_ptr, b_ptr, out_ptr = k.load()
                    lane = V.u32(k.tid, name="lane")
                    a0 = V.u32(n=src_words) if src_words > 1 else V.u32()
                    b0 = V.u32(n=src_words) if src_words > 1 else V.u32()
                    a1 = V.u32(n=src_words) if src_words > 1 else V.u32()
                    b1 = V.u32(n=src_words) if src_words > 1 else V.u32()
                    b.emit(O.global_load(a0, V.u32(lane * (4 * src_words)), a_ptr))
                    b.emit(O.global_load(b0, V.u32(lane * (4 * src_words)), b_ptr))
                    b.emit(O.global_load(a1, V.u32(lane * (4 * src_words)), a_ptr, offset=256))
                    b.emit(O.global_load(b1, V.u32(lane * (4 * src_words)), b_ptr, offset=256))
                    b.wait(a0, b0, a1, b1, comment="ready")
                    acc = b.mfma(form, a0, b0, 0, acc="a")
                    b.mfma(form, a1, b1, acc, dst=acc)
                    v_out = b.read_acc(acc)
                    b.emit(O.global_store(V.u32(lane * 4), v_out[0], out_ptr))

                return KernelDef(
                    f"hz_{form}",
                    Args(a=ptr, b=ptr, out=ptr),
                    body,
                    wgsize=64,
                    max_grid=(1,),
                )

            def body_ins(asm: str) -> list[str]:
                return [
                    ln.strip().split("//")[0].split(";")[0].strip()
                    for ln in asm.splitlines()
                    if ln.startswith("\t") and not ln.strip().startswith((".", ";", "//"))
                ]

            def check_probe(form: str, src_words: int, expected_tail: list[str]) -> None:
                kern = thin.compile(make_hazard_probe, form=form, src_words=src_words)
                ins = body_ins(kern.asm)
                mfmas = [i for i, ln in enumerate(ins) if ln.startswith(f"v_mfma_{form}")]
                assert len(mfmas) == 2
                # Same-op full-register accumulation takes 0 wait states
                assert ins[mfmas[0] + 1 : mfmas[1]] == []
                first_read = next(
                    i
                    for i in range(mfmas[1] + 1, len(ins))
                    if ins[i].startswith("v_accvgpr_read_b32")
                )
                assert ins[mfmas[1] + 1 : first_read] == expected_tail

            # 4-pass XDL -> `s_nop 6` (7 states); 8-pass XDL -> `s_nop 10` (11 states);
            # 16-pass SMFMA -> chunked `s_nop 15` + `s_nop 1` (18 states)
            check_probe("f32_16x16x16_f16", 2, ["s_nop 6"])
            check_probe("f32_32x32x8_f16", 2, ["s_nop 10"])
            check_probe("f32_32x32x2_f32", 1, ["s_nop 15", "s_nop 1"])
    finally:
        thin.use_store(before_s)


def test_gfx942_lds_dma_widths_m0_dedup_oob_and_predication() -> None:
    """Buffer/global -> LDS DMA handles u8/i8/u16/i16/b32, offset, OOB, EXEC, and DropM0."""
    import thin
    from thin.target import targeting

    with targeting("gfx942"):

        def make_dma_widths():
            from thin import KernelDef
            from thin.dsl import Args, O, ptr, u32

            def body(b, k):
                V, S = b.v, b.s  # noqa: N806
                src_g, src_b, n_bytes, out = k.load()
                desc = S.u32(n=4, name="desc")
                b.emit(O.op("s_mov_b32", desc[0], src_b.lo))
                b.emit(O.op("s_and_b32", desc[1], src_b.hi, 0xFFFF))
                b.emit(O.op("s_mov_b32", desc[2], n_bytes))
                b.emit(O.op("s_mov_b32", desc[3], 0x00020000))
                tile = b.lds().region("tile", nbytes=8 * 256, align=16)
                lane = V.u32(k.tid, name="lane")

                # 1. Global DMA: u8, i8, u16, i16, b32 (with source-only `offset=`)
                b.emit(O.global_load(tile + 0, lane, src_g, bytes=1, offset=4))
                # Same constant LDS base (tile + 256) twice with different source offsets:
                # second `s_mov_b32 m0` should be dropped by `Pass::DropM0`!
                b.emit(O.global_load(tile + 256, lane, src_g, bytes=1, signed=True, offset=8))
                b.emit(O.global_load(tile + 256, lane, src_g, bytes=1, signed=True, offset=16))
                b.emit(O.global_load(tile + 512, V.u32(lane * 2), src_g, bytes=2, offset=32))
                b.emit(
                    O.global_load(
                        tile + 768, V.u32(lane * 2), src_g, bytes=2, signed=True, offset=64
                    )
                )
                # Predicated global b32 DMA: pre-fill LDS slot, wait, then overwrite odd lanes
                b.emit(O.ds_write(tile + 1024 + lane * 4, V.u32(0xCAFE0000 + lane)))
                b.wait(tile, "prefill done")
                with b.lanes((lane & 1) == 1):
                    b.emit(O.global_load(tile + 1024, V.u32(lane * 4), src_g, offset=128))

                # 2. Buffer DMA: b32 with OOB lanes zeroed, and offset > 4095 staged into `soffset`
                b.emit(O.buffer_load(tile + 1280, V.u32(lane * 4), desc, 0, offset=16))
                b.emit(O.buffer_load(tile + 1536, lane, desc, 0, bytes=1, signed=True, offset=4100))

                # Wait on `tile` lowers to `vmcnt(0)` because DMAs track on VM counter.
                b.wait(tile, "all DMAs landed")

                for slot in range(7):
                    val = V.u32()
                    b.emit(O.ds_read(val, tile + slot * 256 + lane * 4))
                    b.wait(val, f"read slot {slot}")
                    b.emit(O.global_store(V.u32((slot * 64 + lane) * 4), val, out))

            return KernelDef(
                "dma_widths",
                Args(src_g=ptr, src_b=ptr, n_bytes=u32, out=ptr),
                body,
                wgsize=64,
                max_grid=(1,),
            )

        kern = thin.compile(make_dma_widths)
        asm_lines = [
            ln.strip().split("//")[0].split(";")[0].strip()
            for ln in kern.asm.splitlines()
            if ln.startswith("\t") and not ln.strip().startswith((".", ";", "//"))
        ]
        # Verify DropM0 deduplicated the second `s_mov_b32 m0, 0x100` (256)
        m0_256 = [ln for ln in asm_lines if ln in ("s_mov_b32 m0, 256", "s_mov_b32 m0, 0x100")]
        assert len(m0_256) == 1, f"expected DropM0 to keep one m0=256 write, got {m0_256}"
        # Every surviving `s_mov_b32 m0, ...` has >= 1 wait state (either `s_nop 0` or an
        # intervening instruction like `s_mov_b32 s0, ...`) before the next DMA instruction.
        for idx, ln in enumerate(asm_lines):
            if ln.startswith("s_mov_b32 m0,"):
                nxt = asm_lines[idx + 1]
                assert not nxt.startswith(("global_load_lds_", "buffer_load_")), (
                    f"back-to-back m0 -> DMA without wait state: {ln} -> {nxt}"
                )

        rng = np.random.default_rng(94206)
        raw_g = rng.integers(0, 256, size=1024, dtype=np.uint8)
        raw_b = rng.integers(0, 256, size=4200, dtype=np.uint8)
        out = np.zeros((7, 64), dtype=np.uint32)
        kern.bind(
            (1,),
            {"src_g": raw_g, "src_b": raw_b, "n_bytes": 4132, "out": out.reshape(-1)},
        ).emulate()

        # Slot 0: global u8 from raw_g[4 .. 4+64], zero-extended to u32
        np.testing.assert_array_equal(out[0], raw_g[4 : 4 + 64].astype(np.uint32))
        # Slot 1: second global i8 from raw_g[16 .. 16+64], sign-extended to i32 -> u32
        np.testing.assert_array_equal(
            out[1].view(np.int32), raw_g[16 : 16 + 64].view(np.int8).astype(np.int32)
        )
        # Slot 2: global u16 from raw_g[32 .. 32+128], zero-extended to u32
        np.testing.assert_array_equal(
            out[2], raw_g[32 : 32 + 128].view(np.uint16).astype(np.uint32)
        )
        # Slot 3: global i16 from raw_g[64 .. 64+128], sign-extended to i32
        np.testing.assert_array_equal(
            out[3].view(np.int32), raw_g[64 : 64 + 128].view(np.int16).astype(np.int32)
        )
        # Slot 4: predicated global b32 on odd lanes, prefill on even lanes
        g_u32 = raw_g[128 : 128 + 256].view(np.uint32)
        prefill = 0xCAFE0000 + np.arange(64, dtype=np.uint32)
        expected_slot4 = np.where((np.arange(64) & 1) == 1, g_u32, prefill)
        np.testing.assert_array_equal(out[4], expected_slot4)
        # Slot 5: buffer b32 at offset 16 (all in-bounds since 16 + 256 <= 4132)
        np.testing.assert_array_equal(out[5], raw_b[16 : 16 + 256].view(np.uint32))
        # Slot 6: buffer i8 at offset 4100 (n_bytes=4132) -> lanes 0..31 valid, 32..63 OOB zeroed
        inb = raw_b[4100 : 4100 + 32].view(np.int8).astype(np.int32)
        expected_slot6 = np.concatenate([inb, np.zeros(32, dtype=np.int32)])
        np.testing.assert_array_equal(out[6].view(np.int32), expected_slot6)


def test_gfx942_lds_dma_multiwave_safety_hazards_and_gemm_provenance() -> None:
    """Multi-wave DMA safety checks (static + dynamic) and DMA -> LDS -> MFMA GEMM provenance."""
    import thin
    from thin import Refused
    from thin.emu import Cluster, EmuError, Memory, gemm
    from thin.ir import MemError
    from thin.target import targeting

    with targeting("gfx942"):
        # 1. Multi-wave wave-strided DMA (`tile + k.wave * 256`) compiles and runs race-free.
        def make_multiwave_ok():
            from thin import KernelDef
            from thin.dsl import Args, O, ptr

            def body(b, k):
                V = b.v  # noqa: N806
                src, out = k.load()
                tile = b.lds().region("tile", nbytes=4 * 256, align=16)
                lane = V.u32(k.tid & 63, name="lane")
                g_off = V.u32(k.tid * 4, name="g_off")
                b.emit(O.global_load(tile + k.wave * 256, g_off, src))
                b.wait(tile, "wave DMA landed")
                b.barrier("cross-wave swap")
                partner = (k.wave ^ 1) * 256
                val = V.u32()
                b.emit(O.ds_read(val, tile + partner + lane * 4))
                b.wait(val, "read partner")
                b.emit(O.global_store(g_off, val, out))

            return KernelDef("dma_mw_ok", Args(src=ptr, out=ptr), body, wgsize=256, max_grid=(1,))

        kern_mw = thin.compile(make_multiwave_ok)
        src_mw = np.arange(256, dtype=np.uint32) + 100
        out_mw = np.zeros(256, dtype=np.uint32)
        kern_mw.bind((1,), {"src": src_mw, "out": out_mw}).emulate()
        expected_mw = src_mw.reshape(4, 64)[[1, 0, 3, 2]].reshape(256)
        np.testing.assert_array_equal(out_mw, expected_mw)

        # 2. Multi-wave overlapping DMA (`tile + 0`) is statically refused by LDS race analysis.
        def make_multiwave_race():
            from thin import KernelDef
            from thin.dsl import Args, O, ptr

            def body(b, k):
                src, _ = k.load()
                tile = b.lds().region("tile", nbytes=256, align=16)
                b.emit(O.global_load(tile, b.v.u32(k.tid * 4), src))
                b.wait(tile, "landed")
                b.barrier("done")

            return KernelDef("dma_mw_race", Args(src=ptr, out=ptr), body, wgsize=128, max_grid=(1,))

        with pytest.raises(MemError, match="may touch the same bytes"):
            thin.compile(make_multiwave_race)

        # 3. Unwaited DMA before `b.barrier()` is statically refused by LDS protocol checker.
        def make_unwaited_dma():
            from thin import KernelDef
            from thin.dsl import Args, O, ptr

            def body(b, k):
                src, out = k.load()
                tile = b.lds().region("tile", nbytes=512, align=16)
                lane = b.v.u32(k.tid & 63)
                b.emit(O.global_load(tile + k.wave * 256, b.v.u32(k.tid * 4), src))
                # Missing `b.wait(tile)` before barrier!
                b.barrier("unsafe barrier")
                val = b.v.u32()
                b.emit(O.ds_read(val, tile + ((k.wave ^ 1) * 256) + lane * 4))
                b.wait(val, "read")
                b.emit(O.global_store(b.v.u32(k.tid * 4), val, out))

            return KernelDef(
                "dma_unwaited", Args(src=ptr, out=ptr), body, wgsize=128, max_grid=(1,)
            )

        with pytest.raises(MemError):
            thin.compile(make_unwaited_dma)

        # 4. Dynamic `thin.emu` check: waiting ONLY on `lgkmcnt(0)` (not `vmcnt(0)`)
        # faults in `thin.emu` when another wave reads that LDS region after the barrier!
        bad_asm = """\
\ts_load_dwordx2 s[4:5], s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 2, v0
\tv_lshrrev_b32 v2, 6, v0
\tv_readfirstlane_b32 s6, v2
\ts_lshl_b32 m0, s6, 8
\ts_nop 0
\tglobal_load_lds_dword v1, s[4:5]
\ts_waitcnt lgkmcnt(0)
\ts_barrier
\ts_xor_b32 s7, s6, 1
\ts_lshl_b32 s7, s7, 8
\tv_and_b32 v3, 63, v0
\tv_lshlrev_b32 v3, 2, v3
\tv_add_u32 v3, s7, v3
\tds_read_b32 v4, v3
\ts_waitcnt lgkmcnt(0)
\ts_endpgm
"""
        mem_bad = Memory()
        buf_src = mem_bad.alloc("src", 512)
        ka_bad = mem_bad.alloc("ka", 8, tagged=False)
        ka_bad.write(0, np.array([buf_src.base], dtype=np.uint64).tobytes())
        with pytest.raises(EmuError, match="LDS"):
            Cluster(
                bad_asm,
                {"wgsize": 128, "vgpr": 8, "agpr": 0, "lds": 512, "user_sgprs": 2},
                mem_bad,
                [ka_bad],
                grid=(1,),
            ).run()

        # 5. End-to-end Global->LDS DMA + `ds_read_b64` + `b.mfma("f32_16x16x16_f16", ...)`
        # passes both numerical reference AND `gemm.Problem` + `GemmHooks` provenance!
        def make_dma_gemm():
            from thin import KernelDef
            from thin.dsl import Args, O, ptr

            def body(b, k):
                V = b.v  # noqa: N806
                a_ptr, b_ptr, c_ptr = k.load()
                lds = b.lds()
                lds_a = lds.region("lds_a", nbytes=512, align=16)
                lds_b = lds.region("lds_b", nbytes=512, align=16)
                lane = V.u32(k.tid, name="lane")
                # 16x16 f16 = 256 elements = 512 bytes -> two 64-lane b32 DMAs (256B each)
                b.emit(O.global_load(lds_a + 0, V.u32(lane * 4), a_ptr))
                b.emit(O.global_load(lds_a + 256, V.u32(lane * 4), a_ptr, offset=256))
                b.emit(O.global_load(lds_b + 0, V.u32(lane * 4), b_ptr))
                b.emit(O.global_load(lds_b + 256, V.u32(lane * 4), b_ptr, offset=256))
                b.wait(lds_a, lds_b, "tiles in LDS")

                idx = lane & 15
                grp = lane >> 4
                off_ab = V.u32(idx * 32 + grp * 8, name="off_ab")
                frag_a = V.u32(n=2, name="frag_a")
                frag_b = V.u32(n=2, name="frag_b")
                b.emit(O.ds_read(frag_a, lds_a + off_ab))
                b.emit(O.ds_read(frag_b, lds_b + off_ab))
                b.wait(frag_a, frag_b, "frags ready")

                acc = b.mfma("f32_16x16x16_f16", frag_a, frag_b, 0, acc="a")
                v_c = b.read_acc(acc)
                for j in range(4):
                    row = grp * 4 + j
                    c_off = V.u32((row * 16 + idx) * 4)
                    b.emit(O.global_store(c_off, v_c[j], c_ptr))

            return KernelDef(
                "dma_mfma_16x16x16",
                Args(a=ptr, b=ptr, c=ptr),
                body,
                wgsize=64,
                max_grid=(1,),
            )

        kern_gemm = thin.compile(make_dma_gemm)
        rng = np.random.default_rng(94299)
        mat_a = rng.standard_normal((16, 16)).astype(np.float16)
        mat_b_kn = rng.standard_normal((16, 16)).astype(np.float16)
        mat_b_col = np.ascontiguousarray(mat_b_kn.T)
        mat_c = np.zeros((16, 16), dtype=np.float32)

        from thin.runtime.emulated import BankConflicts

        with pytest.warns(BankConflicts, match="gfx942 bank model"):
            kern_gemm.bind(
                (1,),
                {"a": mat_a.reshape(-1), "b": mat_b_col.reshape(-1), "c": mat_c.reshape(-1)},
            ).emulate()
        ref = mat_a.astype(np.float32) @ mat_b_kn.astype(np.float32)
        np.testing.assert_allclose(mat_c, ref, rtol=1e-3, atol=1e-3)

        # Also check GEMM byte-tag provenance across Global->LDS DMA -> ds_read_b64 -> MFMA!
        mem_g = Memory()
        buf_a = mem_g.alloc("A", 512)
        buf_b = mem_g.alloc("B", 512)
        buf_d = mem_g.alloc("D", 1024)
        ka_g = mem_g.alloc("ka", 24, tagged=False)
        ka_g.write(0, np.array([buf_a.base, buf_b.base, buf_d.base], dtype=np.uint64).tobytes())
        prob = gemm.Problem(
            {
                "A": gemm.rowmajor("A", 32, elem_bytes=2),
                "B": gemm.rowmajor("B", 32, elem_bytes=2),
                "D": gemm.rowmajor_d(64, elem_bytes=4),
            },
            K=16,
            d_rows=np.arange(16),
            d_cols=np.arange(16),
            fail_fast=False,
        )
        hooks = gemm.GemmHooks(prob, mem_g)
        Cluster(
            kern_gemm.asm,
            {
                "wgsize": 64,
                "vgpr": kern_gemm.manifest.vgpr,
                "agpr": kern_gemm.manifest.agpr,
                "lds": kern_gemm.manifest.lds,
                "user_sgprs": 2,
            },
            mem_g,
            [ka_g],
            grid=(1,),
            hooks=hooks,
        ).run()
        assert prob.errors == []
        assert len(prob.stores) == 256

    # 6. Target without VMEM->LDS DMA (`gfx1201`) refuses `O.global_load(tile, ...)` cleanly.
    with targeting("gfx1201"):

        def make_rdna4_dma():
            from thin import KernelDef
            from thin.dsl import Args, O, ptr

            def body(b, k):
                (src,) = k.load()
                tile = b.lds().region("tile", nbytes=128, align=16)
                b.emit(O.global_load(tile, b.v.u32(k.tid * 4), src))

            return KernelDef("no_dma", Args(src=ptr), body, wgsize=32, max_grid=(1,))

        with pytest.raises((Refused, ValueError), match="has no global-to-LDS DMA"):
            thin.compile(make_rdna4_dma)


def test_gfx942_lds_bank_conflict_model_and_ds_2addr() -> None:
    """CDNA3 32-bank LDS model tracks b32/b64/b96/b128, asymmetric b128 read/write, and 2addr."""
    import warnings

    import thin
    from thin.emu import Cluster, Memory
    from thin.runtime.emulated import BankConflicts
    from thin.target import targeting

    def run_ds_snippet(asm_body: str, lds_bytes: int = 16384) -> dict[int, tuple[int, int, int]]:
        mem = Memory()
        ka = mem.alloc("ka", 8, tagged=False)
        c = Cluster(
            asm_body + "\n\ts_endpgm\n",
            {"wgsize": 64, "vgpr": 32, "agpr": 0, "lds": lds_bytes, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        )
        machines = c.run()
        return machines[0].banks

    with targeting("gfx942"):
        # 1. b32: linear stride 4 (0 conflicts), broadcast 0 (0 conflicts),
        #    stride 128 (32-way conflict: 64 cycles, 62 conflicts), padded stride 132 (0 conflicts).
        banks_b32 = run_ds_snippet(
            """\
\tv_lshlrev_b32 v1, 2, v0
\tds_write_b32 v1, v0
\ts_waitcnt lgkmcnt(0)
\tds_read_b32 v2, v1
\tv_mov_b32 v3, 0
\tds_read_b32 v4, v3
\tv_lshlrev_b32 v5, 7, v0
\tds_write_b32 v5, v0
\ts_waitcnt lgkmcnt(0)
\tds_read_b32 v6, v5
\tv_mul_lo_u32 v7, v0, 132
\tds_write_b32 v7, v0
\ts_waitcnt lgkmcnt(0)
\tds_read_b32 v8, v7"""
        )
        vals_b32 = list(banks_b32.values())
        assert vals_b32 == [
            [1, 2, 0],  # linear4 write
            [1, 2, 0],  # linear4 read
            [1, 2, 0],  # broadcast read
            [1, 64, 62],  # stride128 write (32-way per half-wave)
            [1, 64, 62],  # stride128 read
            [1, 2, 0],  # padded stride132 write
            [1, 2, 0],  # padded stride132 read
        ]

        # 2. b96 (12B/lane): linear 12*lane -> (8, 0); stride 128 -> (64, 56).
        banks_b96 = run_ds_snippet(
            """\
\tv_mul_lo_u32 v1, v0, 12
\tds_write_b96 v1, v[8:10]
\ts_waitcnt lgkmcnt(0)
\tds_read_b96 v[12:14], v1
\tv_lshlrev_b32 v2, 7, v0
\tds_write_b96 v2, v[8:10]"""
        )
        assert list(banks_b96.values()) == [[1, 8, 0], [1, 8, 0], [1, 64, 56]]

        # 3. Asymmetric `ds_write_b128` vs `ds_read_b128` on MFMA 16x16 layout:
        #    addr[lane] = (lane & 15) * 32 + (lane >> 4) * 16:
        #    - `ds_write_b128` (sequential 0..7 phases) has a 2-way conflict (16, 8).
        #    - `ds_read_b128` (CDNA3 interleaved 0:3+20:23) is conflict-free (8, 0)!
        banks_asym = run_ds_snippet(
            """\
\tv_and_b32 v1, 15, v0
\tv_lshrrev_b32 v2, 4, v0
\tv_lshlrev_b32 v1, 5, v1
\tv_lshlrev_b32 v2, 4, v2
\tv_add_u32 v3, v1, v2
\tds_write_b128 v3, v[8:11]
\ts_waitcnt lgkmcnt(0)
\tds_read_b128 v[12:15], v3"""
        )
        assert list(banks_asym.values()) == [
            [1, 16, 8],  # ds_write_b128: 2-way conflict
            [1, 8, 0],  # ds_read_b128: conflict-free on CDNA3!
        ]

        # 4. Unswizzled vs XOR-swizzled [16, 64] fp16 (row stride 128B) with `ds_read_b128` in DSL:
        def make_b128_tile(swizzled: bool):
            from thin import KernelDef
            from thin.dsl import Args, O, Xor, ptr

            def body(b, k):
                V = b.v  # noqa: N806
                src, out = k.load()
                sw = Xor(by=0, into=1, mask=7) if swizzled else None
                tile = b.lds().region("tile", (16, 8, 4), elem="u32", swizzle=sw)
                lane = V.u32(k.tid, name="lane")
                # Linear write: 8 chunks of 16B per row -> 64 lanes write rows 0..7, then rows 8..15
                r0 = lane >> 3
                c0 = lane & 7
                v_in = V.u32(n=4)
                b.emit(O.global_load(v_in, V.u32(lane * 16), src))
                b.wait(v_in, "in0")
                b.emit(O.ds_write(tile.at(r0, c0, 0), v_in))
                b.emit(O.ds_write(tile.at(r0 + 8, c0, 0), v_in))
                b.wait(tile, "tile written")

                # MFMA-style read: row = lane & 15, col_chunk = lane >> 4 (0..3)
                row = lane & 15
                col = lane >> 4
                frag = V.u32(n=4)
                b.emit(O.ds_read(frag, tile.at(row, col, 0)))
                b.wait(frag, "frag read")
                b.emit(O.global_store(V.u32(lane * 16), frag, out))

            return KernelDef(
                f"b128_sw_{int(swizzled)}",
                Args(src=ptr, out=ptr),
                body,
                wgsize=64,
                max_grid=(1,),
            )

        k_unsw = thin.compile(make_b128_tile, swizzled=False)
        k_sw = thin.compile(make_b128_tile, swizzled=True)
        src_t = np.arange(256, dtype=np.uint32)
        out_unsw = np.zeros(256, dtype=np.uint32)
        out_sw = np.zeros(256, dtype=np.uint32)

        with pytest.warns(BankConflicts, match="gfx942 bank model"):
            k_unsw.bind((1,), {"src": src_t, "out": out_unsw}).emulate()

        with warnings.catch_warnings():
            warnings.simplefilter("error", BankConflicts)
            k_sw.bind((1,), {"src": src_t, "out": out_sw}).emulate()

        # Both produce the exact same logical MFMA fragment values!
        np.testing.assert_array_equal(out_sw, out_unsw)

        # 5. `ds_write2_b32`/`ds_read2_b32` and `ds_write2st64_b64`/`ds_read2st64_b64`:
        mem2 = Memory()
        out2 = mem2.alloc("out", 64 * 16)
        ka2 = mem2.alloc("ka", 8, tagged=False)
        ka2.write(0, np.array([out2.base], dtype=np.uint64).tobytes())
        asm_2addr = """\
\ts_load_dwordx2 s[4:5], s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 2, v0
\tv_add_u32 v2, 100, v0
\tv_add_u32 v3, 200, v0
\tds_write2_b32 v1, v2, v3 offset0:0 offset1:64
\ts_waitcnt lgkmcnt(0)
\tds_read2_b32 v[4:5], v1 offset0:64 offset1:0
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v6, 3, v0
\tv_mov_b32 v8, v0
\tv_mov_b32 v9, 0x11110000
\tv_mov_b32 v10, v0
\tv_mov_b32 v11, 0x22220000
\tds_write2st64_b64 v6, v[8:9], v[10:11] offset0:1 offset1:2
\ts_waitcnt lgkmcnt(0)
\tds_read2st64_b64 v[12:15], v6 offset0:2 offset1:1
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v7, 4, v0
\tglobal_store_dwordx2 v7, v[4:5], s[4:5] offset:0
\tglobal_store_dwordx2 v7, v[12:13], s[4:5] offset:8
\ts_endpgm
"""
        m2 = Cluster(
            asm_2addr,
            {"wgsize": 64, "vgpr": 24, "agpr": 0, "lds": 4096, "user_sgprs": 2},
            mem2,
            [ka2],
            grid=(1,),
        ).run()[0]
        got2 = np.frombuffer(bytes(out2.data), dtype=np.uint32).reshape(64, 4)
        lanes = np.arange(64, dtype=np.uint32)
        np.testing.assert_array_equal(got2[:, 0], 200 + lanes)
        np.testing.assert_array_equal(got2[:, 1], 100 + lanes)
        np.testing.assert_array_equal(got2[:, 2], lanes)
        np.testing.assert_array_equal(got2[:, 3], np.full(64, 0x22220000, dtype=np.uint32))
        # All 4 2addr instructions executed 2 sub-accesses each with 0 conflicts!
        assert all(v[2] == 0 for v in m2.banks.values())


def test_gfx942_dpp_controls_and_wave64_collectives():
    """GFX9 DPP controls (`row_bcast:15/31`, `wave_*`, `row_newbcast`) + DSL `across`/`scan`."""
    import numpy as np

    import thin
    from thin.emu import Cluster, Memory
    from thin.target import targeting

    lanes = np.arange(64, dtype=np.uint32)
    x_f32 = (lanes.astype(np.float32) + 1.0).copy()

    def run_dpp_asm(body_asm: str, *, dtype=np.float32) -> np.ndarray:
        mem = Memory()
        src_b = mem.alloc("src", 64 * 4)
        src_b.write(0, x_f32.tobytes() if dtype == np.float32 else (lanes + 10).tobytes())
        out_b = mem.alloc("out", 64 * 4, tagged=False)
        ka = mem.alloc("ka", 16, tagged=False)
        ka.write(0, np.array([src_b.base, out_b.base], dtype=np.uint64).tobytes())
        asm = f"""\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 2, v0
\tglobal_load_dword v2, v1, s[4:5]
\ts_waitcnt vmcnt(0)
{body_asm}
\tglobal_store_dword v1, v3, s[6:7]
\ts_endpgm
"""
        with targeting("gfx942"):
            Cluster(
                asm,
                {"wgsize": 64, "vgpr": 16, "agpr": 0, "user_sgprs": 2},
                mem,
                [ka],
                grid=(1,),
            ).run()
        return np.frombuffer(bytes(out_b.data), dtype=dtype).copy()

    # 1. Classic CDNA3 64-lane inclusive scan via DPP `row_shr` + `row_bcast:15` + `row_bcast:31`:
    got_scan = run_dpp_asm("""\
\tv_mov_b32 v3, v2
\tv_add_f32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
\tv_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
\tv_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
\tv_add_f32_dpp v3, v3, v3 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
\tv_add_f32_dpp v3, v3, v3 row_bcast:15 row_mask:0xa bank_mask:0xf
\tv_add_f32_dpp v3, v3, v3 row_bcast:31 row_mask:0xc bank_mask:0xf""")
    np.testing.assert_allclose(got_scan, np.cumsum(x_f32), rtol=0, atol=0)

    # 2. Exact AMD ISA OOB semantics on `row_bcast:15` and `row_bcast:31` with `row_mask:0xf`:
    #    - `row_bcast:15`: Row 0 OOB, Row 1 -> lane 15, Row 2 -> lane 31, Row 3 -> lane 47!
    got_bc15_keep = run_dpp_asm("""\
\tv_mov_b32 v3, -99.0
\tv_mov_b32_dpp v3, v2 row_bcast:15 row_mask:0xf bank_mask:0xf""")
    want_bc15_keep = np.concatenate(
        [
            np.full(16, -99.0, dtype=np.float32),
            np.full(16, x_f32[15], dtype=np.float32),
            np.full(16, x_f32[31], dtype=np.float32),
            np.full(16, x_f32[47], dtype=np.float32),
        ]
    )
    np.testing.assert_array_equal(got_bc15_keep, want_bc15_keep)

    got_bc15_zero = run_dpp_asm("""\
\tv_mov_b32 v3, -99.0
\tv_mov_b32_dpp v3, v2 row_bcast:15 row_mask:0xf bank_mask:0xf bound_ctrl:1""")
    want_bc15_zero = want_bc15_keep.copy()
    want_bc15_zero[:16] = 0.0
    np.testing.assert_array_equal(got_bc15_zero, want_bc15_zero)

    #    - `row_bcast:31`: Rows 0 & 1 OOB, Rows 2 & 3 -> lane 31!
    got_bc31_keep = run_dpp_asm("""\
\tv_mov_b32 v3, -77.0
\tv_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf""")
    want_bc31_keep = np.concatenate(
        [np.full(32, -77.0, dtype=np.float32), np.full(32, x_f32[31], dtype=np.float32)]
    )
    np.testing.assert_array_equal(got_bc31_keep, want_bc31_keep)

    got_bc31_zero = run_dpp_asm("""\
\tv_mov_b32 v3, -77.0
\tv_mov_b32_dpp v3, v2 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1""")
    want_bc31_zero = want_bc31_keep.copy()
    want_bc31_zero[:32] = 0.0
    np.testing.assert_array_equal(got_bc31_zero, want_bc31_zero)

    # 3. `wave_shl:1`, `wave_shr:1`, `wave_rol:1`, `wave_ror:1`, `row_newbcast:7`:
    u_in = lanes + 10
    got_wshl = run_dpp_asm(
        "\tv_mov_b32 v3, 777\n\tv_mov_b32_dpp v3, v2 wave_shl:1 row_mask:0xf bank_mask:0xf",
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(got_wshl, np.append(u_in[1:], np.uint32(777)))

    got_wshl_bc = run_dpp_asm(
        "\tv_mov_b32 v3, 777\n"
        "\tv_mov_b32_dpp v3, v2 wave_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(got_wshl_bc, np.append(u_in[1:], np.uint32(0)))

    got_wshr = run_dpp_asm(
        "\tv_mov_b32 v3, 888\n\tv_mov_b32_dpp v3, v2 wave_shr:1 row_mask:0xf bank_mask:0xf",
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(got_wshr, np.insert(u_in[:-1], 0, np.uint32(888)))

    got_wrol = run_dpp_asm(
        "\tv_mov_b32_dpp v3, v2 wave_rol:1 row_mask:0xf bank_mask:0xf",
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(got_wrol, np.roll(u_in, -1))

    got_wror = run_dpp_asm(
        "\tv_mov_b32_dpp v3, v2 wave_ror:1 row_mask:0xf bank_mask:0xf",
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(got_wror, np.roll(u_in, 1))

    got_newbcast = run_dpp_asm(
        "\tv_add_u32_dpp v3, v2, v2 row_newbcast:7 row_mask:0xf bank_mask:0xf",
        dtype=np.uint32,
    )
    want_newbcast = u_in + u_in[(lanes // 16) * 16 + 7]
    np.testing.assert_array_equal(got_newbcast, want_newbcast)

    got_fmac = run_dpp_asm("""\
\tv_mov_b32 v3, 10.0
\tv_fmac_f32_dpp v3, -v2, v2 row_mirror row_mask:0xf bank_mask:0xf""")
    mirror_idx = (lanes // 16) * 16 + (15 - (lanes % 16))
    np.testing.assert_allclose(got_fmac, 10.0 + (-x_f32[mirror_idx]) * x_f32, rtol=0, atol=0)

    # 4. High-level DSL `across(..., 64)` and `scan(..., 64)` on gfx942:
    def make_wave64_collectives():
        from thin import KernelDef
        from thin.dsl import Args, O, across, ptr, scan

        def body(b, k):
            V = b.v  # noqa: N806
            src, out_red, out_scan = k.load()
            off = V.u32(k.tid * 4)
            x = V.f32()
            b.emit(O.global_load(x, off, src))
            b.wait(x, "x")
            r = V.f32(across(x, "add", 64))
            s = V.f32(scan(x, "add", 64))
            b.emit(O.global_store(off, r, out_red))
            b.emit(O.global_store(off, s, out_scan))

        return KernelDef(
            "wave64_collectives",
            Args(src=ptr, out_red=ptr, out_scan=ptr),
            body,
            wgsize=64,
            max_grid=(1,),
        )

    with targeting("gfx942"):
        ck = thin.compile(make_wave64_collectives)
        out_red = np.zeros(64, dtype=np.float32)
        out_scan = np.zeros(64, dtype=np.float32)
        ck.bind((1,), {"src": x_f32, "out_red": out_red, "out_scan": out_scan}).emulate()
        want_red = np.full(64, x_f32.sum(), dtype=np.float32)
        np.testing.assert_allclose(out_red, want_red, rtol=0, atol=0)
        np.testing.assert_allclose(out_scan, np.cumsum(x_f32), rtol=0, atol=0)


def test_gfx942_packed_f32_ops_and_provenance():
    """CDNA3 `v_pk_fma_f32`, `v_pk_mul_f32`, `v_pk_add_f32`, and `v_pk_mov_b32` in `thin.emu`."""
    import numpy as np

    from thin.emu import Cluster, Memory, gemm
    from thin.target import targeting

    rng = np.random.default_rng(94208)
    s0 = rng.uniform(-2.0, 2.0, size=(64, 2)).astype(np.float32)
    s1 = rng.uniform(-2.0, 2.0, size=(64, 2)).astype(np.float32)
    s2 = rng.uniform(-1.0, 1.0, size=(64, 2)).astype(np.float32)
    # Inject -0.0 and NaN in specific lanes to verify DX10 clamp edge cases:
    s0[0, 0] = np.float32(-0.0)
    s1[0, 0] = np.float32(0.0)
    s2[0, 0] = np.float32(-0.0)
    s0[1, 1] = np.float32("nan")

    pk_fma_mod = "op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,0] neg_hi:[0,1,1] clamp"
    asm = f"""\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_load_dwordx4 s[8:11], s[0:1], 0x10
\ts_waitcnt lgkmcnt(0)
\tv_lshlrev_b32 v1, 3, v0
\tglobal_load_dwordx2 v[2:3], v1, s[4:5]
\tglobal_load_dwordx2 v[4:5], v1, s[6:7]
\tglobal_load_dwordx2 v[6:7], v1, s[8:9]
\ts_waitcnt vmcnt(0)
\t// 0) default v_pk_fma_f32 (op_sel:[0,0,0] op_sel_hi:[1,1,1])
\tv_pk_fma_f32 v[8:9], v[2:3], v[4:5], v[6:7]
\t// 1) mixed selectors + neg_lo + neg_hi + clamp
\tv_pk_fma_f32 v[10:11], v[2:3], v[4:5], v[6:7] {pk_fma_mod}
\t// 2) v_pk_mul_f32 with broadcast s0[0] via op_sel_hi:[0,1] and neg_hi:[0,1]
\tv_pk_mul_f32 v[12:13], v[2:3], v[4:5] op_sel:[0,1] op_sel_hi:[0,1] neg_hi:[0,1]
\t// 3) v_pk_add_f32 with inline float 1.0 in src1 (`op_sel_hi:[1,0]` -> hi!) + clamp
\tv_pk_add_f32 v[14:15], v[2:3], 1.0 op_sel_hi:[1,0] neg_lo:[1,0] clamp
\t// 4) v_pk_mov_b32: copy (`op_sel:[0,1]`) then in-place swap (`op_sel:[1,0]`)
\tv_pk_mov_b32 v[16:17], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[0,0]
\tv_pk_mov_b32 v[16:17], v[16:17], v[16:17] op_sel:[1,0]
\t// 5) v_pk_mov_b32 cross-merge: D.lo = S0[0] = v2, D.hi = S1[1] = v5
\tv_pk_mov_b32 v[18:19], v[2:3], v[4:5] op_sel:[0,1]
\tv_lshlrev_b32 v20, 6, v0
\tglobal_store_dwordx2 v20, v[8:9], s[10:11] offset:0
\tglobal_store_dwordx2 v20, v[10:11], s[10:11] offset:8
\tglobal_store_dwordx2 v20, v[12:13], s[10:11] offset:16
\tglobal_store_dwordx2 v20, v[14:15], s[10:11] offset:24
\tglobal_store_dwordx2 v20, v[16:17], s[10:11] offset:32
\tglobal_store_dwordx2 v20, v[18:19], s[10:11] offset:40
\ts_endpgm
"""
    mem = Memory()
    b0, b1, b2 = mem.alloc("s0", s0.nbytes), mem.alloc("s1", s1.nbytes), mem.alloc("s2", s2.nbytes)
    b0.write(0, s0.tobytes())
    b1.write(0, s1.tobytes())
    b2.write(0, s2.tobytes())
    out = mem.alloc("out", 64 * 64, tagged=False)
    ka = mem.alloc("ka", 32, tagged=False)
    ka.write(0, np.array([b0.base, b1.base, b2.base, out.base], dtype=np.uint64).tobytes())

    with targeting("gfx942"):
        Cluster(
            asm,
            {"wgsize": 64, "vgpr": 24, "agpr": 0, "user_sgprs": 2},
            mem,
            [ka],
            grid=(1,),
        ).run()

    got = np.frombuffer(bytes(out.data), dtype=np.float32).reshape(64, 8, 2)

    def ref_clamp(x: np.ndarray) -> np.ndarray:
        y = np.where(np.isnan(x) | (x <= 0.0), np.float32(0.0), np.minimum(x, np.float32(1.0)))
        # Ensure +0.0 sign bit == 0 (DX10 clamp turns -0.0 and NaN into +0.0):
        bits = y.view(np.uint32).copy()
        bits[y == 0.0] = 0
        return bits.view(np.float32)

    def ref_fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(
            np.float32
        )

    # 0) default v_pk_fma_f32 (check finite lanes + NaN lane 1 hi)
    want0 = ref_fma(s0, s1, s2)
    assert np.isnan(got[1, 0, 1])
    mask_fin = ~np.isnan(want0)
    np.testing.assert_allclose(got[:, 0, :][mask_fin], want0[mask_fin], rtol=0, atol=1e-7)

    # 1) mixed selectors + neg + clamp
    lo1 = ref_clamp(ref_fma(-s0[:, 1], s1[:, 0], s2[:, 1]))
    hi1 = ref_clamp(ref_fma(s0[:, 0], -s1[:, 1], -s2[:, 0]))
    np.testing.assert_allclose(got[:, 1, 0], lo1, rtol=0, atol=1e-7)
    np.testing.assert_allclose(got[:, 1, 1], hi1, rtol=0, atol=1e-7)

    # 2) v_pk_mul_f32
    lo2 = s0[:, 0] * s1[:, 1]
    hi2 = s0[:, 0] * (-s1[:, 1])
    np.testing.assert_array_equal(got[:, 2, 0].view(np.uint32), lo2.view(np.uint32))
    np.testing.assert_array_equal(got[:, 2, 1].view(np.uint32), hi2.view(np.uint32))

    # 3) v_pk_add_f32 with inline 1.0 + clamp (-(-0.0)+1.0=1.0; NaN+1.0 -> +0.0!)
    lo3 = ref_clamp((-s0[:, 0]) + np.float32(1.0))
    hi3 = ref_clamp(s0[:, 1] + np.float32(1.0))
    np.testing.assert_array_equal(got[:, 3, 0].view(np.uint32), lo3.view(np.uint32))
    np.testing.assert_array_equal(got[:, 3, 1].view(np.uint32), hi3.view(np.uint32))
    assert got[1, 3, 1].view(np.uint32) == 0

    # 4) v_pk_mov_b32 in-place swap: <s0[1], s0[0]> bitwise exact (including NaN bits!)
    np.testing.assert_array_equal(got[:, 4, 0].view(np.uint32), s0[:, 1].view(np.uint32))
    np.testing.assert_array_equal(got[:, 4, 1].view(np.uint32), s0[:, 0].view(np.uint32))

    # 5) v_pk_mov_b32 cross-merge: <s0[0], s1[1]>
    np.testing.assert_array_equal(got[:, 5, 0].view(np.uint32), s0[:, 0].view(np.uint32))
    np.testing.assert_array_equal(got[:, 5, 1].view(np.uint32), s1[:, 1].view(np.uint32))

    # 6) Provenance tags flow through `v_pk_mov_b32` + `v_pk_mul_f32` / `v_pk_fma_f32`:
    asm_epilogue = """\
\ts_load_dwordx4 s[4:7], s[0:1], 0x0
\ts_load_dwordx2 s[8:9], s[0:1], 0x10
\ts_waitcnt lgkmcnt(0)
\tv_and_b32 v1, 15, v0
\tv_lshrrev_b32 v2, 4, v0
\tv_lshlrev_b32 v3, 5, v1
\tv_lshl_add_u32 v3, v2, 3, v3
\tglobal_load_dwordx2 v[16:17], v3, s[4:5]
\tglobal_load_dwordx2 v[20:21], v3, s[6:7]
\ts_waitcnt vmcnt(0)
\tv_mfma_f32_16x16x16_f16 a[0:3], v[16:17], v[20:21], 0
\tv_accvgpr_read_b32 v24, a0
\tv_accvgpr_read_b32 v25, a1
\tv_accvgpr_read_b32 v26, a2
\tv_accvgpr_read_b32 v27, a3
\tv_pk_mov_b32 v[28:29], v[24:25], v[24:25] op_sel:[0,1]
\tv_pk_mul_f32 v[28:29], v[28:29], 2.0 op_sel_hi:[1,0]
\tv_pk_mov_b32 v[30:31], v[26:27], v[26:27] op_sel:[0,1]
\tv_pk_fma_f32 v[30:31], v[30:31], 2.0, 0 op_sel_hi:[1,0,0]
\tv_lshlrev_b32 v4, 8, v2
\tv_lshl_add_u32 v4, v1, 2, v4
\tglobal_store_dword v4, v28, s[8:9] offset:0
\tglobal_store_dword v4, v29, s[8:9] offset:64
\tglobal_store_dword v4, v30, s[8:9] offset:128
\tglobal_store_dword v4, v31, s[8:9] offset:192
\ts_endpgm
"""
    mem_ep = Memory()
    ba = mem_ep.alloc("A", 16 * 32)
    bb = mem_ep.alloc("B", 16 * 32)
    bd = mem_ep.alloc("D", 16 * 64)
    ka_ep = mem_ep.alloc("ka", 24, tagged=False)
    ka_ep.write(0, np.array([ba.base, bb.base, bd.base], dtype=np.uint64).tobytes())
    prob = gemm.Problem(
        {
            "A": gemm.rowmajor("A", 32, elem_bytes=2),
            "B": gemm.rowmajor("B", 32, elem_bytes=2),
            "D": gemm.rowmajor_d(64, elem_bytes=4),
        },
        K=16,
        d_rows=np.arange(16),
        d_cols=np.arange(16),
        fail_fast=False,
    )
    with targeting("gfx942"):
        Cluster(
            asm_epilogue,
            {"wgsize": 64, "vgpr": 40, "agpr": 8, "user_sgprs": 2},
            mem_ep,
            [ka_ep],
            grid=(1,),
            hooks=gemm.GemmHooks(prob, mem_ep),
        ).run()
    assert prob.errors == []
    assert len(prob.stores) == 256


def test_gfx942_cdna3_scalar_exec_partial_and_setreg_hazards() -> None:
    """CDNA3 Table 11 / GCNHazardRecognizer wait states on VCC, RWLane, EXEC, PartialDst, setreg."""
    import thin
    from thin.target import targeting

    def body_ins(asm: str) -> list[str]:
        return [
            ln.strip().split("//")[0].split(";")[0].strip()
            for ln in asm.splitlines()
            if ln.startswith("\t") and not ln.strip().startswith((".", ";", "//"))
        ]

    before = thin.use_store(None)
    try:
        with targeting("gfx942"):

            def make_hazards():
                from thin import KernelDef
                from thin.dsl import Args, O, ptr
                from thin.target import PHYS

                vcc = PHYS["vcc"]
                exec_reg = PHYS["exec"]
                exec_lo = PHYS["exec_lo"]

                def body(b, k):
                    V, S = b.v, b.s  # noqa: N806
                    p_in, p_out = k.load()
                    off = V.u32(k.tid * 4, name="off")
                    x = V.u32()
                    y = V.u32()
                    b.emit(O.global_load(x, off, p_in))
                    b.emit(O.global_load(y, off, p_in, offset=256))
                    b.wait(x, y, comment="inputs")

                    # 1. v_add_co_u32 (writes VGPR + VCC) -> v_cndmask_b32: 2 states (`s_nop 1`)
                    sum_v = V.u32()
                    sel_v = V.u32()
                    b.emit(O.op("v_add_co_u32", sum_v, vcc, x, y))
                    b.emit(O.op("v_cndmask_b32", sel_v, x, sum_v, vcc))

                    # 2. v_add_co_u32 (VCC) -> v_div_fmas_f32 (VCC): 4 states (`s_nop 3`)
                    fx = V.f32()
                    fy = V.f32()
                    b.emit(O.op("v_cvt_f32_u32", fx, x))
                    b.emit(O.op("v_cvt_f32_u32", fy, y))
                    fmas_v = V.f32()
                    b.emit(O.op("v_add_co_u32", sum_v, vcc, x, y))
                    b.emit(O.op("v_div_fmas_f32", fmas_v, fx, fy, fx))

                    # 3. VALU SGPR -> v_readlane / v_writelane lane-select: 4 states (`s_nop 3`)
                    lsel = S.u32()
                    got = S.u32()
                    b.emit(O.op("v_readfirstlane_b32", lsel, x))
                    b.emit(O.op("v_readlane_b32", got, y, lsel))
                    b.emit(O.op("v_readfirstlane_b32", lsel, y))
                    b.emit(O.op("v_writelane_b32", sel_v, 7, lsel))

                    # 4. v_cmpx -> exec_lo (`s_nop 1`), readfirstlane (`s_nop 3`), DPP (`s_nop 4`)
                    mb = V.u32(0)
                    s_after = S.u32()
                    dpp_v = V.u32(0)
                    saved = S.u64()
                    with b.raw_exec("cmpx hazard test"):
                        b.emit(O.op("s_mov_b64", saved, exec_reg))
                        b.emit(O.op("v_cmpx_ne_u32", vcc, x, -1))
                        b.emit(O.op("v_mbcnt_lo_u32_b32", mb, exec_lo, 0))
                        b.emit(O.op("v_cmpx_ne_u32", vcc, y, -1))
                        b.emit(O.op("v_readfirstlane_b32", s_after, mb))
                        b.emit(O.op("v_cmpx_ne_u32", vcc, x, -2))
                        b.emit(
                            O.op(
                                "v_mov_b32_dpp",
                                dpp_v,
                                mb,
                                dpp="row_shr:1",
                                bound_ctrl=True,
                            )
                        )
                        b.emit(O.op("s_mov_b64", exec_reg, saved))

                    # 5. PartialDst (`v_cvt_pk_fp8_f32 op_sel:[0,0,1]`) -> next: `s_nop 0`
                    pk = V.u32(0)
                    b.emit(O.op("v_cvt_pk_fp8_f32", pk, fx, fy, op_sel=[0, 0, 1]))
                    b.emit(O.op("v_cvt_pk_fp8_f32", pk, fy, fx))

                    # 6. s_setreg_b32 -> s_getreg_b32 on MODE (0x0801): 2 states (`s_nop 1`)
                    mode_s = S.u32()
                    b.emit(O.op("s_getreg_b32", mode_s, 0x0801))
                    b.emit(O.op("s_setreg_b32", 0x0801, mode_s))
                    b.emit(O.op("s_getreg_b32", mode_s, 0x0801))

                    out_v = V.u32(sel_v + pk + dpp_v + V.u32(got) + V.u32(s_after) + V.u32(mode_s))
                    b.emit(O.global_store(off, out_v, p_out))
                    b.emit(O.global_store(off, fmas_v, p_out, offset=256))

                return KernelDef(
                    "cdna3_misc_hazards",
                    Args(p_in=ptr, p_out=ptr),
                    body,
                    wgsize=64,
                    max_grid=(1,),
                )

            kern = thin.compile(make_hazards)
            ins = body_ins(kern.asm)

            def between(prefix_a: str, prefix_b: str, start: int = 0) -> tuple[int, list[str]]:
                ia = next(i for i in range(start, len(ins)) if ins[i].startswith(prefix_a))
                ib = next(i for i in range(ia + 1, len(ins)) if ins[i].startswith(prefix_b))
                return ib, ins[ia + 1 : ib]

            pos, mid = between("v_add_co_u32", "v_cndmask_b32", 0)
            assert mid == ["s_nop 1"]

            pos, mid = between("v_add_co_u32", "v_div_fmas_f32", pos)
            assert mid == ["s_nop 3"]

            pos, mid = between("v_readfirstlane_b32", "v_readlane_b32", pos)
            assert mid == ["s_nop 3"]

            pos, mid = between("v_readfirstlane_b32", "v_writelane_b32", pos)
            assert mid == ["s_nop 3"]

            pos, mid = between("v_cmpx_", "v_mbcnt_lo_u32_b32", pos)
            assert mid == ["s_nop 1"]

            pos, mid = between("v_cmpx_", "v_readfirstlane_b32", pos)
            assert mid == ["s_nop 3"]

            pos, mid = between("v_cmpx_", "v_mov_b32_dpp", pos)
            assert mid == ["s_nop 4"]

            pos, mid = between("v_cvt_pk_fp8_f32", "v_cvt_pk_fp8_f32", pos)
            assert mid == ["s_nop 0"]

            _, mid = between("s_setreg_b32", "s_getreg_b32", pos)
            assert mid == ["s_nop 1"]
    finally:
        thin.use_store(before)


def test_gfx942_dsl_mfma_validation_and_bounds() -> None:
    """DSL rejects C/D file mismatch and bad cbsz, and accepts valid sparse cbsz+abid."""
    import pytest

    from thin.dsl import Builder, O
    from thin.target import targeting

    with targeting("gfx942"):
        b = Builder()
        va = b.v.u32(n=2)
        vb = b.v.u32(n=2)
        vc = b.v.f32(n=4)
        ac = b.a.f32(n=4)

        # Mixing AGPR dst and VGPR src2 (or vice versa) is rejected immediately:
        with pytest.raises(ValueError, match="same register file"):
            b.mfma("f32_16x16x16_f16", va, vb, vc, dst=ac)
        with pytest.raises(ValueError, match="same register file"):
            b.emit(O.op("v_mfma_f32_16x16x16_f16", ac, va, vb, vc))

        # 2-block `32x32x4_2b_f16` allows cbsz=1 but rejects cbsz=2:
        acc32 = b.a.f32(n=32)
        b.mfma("f32_32x32x4_2b_f16", va, vb, acc32, dst=acc32, cbsz=1, abid=1)
        with pytest.raises(ValueError, match=r"cbsz 2 out of range \(0\.\.=1\)"):
            b.mfma("f32_32x32x4_2b_f16", va, vb, acc32, dst=acc32, cbsz=2)

        # Sparse `v_smfmac_*` allows cbsz in 0..=3 + abid, and requires VGPR idx:
        vb4 = b.v.u32(n=4)
        vidx = b.v.u32()
        aidx = b.a.u32()
        b.smfmac("f32_16x16x32_f16", va, vb4, vidx, ac, cbsz=2, abid=1)
        b.smfmac("f32_16x16x64_fp8_fp8", va, vb4, vidx, ac, cbsz=3, abid=1)
        with pytest.raises(ValueError, match="sparse index takes 'v'"):
            b.smfmac("f32_16x16x32_f16", va, vb4, aidx, ac)
