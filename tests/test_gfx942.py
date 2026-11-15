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

    return KernelDef(
        "gfx942_party", Args(out=ptr, **decl.args()), body, wgsize=128, max_grid=(1,)
    )


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
