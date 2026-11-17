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

    # Common FP8/BF8 MFMA forms on gfx950 use OCP e4m3fn/e5m2 (not FNUZ) and are marked provisional.
    fp8_mf = t.matrix_form("v_mfma_f32_16x16x32_fp8_bf8")
    assert fp8_mf is not None
    assert (fp8_mf.a, fp8_mf.b) == ("e4m3fn", "e5m2")
    assert fp8_mf.provisional is True

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
