"""The ISA table generator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.isagen import bundle, canonical, isa_tables, operand_table
from tools.isagen.targets import TARGETS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads(
    (ROOT / "crates/thin-targets/tests/data/target/tables_manifest.json").read_text()
)

# XML-listed DPP/SDWA variants that LLVM omits on all GFX9 targets (gfx942 and gfx950).
_LLVM_OMITTED_GFX9_VARIANTS = frozenset(
    {
        "v_accvgpr_mov_b32_dpp",
        "v_accvgpr_mov_b32_sdwa",
        "v_mov_b64_sdwa",
        "v_nop_dpp",
    }
)

# Upstream llvm-mc --disassemble omits the _e64/_sdwa suffix on 0-operand v_nop/v_clrexcp.
_LLVM_ZERO_OP_BARE_DISASM = {
    "v_clrexcp_e64": "v_clrexcp",
    "v_nop_e64": "v_nop",
    "v_nop_sdwa": "v_nop",
}

# 64-bit / FP8 DPP16 ops where GFX9 hardware only permits row_newbcast.
_DP_DPP_OPS = frozenset(
    {
        "v_ceil_f64_dpp",
        "v_cvt_f32_f64_dpp",
        "v_cvt_f64_i32_dpp",
        "v_cvt_f64_u32_dpp",
        "v_cvt_i32_f64_dpp",
        "v_cvt_pk_f32_bf8_dpp",
        "v_cvt_pk_f32_fp8_dpp",
        "v_cvt_u32_f64_dpp",
        "v_floor_f64_dpp",
        "v_fmac_f64_dpp",
        "v_fract_f64_dpp",
        "v_frexp_exp_i32_f64_dpp",
        "v_frexp_mant_f64_dpp",
        "v_mov_b64_dpp",
        "v_trunc_f64_dpp",
    }
)


def _reg(file_ch: str, base_idx: int, dwords: int) -> str:
    if dwords <= 1:
        return f"{file_ch}{base_idx}"
    return f"{file_ch}[{base_idx}:{base_idx + dwords - 1}]"


def _isa_op_line(op: dict, short_names: set[str]) -> str | None:
    m = op["mnemonic"]
    cls = op["class"]
    if cls == "MTBUF" or (cls.endswith("_LIT") and m != "s_setreg_imm32_b32"):
        return None
    if m == "s_setreg_imm32_b32":
        return "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 1), 0x12345678"
    if m == "s_setreg_b32":
        return "s_setreg_b32 hwreg(HW_REG_MODE, 0, 1), s0"
    if m == "s_getreg_b32":
        return "s_getreg_b32 s0, hwreg(HW_REG_MODE, 0, 1)"
    if m == "s_waitcnt":
        return "s_waitcnt vmcnt(0)"
    opmap = {o["field"]: o for o in op["operands"]}
    if cls == "SOPP":
        return m if not opmap else f"{m} 0"
    if cls in ("SOP1", "SOP2", "SOPC", "SOPK"):
        ops = []
        sidx = 0
        for fn in ("sdst", "ssrc0", "ssrc1", "simm16"):
            if fn not in opmap:
                continue
            o = opmap[fn]
            dw = max(1, (64 if o["format"] == "FMT_NUM_M64" else o["bits"]) // 32)
            if o["type"] in ("OPR_SIMM16", "OPR_VERSION", "OPR_LABEL", "OPR_SIMM4"):
                ops.append("0")
            else:
                align = 4 if dw > 2 else (2 if dw == 2 else 1)
                sidx = ((sidx + align - 1) // align) * align
                ops.append(_reg("s", sidx, dw))
                sidx += dw
        return f"{m} {', '.join(ops)}"
    if cls == "SMEM":
        has_d, has_b = "sdata" in opmap, "sbase" in opmap
        has_o = "offset" in opmap or "soffset" in opmap
        if not has_d and not has_b and not has_o:
            return m
        if has_d and not has_b:
            return f"{m} {_reg('s', 0, max(1, opmap['sdata']['bits'] // 32))}"
        if not has_d and has_b:
            return f"{m} {_reg('s', 0, max(1, opmap['sbase']['bits'] // 32))}, 0x0"
        d, b = opmap["sdata"], opmap["sbase"]
        d_str = (
            "0"
            if d["type"] == "OPR_SIMM8" or d["format"] == "FMT_NUM_U8"
            else _reg("s", 4, max(1, d["bits"] // 32))
        )
        return f"{m} {d_str}, {_reg('s', 0, max(1, b['bits'] // 32))}, 0x0"
    if cls == "MUBUF":
        if not opmap:
            return m
        return f"{m} {_reg('v', 4, max(1, opmap['vdata']['bits'] // 32))}, v0, s[8:11], 0 offen"
    if cls == "FLAT":
        if not opmap:
            return m
        if "vdst" in opmap and "data" in opmap:
            vd = _reg("v", 4, max(1, opmap["vdst"]["bits"] // 32))
            dt = _reg("v", 8, max(1, opmap["data"]["bits"] // 32))
            return f"{m} {vd}, v[0:1], {dt} sc0"
        if "vdst" in opmap:
            return f"{m} {_reg('v', 4, max(1, opmap['vdst']['bits'] // 32))}, v[0:1]"
        return f"{m} v[0:1], {_reg('v', 4, max(1, opmap['data']['bits'] // 32))}"
    if cls in ("GLOBAL", "SCRATCH"):
        if not opmap:
            return m
        saddr = "s[4:5]" if cls == "GLOBAL" else "off"
        if "_load_lds_" in m:
            return f"{m} v0, {saddr}"
        if "vdst" in opmap and "data" in opmap:
            vd = _reg("v", 4, max(1, opmap["vdst"]["bits"] // 32))
            dt = _reg("v", 8, max(1, opmap["data"]["bits"] // 32))
            return f"{m} {vd}, v0, {dt}, {saddr} sc0"
        if "vdst" in opmap:
            return f"{m} {_reg('v', 4, max(1, opmap['vdst']['bits'] // 32))}, v0, {saddr}"
        return f"{m} v0, {_reg('v', 4, max(1, opmap['data']['bits'] // 32))}, {saddr}"
    if cls == "DS":
        ops = []
        vidx = 4
        for fn in ("vdst", "addr", "data0", "data1"):
            if fn not in opmap:
                continue
            if fn == "addr":
                ops.append("v0")
            else:
                dw = max(1, opmap[fn]["bits"] // 32)
                ops.append(_reg("v", vidx, dw))
                vidx += max(4, dw)
        return f"{m} {', '.join(ops)}"
    if cls in ("VOP1", "VOP1_SDST", "VOP2", "VOPC"):
        base_m = m.removesuffix("_e32")
        if not opmap:
            return m
        if cls == "VOPC":
            v0 = _reg("v", 0, max(1, opmap["src0"]["bits"] // 32))
            v1 = _reg("v", 4, max(1, opmap["vsrc1"]["bits"] // 32))
            return f"{m} {v0}, {v1}" if base_m.startswith("v_cmpx") else f"{m} vcc, {v0}, {v1}"
        if cls == "VOP2":
            vd = _reg("v", 0, max(1, opmap["vdst"]["bits"] // 32))
            s0 = _reg("v", 4, max(1, opmap["src0"]["bits"] // 32))
            s1 = _reg("v", 8, max(1, opmap["vsrc1"]["bits"] // 32))
            if base_m.startswith("v_cndmask"):
                return f"{m} {vd}, {s0}, {s1}, vcc"
            if base_m in ("v_addc_co_u32", "v_subb_co_u32", "v_subbrev_co_u32"):
                return f"{m} {vd}, vcc, {s0}, {s1}, vcc"
            if base_m in ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32"):
                return f"{m} {vd}, vcc, {s0}, {s1}"
            return f"{m} {vd}, {s0}, {s1}"
        if cls == "VOP1_SDST":
            return f"{m} s0, v0"
        if base_m == "v_accvgpr_mov_b32":
            return f"{m} a0, a1"
        vd = _reg("v", 0, max(1, opmap["vdst"]["bits"] // 32))
        s0 = _reg("v", 4, max(1, opmap["src0"]["bits"] // 32))
        return f"{m} {vd}, {s0}"
    if cls in ("VOP3", "VOP3_SDST", "VOP3SD", "VOP3P", "VOP3P_MFMA", "VOP3PX2"):
        print_m = f"{m}_e64" if (cls == "VOP3SD" and f"{m}_e32" in short_names) else m
        ops = []
        vidx, sidx = 0, 0
        for fn in ("vdst", "sdst", "src0", "src1", "src2", "scale_src0", "scale_src1"):
            if fn not in opmap:
                continue
            o = opmap[fn]
            ot = o["type"]
            if ot == "OPR_EXEC":
                continue
            dw = max(1, (64 if o["format"] == "FMT_NUM_M64" else o["bits"]) // 32)
            if m == "v_writelane_b32" and fn == "src1":
                ops.append("0")
            elif ot == "OPR_SREG_M0":
                ops.append("m0")
            elif ("ACCVGPR" in ot and "VGPR_OR" not in ot) or (
                cls in ("VOP3P_MFMA", "VOP3PX2")
                and fn in ("vdst", "src2")
                and not m.startswith("v_smfmac_")
            ):
                ops.append(_reg("a", 0, dw))
            elif ot.startswith(("OPR_SREG", "OPR_SDST", "OPR_SSRC")):
                align = 4 if dw > 2 else (2 if dw == 2 else 1)
                sidx = ((sidx + align - 1) // align) * align
                ops.append(_reg("s", sidx, dw))
                sidx += dw
            else:
                align = 2 if dw > 1 else 1
                vidx = ((vidx + align - 1) // align) * align
                ops.append(_reg("v", vidx, dw))
                vidx += dw
        return f"{print_m} {', '.join(ops)}"
    raise AssertionError(f"unhandled class {cls} on {m}")


def _operand_row_line(row: dict) -> str:
    m = row["mnemonic"]
    rtn = m.endswith("+rtn")
    op = m[:-4] if rtn else m
    enc = row["encoding"]
    cnt = row["count"]
    slots = row["slots"]
    if cnt == 0:
        return "s_waitcnt vmcnt(0)" if op == "s_waitcnt" else op
    if op == "s_setreg_imm32_b32":
        return "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 1), 0"
    if op == "s_setreg_b32":
        return "s_setreg_b32 hwreg(HW_REG_MODE, 0, 1), s0"
    if op == "s_getreg_b32":
        return "s_getreg_b32 s0, hwreg(HW_REG_MODE, 0, 1)"
    if enc == "ENC_SOPP":
        return f"{op} 0"
    if enc == "ENC_MUBUF lds":
        return f"{op} v0, s[8:11], 0 offen lds"
    if enc == "ENC_MUBUF":
        vd = _reg("v", 4, max(1, slots[0][1] // 32))
        mod = " offen sc0" if rtn else " offen"
        return f"{op} {vd}, v0, s[8:11], 0{mod}"
    if enc == "ENC_FLAT":
        if "atomic" in op:
            if cnt == 3:
                vd = _reg("v", 4, max(1, slots[0][1] // 32))
                dt = _reg("v", 8, max(1, slots[2][1] // 32))
                return f"{op} {vd}, v[0:1], {dt} sc0"
            return f"{op} v[0:1], {_reg('v', 8, max(1, slots[1][1] // 32))}"
        ops = []
        vidx = 4
        for idx, (_, sz) in enumerate(slots):
            if idx == (1 if row["roles"][0] == "w" else 0):
                ops.append("v[0:1]")
            else:
                dw = max(1, sz // 32)
                ops.append(_reg("v", vidx, dw))
                vidx += max(4, dw)
        return f"{op} {', '.join(ops)}"
    if enc in ("ENC_FLAT_GLBL", "ENC_FLAT_SCRATCH"):
        saddr = "s[4:5]" if enc == "ENC_FLAT_GLBL" else "off"
        if "_load_lds_" in op:
            return f"{op} v0, {saddr}"
        if "atomic" in op:
            if cnt == 4:
                vd = _reg("v", 4, max(1, slots[0][1] // 32))
                dt = _reg("v", 8, max(1, slots[2][1] // 32))
                return f"{op} {vd}, v0, {dt}, {saddr} sc0"
            return f"{op} v0, {_reg('v', 8, max(1, slots[1][1] // 32))}, {saddr}"
        ops = []
        vidx = 4
        has_w = row["roles"][0] == "w"
        for idx, (_, sz) in enumerate(slots):
            if idx == len(slots) - 1:
                ops.append(saddr)
            elif idx == (1 if has_w else 0):
                ops.append("v0")
            else:
                dw = max(1, sz // 32)
                ops.append(_reg("v", vidx, dw))
                vidx += max(4, dw)
        return f"{op} {', '.join(ops)}"
    if enc == "ENC_SMEM":
        ops = []
        sidx = 4
        for idx, (st, sz) in enumerate(slots):
            dw = max(1, sz // 32)
            if st == "OPR_SIMM8":
                ops.append("0")
            elif idx == len(slots) - 1 and len(slots) >= 2:
                ops.append("0x0")
            else:
                align = 4 if dw > 2 else (2 if dw == 2 else 1)
                sidx = ((sidx + align - 1) // align) * align
                ops.append(_reg("s", sidx, dw))
                sidx += dw
        return f"{op} {', '.join(ops)}"
    if enc == "ENC_DS":
        ops = []
        vidx = 4
        for st, sz in slots:
            dw = max(1, sz // 32)
            if st == "OPR_VGPR" and dw == 1:
                ops.append("v0")
            else:
                ops.append(_reg("v", vidx, dw))
                vidx += max(4, dw)
        return f"{op} {', '.join(ops)}"
    ops = []
    vidx, sidx = 0, 0
    for idx, (st, sz) in enumerate(slots):
        dw = max(1, sz // 32)
        if op == "v_writelane_b32" and idx == 1:
            ops.append("0")
        elif st == "OPR_VCC" or (enc == "VOP2_VOP_SDWA_SDST_ENC" and idx == 1):
            ops.append("vcc")
        elif st == "OPR_SREG_M0":
            ops.append("m0")
        elif st in ("OPR_SIMM4", "OPR_SIMM16", "OPR_SIMM32", "OPR_VERSION", "OPR_LABEL"):
            ops.append("0")
        elif "ACCVGPR" in st and "VGPR_OR" not in st:
            ops.append(_reg("a", vidx, dw))
            vidx += dw
        elif st.startswith(("OPR_SREG", "OPR_SDST", "OPR_SSRC")):
            align = 4 if dw > 2 else (2 if dw == 2 else 1)
            sidx = ((sidx + align - 1) // align) * align
            ops.append(_reg("s", sidx, dw))
            sidx += dw
        else:
            align = 2 if dw > 1 else 1
            vidx = ((vidx + align - 1) // align) * align
            ops.append(_reg("v", vidx, dw))
            vidx += dw
    extra = (
        " row_newbcast:0"
        if op in _DP_DPP_OPS
        else (" quad_perm:[0,1,2,3]" if op.endswith("_dpp") else "")
    )
    return f"{op} {', '.join(ops)}{extra}"


def _llvm_mc_encode(cpu: str, lines: list[str]) -> list[bytes]:
    proc = subprocess.run(
        ["llvm-mc", "-triple=amdgcn-amd-amdhsa", f"-mcpu={cpu}", "--show-encoding"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    out = []
    for ln in proc.stdout.splitlines():
        if "encoding: [" in ln:
            hex_part = ln.rsplit("encoding: [", 1)[1].rstrip("]")
            out.append(bytes(int(tok.strip(), 16) for tok in hex_part.split(",")))
    assert len(out) == len(lines)
    return out


def _llvm_mc_disassemble(cpu: str, encoded: list[bytes]) -> list[str]:
    hex_lines = [" ".join(f"0x{b:02x}" for b in raw) for raw in encoded]
    proc = subprocess.run(
        ["llvm-mc", "-triple=amdgcn-amd-amdhsa", f"-mcpu={cpu}", "--disassemble"],
        input="\n".join(hex_lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert "warning:" not in proc.stderr, proc.stderr
    lines = [
        ln.strip()
        for ln in proc.stdout.splitlines()
        if ln.strip() and not ln.strip().startswith(".")
    ]
    assert len(lines) == len(encoded)
    return lines


def test_the_pin_matches_the_manifest():
    assert (
        MANIFEST["bundle"]["file"],
        MANIFEST["bundle"]["sha256"],
    ) == (bundle.FILE, bundle.SHA256)


def test_another_bundle_is_refused(tmp_path):
    other = tmp_path / bundle.FILE
    other.write_bytes(b"not the pinned bundle")
    with pytest.raises(bundle.BundleError):
        bundle.check(other)


@pytest.mark.skipif("THIN_ISA_BUNDLE" not in os.environ, reason="set THIN_ISA_BUNDLE to the bundle")
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_a_clean_run_reproduces_checked_in_tables(target):
    rules = TARGETS[target]
    isa = bundle.isa_root(Path(os.environ["THIN_ISA_BUNDLE"]), rules.member)
    data_dir = ROOT / "crates/thin-targets/data" / target
    got_isa = canonical.sha256(isa_tables.generate(isa, rules))
    got_ops = canonical.sha256(operand_table.generate(isa, rules))
    assert got_isa == hashlib.sha256((data_dir / "isa_tables.json").read_bytes()).hexdigest()
    assert got_ops == hashlib.sha256((data_dir / "operand_table.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("table", ["isa_tables", "operand_table"])
def test_the_checked_in_tables_are_the_manifests(table):
    """The generator's checked-in output, without the bundle."""
    path = ROOT / "crates/thin-targets/data/gfx1201" / f"{table}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == MANIFEST[table]["sha256_canonical_json"]


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("table", ["isa_tables", "operand_table"])
def test_every_targets_checked_in_tables_are_canonical_json(target, table):
    path = ROOT / "crates/thin-targets/data" / target / f"{table}.json"
    parsed = json.loads(path.read_text())
    assert canonical.dumps(parsed).encode() == path.read_bytes()


def test_gfx950_tables_preserve_cdna_common_roles_and_cdna4_delta():
    it942 = json.loads((ROOT / "crates/thin-targets/data/gfx942/isa_tables.json").read_text())
    ot942 = json.loads((ROOT / "crates/thin-targets/data/gfx942/operand_table.json").read_text())
    it950 = json.loads((ROOT / "crates/thin-targets/data/gfx950/isa_tables.json").read_text())
    ot950 = json.loads((ROOT / "crates/thin-targets/data/gfx950/operand_table.json").read_text())

    enc950 = {e["class"]: e for e in it950["encodings"]}
    assert "VOP3PX2" in enc950
    vop3px2 = enc950["VOP3PX2"]
    assert vop3px2["bytes"] == 16
    fields_px2 = {f["name"]: f for f in vop3px2["fields"]}
    assert fields_px2["x2encoding"] == {
        "name": "x2encoding",
        "hi": 31,
        "lo": 16,
        "kind": "fixed",
        "value": 0xD3AC,
    }
    assert fields_px2["encoding"] == {
        "name": "encoding",
        "hi": 95,
        "lo": 87,
        "kind": "fixed",
        "value": 0x1A7,
    }
    assert fields_px2["abid"]["value"] == 1
    assert "opsel_hi2" in fields_px2
    assert "opsel_hi2" in {f["name"] for f in enc950["VOP3P"]["fields"]}

    ops942 = {o["mnemonic"]: o for o in it942["ops"]}
    ops950 = {o["mnemonic"]: o for o in it950["ops"]}
    removed = set(ops942) - set(ops950)
    assert removed == {
        "ds_gws_barrier",
        "ds_gws_init",
        "ds_gws_sema_br",
        "ds_gws_sema_p",
        "ds_gws_sema_release_all",
        "ds_gws_sema_v",
        "v_mfma_f32_16x16x8_xf32",
        "v_mfma_f32_32x32x4_xf32",
    }
    for common in set(ops942) & set(ops950):
        assert (ops942[common]["class"], ops942[common]["opcode"]) == (
            ops950[common]["class"],
            ops950[common]["opcode"],
        )

    rows942 = {(r["mnemonic"], r["count"]): r for r in ot942}
    rows950 = {(r["mnemonic"], r["count"]): r for r in ot950}
    for key in set(rows942) & set(rows950):
        for attr in ("roles", "defs", "uses", "sgpr_widths", "encoding"):
            assert rows942[key][attr] == rows950[key][attr], f"{key}: {attr}"


@pytest.mark.skipif(shutil.which("llvm-mc") is None, reason="llvm-mc required")
def test_gfx950_isa_tables_bidirectional_round_trip_against_llvm_mc():
    it950 = json.loads((ROOT / "crates/thin-targets/data/gfx950/isa_tables.json").read_text())
    encodings = {e["class"]: e for e in it950["encodings"]}
    short_names = {o["mnemonic"] for o in it950["ops"] if o["mnemonic"].endswith("_e32")}

    checked_ops = []
    lines = []
    for op in it950["ops"]:
        line = _isa_op_line(op, short_names)
        if line is not None:
            checked_ops.append(op)
            lines.append(line)

    assert len(checked_ops) == 1562
    bytes1 = _llvm_mc_encode("gfx950", lines)
    for op, raw in zip(checked_ops, bytes1, strict=True):
        enc = encodings[op["class"]]
        assert len(raw) == enc["bytes"], op["mnemonic"]
        word = int.from_bytes(raw, "little")
        for f in enc["fields"]:
            val = (word >> f["lo"]) & ((1 << (f["hi"] - f["lo"] + 1)) - 1)
            if f["kind"] == "fixed":
                assert val == f["value"], f"{op['mnemonic']} fixed {f['name']}"
            elif f["kind"] == "op":
                assert val == op["opcode"], f"{op['mnemonic']} opcode"

    disasm = _llvm_mc_disassemble("gfx950", bytes1)
    bytes2 = _llvm_mc_encode("gfx950", disasm)
    for op, b1, text, b2 in zip(checked_ops, bytes1, disasm, bytes2, strict=True):
        m = op["mnemonic"]
        if m in _LLVM_ZERO_OP_BARE_DISASM:
            assert text == _LLVM_ZERO_OP_BARE_DISASM[m]
        else:
            assert b2 == b1, f"{m}: {text}"


@pytest.mark.skipif(shutil.which("llvm-mc") is None, reason="llvm-mc required")
def test_gfx950_operand_table_bidirectional_round_trip_against_llvm_mc():
    ot950 = json.loads((ROOT / "crates/thin-targets/data/gfx950/operand_table.json").read_text())
    rows = [r for r in ot950 if r["mnemonic"] not in _LLVM_OMITTED_GFX9_VARIANTS]
    assert len(rows) == 2051

    lines = [_operand_row_line(r) for r in rows]
    bytes1 = _llvm_mc_encode("gfx950", lines)
    disasm = _llvm_mc_disassemble("gfx950", bytes1)
    bytes2 = _llvm_mc_encode("gfx950", disasm)
    for row, b1, text, b2 in zip(rows, bytes1, disasm, bytes2, strict=True):
        m = row["mnemonic"]
        if m in _LLVM_ZERO_OP_BARE_DISASM:
            assert text == _LLVM_ZERO_OP_BARE_DISASM[m]
        else:
            assert b2 == b1, f"{m}: {text}"
