//! Comprehensive CDNA 4 (`gfx950`, MI350X/MI355X) target, encoder/decoder parity
//! against `llvm-mc`, 128-bit `VOP3PX2` & `f8f6f4` variable tuple widths, negative
//! refusals, code-object `e_flags = 0x54F`, 160 KiB / 64-bank LDS, and full-table
//! round-trip tests.

use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;
use thin_target::{assemble, descriptor, read_s, CodeItem, Elem, Encoded, Flags, Footer};

const ARCH: &str = "gfx950";

fn llvm_mc_available() -> bool {
    Command::new("llvm-mc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn llvm_mc_bytes(lines: &[String]) -> Vec<Vec<u8>> {
    let mut child = Command::new("llvm-mc")
        .args([
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "--show-encoding",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn llvm-mc");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for line in lines {
            writeln!(stdin, "\t{line}").expect("write line");
        }
    }
    let out = child.wait_with_output().expect("wait llvm-mc");
    assert!(
        out.status.success(),
        "llvm-mc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    let mut encoded = Vec::new();
    for line in stdout.lines() {
        if let Some((_, hex_list)) = line.rsplit_once("encoding: [") {
            let inner = hex_list.trim_end_matches(']');
            let bytes: Vec<u8> = inner
                .split(',')
                .map(|tok| {
                    let h = tok.trim().strip_prefix("0x").expect("0x prefix");
                    u8::from_str_radix(h, 16).expect("hex byte")
                })
                .collect();
            encoded.push(bytes);
        }
    }
    assert_eq!(encoded.len(), lines.len(), "encoding count mismatch");
    encoded
}

#[test]
fn target_facts_and_resources_match_cdna4() {
    let t = thin_targets::get(ARCH).expect("gfx950 builds");
    let def = t.definition();
    assert_eq!(def.wave, 64);
    assert_eq!(def.lds_max, 163_840);
    assert_eq!(def.chip.lds, 163_840);
    assert_eq!(def.chip.xcds, 8);
    assert_eq!(def.chip.cus, 256);
    assert_eq!(def.chip.simds, 4);
    assert!(!def.operands.vop3_literal);
    assert_eq!(def.operands.constant_bus, 1);
    assert_eq!(def.operands.smem_offset_bits, 21);
    assert!(!t.hooks().fnuz_fp8());
    assert_eq!(t.hooks().e_flags(), 0x54F);
    assert!(thin_targets::MACHINES
        .iter()
        .any(|m| m.number == 0x4F && m.target == ARCH));

    let v = t.file('v').expect("v file");
    let s = t.file('s').expect("s file");
    let a = t.file('a').expect("a file");
    assert_eq!((v.size, v.granule, v.align64), (256, 8, 2));
    assert_eq!((s.size, s.granule, s.align64), (102, 8, 1));
    assert_eq!((a.size, a.granule, a.align64), (256, 4, 2));

    assert_eq!(t.allocated_vgprs(8, 0), 8);
    assert_eq!(t.allocated_vgprs(8, 4), 16);
    assert!(t.hooks().special("null").is_none());
    assert_eq!(t.hooks().special("m0"), Some((124, 1)));
    assert_eq!(t.hooks().special("vcc"), Some((106, 2)));
    assert_eq!(t.hooks().special("exec"), Some((126, 2)));

    // Common MFMA forms on gfx950 use OCP FP8/BF8 (`Elem::Fp8` / `Elem::Bf8`) and `provisional = true`.
    let mf = t
        .hooks()
        .matrix_form("v_mfma_f32_16x16x32_fp8_bf8")
        .expect("fp8_bf8 mfma");
    assert_eq!((mf.a, mf.b), (Elem::Fp8, Elem::Bf8));
    assert!(mf.provisional);

    // Variable widths on f8f6f4 MFMA and scaled MFMA forms.
    for (op, arity) in [
        ("v_mfma_f32_16x16x128_f8f6f4", 4),
        ("v_mfma_f32_32x32x64_f8f6f4", 4),
        ("v_mfma_scale_f32_16x16x128_f8f6f4", 6),
        ("v_mfma_scale_f32_32x32x64_f8f6f4", 6),
    ] {
        let f = t.form(op, arity, false).expect(op);
        assert!(f.flags.has(Flags::MATRIX), "{op} MATRIX");
        assert_eq!(f.slots[1].widths, vec![4, 6, 8], "{op} src0 widths");
        assert_eq!(f.slots[2].widths, vec![4, 6, 8], "{op} src1 widths");
    }

    // Transpose LDS reads carry transfer bytes per lane (8 for b64_tr_*, 12 for b96_tr_b6).
    for op in [
        "ds_read_b64_tr_b4",
        "ds_read_b64_tr_b8",
        "ds_read_b64_tr_b16",
    ] {
        let f = t.form(op, 2, false).expect(op);
        assert_eq!(f.mem_bytes, 8, "{op} mem_bytes");
        assert!(f.flags.has(Flags::DS | Flags::LOAD | Flags::LDS_MEM));
    }
    let f96 = t.form("ds_read_b96_tr_b6", 2, false).expect("b96_tr_b6");
    assert_eq!(f96.mem_bytes, 12);
}

#[test]
fn cdna4_new_instructions_and_vop3px2_match_llvm_mc_byte_for_byte() {
    let mut lines: Vec<String> = [
        // 128-bit VOP3PX2 MX-scaled MFMA across cbsz/blgp/op_sel/op_sel_hi combinations
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17",
        "v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[0:5], v[8:11], 0, v16, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:4",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[0:3], a[8:13], 1.0, v18, v19 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:3",
        "v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[0:7], v[8:15], a[0:15], v20, v21 cbsz:1 blgp:1",
        "v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], a[16:21], a[24:27], v[0:15], v22, v23 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:3 blgp:4",
        // Unscaled F8F6F4 MFMA across 8/6/4-dword tuple combinations
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]",
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[6:11], a[0:3] cbsz:2 blgp:3",
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:3], v[4:7], 0 cbsz:4 blgp:4",
        "v_mfma_f32_32x32x64_f8f6f4 a[0:15], v[0:5], v[6:9], a[0:15] cbsz:3 blgp:4",
        // New dense MFMA & sparse SMFMAC
        "v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]",
        "v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[4:7], a[0:3]",
        "v_mfma_f32_32x32x16_bf16 a[0:15], v[0:3], v[4:7], a[0:15]",
        "v_mfma_f32_32x32x16_f16 a[0:15], v[0:3], v[4:7], a[0:15]",
        "v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], v[4:7], a[0:3]",
        "v_mfma_i32_32x32x32_i8 a[0:15], v[0:3], v[4:7], a[0:15]",
        "v_smfmac_f32_16x16x64_bf16 a[0:3], v[0:3], v[4:11], v12",
        "v_smfmac_f32_32x32x32_f16 a[0:15], v[0:3], v[4:11], v12 cbsz:1 abid:2",
        "v_smfmac_f32_16x16x128_fp8_bf8 a[0:3], v[0:3], v[4:11], v12",
        "v_smfmac_f32_32x32x64_bf8_fp8 a[0:15], v[0:3], v[4:11], v12",
        "v_smfmac_i32_16x16x128_i8 a[0:3], v[0:3], v[4:11], v12",
        "v_smfmac_i32_32x32x64_i8 a[0:15], v[0:3], v[4:11], v12",
        // New VALU: bitop3, permlane_swap, scalef32, cvt, dot2_bf16, min/max3, prng, ashr_pk
        "v_bitop3_b32 v0, v1, v2, s4 bitop3:0xa8",
        "v_bitop3_b32 v0, v1, v2, v3 bitop3:17",
        "v_bitop3_b16 v0, v1, v2, v3 bitop3:0xff op_sel:[1,0,1,1]",
        "v_permlane16_swap_b32_e32 v0, v1",
        "v_permlane32_swap_b32_e32 v2, v3",
        "v_permlane16_swap_b32_e64 v0, v1 bound_ctrl:1",
        "v_permlane16_swap_b32_e64 v0, v1 fi:1",
        "v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1 fi:1",
        "v_cvt_f32_bf16_e32 v0, v1",
        "v_cvt_f32_bf16_e64 v0, -|v1| clamp mul:2",
        "v_cvt_pk_bf16_f32 v0, -v1, |v2| clamp",
        "v_cvt_pk_f16_f32 v0, v1, 1.0 div:2",
        "v_cvt_sr_bf16_f32 v0, -v1, v2 op_sel:[1,0,1]",
        "v_cvt_sr_f16_f32 v0, v1, s4 op_sel:[0,1,1]",
        "v_cvt_scalef32_f32_fp8 v0, v1, -v2 op_sel:[1,0,0]",
        "v_cvt_scalef32_f16_bf8 v0, v1, |v2| op_sel:[0,0,1]",
        "v_cvt_scalef32_pk_fp8_f32 v0, -v1, |v2|, v3 op_sel:[0,0,0,1]",
        "v_cvt_scalef32_pk_fp4_f16 v0, -v1, v2 op_sel:[1,0,0,1]",
        "v_cvt_scalef32_pk_fp4_bf16 v0, |v1|, v2 op_sel:[0,1,0,0]",
        "v_cvt_scalef32_sr_fp8_f32 v0, -v1, v2, v3 op_sel:[0,0,1,0]",
        "v_cvt_scalef32_sr_pk_fp4_f32 v0, -v[2:3], v4, v5 op_sel:[0,0,0,1]",
        "v_cvt_scalef32_pk32_fp6_f16 v[0:5], v[6:21], v22",
        "v_cvt_scalef32_pk32_f32_bf6 v[0:31], v[32:37], v38",
        "v_cvt_scalef32_2xpk16_fp6_f32 v[0:5], v[6:21], v[22:37], v38",
        "v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[8:39], v40, v41",
        "v_dot2_f32_bf16 v0, v1, v2, v3 neg_lo:[1,0] neg_hi:[0,1] clamp",
        "v_dot2c_f32_bf16_e32 v0, v1, v2",
        "v_dot2c_f32_bf16_e64 v0, -v1, |v2| clamp",
        "v_dot2c_f32_bf16_dpp v0, -v1, |v2| quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
        "v_minimum3_f32 v0, -v1, |v2|, 0.5 clamp",
        "v_maximum3_f32 v0, v1, v2, v3 mul:4",
        "v_pk_minimum3_f16 v0, v1, v2, v3 op_sel:[1,0,1] op_sel_hi:[0,1,0]",
        "v_pk_maximum3_f16 v0, v1, v2, v3 neg_lo:[1,0,0] clamp",
        "v_ashr_pk_i8_i32 v0, v1, v2, v3 op_sel:[0,0,0,1]",
        "v_ashr_pk_u8_i32 v0, v1, v2, 4 op_sel:[1,0,0,0]",
        "v_prng_b32_e32 v0, v1",
        "v_prng_b32_e64 v0, s4",
        "v_exp_legacy_f32_e32 v0, v1",
        "v_log_legacy_f32_e64 v0, |v1| clamp",
        // New memory ops: transpose LDS reads, 96/128-bit LDS DMA, buffer_atomic_pk_add_bf16
        "ds_read_b64_tr_b4 v[0:1], v2 offset:32",
        "ds_read_b64_tr_b8 a[2:3], v4 offset:64",
        "ds_read_b64_tr_b16 v[4:5], v6 offset:128",
        "ds_read_b96_tr_b6 v[0:2], v7 offset:192",
        "ds_read_b96_tr_b6 v[1:3], v7 offset:256",
        "ds_read_b96_tr_b6 a[1:3], v7",
        "global_load_lds_dwordx3 v0, s[4:5] offset:16",
        "global_load_lds_dwordx4 v[0:1], off offset:-64 sc0",
        "buffer_load_dwordx3 v0, s[8:11], 0 offen offset:16 lds",
        "buffer_load_dwordx4 off, s[8:11], s12 offset:32 lds",
        "buffer_atomic_pk_add_bf16 v0, v1, s[8:11], 0 offen",
        "buffer_atomic_pk_add_bf16 v0, v1, s[8:11], 0 offen sc0",
        // Regression check for FMT_NUM_U8 on v_alignbit_b32 / v_alignbyte_b32
        "v_alignbit_b32 v0, v1, v2, v3",
        "v_alignbit_b32 v0, v1, v2, 12",
        "v_alignbyte_b32 v0, v1, v2, s4",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // Also include one synthesized canonical line for every encodable op in gfx950/isa_tables.json!
    let isa: Value =
        serde_json::from_str(include_str!("../data/gfx950/isa_tables.json")).expect("json");
    for op in isa["ops"].as_array().expect("ops") {
        let class = op["class"].as_str().unwrap();
        let m = op["mnemonic"].as_str().unwrap();
        let special_syntax = op["operands"].as_array().unwrap().iter().any(|o| {
            matches!(
                o["type"].as_str().unwrap(),
                "OPR_LABEL"
                    | "OPR_SIMM4"
                    | "OPR_SIMM5"
                    | "OPR_TGT"
                    | "OPR_ATTR"
                    | "OPR_SENDMSG_RTN"
            )
        });
        if class == "MTBUF"
            || special_syntax
            || class.ends_with("_SDWA")
            || class.ends_with("_SDWA_SDST")
            || class.ends_with("_DPP16")
            || (class.ends_with("_LIT") && m != "s_setreg_imm32_b32")
            || m == "s_cbranch_i_fork"
            || m == "s_rfe_restore_b64"
        {
            continue;
        }
        lines.push(sample_line_for_op(op));
    }

    let t = thin_targets::get(ARCH).unwrap();
    let mut our_bytes = Vec::with_capacity(lines.len());
    for line in &lines {
        match t.hooks().encode(line) {
            Ok(Encoded::Bytes(bytes)) => {
                let decoded = thin_targets::decode(ARCH, &bytes)
                    .unwrap_or_else(|e| panic!("decode failed for `{line}` ({bytes:02x?}): {e}"));
                assert_eq!(decoded.len(), 1, "expected 1 instruction for `{line}`");
                let reencoded = thin_targets::encode_one(ARCH, &decoded[0]).unwrap_or_else(|e| {
                    panic!("re-encode failed for `{line}` -> `{}`: {e}", decoded[0])
                });
                assert_eq!(
                    reencoded, bytes,
                    "round-trip mismatch for `{line}` (decoded `{}`)",
                    decoded[0]
                );
                thin_targets::physical_form(ARCH, &decoded[0]).unwrap_or_else(|e| {
                    panic!("physical_form failed for `{line}` -> `{}`: {e}", decoded[0])
                });
                our_bytes.push(bytes);
            }
            Ok(Encoded::Branch { .. }) => unreachable!("branches use assemble()"),
            Err(e) => panic!("encode failed for `{line}`: {e}"),
        }
    }

    if llvm_mc_available() {
        let mc_bytes = llvm_mc_bytes(&lines);
        for ((line, ours), mc) in lines.iter().zip(&our_bytes).zip(&mc_bytes) {
            assert_eq!(
                ours, mc,
                "llvm-mc byte mismatch for `{line}`:\n  ours: {ours:02x?}\n  mc:   {mc:02x?}"
            );
        }
    }
}

fn reg_str(prefix: char, start: u32, dwords: u32) -> String {
    if dwords <= 1 {
        format!("{prefix}{start}")
    } else {
        format!("{prefix}[{start}:{}]", start + dwords - 1)
    }
}

fn sample_line_for_op(op: &Value) -> String {
    let m = op["mnemonic"].as_str().unwrap();
    let cls = op["class"].as_str().unwrap();
    let op_map: std::collections::HashMap<&str, &Value> = op["operands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| (o["field"].as_str().unwrap(), o))
        .collect();
    let dw = |f: &str| -> u32 {
        if op_map[f]["format"].as_str() == Some("FMT_NUM_M64") {
            2
        } else {
            ((op_map[f]["bits"].as_u64().unwrap() as u32) / 32).max(1)
        }
    };

    if m == "s_waitcnt" {
        return "s_waitcnt vmcnt(0)".into();
    }
    if m == "s_setreg_imm32_b32" {
        return "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 32), 0x12345678".into();
    }
    if m == "s_setreg_b32" {
        return "s_setreg_b32 hwreg(HW_REG_MODE, 0, 32), s0".into();
    }
    match cls {
        "SOPP" => {
            if let Some(simm) = op_map.get("simm16") {
                if simm["type"].as_str().unwrap() == "OPR_LABEL" {
                    // Test non-branch SOPP here; branches tested in `branches_and_vop3px2_offsets`.
                    return "s_nop 0".into();
                }
                return format!("{m} 1");
            }
            m.into()
        }
        "SOP1" | "SOP2" | "SOPC" | "SOPK" => {
            let mut parts = Vec::new();
            let mut s_next = 0u32;
            for f in ["sdst", "ssrc0", "ssrc1", "simm16"] {
                let Some(o) = op_map.get(f) else { continue };
                if f == "simm16" {
                    let t = o["type"].as_str().unwrap();
                    parts.push(if t == "OPR_HWREG" {
                        "hwreg(HW_REG_MODE, 0, 32)".into()
                    } else {
                        "1".into()
                    });
                } else if o["type"].as_str().unwrap() == "OPR_SREG_M0" {
                    parts.push("m0".into());
                } else {
                    let w = dw(f);
                    let align = if w > 2 { 4 } else { w };
                    s_next = s_next.div_ceil(align) * align;
                    parts.push(reg_str('s', s_next, w));
                    s_next += w;
                }
            }
            format!("{m} {}", parts.join(", "))
        }
        "SMEM" => {
            let hd = op_map.contains_key("sdata");
            let hb = op_map.contains_key("sbase");
            let ho = op_map.contains_key("offset") || op_map.contains_key("soffset");
            match (hd, hb, ho) {
                (false, false, false) => m.into(),
                (true, false, false) => format!("{m} {}", reg_str('s', 0, dw("sdata"))),
                (false, true, true) => format!("{m} {}, 0x0", reg_str('s', 0, dw("sbase"))),
                (true, true, true) => {
                    let d = if op_map["sdata"]["type"].as_str().unwrap() == "OPR_SIMM8" {
                        "1".into()
                    } else {
                        reg_str('s', 0, dw("sdata"))
                    };
                    format!("{m} {d}, {}, 0x0", reg_str('s', 16, dw("sbase")))
                }
                _ => unreachable!("{m}"),
            }
        }
        "VOP1" | "VOP1_SDST" | "VOP2" | "VOPC" => {
            if op_map.is_empty() {
                return m.into();
            }
            let base = m.strip_suffix("_e32").unwrap_or(m);
            match cls {
                "VOPC" if base.starts_with("v_cmpx") => format!(
                    "{m} {}, {}",
                    reg_str('v', 0, dw("src0")),
                    reg_str('v', 4, dw("vsrc1"))
                ),
                "VOPC" => format!(
                    "{m} vcc, {}, {}",
                    reg_str('v', 0, dw("src0")),
                    reg_str('v', 4, dw("vsrc1"))
                ),
                "VOP1_SDST" => format!(
                    "{m} {}, {}",
                    reg_str('s', 0, dw("vdst")),
                    reg_str('v', 0, dw("src0"))
                ),
                "VOP1" if base == "v_accvgpr_mov_b32" => format!("{m} a0, a2"),
                "VOP1" => format!(
                    "{m} {}, {}",
                    reg_str('v', 0, dw("vdst")),
                    reg_str('v', 4, dw("src0"))
                ),
                _ if matches!(base, "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32") => {
                    format!("{m} v0, vcc, v2, v4, vcc")
                }
                _ if matches!(base, "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32") => {
                    format!("{m} v0, vcc, v2, v4")
                }
                _ if base.starts_with("v_cndmask") => format!("{m} v0, v2, v4, vcc"),
                _ => format!(
                    "{m} {}, {}, {}",
                    reg_str('v', 0, dw("vdst")),
                    reg_str('v', 4, dw("src0")),
                    reg_str('v', 8, dw("vsrc1"))
                ),
            }
        }
        "VOP3" | "VOP3_SDST" | "VOP3SD" | "VOP3P" => {
            let print_m = if cls == "VOP3SD"
                && matches!(
                    m,
                    "v_add_co_u32"
                        | "v_sub_co_u32"
                        | "v_subrev_co_u32"
                        | "v_addc_co_u32"
                        | "v_subb_co_u32"
                        | "v_subbrev_co_u32"
                ) {
                format!("{m}_e64")
            } else {
                m.to_owned()
            };
            let mut parts = Vec::new();
            let mut v_next = 0u32;
            let mut s_next = 0u32;
            for f in ["vdst", "sdst", "src0", "src1", "src2"] {
                let Some(o) = op_map.get(f) else { continue };
                let t = o["type"].as_str().unwrap();
                if t == "OPR_EXEC" {
                    continue;
                }
                if m == "v_writelane_b32" && f == "src1" {
                    parts.push("0".into());
                    continue;
                }
                if t == "OPR_SREG_M0" {
                    parts.push("m0".into());
                    continue;
                }
                let w = dw(f);
                if matches!(
                    t,
                    "OPR_SREG"
                        | "OPR_SREG_NOVCC"
                        | "OPR_SDST"
                        | "OPR_SSRC"
                        | "OPR_SSRC_NOLIT"
                        | "OPR_SSRC_LANESEL"
                ) || (f == "sdst")
                {
                    let align = if w > 2 { 4 } else { w };
                    s_next = s_next.div_ceil(align) * align;
                    parts.push(reg_str('s', s_next, w));
                    s_next += w;
                } else if t.contains("ACCVGPR") && !t.contains("VGPR_OR") {
                    v_next = v_next.div_ceil(2) * 2;
                    parts.push(reg_str('a', v_next, w));
                    v_next += w.max(2);
                } else {
                    if w > 1 {
                        v_next = v_next.div_ceil(2) * 2;
                    }
                    parts.push(reg_str('v', v_next, w));
                    v_next += w.max(2);
                }
            }
            if parts.is_empty() {
                print_m
            } else {
                format!("{print_m} {}", parts.join(", "))
            }
        }
        "VOP3P_MFMA" => {
            let mut parts = Vec::new();
            let mut v_next = 0u32;
            for f in ["vdst", "src0", "src1", "src2"] {
                let w = dw(f);
                v_next = v_next.div_ceil(2) * 2;
                let file = if f == "vdst" || (f == "src2" && !m.starts_with("v_smfmac_")) {
                    'a'
                } else {
                    'v'
                };
                parts.push(reg_str(file, v_next, w));
                v_next += w.max(2);
            }
            format!("{m} {}", parts.join(", "))
        }
        "VOP3PX2" => {
            let wd = dw("vdst");
            format!(
                "{m} {}, v[0:7], v[8:15], {}, v16, v17",
                reg_str('a', 0, wd),
                reg_str('a', 0, wd)
            )
        }
        "DS" => {
            let mut parts = Vec::new();
            let mut v_next = 0u32;
            for f in ["vdst", "addr", "data0", "data1"] {
                if op_map.contains_key(f) {
                    let w = dw(f);
                    v_next = v_next.div_ceil(2) * 2;
                    parts.push(reg_str('v', v_next, w));
                    v_next += w.max(2);
                }
            }
            if parts.is_empty() {
                m.into()
            } else {
                format!("{m} {}", parts.join(", "))
            }
        }
        "FLAT" => {
            let hd = op_map.contains_key("vdst");
            let hdata = op_map.contains_key("data");
            let ha = op_map.contains_key("addr");
            match (hd, ha, hdata) {
                (false, false, false) => m.into(),
                (true, true, false) => format!("{m} {}, v[0:1]", reg_str('v', 4, dw("vdst"))),
                (false, true, true) => format!("{m} v[0:1], {}", reg_str('v', 4, dw("data"))),
                (true, true, true) => format!(
                    "{m} {}, v[0:1], {} sc0",
                    reg_str('v', 4, dw("vdst")),
                    reg_str('v', 8, dw("data"))
                ),
                _ => unreachable!("{m}"),
            }
        }
        "GLOBAL" | "SCRATCH" => {
            let hd = op_map.contains_key("vdst");
            let hdata = op_map.contains_key("data");
            let ha = op_map.contains_key("addr");
            let sref = if cls == "GLOBAL" { "s[0:1]" } else { "off" };
            if m.contains("_load_lds_") {
                return format!("{m} v0, {sref}");
            }
            match (hd, ha, hdata) {
                (false, false, false) => m.into(),
                (false, true, false) => format!("{m} v0, {sref}"),
                (true, true, false) => format!("{m} {}, v0, {sref}", reg_str('v', 4, dw("vdst"))),
                (false, true, true) => format!("{m} v0, {}, {sref}", reg_str('v', 4, dw("data"))),
                (true, true, true) => format!(
                    "{m} {}, v0, {}, {sref} sc0",
                    reg_str('v', 4, dw("vdst")),
                    reg_str('v', 8, dw("data"))
                ),
                _ => unreachable!("{m}"),
            }
        }
        "MUBUF" => {
            if op_map.is_empty() {
                m.into()
            } else {
                format!("{m} {}, v0, s[0:3], 0 offen", reg_str('v', 4, dw("vdata")))
            }
        }
        other => panic!("unexpected class {other}"),
    }
}

#[test]
fn branches_and_vop3px2_offsets_assemble_decode_and_reencode() {
    let t = thin_targets::get(ARCH).unwrap();
    let items = vec![
        CodeItem::Label {
            name: ".Ltop".into(),
            line: 1,
        },
        CodeItem::Ins {
            text: "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17"
                .into(),
            comment: String::new(),
            line: 2,
        },
        CodeItem::Ins {
            text: "s_sub_u32 s0, s0, 1".into(),
            comment: String::new(),
            line: 3,
        },
        CodeItem::Ins {
            text: "s_cbranch_scc1 .Ltop".into(),
            comment: String::new(),
            line: 4,
        },
        CodeItem::Ins {
            text: "s_endpgm".into(),
            comment: String::new(),
            line: 5,
        },
    ];
    let bytes = assemble(&items, t.hooks()).expect("assemble loop with 16-byte VOP3PX2");
    assert_eq!(bytes.len(), 16 + 4 + 4 + 4);
    let decoded = thin_targets::decode(ARCH, &bytes).expect("decode loop");
    assert_eq!(decoded.len(), 4);
    assert_eq!(decoded[0].size, 16);
    assert_eq!(decoded[1].offset, 16);
    assert_eq!(decoded[2].offset, 20);
    assert_eq!(decoded[2].branch_target(), Some(0));
    let reencoded = thin_targets::encode(ARCH, &decoded).expect("reencode loop");
    assert_eq!(reencoded, bytes);
}

#[test]
fn cdna4_negative_refusals_enforce_isa_and_encoding_constraints() {
    let t = thin_targets::get(ARCH).unwrap();
    let bad_lines = [
        // Removed CDNA 3 instructions
        "ds_gws_barrier v0 gds",
        "v_mfma_f32_16x16x8_xf32 a[0:3], v[0:1], v[2:3], a[0:3]",
        "v_mfma_f32_32x32x4_xf32 a[0:15], v[0:1], v[2:3], a[0:15]",
        // VOP3PX2 refusals
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 abid:1",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], s0, v17",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], 0, v17",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], a0, v17",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 op_sel:[0,0,1]",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 op_sel_hi:[0,0,1]",
        // F8F6F4 tuple size mismatches, abid refusal & odd alignment
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] abid:1",
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] cbsz:2",
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3] blgp:4",
        "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[1:6], v[8:15], a[0:3] cbsz:2",
        "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 cbsz:4",
        // bitop3 / permlane_swap / scalef32 modifier refusals
        "v_bitop3_b32 v0, -v1, v2, v3",
        "v_bitop3_b32 v0, v1, v2, v3 clamp",
        "v_bitop3_b32 v0, v1, v2, v3 mul:2",
        "v_bitop3_b32 v0, v1, v2, v3 bitop3:256",
        "v_permlane16_swap_b32_e64 v0, v1 clamp",
        "v_permlane16_swap_b32_e64 v0, v1 bound_ctrl:2",
        "v_permlane16_swap_b32_e64 v0, v1 fi:2",
        "v_cvt_scalef32_pk_fp8_f32 v0, v1, v2, v3 clamp",
        "v_cvt_scalef32_pk_fp8_f32 v0, v1, v2, v3 mul:2",
        "v_cvt_scalef32_2xpk16_fp6_f32 v[0:5], -v[6:21], v[22:37], v38",
        // MUBUF lds still rejects dwordx2
        "buffer_load_dwordx2 v0, s[8:11], 0 offen lds",
        // ds_read_b96 (non-tr) still requires even VGPR alignment
        "ds_read_b96 v[1:3], v4",
    ];
    for line in bad_lines {
        assert!(
            t.hooks().encode(line).is_err(),
            "expected `{line}` to be rejected on gfx950"
        );
    }

    // Decoder refusals on VOP3PX2: truncated <16 bytes, abid != 1, opsel_hi2 != 0
    let Encoded::Bytes(valid) = t
        .hooks()
        .encode("v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17")
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(valid.len(), 16);
    assert_eq!(
        thin_targets::decode(ARCH, &valid[..12]).unwrap_err().why,
        thin_targets::decode::Why::Truncated
    );
    let mut bad_abid = valid.clone();
    bad_abid[9] &= !0x78; // clear abid [78:75] from 1 to 0
    assert_eq!(
        thin_targets::decode(ARCH, &bad_abid).unwrap_err().why,
        thin_targets::decode::Why::Field
    );
    let mut bad_hi2 = valid;
    bad_hi2[1] |= 0x40; // set opsel_hi2 (bit 14) = 1
    assert_eq!(
        thin_targets::decode(ARCH, &bad_hi2).unwrap_err().why,
        thin_targets::decode::Why::Field
    );
}

#[test]
fn cdna4_descriptor_and_160kb_lds_match_llvm_mc_and_elf_0x54f() {
    let t = thin_targets::get(ARCH).unwrap();
    for (vgpr, agpr, lds) in [(8, 0, 0), (32, 32, 65_536), (64, 128, 163_840)] {
        let footer = Footer {
            name: "kern950".into(),
            target_name: ARCH.into(),
            lds,
            kernarg: 32,
            vgpr,
            agpr,
            sgpr: 24,
            wgsize: 256,
            wgy: 1,
            preload: 0,
            ends: true,
        };
        let asm = format!(
            "\t.amdgcn_target \"amdgcn-amd-amdhsa--gfx950\"\n\
             \t.text\n\
             \t.globl kern950\n\
             \t.p2align 8\n\
             \t.type kern950,@function\n\
             kern950:\n\
             \ts_endpgm\n\
             {}",
            t.hooks().footer(&footer).unwrap()
        );
        let parsed = read_s(&asm, true).expect("read_s");
        let desc = descriptor(&parsed, t).expect("descriptor");
        if llvm_mc_available() {
            let tmp = std::env::temp_dir().join(format!(
                "thin_gfx950_desc_{}_{}_{}.s",
                std::process::id(),
                vgpr,
                lds
            ));
            let obj = tmp.with_extension("o");
            let rod = tmp.with_extension("rodata");
            std::fs::write(&tmp, &asm).unwrap();
            assert!(Command::new("llvm-mc")
                .args(["-filetype=obj", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx950"])
                .arg(&tmp)
                .arg("-o")
                .arg(&obj)
                .status()
                .unwrap()
                .success());
            let elf_bytes = std::fs::read(&obj).unwrap();
            // ELF64 header e_flags is at bytes [48..52] little-endian (`0x54F`).
            let e_flags = u32::from_le_bytes(elf_bytes[48..52].try_into().unwrap());
            assert_eq!(e_flags, t.hooks().e_flags());
            assert!(Command::new("llvm-objcopy")
                .arg("--dump-section")
                .arg(format!(".rodata={}", rod.display()))
                .arg(&obj)
                .status()
                .unwrap()
                .success());
            let mut mc_desc = std::fs::read(&rod).unwrap();
            let _ = std::fs::remove_file(&tmp);
            let _ = std::fs::remove_file(&obj);
            let _ = std::fs::remove_file(&rod);
            mc_desc[16..24].fill(0);
            let mut ours = desc.to_vec();
            ours[16..24].fill(0);
            assert_eq!(ours, mc_desc, "descriptor mismatch for lds={lds}");
        }
    }
}

#[test]
fn cdna4_lds_banks_hook_models_64_banks() {
    let t = thin_targets::get(ARCH).unwrap();
    let h = t.hooks();
    assert!(h.lds_banks(true, 0).is_none());
    assert!(h.lds_banks(true, 32).is_none());

    for &b in &[1u32, 2, 4] {
        for read in [false, true] {
            let (banks, phase) = h.lds_banks(read, b).unwrap();
            assert_eq!(banks, 64);
            assert_eq!(phase, vec![0; 64]);
        }
    }
    for read in [false, true] {
        let (banks, phase) = h.lds_banks(read, 8).unwrap();
        assert_eq!(banks, 64);
        assert!(phase[..32].iter().all(|&p| p == 0));
        assert!(phase[32..].iter().all(|&p| p == 1));
    }
    for &b in &[12u32, 16] {
        for read in [false, true] {
            let (banks, phase) = h.lds_banks(read, b).unwrap();
            assert_eq!(banks, 64);
            for q in 0..4usize {
                assert!(phase[q * 16..(q + 1) * 16].iter().all(|&p| p == q as u32));
            }
        }
    }
}
