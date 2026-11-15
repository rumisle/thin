//! Comprehensive CDNA 3 (`gfx942`, MI300X) target, encoder/decoder parity
//! against `llvm-mc`, negative refusals, hazard rules, and descriptor tests.

use std::process::Command;
use thin_target::{assemble, descriptor, read_s, CodeItem, Encoded, Footer};

const ARCH: &str = "gfx942";

fn llvm_mc_available() -> bool {
    Command::new("llvm-mc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Assembles `lines` with `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942 --show-encoding`
/// and extracts each line's encoded bytes in order.
fn llvm_mc_bytes(lines: &[&str]) -> Vec<Vec<u8>> {
    use std::io::Write;
    let mut child = Command::new("llvm-mc")
        .args([
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx942",
            "--show-encoding",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
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
fn target_facts_and_resources_match_cdna3() {
    let t = thin_targets::get(ARCH).expect("gfx942 builds");
    let def = t.definition();
    assert_eq!(def.wave, 64);
    assert_eq!(def.lds_max, 65536);
    assert_eq!(def.chip.xcds, 8);
    assert_eq!(def.chip.cus, 304);
    assert_eq!(def.chip.simds, 4);
    assert!(!def.operands.vop3_literal);
    assert_eq!(def.operands.constant_bus, 1);
    assert_eq!(def.operands.smem_offset_bits, 21);

    let v = t.file('v').expect("v file");
    let s = t.file('s').expect("s file");
    let a = t.file('a').expect("a file");
    assert_eq!((v.size, v.granule, v.align64), (256, 8, 2));
    assert_eq!((s.size, s.granule, s.align64), (102, 8, 1));
    assert_eq!((a.size, a.granule, a.align64), (256, 4, 2));

    // CDNA 3 unified archVGPR + AGPR allocation: round_up(vgpr, 4) + round_up(agpr, 4), min 4, granule 8 in combined mode when >0 AGPRs?
    // Verify hooks.allocated_vgprs:
    assert_eq!(t.allocated_vgprs(8, 0), 8);
    assert_eq!(t.allocated_vgprs(4, 4), 8);
    assert_eq!(t.allocated_vgprs(8, 4), 16);
    assert_eq!(t.allocated_vgprs(5, 3), 16);
    assert!(t.hooks().special("null").is_none());
    assert_eq!(t.hooks().special("m0"), Some((124, 1)));
    assert_eq!(t.hooks().special("vcc"), Some((106, 2)));
    assert_eq!(t.hooks().special("exec"), Some((126, 2)));
}

#[test]
fn encoder_and_decoder_match_llvm_mc_byte_for_byte() {
    let lines: &[&str] = &[
        // SOP1 / SOP2 / SOPC / SOPK / SOPK_LIT / SOPP
        "s_mov_b32 s0, s1",
        "s_mov_b32 s0, 0x12345678",
        "s_mov_b64 s[0:1], s[2:3]",
        "s_not_b64 exec, s[4:5]",
        "s_and_saveexec_b64 s[6:7], vcc",
        "s_add_u32 s2, s0, s1",
        "s_addc_u32 s3, s2, 16",
        "s_lshl_b64 s[4:5], s[2:3], 3",
        "s_and_b64 s[0:1], exec, vcc",
        "s_cmp_eq_u32 s0, 42",
        "s_cmp_lg_u64 s[0:1], 0",
        "s_movk_i32 s4, -12",
        "s_addk_i32 s4, 1024",
        "s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s1",
        "s_getreg_b32 s2, hwreg(HW_REG_XCC_ID, 0, 4)",
        "s_nop 0",
        "s_nop 17",
        "s_sleep 4",
        "s_barrier",
        "s_icache_inv",
        "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
        "s_waitcnt vmcnt(45)",
        "s_waitcnt lgkmcnt(3)",
        "s_waitcnt vmcnt(18) lgkmcnt(0)",
        "s_endpgm",
        // SMEM
        "s_load_dword s4, s[0:1], 0x0",
        "s_load_dwordx2 s[4:5], s[0:1], 0x10",
        "s_load_dwordx4 s[8:11], s[0:1], 0x40 glc",
        "s_load_dwordx8 s[16:23], s[0:1], s2",
        "s_buffer_load_dword s5, s[8:11], 0x20",
        "s_memtime s[6:7]",
        "s_memrealtime s[8:9]",
        "s_dcache_inv",
        "s_dcache_wb",
        // VOP1 / VOP1_SDST / VOP2 / VOPC
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0x3f800000",
        "v_cvt_f16_f32 v2, v1",
        "v_rcp_f32 v3, v1",
        "v_readfirstlane_b32 s4, v2",
        "v_accvgpr_mov_b32 a0, a1",
        "v_add_f32 v4, v1, v3",
        "v_add_u32 v5, s4, v0",
        "v_lshlrev_b32 v6, 4, v0",
        "v_fmac_f32 v7, 0.5, v4",
        "v_cndmask_b32_e32 v8, v1, v3, vcc",
        "v_add_co_u32_e32 v9, vcc, v5, v6",
        "v_addc_co_u32_e32 v10, vcc, v5, v6, vcc",
        "v_cmp_eq_f32_e32 vcc, v1, v3",
        "v_cmp_gt_u32_e32 vcc, s4, v0",
        "v_cmpx_lt_i32_e32 vcc, v5, v6",
        // VOP3 / VOP3_SDST / VOP3SD
        "v_fma_f32 v12, -v1, |v3|, v4 clamp",
        "v_fma_f32 v12, s4, s4, v4",
        "v_fma_f32 v13, v1, v3, 1.0 mul:2",
        "v_add3_u32 v14, v5, v6, s4",
        "v_lshl_add_u32 v15, v0, 2, s4",
        "v_alignbit_b32 v16, v5, v6, 8",
        "v_cmp_gt_i32_e64 s[12:13], v5, v6",
        "v_cmpx_eq_u32_e64 s[14:15], v5, 0",
        "v_cndmask_b32_e64 v17, v1, v3, s[12:13]",
        "v_add_co_u32_e64 v18, s[14:15], v5, 1",
        "v_div_scale_f32 v19, s[12:13], v1, v3, v4",
        "v_writelane_b32 v1, s14, m0",
        // VOP3P & VOP3P_MFMA
        "v_pk_fma_f16 v20, v1, v2, v3",
        "v_pk_add_f16 v21, v1, v2 op_sel:[1,0] op_sel_hi:[0,1] clamp",
        "v_pk_mul_f32 v[22:23], v[2:3], v[4:5]",
        "v_dot2_f32_f16 v24, v1, v2, v3 neg_lo:[1,0] neg_hi:[0,1]",
        "v_accvgpr_write_b32 a4, v1",
        "v_accvgpr_write_b32 a5, 0",
        "v_accvgpr_read_b32 v25, a4",
        "v_mfma_f32_16x16x16_f16 a[0:3], v[2:3], v[4:5], a[0:3]",
        "v_mfma_f32_16x16x16_bf16 a[4:7], a[0:1], a[2:3], 0",
        "v_mfma_f32_32x32x8_f16 a[0:15], v[2:3], v[4:5], a[0:15] cbsz:1 abid:2 blgp:3",
        "v_mfma_i32_16x16x32_i8 a[8:11], v[2:3], v[4:5], a[8:11]",
        "v_mfma_f64_16x16x4_f64 a[0:7], v[2:3], v[4:5], a[0:7]",
        // DPP16
        "v_mov_b32_dpp v26, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
        "v_add_f32_dpp v27, -v1, |v3| row_shl:1 row_mask:0xa bank_mask:0x5 bound_ctrl:1",
        "v_max_f32_dpp v28, v1, v3 row_bcast:15 row_mask:0xf bank_mask:0xf",
        "v_min_f32_dpp v29, v1, v3 row_newbcast:7 row_mask:0xf bank_mask:0xf",
        // GLOBAL / FLAT / SCRATCH
        "global_load_dword v30, v0, s[0:1]",
        "global_load_dwordx4 v[32:35], v0, s[0:1] offset:64 sc0 nt",
        "global_load_dwordx4 a[12:15], v0, s[0:1] offset:-128",
        "global_store_dword v0, v30, s[0:1] sc1",
        "global_store_dwordx4 v0, a[12:15], s[0:1]",
        "global_atomic_add v31, v0, v5, s[0:1] sc0",
        "global_atomic_add v0, v5, s[0:1] sc1",
        "global_load_lds_dword v0, s[0:1] offset:16",
        "flat_load_dword v36, v[2:3] offset:32",
        "flat_store_dword v[2:3], v36",
        "flat_atomic_add v37, v[2:3], v5 sc0",
        "flat_atomic_add v[2:3], v5",
        "scratch_load_dword v38, v0, s2 offset:-16",
        "scratch_store_dword v0, v38, off offset:32",
        // MUBUF
        "buffer_load_dword v39, off, s[8:11], 0 offset:128",
        "buffer_load_dwordx4 a[16:19], v0, s[8:11], s4 offen",
        "buffer_store_dword v39, v[0:1], s[8:11], 0 idxen offen",
        "buffer_load_dword v0, s[8:11], s4 offen lds",
        "buffer_wbl2 sc1",
        "buffer_inv sc1",
        // DS
        "ds_read_b32 v40, v0 offset:256",
        "ds_read_b128 a[20:23], v0",
        "ds_write_b32 v0, v40 offset:64",
        "ds_write_b128 v0, a[20:23]",
        "ds_read2_b32 v[42:43], v0 offset0:1 offset1:5",
        "ds_write2st64_b64 v0, v[2:3], v[4:5] offset0:2 offset1:3",
        "ds_bpermute_b32 v41, v0, v1",
        "ds_swizzle_b32 v44, v1 offset:0x801f",
        "ds_add_u32 v0, v5",
        "ds_add_rtn_u32 v45, v0, v5",
        "ds_gws_barrier v0 gds",
    ];

    let t = thin_targets::get(ARCH).unwrap();
    let mut our_bytes = Vec::with_capacity(lines.len());
    for &line in lines {
        let Encoded::Bytes(bytes) = t
            .hooks()
            .encode(line)
            .unwrap_or_else(|e| panic!("encode failed for `{line}`: {e}"))
        else {
            panic!("expected Bytes for `{line}`");
        };
        let decoded = thin_targets::decode(ARCH, &bytes)
            .unwrap_or_else(|e| panic!("decode failed for `{line}` ({bytes:02x?}): {e}"));
        assert_eq!(
            decoded.len(),
            1,
            "expected 1 decoded instruction for `{line}`"
        );
        let reencoded = thin_targets::encode_one(ARCH, &decoded[0])
            .unwrap_or_else(|e| panic!("re-encode failed for `{line}` -> `{}`: {e}", decoded[0]));
        assert_eq!(
            reencoded, bytes,
            "decode -> re-encode mismatch for `{line}` (decoded as `{}`)",
            decoded[0]
        );
        thin_targets::physical_form(ARCH, &decoded[0]).unwrap_or_else(|e| {
            panic!("physical_form failed for `{line}` -> `{}`: {e}", decoded[0])
        });
        our_bytes.push(bytes);
    }

    if llvm_mc_available() {
        let mc_bytes = llvm_mc_bytes(lines);
        for ((&line, ours), mc) in lines.iter().zip(&our_bytes).zip(&mc_bytes) {
            assert_eq!(
                ours, mc,
                "llvm-mc byte mismatch for `{line}`:\n  ours: {ours:02x?}\n  mc:   {mc:02x?}"
            );
        }
    }
}

#[test]
fn branches_assemble_decode_and_reencode() {
    let t = thin_targets::get(ARCH).unwrap();
    let items = vec![
        CodeItem::Label {
            name: ".Ltop".into(),
            line: 1,
        },
        CodeItem::Ins {
            text: "s_add_u32 s0, s0, 1".into(),
            comment: String::new(),
            line: 2,
        },
        CodeItem::Ins {
            text: "s_cmp_lt_u32 s0, 10".into(),
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
    let bytes = assemble(&items, t.hooks()).expect("assemble loop");
    let decoded = thin_targets::decode(ARCH, &bytes).expect("decode loop");
    assert_eq!(decoded.len(), 4);
    assert_eq!(decoded[2].branch_target(), Some(0));
    let roundtrip = thin_targets::encode(ARCH, &decoded).expect("re-encode loop");
    assert_eq!(roundtrip, bytes);
}

#[test]
fn cdna3_refusals_reject_invalid_encodings() {
    let t = thin_targets::get(ARCH).unwrap();
    let bad = [
        // `null` does not exist on CDNA 3
        "s_mov_b32 s0, null",
        "global_load_dword v0, v1, null",
        // 64-bit / 128-bit VGPR and AGPR pairs/tuples require even register base on CDNA 3
        "global_load_dwordx2 v[1:2], v0, s[0:1]",
        "global_load_dwordx4 a[1:4], v0, s[0:1]",
        "v_mfma_f32_16x16x16_f16 a[1:4], v[2:3], v[4:5], a[1:4]",
        // VOP3 has no literal on CDNA 3
        "v_fma_f32 v0, v1, v2, 0x12345678",
        // Constant bus limit is 1 scalar/literal source on CDNA 3
        "v_fma_f32 v0, s0, s1, v2",
        "v_cndmask_b32_e32 v0, s0, v2, vcc",
        // RDNA-only DPP controls must be rejected on CDNA 3
        "v_mov_b32_dpp v0, v1 row_share:0 row_mask:0xf bank_mask:0xf",
        "v_mov_b32_dpp v0, v1 row_xmask:0 row_mask:0xf bank_mask:0xf",
        // Returning vs non-returning global/flat atomics require/forbid `sc0`
        "global_atomic_add v1, v0, v2, s[0:1]",
        "global_atomic_add v0, v2, s[0:1] sc0",
        "flat_atomic_add v1, v[0:1], v2",
        "flat_atomic_add v[0:1], v2 sc0",
    ];
    for line in bad {
        assert!(
            t.hooks().encode(line).is_err(),
            "expected `{line}` to be refused on gfx942"
        );
    }
}

#[test]
fn cdna3_hazard_rules_insert_required_wait_states() {
    let t = thin_targets::get(ARCH).unwrap();
    let model = thin_targets::hazards(ARCH).unwrap();

    // 1. v_accvgpr_write_b32 -> MFMA reading same AGPR requires 2 wait states (s_nop 1)
    // 2. 4-pass MFMA -> v_accvgpr_read_b32 reading written AGPR requires 18 wait states (s_nop 17)
    let mfma_lines = [
        "v_accvgpr_write_b32 a0, 0",
        "v_mfma_f32_16x16x16_f16 a[0:3], v[2:3], v[4:5], a[0:3]",
        "v_accvgpr_read_b32 v6, a0",
    ];
    let snap =
        thin_test_support::asm::snapshot(t, thin_test_support::asm::read(t, &mfma_lines).unwrap());
    let needs = thin_passes::hazards::needs(&snap, model);
    assert_eq!(needs[0].states, 0);
    assert_eq!(needs[1].states, 2);
    assert_eq!(needs[2].states, 18);

    // 3. SALU writing M0 -> DS reading M0 (`ds_add_u32` / `global_load_lds_dword`) requires 1 state
    let m0_lines = ["s_mov_b32 m0, 0", "global_load_lds_dword v0, s[0:1]"];
    let snap_m0 =
        thin_test_support::asm::snapshot(t, thin_test_support::asm::read(t, &m0_lines).unwrap());
    let needs_m0 = thin_passes::hazards::needs(&snap_m0, model);
    assert_eq!(needs_m0[1].states, 1);

    // 4. VALU writing SGPR (`v_readfirstlane_b32` / `v_cmp_*_e64`) -> VMEM reading SADDR requires 5 states
    let saddr_lines = [
        "v_readfirstlane_b32 s4, v0",
        "v_readfirstlane_b32 s5, v1",
        "global_load_dword v2, v0, s[4:5]",
    ];
    let snap_saddr =
        thin_test_support::asm::snapshot(t, thin_test_support::asm::read(t, &saddr_lines).unwrap());
    let needs_saddr = thin_passes::hazards::needs(&snap_saddr, model);
    assert_eq!(needs_saddr[2].states, 5);
}

#[test]
fn cdna3_descriptor_matches_llvm_mc_with_and_without_agprs() {
    let t = thin_targets::get(ARCH).unwrap();
    for (vgpr, agpr, sgpr, lds) in [(8, 0, 12, 0), (16, 8, 24, 4096), (64, 32, 40, 16384)] {
        let footer = Footer {
            name: "desc_check".into(),
            target_name: ARCH.into(),
            lds,
            kernarg: 16,
            vgpr,
            agpr,
            sgpr,
            wgsize: 256,
            wgy: 1,
            preload: 0,
            ends: true,
        };
        let asm = format!(
            "\t.amdgcn_target \"amdgcn-amd-amdhsa--gfx942\"\n\
             \t.amdhsa_code_object_version 6\n\
             \t.text\n\
             \t.globl desc_check\n\
             \t.p2align 8\n\
             \t.type desc_check,@function\n\
             desc_check:\n\
             \ts_endpgm\n\
             {}",
            t.hooks().footer(&footer).unwrap()
        );
        let parsed = read_s(&asm, true).expect("read_s");
        let desc = descriptor(&parsed, t).expect("descriptor");
        if llvm_mc_available() {
            let tmp = std::env::temp_dir().join(format!(
                "thin_gfx942_desc_{}_{}_{}.s",
                std::process::id(),
                vgpr,
                agpr
            ));
            let obj = tmp.with_extension("o");
            let rod = tmp.with_extension("rodata");
            std::fs::write(&tmp, &asm).unwrap();
            assert!(Command::new("llvm-mc")
                .args(["-filetype=obj", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx942",])
                .arg(&tmp)
                .arg("-o")
                .arg(&obj)
                .status()
                .unwrap()
                .success());
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
            assert_eq!(
                ours, mc_desc,
                "descriptor mismatch for vgpr={vgpr}, agpr={agpr}"
            );
        }
    }
}
