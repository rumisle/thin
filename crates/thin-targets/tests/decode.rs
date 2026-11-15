//! The decoder's law: `encode(decode(b)) == b` byte for byte, and
//! `parse(print(decode(b)))` is the decoded line, on the encoder's own test
//! lines, on whole programs with branches, and on random words (where the
//! decoder may refuse, but never accept what does not re-encode).

use std::io::Read;

use thin_target::{assemble, parse_line, read_s, Encoded};
use thin_targets::decode::Why;
use thin_targets::{decode, decoder, encode, encode_one, get, Instruction};

const T: &str = "gfx1201";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// Every law for one decoded instruction of `bytes` (its own bytes).
fn laws(i: &Instruction, bytes: &[u8]) -> Result<(), String> {
    let text = i.to_string();
    let parsed = parse_line(get(T).unwrap().hooks(), &text)?;
    if parsed != i.line() {
        return Err(format!("`{text}` parses to {parsed:?}, not {:?}", i.line()));
    }
    let again = encode_one(T, i)?;
    if again != bytes {
        return Err(format!(
            "`{text}` encodes to {}, not {}",
            hex(&again),
            hex(bytes)
        ));
    }
    Ok(())
}

fn encoder_lines() -> Vec<(String, Vec<u8>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/target/encoder_lines.json.gz"
    );
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).expect("the vectors"))
        .read_to_string(&mut text)
        .expect("gzip");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    let rows = doc["rows"].as_array().expect("rows");
    rows.iter()
        .filter_map(|r| Some((r[0].as_str()?.to_owned(), unhex(r[1].as_str()?))))
        .collect()
}

/// Every line the encoder's vectors encode decodes, re-encodes to its
/// bytes and prints a line that parses back to the decoded one.
#[test]
fn every_encoder_line_round_trips() {
    let lines = encoder_lines();
    assert!(lines.len() > 20_000);
    let mut failures = Vec::new();
    let mut same_text = 0;
    for (line, bytes) in &lines {
        let got = decode(T, bytes)
            .map_err(|e| e.to_string())
            .and_then(|is| match is.as_slice() {
                [i] => laws(i, bytes).map(|()| i.to_string()),
                _ => Err(format!("{} instructions", is.len())),
            });
        match got {
            Ok(text) => same_text += usize::from(&text == line),
            Err(e) => failures.push(format!("{line}: {e}")),
        }
    }
    eprintln!(
        "encoder lines: {} decoded and re-encoded, {} printed as written",
        lines.len() - failures.len(),
        same_text
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

/// A program with branches both ways decodes whole and assembles back.
#[test]
fn a_program_with_branches_round_trips() {
    let text = "\
        .text\n\
        s_mov_b32 s2, 0\n\
        .L_loop:\n\
        s_add_co_u32 s2, s2, 1\n\
        v_add_f32 v1, 0x41200000, v2\n\
        s_cmp_lt_u32 s2, 10\n\
        s_cbranch_scc1 .L_loop\n\
        s_branch .L_end\n\
        s_nop 0\n\
        .L_end:\n\
        s_endpgm\n";
    let file = read_s(text, false).expect("reads");
    let bytes = assemble(&file.code, get(T).unwrap().hooks()).expect("assembles");
    let is = decode(T, &bytes).expect("decodes");
    assert_eq!(is.len(), 8);
    assert_eq!(is[4].to_string(), "s_cbranch_scc1 .L4");
    assert_eq!(is[5].to_string(), "s_branch .L32");
    assert_eq!(encode(T, &is).expect("encodes"), bytes);
    let mut short = bytes.clone();
    short.truncate(bytes.len() - 8);
    let e = decode(T, &short).expect_err("the branch to the end lands past it now");
    assert_eq!(e.why, Why::Branch);
}

/// A refusal says where, the bits and why.
#[test]
fn refusals_are_precise() {
    // v_add_nc_u32_e64 v1, |v2|, v3: no abs on an integer input.
    let e = decode(T, &unhex("010125d502070200")).expect_err("abs");
    assert_eq!((e.offset, e.why, e.field), (0, Why::Field, Some("abs")));
    assert_eq!(
        e.to_string(),
        "0x0: 010125d502070200: v_add_nc_u32_e64: abs on src0: v_add_nc_u32_e64 takes none there"
    );
    let e = decode(T, &unhex("0000b0bf0000")).expect_err("truncated");
    assert_eq!((e.offset, e.why), (4, Why::Truncated));
    // global_load_b32 v1, v0, s[4:5] cut short.
    let e = decode(T, &unhex("040005ee01000000")).expect_err("truncated");
    assert_eq!((e.why, e.size), (Why::Truncated, Some(12)));
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random words and bit flips of the encoder's lines: whatever the
/// decoder accepts re-encodes to exactly its bytes.
#[test]
fn whatever_decodes_re_encodes() {
    let d = decoder(T).unwrap();
    let lines = encoder_lines();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut accepted, mut tried) = (0, 0);
    let mut failures = Vec::new();
    // THIN_DECODE_FUZZ sets the rounds (a release run takes millions).
    let rounds = std::env::var("THIN_DECODE_FUZZ")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(400_000);
    for round in 0..rounds {
        let mut b = [0u8; 16];
        for chunk in b.chunks_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes()[..chunk.len()]);
        }
        if round % 4 != 0 {
            let (_, bytes) = &lines[(rng.next() % lines.len() as u64) as usize];
            b[..bytes.len()].copy_from_slice(bytes);
            for _ in 0..1 + rng.next() % 3 {
                let bit = rng.next() % (8 * bytes.len() as u64);
                b[(bit / 8) as usize] ^= 1 << (bit % 8);
            }
        }
        tried += 1;
        let Ok(i) = d.decode_at(&b, 0) else { continue };
        accepted += 1;
        if let Err(e) = laws(&i, &b[..usize::from(i.size)]) {
            failures.push(format!("{}: {e}", hex(&b[..usize::from(i.size)])));
        }
    }
    eprintln!("random: {accepted} of {tried} decoded, all re-encoded");
    assert!(accepted > 50_000);
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

/// The encoder's own branch words decode to their label.
#[test]
fn branches_decode_to_labels() {
    let Ok(Encoded::Branch { raw, .. }) = get(T).unwrap().hooks().encode("s_branch .L0") else {
        panic!("a branch");
    };
    let word = (raw | 0xFFFF).to_le_bytes(); // a branch to itself
    let is = decode(T, &word).expect("decodes");
    assert_eq!(is[0].to_string(), "s_branch .L0");
}

/// Lines as llvm-mc encodes them (`-triple=amdgcn -mcpu=gfx1201
/// -show-encoding`): thin encodes each to llvm-mc's bytes, and those bytes
/// decode to a line that parses back and re-encodes to them.
fn agree_with_llvm_mc(lines: &[(&str, &str)]) {
    let hooks = get(T).unwrap().hooks();
    for (line, want) in lines {
        let Ok(Encoded::Bytes(bytes)) = hooks.encode(line) else {
            panic!("{line}: {:?}", hooks.encode(line));
        };
        assert_eq!(hex(&bytes), *want, "{line}");
        let is = decode(T, &bytes).unwrap_or_else(|e| panic!("{line}: {e}"));
        laws(&is[0], &bytes).unwrap_or_else(|e| panic!("{line}: {e}"));
    }
}

/// Buffer loads, stores and atomics.
#[test]
fn buffer_ops_round_trip() {
    agree_with_llvm_mc(&[
        (
            "buffer_load_b32 v1, v2, s[4:7], null offen",
            "7c0005c40108804002000000",
        ),
        (
            "buffer_load_b32 v1, off, s[4:7], s2 offset:16",
            "020005c40108800000100000",
        ),
        (
            "buffer_load_b128 v[8:11], v4, s[0:3], s3 offen offset:4095",
            "03c005c40800804004ff0f00",
        ),
        (
            "buffer_load_b64 v[1:2], v[2:3], s[4:7], null idxen offen",
            "7c4005c4010880c002000000",
        ),
        (
            "buffer_load_b32 v1, v2, s[4:7], m0 idxen",
            "7d0005c40108808002000000",
        ),
        (
            "buffer_load_b32 v1, v2, s[4:7], vcc_lo offen",
            "6a0005c40108804002000000",
        ),
        (
            "buffer_store_b32 v1, v2, s[4:7], null offen th:TH_STORE_NT scope:SCOPE_SYS",
            "7c8006c401089c4002000000",
        ),
        (
            "buffer_store_b128 v[4:7], v2, s[8:11], s9 offen offset:8388607",
            "094007c40410804002ffff7f",
        ),
        (
            "buffer_load_d16_hi_b16 v1, v2, s[4:7], s70 offen offset:2",
            "46c008c40108804002020000",
        ),
        (
            "buffer_atomic_add_u32 v1, v2, s[4:7], m0 idxen th:TH_ATOMIC_RETURN",
            "7d400dc40108908002000000",
        ),
        (
            "buffer_atomic_add_u32 v1, v2, s[4:7], null offen",
            "7c400dc40108804002000000",
        ),
        (
            "buffer_atomic_cmpswap_b64 v[0:3], v4, s[4:7], null offen th:TH_ATOMIC_RETURN",
            "7c8010c40008904004000000",
        ),
    ]);
    let hooks = get(T).unwrap().hooks();
    for refused in [
        "buffer_load_b32 v1, v2, s[4:7], 0 offen",
        "buffer_load_b32 v1, v2, s[5:8], null offen",
        "buffer_load_b32 v1, v2, s[4:7], null offen offset:8388608",
        "buffer_load_b32 v1, v2, s[4:7], null",
        "buffer_load_b32 v1, off, s[4:7], null offen",
        "buffer_load_b32 v1, v2, s[4:7], exec_lo offen",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
}

/// VOP3 op_sel on the fp8 and bf8 packs and unpacks, only the bit each
/// reads.
#[test]
fn fp8_op_sel_round_trips() {
    agree_with_llvm_mc(&[
        (
            "v_cvt_pk_fp8_f32 v1, v2, v3 op_sel:[0,0,1]",
            "014069d702070200",
        ),
        (
            "v_cvt_pk_bf8_f32 v1, v2, v3 op_sel:[0,0,1]",
            "01406ad702070200",
        ),
        (
            "v_cvt_pk_fp8_f32 v1, v2, 1.0 op_sel:[0,0,1]",
            "014069d702e50100",
        ),
        (
            "v_cvt_pk_f32_fp8_e64 v[1:2], v3 op_sel:[1,0]",
            "0108eed503010000",
        ),
        (
            "v_cvt_pk_f32_bf8_e64 v[1:2], v3 op_sel:[1,0]",
            "0108efd503010000",
        ),
        (
            "v_cvt_pk_f32_fp8_e64 v[1:2], s3 op_sel:[1,0]",
            "0108eed503000000",
        ),
    ]);
    let hooks = get(T).unwrap().hooks();
    for refused in [
        "v_cvt_pk_fp8_f32 v1, v2, v3 op_sel:[1,0,0]",
        "v_cvt_pk_f32_fp8_e64 v[1:2], v3 op_sel:[0,1]",
        "v_add_f32_e64 v1, v2, v3 op_sel:[0,0,1]",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
}

/// Sign modifiers on VOP3 float sources, the moves and the select (RDNA4
/// ISA guide).
#[test]
fn vop3_signs_round_trip() {
    agree_with_llvm_mc(&[
        ("v_add_f32_e64 v1, -v2, |v3|", "010203d502070220"),
        ("v_max_num_f16_e64 v1, |v2|, v3", "010131d502070200"),
        ("v_max_num_f32_e64 v1, |v2|, |v3|", "010316d502070200"),
        ("v_fma_f32 v1, -|v2|, v3, -v4", "010113d6020712a4"),
        ("v_cmp_lt_f32_e64 s0, -v1, |v2|", "000211d401050220"),
        ("v_cndmask_b32_e64 v1, -v2, |v3|, s0", "010201d502070220"),
        ("v_add_f32 v1, -v2, v3", "010003d502070220"),
        (
            "v_div_scale_f32 v1, vcc_lo, -v2, v3, v4",
            "016afcd602071224",
        ),
        ("v_add_f32_e64 v1, |-1|, v2", "010103d5c1040200"),
        ("v_add_f32_e64 v1, -|0.5|, v2", "010103d5f0040220"),
        ("v_add_f32_e64 v1, -s2, v3", "010003d502060220"),
        (
            "v_add_f32_e64 v1, -|0x41200000|, v3",
            "010103d5ff06022000002041",
        ),
        (
            "v_fma_f64 v[0:1], -v[2:3], |v[4:5]|, v[6:7]",
            "000214d602091a24",
        ),
    ]);
    let hooks = get(T).unwrap().hooks();
    for refused in [
        "v_add_nc_u32_e64 v1, -v2, v3",
        "v_cvt_pk_fp8_f32 v1, -v2, v3",
        "v_ldexp_f32 v1, -v2, -v3",
        "v_add_f32_e64 -v1, v2, v3",
        "v_div_scale_f32 v1, vcc_lo, |v2|, v3, v4",
        "v_add_f32_e32 v1, -v2, v3",
        "v_cndmask_b32_e64 v1, v2, v3, -s0",
        "v_add_f32_dpp v1, -v2, v3 quad_perm:[0,1,2,3]",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
    // A negated constant without abs has no spelling (`-5` is the constant
    // -5): v_add_f32_e64 v1, neg(5), v2 is refused, not mis-decoded.
    let e = decode(T, &unhex("010003d585040220")).expect_err("neg on 5");
    assert_eq!(e.why, Why::Operand);
}

/// `s_setreg_imm32_b32 hwreg(...), imm32`.
#[test]
fn setreg_imm32_round_trips() {
    agree_with_llvm_mc(&[
        (
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), 2",
            "011880b902000000",
        ),
        (
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE), 0x12345678",
            "01f880b978563412",
        ),
        (
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_USER, 3, 5), 0",
            "d22080b900000000",
        ),
        (
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), -1",
            "011880b9ffffffff",
        ),
    ]);
    let hooks = get(T).unwrap().hooks();
    for refused in [
        "s_setreg_imm32_b32 0x10",
        "s_setreg_imm32_b32 hwreg(64), 0",
        "s_setreg_imm32_b32 hwreg(0x1a), 0",
        "s_setreg_imm32_b32 hwreg(26, 30, 4), 0",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
}

/// `s_getreg_b32` / `s_setreg_b32` with hwreg(...) (the hardware register
/// first in s_setreg_b32), and messages by their ids.
#[test]
fn hardware_registers_and_messages_round_trip() {
    agree_with_llvm_mc(&[
        (
            "s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES_LO)",
            "1df882b8",
        ),
        (
            "s_getreg_b32 s5, hwreg(HW_REG_WAVE_HW_ID1, 8, 4)",
            "171a85b8",
        ),
        ("s_setreg_b32 hwreg(HW_REG_WAVE_MODE), s2", "01f802b9"),
        ("s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), s7", "011807b9"),
        ("s_sendmsg_rtn_b64 s[0:1], 131", "834d80be"),
        ("s_sendmsg_rtn_b32 s4, 128", "804c84be"),
        ("s_sendmsg 3", "0300b6bf"),
    ]);
    let hooks = get(T).unwrap().hooks();
    for refused in [
        "s_setreg_b32 s2, hwreg(HW_REG_WAVE_MODE)",
        "s_sendmsg_rtn_b64 s[0:1], 256",
        "s_sendmsg_rtn_b64 s[0:1], s4",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
}

/// Constants as llvm-mc reads them: +0.0 and 1/(2π) are inline codes, and
/// a sign before a number is the number's (`-.5` is the inline -0.5).
#[test]
fn float_constants_are_inline_where_llvm_has_them() {
    agree_with_llvm_mc(&[
        ("v_mov_b32 v0, 0.0", "8002007e"),
        ("v_mov_b32 v0, -.5", "f102007e"),
        ("v_mov_b32 v0, 0.15915494", "f802007e"),
        ("s_add_co_u32 s0, s1, 0.0", "01800080"),
    ]);
    // -0.0 is no inline constant: its bits are a literal.
    agree_with_llvm_mc(&[("v_mov_b32 v0, -0.0", "ff02007e00000080")]);
}

/// Every temporal hint LLVM names for GFX12, BYPASS only at SCOPE_SYS
/// and LU or WB only below it; atomics with and without a return.
#[test]
fn cache_hints_round_trip() {
    agree_with_llvm_mc(&[
        (
            "global_atomic_add_u32 v1, v0, s[0:1]",
            "00400dee0000000001000000",
        ),
        (
            "global_atomic_add_u32 v1, v0, s[0:1] th:TH_ATOMIC_NT",
            "00400dee0000200001000000",
        ),
        (
            "global_atomic_add_u32 v2, v1, v0, s[0:1] th:TH_ATOMIC_NT_RETURN",
            "00400dee0200300001000000",
        ),
        (
            "global_atomic_add_u32 v1, v0, s[0:1] th:TH_ATOMIC_CASCADE_NT scope:SCOPE_DEV",
            "00400dee0000680001000000",
        ),
        (
            "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_BYPASS scope:SCOPE_SYS",
            "000005ee00003c0001000000",
        ),
        (
            "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_NT_RT",
            "000005ee0000400001000000",
        ),
        (
            "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_LU",
            "000005ee0000300001000000",
        ),
        (
            "global_store_b32 v1, v0, s[0:1] th:TH_STORE_WB",
            "008006ee0000300001000000",
        ),
        (
            "global_store_b32 v1, v0, s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS",
            "008006ee00003c0001000000",
        ),
        (
            "global_store_b32 v1, v0, s[0:1] th:TH_STORE_NT_RT",
            "008006ee0000400001000000",
        ),
    ]);
    let hooks = get(T).unwrap().hooks();
    for refused in [
        "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_BYPASS",
        "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_LU scope:SCOPE_SYS",
        "global_store_b32 v1, v0, s[0:1] th:TH_STORE_WB scope:SCOPE_SYS",
        "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_RT_WB",
        "global_store_b32 v1, v0, s[0:1] th:TH_STORE_LU",
        "global_load_b32 v0, v1, s[0:1] th:263",
        "global_atomic_add_u32 v1, v0, s[0:1] th:TH_ATOMIC_RETURN",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
}

/// Scalar ranges are aligned as sources too, as llvm-mc requires; labels
/// take no sign; ops LLVM spells otherwise are neither spelled nor encoded.
#[test]
fn malformed_operands_are_refused() {
    let t = get(T).unwrap();
    let hooks = t.hooks();
    for refused in [
        "s_mov_b64 s[0:1], s[1:2]",
        "s_load_b32 s0, s[1:2], 0x0",
        "s_and_b64 s[0:1], s[2:3], s[5:6]",
        "s_branch -.L1",
        "s_waitcnt 0",
    ] {
        assert!(hooks.encode(refused).is_err(), "{refused}");
    }
    assert_eq!(t.spell("s.waitcnt"), None);
    assert_eq!(t.spell("s.wait.loadcnt").as_deref(), Some("s_wait_loadcnt"));
}

/// A lane select takes an SGPR, M0 or an inline constant, never a literal.
#[test]
fn lane_selects_round_trip() {
    agree_with_llvm_mc(&[
        ("v_readlane_b32 s0, v2, 15", "000060d7021f0100"),
        ("v_readlane_b32 s0, v2, s5", "000060d7020b0000"),
        ("v_writelane_b32 v1, s0, 3", "010061d700060100"),
    ]);
    let hooks = get(T).unwrap().hooks();
    assert!(hooks.encode("v_readlane_b32 s0, v2, 100").is_err());
}
