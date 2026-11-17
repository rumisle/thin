//! gfx942: CDNA 3 as on the AMD Instinct MI300X, in wave64.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use thin_target::{Elem, Error, Form, MatrixForm, Summation, Target};

use crate::cdna::{
    self,
    decoder::Decoder,
    hazards::{cdna3_rule_count, Cdna3Model, CDNA3_WINDOW},
    Data, PhysicalFormCache, Spec, COMMON_HWREG, COMMON_SPECIALS,
};
use crate::decode::Instruction;

/// The target's name.
pub const NAME: &str = "gfx942";

/// gfx942's hazard model.
pub static HAZARD_MODEL: Cdna3Model = Cdna3Model;

static MATRIX: [&str; 46] = [
    "v_mfma_f32_16x16x16_bf16",
    "v_mfma_f32_16x16x16_f16",
    "v_mfma_f32_16x16x1_4b_f32",
    "v_mfma_f32_16x16x32_bf8_bf8",
    "v_mfma_f32_16x16x32_bf8_fp8",
    "v_mfma_f32_16x16x32_fp8_bf8",
    "v_mfma_f32_16x16x32_fp8_fp8",
    "v_mfma_f32_16x16x4_4b_bf16",
    "v_mfma_f32_16x16x4_4b_f16",
    "v_mfma_f32_16x16x4_f32",
    "v_mfma_f32_16x16x8_xf32",
    "v_mfma_f32_32x32x16_bf8_bf8",
    "v_mfma_f32_32x32x16_bf8_fp8",
    "v_mfma_f32_32x32x16_fp8_bf8",
    "v_mfma_f32_32x32x16_fp8_fp8",
    "v_mfma_f32_32x32x1_2b_f32",
    "v_mfma_f32_32x32x2_f32",
    "v_mfma_f32_32x32x4_2b_bf16",
    "v_mfma_f32_32x32x4_2b_f16",
    "v_mfma_f32_32x32x4_xf32",
    "v_mfma_f32_32x32x8_bf16",
    "v_mfma_f32_32x32x8_f16",
    "v_mfma_f32_4x4x1_16b_f32",
    "v_mfma_f32_4x4x4_16b_bf16",
    "v_mfma_f32_4x4x4_16b_f16",
    "v_mfma_f64_16x16x4_f64",
    "v_mfma_f64_4x4x4_4b_f64",
    "v_mfma_i32_16x16x32_i8",
    "v_mfma_i32_16x16x4_4b_i8",
    "v_mfma_i32_32x32x16_i8",
    "v_mfma_i32_32x32x4_2b_i8",
    "v_mfma_i32_4x4x4_16b_i8",
    "v_smfmac_f32_16x16x32_bf16",
    "v_smfmac_f32_16x16x32_f16",
    "v_smfmac_f32_16x16x64_bf8_bf8",
    "v_smfmac_f32_16x16x64_bf8_fp8",
    "v_smfmac_f32_16x16x64_fp8_bf8",
    "v_smfmac_f32_16x16x64_fp8_fp8",
    "v_smfmac_f32_32x32x16_bf16",
    "v_smfmac_f32_32x32x16_f16",
    "v_smfmac_f32_32x32x32_bf8_bf8",
    "v_smfmac_f32_32x32x32_bf8_fp8",
    "v_smfmac_f32_32x32x32_fp8_bf8",
    "v_smfmac_f32_32x32x32_fp8_fp8",
    "v_smfmac_i32_16x16x64_i8",
    "v_smfmac_i32_32x32x32_i8",
];

/// Authoritative CDNA 3 matrix form table covering all 32 dense `v_mfma_*` and
/// 14 4:2 sparse `v_smfmac_*` instructions (`passes = cycles / 4`).
fn mfma_form(mnemonic: &str) -> Option<MatrixForm> {
    if let Some(mf) = cdna::common_mfma_form(mnemonic, true, false) {
        return Some(mf);
    }
    let (m, n, k, passes) = match mnemonic {
        "v_mfma_f32_16x16x8_xf32" => (16, 16, 8, 4),
        "v_mfma_f32_32x32x4_xf32" => (32, 32, 4, 8),
        _ => return None,
    };
    Some(MatrixForm {
        a: Elem::Xf32,
        b: Elem::Xf32,
        d: Elem::F32,
        m,
        n,
        k,
        blocks: 1,
        wave: 64,
        passes,
        sparse: false,
        cbsz_abid: false,
        blgp: false,
        sum: Summation::Grouped {
            group_half: 8,
            group_byte: 16,
            fraction: 24,
        },
        provisional: false,
    })
}

/// CDNA 3 LDS bank model (32 physical banks x 4 bytes/bank = 128 B/cycle):
/// - `1 | 2 | 4` bytes (`b8`/`b16`/`b32`): 2 half-wave phases (`lanes 0..31`, `32..63`).
/// - `8` bytes (`b64`): 4 quarter-wave phases of 16 lanes (`0..15`, `16..31`, `32..47`, `48..63`).
/// - `12` bytes (`b96`): 8 eighth-wave phases of 8 lanes (`l / 8`).
/// - `16` bytes (`b128`):
///   - writes (`ds_write_b128`): 8 sequential 8-lane phases (`l / 8`).
///   - reads (`ds_read_b128`): CDNA 3 crossbar-interleaved 8-lane quad pairs per half-wave:
///     `(0:3 + 20:23)`, `(4:7 + 16:19)`, `(8:11 + 28:31)`, `(12:15 + 24:27)` (and `+32`).
fn lds_banks(read: bool, bytes: u32) -> Option<(u64, Vec<u32>)> {
    let phase: Vec<u32> = match bytes {
        1 | 2 | 4 => (0..64).map(|l| l / 32).collect(),
        8 => (0..64).map(|l| l / 16).collect(),
        12 => (0..64).map(|l| l / 8).collect(),
        16 if !read => (0..64).map(|l| l / 8).collect(),
        16 => (0..64)
            .map(|l| {
                let half = l / 32;
                let q = (l % 32) / 4;
                let sub = if q < 4 { q } else { q ^ 5 };
                half * 4 + sub
            })
            .collect(),
        _ => return None,
    };
    Some((32, phase))
}

fn data() -> Result<&'static Data, Error> {
    static DATA: OnceLock<Result<Data, Error>> = OnceLock::new();
    let d = DATA.get_or_init(|| {
        cdna::load_data(
            include_str!("../../data/gfx942/isa_tables.json"),
            include_str!("../../data/gfx942/operand_table.json"),
        )
    });
    d.as_ref().map_err(Clone::clone)
}

fn signs(mnemonic: &str) -> Vec<usize> {
    cdna::signs_for(data(), mnemonic)
}

static SPEC: Spec = Spec {
    name: NAME,
    e_flags: 0x54C,
    xcds: 8,
    cus: 304,
    lds_max: 65536,
    fnuz_fp8: true,
    wide_lds_dma: false,
    specials: &COMMON_SPECIALS,
    hwreg: &COMMON_HWREG,
    spelled_otherwise: &[],
    extra_spellings: &[],
    matrix: &MATRIX,
    matrix_form: mfma_form,
    fields: cdna::fields,
    signs,
    widths: cdna::no_widths,
    lds_banks,
    data,
    hazard_rules_len: cdna3_rule_count,
    hazard_window: CDNA3_WINDOW,
};

/// Builds gfx942 from its checked-in tables.
pub fn build() -> Result<Target, Error> {
    cdna::build(&SPEC)
}

pub(crate) fn decoder() -> Result<&'static Decoder, Error> {
    static DECODER: OnceLock<Result<Decoder, Error>> = OnceLock::new();
    let built = DECODER.get_or_init(|| Decoder::build(&SPEC, data()?));
    built.as_ref().map_err(Clone::clone)
}

pub(crate) fn physical_form(i: &Instruction) -> Result<&'static Form, Error> {
    static FORMS: OnceLock<PhysicalFormCache> = OnceLock::new();
    let cache = FORMS.get_or_init(|| Mutex::new(HashMap::new()));
    cdna::decoder::physical_form(NAME, cache, i)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use thin_target::Encoded;

    use super::*;
    use crate::cdna::COMMON_SPELLINGS;

    fn target() -> &'static Target {
        crate::get(NAME).expect("gfx942 builds")
    }

    #[test]
    fn every_spelling_round_trips() {
        for (op, mnemonic) in COMMON_SPELLINGS {
            assert_eq!(target().spell(op).as_deref(), Some(mnemonic), "{op}");
            assert_eq!(target().canon(mnemonic), op, "{mnemonic}");
        }
    }

    #[test]
    fn forms_have_unique_keys_and_ops_are_sorted() {
        let forms = target().forms();
        let keys: HashSet<_> = forms
            .iter()
            .map(|f| (&f.op, f.arity, f.returning))
            .collect();
        assert_eq!(keys.len(), forms.len());
        assert_eq!(forms.len(), 1944);
    }

    #[test]
    fn representative_instructions_encode_and_decode_round_trip() {
        let lines = [
            "s_load_dwordx4 s[4:7], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(15)",
            "v_lshl_add_u32 v1, s2, 6, v0",
            "v_cmp_gt_u32_e32 vcc, s3, v1",
            "s_and_saveexec_b64 s[8:9], vcc",
            "global_load_dword v2, v[0:1], off",
            "global_load_dword v3, v0, s[4:5] offset:-16 sc0",
            "global_store_dword v0, v2, s[6:7] sc1",
            "global_atomic_add v4, v0, v2, s[6:7] sc0",
            "global_atomic_add v0, v2, s[6:7]",
            "buffer_load_dword v5, v0, s[8:11], 0 offen offset:32",
            "buffer_load_dword off, s[8:11], s12 offset:16 lds",
            "buffer_wbl2 sc1",
            "buffer_inv sc0 sc1",
            "ds_write_b32 v0, v2 offset:256",
            "ds_read_b32 v3, v0 offset:256",
            "ds_read2_b64 v[4:7], v0 offset0:1 offset1:2",
            "v_accvgpr_write_b32 a0, v0",
            "v_accvgpr_read_b32 v0, a0",
            "v_accvgpr_mov_b32 a1, a0",
            "v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]",
            "v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], v[2:3], 0 cbsz:1 abid:2 blgp:3",
            "v_smfmac_f32_16x16x32_f16 a[0:3], v[0:1], v[2:5], v6",
            "v_pk_add_f16 v0, v1, v2 op_sel_hi:[1,0]",
            "v_fma_mix_f32 v0, -v1, |v2|, -|v3| op_sel_hi:[1,0,1] clamp",
            "v_add_f32_dpp v0, -v1, |v2| row_newbcast:3 row_mask:15 bank_mask:15 bound_ctrl:1",
            "s_memtime s[10:11]",
            "s_memrealtime s[12:13]",
            "s_barrier",
            "s_endpgm",
        ];
        let d = decoder().expect("gfx942 decoder builds");
        for line in lines {
            let Ok(Encoded::Bytes(bytes)) = target().hooks().encode(line) else {
                panic!("{line} failed to encode");
            };
            let decoded = d.decode(&bytes).unwrap_or_else(|e| panic!("{line}: {e}"));
            assert_eq!(decoded.len(), 1, "{line}");
            let reencoded =
                crate::encode_one(NAME, &decoded[0]).unwrap_or_else(|e| panic!("{line}: {e}"));
            assert_eq!(reencoded, bytes, "{line} -> {}", decoded[0]);
            let pform =
                physical_form(&decoded[0]).unwrap_or_else(|e| panic!("physical_form({line}): {e}"));
            assert_eq!(pform.arity, decoded[0].operands().len(), "{line}");
        }
    }
}
