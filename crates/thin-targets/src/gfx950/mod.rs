//! gfx950: CDNA 4 as on the AMD Instinct MI350X/MI355X, in wave64.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use thin_target::{Error, Form, MatrixForm, Target};

use crate::cdna::{
    self,
    decoder::Decoder,
    hazards::{cdna3_rule_count, Cdna3Model, CDNA3_WINDOW},
    Data, PhysicalFormCache, Spec, COMMON_HWREG, COMMON_SPECIALS,
};
use crate::decode::Instruction;

/// The target's name.
pub const NAME: &str = "gfx950";

/// gfx950's hazard model (upgraded to `Cdna4Model` in the hazards commit).
pub static HAZARD_MODEL: Cdna3Model = Cdna3Model;

static EXTRA_SPELLINGS: [(&str, &str); 6] = [
    ("global.load.lds.b96", "global_load_lds_dwordx3"),
    ("global.load.lds.b128", "global_load_lds_dwordx4"),
    ("ds.load.b64.tr.b4", "ds_read_b64_tr_b4"),
    ("ds.load.b64.tr.b8", "ds_read_b64_tr_b8"),
    ("ds.load.b64.tr.b16", "ds_read_b64_tr_b16"),
    ("ds.load.b96.tr.b6", "ds_read_b96_tr_b6"),
];

static MATRIX: [&str; 68] = [
    "v_mfma_f32_16x16x128_f8f6f4",
    "v_mfma_f32_16x16x16_bf16",
    "v_mfma_f32_16x16x16_f16",
    "v_mfma_f32_16x16x1_4b_f32",
    "v_mfma_f32_16x16x32_bf16",
    "v_mfma_f32_16x16x32_bf8_bf8",
    "v_mfma_f32_16x16x32_bf8_fp8",
    "v_mfma_f32_16x16x32_f16",
    "v_mfma_f32_16x16x32_fp8_bf8",
    "v_mfma_f32_16x16x32_fp8_fp8",
    "v_mfma_f32_16x16x4_4b_bf16",
    "v_mfma_f32_16x16x4_4b_f16",
    "v_mfma_f32_16x16x4_f32",
    "v_mfma_f32_32x32x16_bf16",
    "v_mfma_f32_32x32x16_bf8_bf8",
    "v_mfma_f32_32x32x16_bf8_fp8",
    "v_mfma_f32_32x32x16_f16",
    "v_mfma_f32_32x32x16_fp8_bf8",
    "v_mfma_f32_32x32x16_fp8_fp8",
    "v_mfma_f32_32x32x1_2b_f32",
    "v_mfma_f32_32x32x2_f32",
    "v_mfma_f32_32x32x4_2b_bf16",
    "v_mfma_f32_32x32x4_2b_f16",
    "v_mfma_f32_32x32x64_f8f6f4",
    "v_mfma_f32_32x32x8_bf16",
    "v_mfma_f32_32x32x8_f16",
    "v_mfma_f32_4x4x1_16b_f32",
    "v_mfma_f32_4x4x4_16b_bf16",
    "v_mfma_f32_4x4x4_16b_f16",
    "v_mfma_f64_16x16x4_f64",
    "v_mfma_f64_4x4x4_4b_f64",
    "v_mfma_i32_16x16x32_i8",
    "v_mfma_i32_16x16x4_4b_i8",
    "v_mfma_i32_16x16x64_i8",
    "v_mfma_i32_32x32x16_i8",
    "v_mfma_i32_32x32x32_i8",
    "v_mfma_i32_32x32x4_2b_i8",
    "v_mfma_i32_4x4x4_16b_i8",
    "v_mfma_scale_f32_16x16x128_f8f6f4",
    "v_mfma_scale_f32_32x32x64_f8f6f4",
    "v_smfmac_f32_16x16x128_bf8_bf8",
    "v_smfmac_f32_16x16x128_bf8_fp8",
    "v_smfmac_f32_16x16x128_fp8_bf8",
    "v_smfmac_f32_16x16x128_fp8_fp8",
    "v_smfmac_f32_16x16x32_bf16",
    "v_smfmac_f32_16x16x32_f16",
    "v_smfmac_f32_16x16x64_bf16",
    "v_smfmac_f32_16x16x64_bf8_bf8",
    "v_smfmac_f32_16x16x64_bf8_fp8",
    "v_smfmac_f32_16x16x64_f16",
    "v_smfmac_f32_16x16x64_fp8_bf8",
    "v_smfmac_f32_16x16x64_fp8_fp8",
    "v_smfmac_f32_32x32x16_bf16",
    "v_smfmac_f32_32x32x16_f16",
    "v_smfmac_f32_32x32x32_bf16",
    "v_smfmac_f32_32x32x32_bf8_bf8",
    "v_smfmac_f32_32x32x32_bf8_fp8",
    "v_smfmac_f32_32x32x32_f16",
    "v_smfmac_f32_32x32x32_fp8_bf8",
    "v_smfmac_f32_32x32x32_fp8_fp8",
    "v_smfmac_f32_32x32x64_bf8_bf8",
    "v_smfmac_f32_32x32x64_bf8_fp8",
    "v_smfmac_f32_32x32x64_fp8_bf8",
    "v_smfmac_f32_32x32x64_fp8_fp8",
    "v_smfmac_i32_16x16x128_i8",
    "v_smfmac_i32_16x16x64_i8",
    "v_smfmac_i32_32x32x32_i8",
    "v_smfmac_i32_32x32x64_i8",
];

/// CDNA 4 matrix form table (`provisional: true`, derived from the CDNA 4 ISA manual
/// without MI350X/MI355X silicon confirmation; OCP `e4m3fn`/`e5m2` FP8/BF8).
fn mfma_form(mnemonic: &str) -> Option<MatrixForm> {
    cdna::common_mfma_form(mnemonic, false, true)
}

/// Variable tuple widths on `*_f8f6f4` MFMA `src0` (slot 1, selected by `cbsz`)
/// and `src1` (slot 2, selected by `blgp`): 4 dwords (FP4), 6 dwords (FP6/BF6),
/// or 8 dwords (FP8/BF8).
fn widths(mnemonic: &str, slot: usize) -> Vec<u32> {
    if mnemonic.ends_with("_f8f6f4") && (slot == 1 || slot == 2) {
        vec![4, 6, 8]
    } else {
        Vec::new()
    }
}

/// Derived CDNA 4 LDS bank model (64 physical banks x 4 bytes/bank = 256 B/cycle,
/// 160 KiB = 64 banks x 640 dwords):
/// - `1 | 2 | 4` bytes (`b8`/`b16`/`b32`): 1 full-wave phase (`0..63`).
/// - `8` bytes (`b64`): 2 half-wave phases of 32 lanes (`l / 32`).
/// - `12 | 16` bytes (`b96`/`b128`): 4 quarter-wave phases of 16 lanes (`l / 16`).
fn lds_banks(_read: bool, bytes: u32) -> Option<(u64, Vec<u32>)> {
    let phase: Vec<u32> = match bytes {
        1 | 2 | 4 => vec![0; 64],
        8 => (0..64).map(|l| l / 32).collect(),
        12 | 16 => (0..64).map(|l| l / 16).collect(),
        _ => return None,
    };
    Some((64, phase))
}

fn data() -> Result<&'static Data, Error> {
    static DATA: OnceLock<Result<Data, Error>> = OnceLock::new();
    let d = DATA.get_or_init(|| {
        cdna::load_data(
            include_str!("../../data/gfx950/isa_tables.json"),
            include_str!("../../data/gfx950/operand_table.json"),
        )
    });
    d.as_ref().map_err(Clone::clone)
}

fn signs(mnemonic: &str) -> Vec<usize> {
    cdna::signs_for(data(), mnemonic)
}

static SPEC: Spec = Spec {
    name: NAME,
    // EF_AMDGPU_MACH_AMDGCN_GFX950 (0x4F) | SRAMECC_ANY_V4 (0x400) | XNACK_ANY_V4 (0x100)
    e_flags: 0x54F,
    xcds: 8,
    cus: 256,
    lds_max: 163_840,
    fnuz_fp8: false,
    wide_lds_dma: true,
    specials: &COMMON_SPECIALS,
    hwreg: &COMMON_HWREG,
    spelled_otherwise: &[],
    extra_spellings: &EXTRA_SPELLINGS,
    matrix: &MATRIX,
    matrix_form: mfma_form,
    fields: cdna::fields,
    signs,
    widths,
    lds_banks,
    data,
    hazard_rules_len: cdna3_rule_count,
    hazard_window: CDNA3_WINDOW,
};

/// Builds gfx950 from its checked-in tables.
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
        crate::get(NAME).expect("gfx950 builds")
    }

    #[test]
    fn every_spelling_round_trips() {
        for &(op, mnemonic) in COMMON_SPELLINGS.iter().chain(EXTRA_SPELLINGS.iter()) {
            assert_eq!(target().spell(op).as_deref(), Some(mnemonic), "{op}");
            assert_eq!(target().canon(mnemonic), op, "{mnemonic}");
        }
    }

    #[test]
    fn forms_have_unique_keys() {
        let forms = target().forms();
        let keys: HashSet<_> = forms
            .iter()
            .map(|f| (&f.op, f.arity, f.returning))
            .collect();
        assert_eq!(keys.len(), forms.len());
        assert_eq!(forms.len(), 2055);
    }

    #[test]
    fn representative_cdna4_instructions_encode_and_decode_round_trip() {
        let lines = [
            "s_load_dwordx4 s[4:7], s[0:1], 0x0",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "global_load_lds_dwordx3 v[0:1], off offset:16",
            "global_load_lds_dwordx4 v0, s[4:5] offset:-32",
            "buffer_load_dwordx3 v0, s[8:11], 0 offen offset:16 lds",
            "buffer_load_dwordx4 v0, s[8:11], s12 offen offset:32 lds",
            "buffer_atomic_pk_add_bf16 v0, v1, s[8:11], 0 offen sc0",
            "ds_read_b64_tr_b4 v[0:1], v2 offset:64",
            "ds_read_b64_tr_b8 v[2:3], v4 offset:128",
            "ds_read_b64_tr_b16 a[0:1], v5",
            "ds_read_b96_tr_b6 v[1:3], v6 offset:256",
            "ds_read_b96_tr_b6 a[1:3], v6",
            "v_bitop3_b32 v0, v1, v2, v3 bitop3:0xa8",
            "v_bitop3_b16 v0, v1, v2, v3 bitop3:0x41 op_sel:[1,0,1,0]",
            "v_permlane16_swap_b32_e32 v0, v1",
            "v_permlane16_swap_b32_e64 v0, v1 bound_ctrl:1 fi:1",
            "v_permlane32_swap_b32_e64 v2, v3 fi:1",
            "v_cvt_scalef32_pk_fp4_f32 v0, -v1, |v2|, v3 op_sel:[1,0,0,1]",
            "v_cvt_scalef32_pk_fp4_f16 v0, -v1, v2 op_sel:[0,1,0,1]",
            "v_cvt_scalef32_2xpk16_fp6_f32 v[0:5], v[6:21], v[22:37], v38",
            "v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]",
            "v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[6:9], a[0:3] cbsz:2 blgp:4",
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17",
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:5], v[8:11], 0, v16, v17 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:3 blgp:4",
            "v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], v[0:5], a[0:15], v10, v11 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:1 blgp:2",
            "s_barrier",
            "s_endpgm",
        ];
        let d = decoder().expect("gfx950 decoder builds");
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
