//! gfx1201: RDNA4 as on the Radeon RX 9070 XT, in wave32.

mod code_object;
pub(crate) mod decoder;
mod encoder;
pub(crate) mod hazards;

use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

use thin_target::{
    Block, Chip, ClockRead, Counter, Definition, Elem, Encoded, Error, Field, FieldKind, File,
    Flags, Footer, FormRules, Hazards, Hooks, LibHooks, MatrixForm, Mem, Operands, Resources,
    Summation, Syntax, Target, WaitInsn, WaitText,
};

use crate::tables::{self, IsaTables};
use thin_target::Row;

/// The target's name.
pub const NAME: &str = "gfx1201";

/// The scalar source-space names: (name, number, dwords).
pub static SPECIALS: [(&str, u32, u32); 13] = [
    ("vcc", 106, 2),
    ("vcc_lo", 106, 1),
    ("vcc_hi", 107, 1),
    ("ttmp7", 115, 1),
    ("ttmp9", 117, 1),
    ("null", 124, 1),
    ("m0", 125, 1),
    ("exec", 126, 2),
    ("exec_lo", 126, 1),
    ("exec_hi", 127, 1),
    ("scc", 253, 1),
    ("vccz", 251, 1),
    ("execz", 252, 1),
];

/// The hardware registers `hwreg(...)` names, with their ids: the
/// OPR_HWREG ID values of AMD's RDNA4 ISA XML (its names upper-cased),
/// with IB_STS2 (28), which only the RDNA4 ISA guide's table lists ("Wave
/// State Registers"). S_GETREG / S_SETREG take #SIMM16 =
/// {size-1[4:0], offset[4:0], id[5:0]}. No other id is named (the
/// AMD documents name none); `hwreg(26, 0, 2)` writes any id by number.
pub static HWREG: [(&str, u32); 20] = [
    ("HW_REG_WAVE_MODE", 1),
    ("HW_REG_WAVE_STATUS", 2),
    ("HW_REG_WAVE_STATE_PRIV", 4),
    ("HW_REG_WAVE_GPR_ALLOC", 5),
    ("HW_REG_WAVE_LDS_ALLOC", 6),
    ("HW_REG_PERF_SNAPSHOT_DATA", 10),
    ("HW_REG_PERF_SNAPSHOT_PC_LO", 11),
    ("HW_REG_PERF_SNAPSHOT_PC_HI", 12),
    ("HW_REG_PERF_SNAPSHOT_DATA1", 15),
    ("HW_REG_PERF_SNAPSHOT_DATA2", 16),
    ("HW_REG_WAVE_EXCP_FLAG_PRIV", 17),
    ("HW_REG_WAVE_EXCP_FLAG_USER", 18),
    ("HW_REG_WAVE_TRAP_CTRL", 19),
    ("HW_REG_WAVE_SCRATCH_BASE_LO", 20),
    ("HW_REG_WAVE_SCRATCH_BASE_HI", 21),
    ("HW_REG_WAVE_HW_ID1", 23),
    ("HW_REG_WAVE_HW_ID2", 24),
    ("HW_REG_SHADER_CYCLES_LO", 29),
    ("HW_REG_SHADER_CYCLES_HI", 30),
    ("HW_REG_IB_STS2", 28),
];

/// The returning message that reads the constant 100 MHz clock
/// (`sendmsg(MSG_RTN_GET_REALTIME)`).
pub const MSG_RTN_GET_REALTIME: u32 = 0x83;

/// The `s_wait_alu` dependency counters: (field, shift, bits), in the
/// order the assembler prints them.
pub static DEPCTR: [(&str, u32, u32); 7] = [
    ("depctr_hold_cnt", 7, 1),
    ("depctr_sa_sdst", 0, 1),
    ("depctr_va_vdst", 12, 4),
    ("depctr_va_sdst", 9, 3),
    ("depctr_va_ssrc", 8, 1),
    ("depctr_va_vcc", 1, 1),
    ("depctr_vm_vsrc", 2, 3),
];

/// The packed `s_wait_alu` value that waits for nothing.
pub const DEPCTR_NONE: u32 = 0xFF9F;

static WAIT_FIELDS: [[&str; 1]; 4] = [["loadcnt"], ["storecnt"], ["dscnt"], ["kmcnt"]];
static DEPCTR_FIELDS: [&str; 7] = [
    "depctr_hold_cnt",
    "depctr_sa_sdst",
    "depctr_va_vdst",
    "depctr_va_sdst",
    "depctr_va_ssrc",
    "depctr_va_vcc",
    "depctr_vm_vsrc",
];

fn waits() -> Vec<WaitInsn> {
    let bare = |mnemonic, fields| WaitInsn {
        mnemonic,
        fields,
        text: WaitText::Bare,
    };
    vec![
        bare("s_wait_loadcnt", &WAIT_FIELDS[0]),
        bare("s_wait_storecnt", &WAIT_FIELDS[1]),
        bare("s_wait_dscnt", &WAIT_FIELDS[2]),
        bare("s_wait_kmcnt", &WAIT_FIELDS[3]),
        WaitInsn {
            mnemonic: "s_wait_alu",
            fields: &DEPCTR_FIELDS,
            text: WaitText::Named,
        },
    ]
}

fn counters() -> Vec<Counter> {
    // Register results land at the retiring wait; the store counter returns none.
    let counter = |name, max, field, units, lands| Counter {
        name,
        max,
        field,
        units,
        lands,
    };
    vec![
        counter("load", 63, "loadcnt", Flags::VMEM, thin_target::Lands::Wait),
        counter(
            "store",
            63,
            "storecnt",
            Flags::VMEM,
            thin_target::Lands::Issue,
        ),
        counter("ds", 63, "dscnt", Flags::DS, thin_target::Lands::Wait),
        counter("km", 31, "kmcnt", Flags::SMEM, thin_target::Lands::Wait),
    ]
}

/// The spellings where the default rule does not apply.
pub static SPELLINGS: [(&str, &str); 21] = [
    ("v.add.u32", "v_add_nc_u32"),
    ("v.sub.u32", "v_sub_nc_u32"),
    ("v.subrev.u32", "v_subrev_nc_u32"),
    ("v.addc.co.u32", "v_add_co_ci_u32"),
    ("v.subb.co.u32", "v_sub_co_ci_u32"),
    ("v.subbrev.co.u32", "v_subrev_co_ci_u32"),
    ("v.max.f32", "v_max_num_f32"),
    ("v.min.f32", "v_min_num_f32"),
    ("v.max3.f32", "v_max3_num_f32"),
    ("v.min3.f32", "v_min3_num_f32"),
    ("v.med3.f32", "v_med3_num_f32"),
    ("s.add.u32", "s_add_co_u32"),
    ("s.addc.u32", "s_add_co_ci_u32"),
    ("s.sub.u32", "s_sub_co_u32"),
    ("s.subb.u32", "s_sub_co_ci_u32"),
    ("s.andn2.b32", "s_and_not1_b32"),
    ("s.andn2.b64", "s_and_not1_b64"),
    ("s.orn2.b32", "s_or_not1_b32"),
    ("s.orn2.b64", "s_or_not1_b64"),
    ("s.ff1.i32.b32", "s_ctz_i32_b32"),
    ("s.ff1.i32.b64", "s_ctz_i32_b64"),
];

/// The `th` choices of the forms' fields: the names of [`TH_SYMBOLS`].
pub static TH: [&str; 23] = [
    "TH_LOAD_RT",
    "TH_LOAD_NT",
    "TH_LOAD_HT",
    "TH_LOAD_LU",
    "TH_LOAD_NT_RT",
    "TH_LOAD_RT_NT",
    "TH_LOAD_NT_HT",
    "TH_LOAD_BYPASS",
    "TH_STORE_RT",
    "TH_STORE_NT",
    "TH_STORE_HT",
    "TH_STORE_WB",
    "TH_STORE_NT_RT",
    "TH_STORE_RT_NT",
    "TH_STORE_NT_HT",
    "TH_STORE_NT_WB",
    "TH_STORE_BYPASS",
    "TH_ATOMIC_RT",
    "TH_ATOMIC_RETURN",
    "TH_ATOMIC_NT",
    "TH_ATOMIC_NT_RETURN",
    "TH_ATOMIC_CASCADE_RT",
    "TH_ATOMIC_CASCADE_NT",
];

/// The `scope` choices.
pub static SCOPE: [&str; 4] = ["SCOPE_CU", "SCOPE_SE", "SCOPE_DEV", "SCOPE_SYS"];

/// The temporal hints a line may name, as LLVM spells them for GFX12.
///
/// A value is the op kind in bits 8-9 (load, store, atomic) and the 3-bit
/// TH field below. Loads and stores share the field's codes; an atomic's
/// bits are RETURN (0), NT (1) and CASCADE (2). Code 3 of a load or store
/// is LU or WB at a narrower scope and BYPASS at SCOPE_SYS; BYPASS carries
/// [`TH_BYPASS`] so the encoder can require that scope. Load code 7 is
/// reserved.
pub(crate) static TH_SYMBOLS: [(&str, i128); 23] = [
    ("TH_LOAD_RT", 0x100),
    ("TH_LOAD_NT", 0x101),
    ("TH_LOAD_HT", 0x102),
    ("TH_LOAD_LU", 0x103),
    ("TH_LOAD_NT_RT", 0x104),
    ("TH_LOAD_RT_NT", 0x105),
    ("TH_LOAD_NT_HT", 0x106),
    ("TH_LOAD_BYPASS", 0x103 | TH_BYPASS),
    ("TH_STORE_RT", 0x200),
    ("TH_STORE_NT", 0x201),
    ("TH_STORE_HT", 0x202),
    ("TH_STORE_WB", 0x203),
    ("TH_STORE_NT_RT", 0x204),
    ("TH_STORE_RT_NT", 0x205),
    ("TH_STORE_NT_HT", 0x206),
    ("TH_STORE_NT_WB", 0x207),
    ("TH_STORE_BYPASS", 0x203 | TH_BYPASS),
    ("TH_ATOMIC_RT", 0x300),
    ("TH_ATOMIC_RETURN", 0x301),
    ("TH_ATOMIC_NT", 0x302),
    ("TH_ATOMIC_NT_RETURN", 0x303),
    ("TH_ATOMIC_CASCADE_RT", 0x304),
    ("TH_ATOMIC_CASCADE_NT", 0x306),
];

/// The mark of a BYPASS hint in a [`TH_SYMBOLS`] value: code 3, at SCOPE_SYS.
pub(crate) const TH_BYPASS: i128 = 0x8;

fn counter(flags: Flags, name: &str, writes: bool) -> Option<usize> {
    // A returning message (s_sendmsg_rtn_*) counts on KMcnt as a scalar
    // load does: its result lands when the count reaches it.
    if flags.has(Flags::SMEM) || name.starts_with("s_sendmsg_rtn") {
        Some(3)
    } else if flags.has(Flags::DS) {
        Some(2)
    } else if !flags.has(Flags::VMEM) {
        None
    } else if flags.has(Flags::ATOMIC) {
        Some(if writes { 0 } else { 1 })
    } else if flags.has(Flags::STORE) {
        Some(1)
    } else {
        Some(0)
    }
}

fn cache_fields() -> Result<Vec<Field>, Error> {
    Ok(vec![
        Field::new("th", FieldKind::Choice, 0, false)?.choices(&TH),
        Field::new("scope", FieldKind::Choice, 0, false)?.choices(&SCOPE),
    ])
}

fn fields(spelling: &str, encoding: &str, nsrc: usize) -> Result<Vec<Field>, Error> {
    let wait = waits().into_iter().find(|w| w.mnemonic == spelling);
    if let (Some(wait), 0) = (wait, nsrc) {
        return wait
            .fields
            .iter()
            .map(|&f| {
                let bits = DEPCTR.iter().find(|d| d.0 == f).map_or(16, |d| d.2);
                Field::new(f, FieldKind::Int, bits, false)
            })
            .collect();
    }
    let int = |name, bits, signed| Field::new(name, FieldKind::Int, bits, signed);
    let flag = |name| Field::new(name, FieldKind::Flag, 0, false);
    let bits = |name| Field::new(name, FieldKind::Bits, nsrc as u32, false);
    Ok(match encoding {
        "ENC_VGLOBAL" | "ENC_VFLAT" | "ENC_VSCRATCH" => {
            [vec![int("offset", 24, true)?], cache_fields()?].concat()
        }
        "ENC_VBUFFER" => [
            vec![flag("offen")?, flag("idxen")?, int("offset", 23, false)?],
            cache_fields()?,
        ]
        .concat(),
        "ENC_SMEM" => cache_fields()?,
        "ENC_VDS" if spelling.contains("2addr") => {
            vec![int("offset0", 8, false)?, int("offset1", 8, false)?]
        }
        "ENC_VDS" => vec![int("offset", 16, false)?],
        "ENC_VOP3P" => vec![
            bits("op_sel")?,
            bits("op_sel_hi")?,
            bits("neg_lo")?,
            bits("neg_hi")?,
            flag("clamp")?,
        ],
        // The fp8 / bf8 packs and unpacks choose a word or byte with
        // op_sel: one bit per source, then the destination's.
        "ENC_VOP3" if encoder::opsel_bits(spelling) != 0 => vec![
            Field::new("op_sel", FieldKind::Bits, nsrc as u32 + 1, false)?,
            flag("clamp")?,
        ],
        "ENC_VOP3" | "VOP3_SDST_ENC" => vec![flag("clamp")?],
        "VOP1_VOP_DPP16" | "VOP2_VOP_DPP16" => vec![
            Field::new("dpp", FieldKind::Dpp, 0, false)?,
            int("row_mask", 4, false)?,
            int("bank_mask", 4, false)?,
            flag("bound_ctrl")?,
        ],
        _ => Vec::new(),
    })
}

/// The slots of a form that take sign modifiers (neg, abs): a VOP3
/// source the encoder signs (`encoder::signs_of`). Slots follow the op's
/// operand fields in order, so slot k is the k-th field present.
fn signs(mnemonic: &str) -> Vec<usize> {
    let Ok(Data { isa, .. }) = data() else {
        return Vec::new();
    };
    let Some(op) = isa.ops.get(mnemonic) else {
        return Vec::new();
    };
    if !op.class.starts_with("VOP3") || op.class == "VOP3P" {
        return Vec::new();
    }
    let present = ["vdst", "sdst", "src0", "src1", "src2"]
        .iter()
        .filter_map(|n| op.operand(n));
    (present.enumerate())
        .filter(|(_, f)| f.field.starts_with("src") && encoder::signs_of(mnemonic, &op.class, f).0)
        .map(|(k, _)| k)
        .collect()
}

fn no_widths(_: &str, _: usize) -> Vec<u32> {
    Vec::new()
}

static SMEM: [&str; 6] = [
    "s_load",
    "s_buffer_load",
    "s_prefetch",
    "s_buffer_prefetch",
    "s_dcache",
    "s_atc_probe",
];

/// gfx1201's form rules.
fn form_rules() -> FormRules {
    FormRules {
        matrix: &[],
        smem: &SMEM,
        trans: &[
            "v_exp_f32",
            "v_log_f32",
            "v_rcp_f32",
            "v_rsq_f32",
            "v_sqrt_f32",
            "v_sin_f32",
            "v_cos_f32",
            "v_rcp_iflag_f32",
            "v_exp_f16",
            "v_log_f16",
            "v_rcp_f16",
            "v_rsq_f16",
            "v_sqrt_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_rcp_f64",
            "v_rsq_f64",
            "v_sqrt_f64",
        ],
        waits: &["s_wait_"],
        barrier: "s_barrier_wait",
        carry_in: &["s_add_co_ci_u32", "s_sub_co_ci_u32"],
        wave_varying: &["s_getreg", "s_sendmsg_rtn"],
        counter,
        fields,
        signs,
        widths: no_widths,
    }
}

/// gfx1201's hooks: its ISA tables, the operand table's mnemonics, and
/// the instruction form rules.
pub struct Gfx1201 {
    isa: &'static IsaTables,
    rows: HashSet<&'static str>,
}

/// The checked-in tables, read once per process: the target, its encoder
/// and its decoder share them.
pub(crate) struct Data {
    /// The ISA tables.
    pub isa: IsaTables,
    /// The operand table's rows.
    pub rows: Vec<Row>,
}

/// The tables, read on first use.
pub(crate) fn data() -> Result<&'static Data, Error> {
    static DATA: OnceLock<Result<Data, Error>> = OnceLock::new();
    let data = DATA.get_or_init(|| {
        Ok(Data {
            isa: tables::isa_tables(include_str!("../../data/gfx1201/isa_tables.json"))?,
            rows: tables::operand_table(include_str!("../../data/gfx1201/operand_table.json"))?,
        })
    });
    data.as_ref().map_err(Clone::clone)
}

impl Syntax for Gfx1201 {
    fn reg_count(&self, file: char) -> u32 {
        match file {
            'v' => 256,
            's' => 106,
            _ => 0,
        }
    }

    fn special(&self, name: &str) -> Option<(u32, u32)> {
        SPECIALS.iter().find(|s| s.0 == name).map(|s| (s.1, s.2))
    }

    fn hwreg(&self) -> &[(&'static str, u32)] {
        &HWREG
    }

    fn mod_flag(&self, token: &str) -> bool {
        matches!(
            token,
            "offen" | "idxen" | "clamp" | "row_mirror" | "row_half_mirror"
        )
    }

    fn counted(&self, mnemonic: &str) -> Option<Vec<(&'static str, u64)>> {
        (mnemonic == "s_wait_alu").then(|| DEPCTR.iter().map(|d| (d.0, (1 << d.2) - 1)).collect())
    }

    fn symbol(&self, modifier: &str, name: &str) -> Option<i128> {
        let table: &[(&str, i128)] = match modifier {
            "th" => &TH_SYMBOLS,
            "scope" => &[
                ("SCOPE_CU", 0),
                ("SCOPE_SE", 1),
                ("SCOPE_DEV", 2),
                ("SCOPE_SYS", 3),
            ],
            _ => return None,
        };
        table.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }
}

impl Hooks for Gfx1201 {
    fn mem_of(&self, mnemonic: &str) -> Option<Mem> {
        match self.isa.ops.get(mnemonic)?.class.as_str() {
            "DS" => Some(Mem::Ds),
            "VFLAT" => Some(Mem::Flat),
            "VGLOBAL" => Some(Mem::Global),
            "VSCRATCH" => Some(Mem::Scratch),
            "VBUFFER" => Some(Mem::Buffer),
            "SMEM" => Some(Mem::Smem),
            _ => None,
        }
    }

    fn returning(&self, mnemonic: &str, th: Option<&str>) -> bool {
        // A buffer atomic returns when its hint's RETURN bit is set; a
        // global or flat one by its destination operand instead.
        let hint = th.and_then(|t| TH_SYMBOLS.iter().find(|s| s.0 == t));
        mnemonic.starts_with("buffer_atomic")
            && hint.is_some_and(|&(_, v)| v >> 8 == 3 && v & 1 == 1)
    }

    fn nop_word(&self) -> u32 {
        0xBF80_0000
    }

    fn encode(&self, line: &str) -> Result<Encoded, String> {
        encoder::encode(self, line)
    }

    fn e_flags(&self) -> u32 {
        0x4E
    }

    fn granule(&self) -> u32 {
        8
    }

    fn directives(&self) -> &[&'static str] {
        &code_object::DIRECTIVES
    }

    fn footer(&self, footer: &Footer) -> Result<String, Error> {
        code_object::footer(footer)
    }

    fn resources(&self, block: &Block) -> Result<Resources, Error> {
        code_object::resources(block)
    }

    fn interface_resources(&self, block: &Block) -> Result<Resources, Error> {
        code_object::interface_resources(block)
    }

    fn matrix_form(&self, mnemonic: &str) -> Option<MatrixForm> {
        wmma_form(mnemonic)
    }

    /// The `scope` field (SCOPE_CU 0, SCOPE_SE 1, SCOPE_DEV 2, SCOPE_SYS 3:
    /// the CU and SE scopes are the workgroup's, an atomic is at least the
    /// device's); write-backs and invalidates reach the L2 at SCOPE_SYS and
    /// count as a store and a load. The non-temporal `th` values (RDNA4 ISA
    /// guide Tables 13-14): NT 1, LU 3, NT_RT 4, RT_NT 5, NT_HT 6, and for
    /// stores NT_WB 7 (a load's 7 is RT_WB). IPC memory takes SCOPE_DEV.
    fn scope_rule(&self) -> Option<thin_target::ScopeRule> {
        use thin_target::Scope::{Agent, System, Workgroup};
        Some(thin_target::ScopeRule {
            code: &[("scope", 0)],
            plain: &[Workgroup, Workgroup, Agent, System],
            atomic: &[Agent, Agent, Agent, System],
            writeback_l2: Some(3),
            invalidate_l2: 3,
            maintenance_counted: true,
            nontemporal: ("th", [0b0111_1010, 0b1111_1010]),
            ipc: Agent,
            ipc_text: "scope ≥ SCOPE_DEV or a non-temporal th",
            returns: None,
        })
    }

    /// Invalidate instruction cache, then leave sixteen nop slots before use.
    /// The launch covers eight workgroups per hardware unit.
    fn icache_invalidate(&self) -> Option<thin_target::IcacheInvalidate> {
        Some(thin_target::IcacheInvalidate {
            instruction: "s_icache_inv",
            nop: "s_nop",
            nops: 16,
            per_cu: 8,
        })
    }

    fn lib_hooks(&self) -> LibHooks {
        // A whole hardware register in s_getreg_b32's selector: the id,
        // offset 0, size 32 (encoded as size - 1 at bit 11).
        let whole = |name: &str| {
            let (_, id) = HWREG.iter().find(|(n, _)| *n == name).expect("a hwreg id");
            id | 31 << 11
        };
        LibHooks {
            lane_granule: Some(32),
            // The core's cycle counter is two hardware registers, read in
            // order; the constant clock a returning message, on KMcnt.
            core_clock: Some(ClockRead::Halves {
                op: "s_getreg_b32",
                lo: whole("HW_REG_SHADER_CYCLES_LO"),
                hi: whole("HW_REG_SHADER_CYCLES_HI"),
            }),
            real_clock: Some(ClockRead::Pair {
                op: "s_sendmsg_rtn_b64",
                imm: Some(MSG_RTN_GET_REALTIME),
            }),
            // The constant clock MSG_RTN_GET_REALTIME reads runs at 100 MHz.
            real_clock_mhz: Some(100),
            coherent: Some("SCOPE_DEV"),
            system: Some("SCOPE_SYS"),
            returning_atomic_add: Some("global_atomic_add_u32"),
            atomic_return: Some("TH_ATOMIC_RETURN"),
            scalar_add: Some("s_add_co_u32"),
            mask_shr: Some("s_lshr_b32"),
            sleep: Some("s_sleep"),
        }
    }
}

/// WMMA forms (kind `valu`) use the measured grouped summation model (the
/// emulator runs f16 and bf16 inputs with an f32 or bf16 D through its dot
/// unit); `v_swmmac_*` is not one.
fn wmma_form(mnemonic: &str) -> Option<MatrixForm> {
    use Elem::{Bf16, Bf8, Fp8, Iu4, Iu8, F16, F32, I32};
    let rest = mnemonic.strip_prefix("v_wmma_")?;
    let (d, a, b, k) = match rest {
        "f32_16x16x16_f16" => (F32, F16, F16, 16),
        "f32_16x16x16_bf16" => (F32, Bf16, Bf16, 16),
        "f16_16x16x16_f16" => (F16, F16, F16, 16),
        "bf16_16x16x16_bf16" => (Bf16, Bf16, Bf16, 16),
        "i32_16x16x16_iu8" => (I32, Iu8, Iu8, 16),
        "i32_16x16x16_iu4" => (I32, Iu4, Iu4, 16),
        "i32_16x16x32_iu4" => (I32, Iu4, Iu4, 32),
        "f32_16x16x16_fp8_fp8" => (F32, Fp8, Fp8, 16),
        "f32_16x16x16_fp8_bf8" => (F32, Fp8, Bf8, 16),
        "f32_16x16x16_bf8_fp8" => (F32, Bf8, Fp8, 16),
        "f32_16x16x16_bf8_bf8" => (F32, Bf8, Bf8, 16),
        _ => return None,
    };
    Some(MatrixForm {
        a,
        b,
        d,
        m: 16,
        n: 16,
        k,
        blocks: 1,
        wave: 32,
        passes: 1,
        sparse: false,
        scaled: false,
        cbsz_abid: false,
        blgp: false,
        // Every float form, measured on an RX 9070 XT bit for bit (integer
        // forms ignore it).
        sum: Summation::DotUnit,
        provisional: false,
    })
}

/// Builds gfx1201 from its checked-in tables.
pub fn build() -> Result<Target, Error> {
    let Data { isa, rows } = data()?;
    let forms = form_rules().forms(rows, &waits())?;
    // Ops LLVM spells otherwise (s_wait_* for s_waitcnt) are not this
    // target's mnemonics: neither spelled nor encoded.
    let encodings: BTreeSet<String> = (isa.ops.keys())
        .filter(|m| !encoder::SPELLED_OTHERWISE.contains(&m.as_str()))
        .cloned()
        .collect();
    let names = rows.iter().map(|r| r.mnemonic.as_str()).collect();
    let counters = counters();
    Target::build(Definition {
        name: NAME,
        forms,
        files: vec![
            File {
                letter: 'v',
                size: 256,
                granule: 24,
                align64: 1,
            },
            File {
                letter: 's',
                size: 106,
                granule: 1,
                align64: 1,
            },
        ],
        encodings,
        wave: 32,
        lds_max: 65536,
        hazards: Hazards {
            rules: 0,
            pass: true,
            window: 0,
            delay_hint: Some("s_delay_alu"),
        },
        counter_names: counters.iter().map(|c| c.name).collect(),
        counters,
        waits: waits(),
        unit_prefixes: (&[], &SMEM),
        operands: Operands {
            vop3_literal: true,
            constant_bus: 2,
            valu_sgpr_write_waits_smem: false,
            smem_offset_bits: 24,
        },
        chip: Chip {
            xcds: 1,
            cus: 32,
            simds: 4,
            lds: 131072,
            vgprs: 256,
            vgprs_per_lane: 1536,
            sgprs: 106,
            user_sgprs: 16,
            preload: 0,
            wave_slots: 16,
            wgids: &["ttmp9", "ttmp7"],
        },
        spellings: SPELLINGS.to_vec(),
        barrier: vec![
            ("s_barrier_signal", vec![-1]),
            ("s_barrier_wait", vec![65535]),
        ],
        hooks: Box::new(Gfx1201 { isa, rows: names }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> &'static Target {
        crate::get(NAME).expect("gfx1201 builds")
    }

    #[test]
    fn depctr_fields_tile_the_value_that_waits_for_nothing() {
        let ones = DEPCTR
            .iter()
            .fold(0u32, |v, &(_, shift, bits)| v | ((1 << bits) - 1) << shift);
        assert_eq!(ones, DEPCTR_NONE);
        assert_eq!(
            target().form("s_wait_alu", 0, false).unwrap().fields.len(),
            7
        );
    }

    #[test]
    fn a_vector_add_encodes_as_llvm_mc_does() {
        let lines = [
            ("s_mov_b32 s2, ttmp9", "750082be"),
            ("s_load_b128 s[4:7], s[0:1], 0x0", "004100f4000000f8"),
            ("v_lshl_add_u32 v1, s2, 8, v0", "010046d602100104"),
            ("s_wait_kmcnt 0", "0000c7bf"),
            ("v_cmp_gt_u32 vcc_lo, s3, v1", "0302987c"),
            ("s_mov_b32 exec_lo, -1", "c100febe"),
            ("global_load_b32 v1, v0, s[4:5]", "040005ee0100000000000000"),
            (
                "global_store_b32 v0, v1, s[8:9]",
                "088006ee0000800000000000",
            ),
            ("v_add_f32 v1, v1, v2", "01050206"),
            ("s_endpgm", "0000b0bf"),
        ];
        for (line, hex) in lines {
            let Ok(Encoded::Bytes(bytes)) = target().hooks().encode(line) else {
                panic!("{line} does not encode");
            };
            let got: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(got, hex, "{line}");
        }
    }

    #[test]
    fn every_spelling_round_trips() {
        for (op, mnemonic) in SPELLINGS {
            assert_eq!(target().spell(op).as_deref(), Some(mnemonic));
            assert_eq!(target().canon(mnemonic), op);
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
        assert_eq!(forms.len(), 1893);
    }
}
