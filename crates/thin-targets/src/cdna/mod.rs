//! Shared parameterized CDNA (wave64, unified VGPR+AGPR, XDLOP/MFMA) target family.

pub(crate) mod code_object;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod hazards;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Mutex;

use thin_target::{
    Block, Chip, ClockRead, Counter, Definition, Elem, Encoded, Error, Field, FieldKind, File,
    Flags, Footer, Form, FormRules, Hazards, Hooks, LibHooks, MatrixForm, Mem, Operands, Resources,
    Row, Summation, Syntax, Target, WaitInsn, WaitText,
};

use crate::tables::{self, IsaTables};

/// Packed `s_waitcnt` immediate that waits for nothing (`vmcnt(63) expcnt(7) lgkmcnt(15)`).
pub const WAITCNT_NONE: u32 = 0xCF7F;

static WAITCNT_FIELDS: [&str; 3] = ["vmcnt", "expcnt", "lgkmcnt"];

/// Shared scalar source-space names across CDNA 3 and CDNA 4: `(name, number, dwords)`.
/// Note: `null` (`125`) is reserved on CDNA (`124` is `m0`).
pub static COMMON_SPECIALS: [(&str, u32, u32); 12] = [
    ("vcc", 106, 2),
    ("vcc_lo", 106, 1),
    ("vcc_hi", 107, 1),
    ("ttmp7", 115, 1),
    ("ttmp9", 117, 1),
    ("m0", 124, 1),
    ("exec", 126, 2),
    ("exec_lo", 126, 1),
    ("exec_hi", 127, 1),
    ("scc", 253, 1),
    ("vccz", 251, 1),
    ("execz", 252, 1),
];

/// Common `hwreg(...)` names and IDs shared across CDNA 3 and CDNA 4.
pub static COMMON_HWREG: [(&str, u32); 24] = [
    ("HW_REG_MODE", 1),
    ("HW_REG_STATUS", 2),
    ("HW_REG_TRAPSTS", 3),
    ("HW_REG_HW_ID", 4),
    ("HW_REG_GPR_ALLOC", 5),
    ("HW_REG_LDS_ALLOC", 6),
    ("HW_REG_IB_STS", 7),
    ("HW_REG_PC_LO", 8),
    ("HW_REG_PC_HI", 9),
    ("HW_REG_INST_DW0", 10),
    ("HW_REG_INST_DW1", 11),
    ("HW_REG_IB_DBG0", 12),
    ("HW_REG_IB_DBG1", 13),
    ("HW_REG_FLUSH_IB", 14),
    ("HW_REG_SH_MEM_BASES", 15),
    ("HW_REG_SQ_SHADER_TBA_LO", 16),
    ("HW_REG_SQ_SHADER_TBA_HI", 17),
    ("HW_REG_SQ_SHADER_TMA_LO", 18),
    ("HW_REG_SQ_SHADER_TMA_HI", 19),
    ("HW_REG_XCC_ID", 20),
    ("HW_REG_SQ_PERF_SNAPSHOT_DATA", 21),
    ("HW_REG_SQ_PERF_SNAPSHOT_DATA1", 22),
    ("HW_REG_SQ_PERF_SNAPSHOT_PC_LO", 23),
    ("HW_REG_SQ_PERF_SNAPSHOT_PC_HI", 24),
];

/// Canonical dotted-op spellings common to all CDNA targets where replacing `.` with `_` does not apply.
pub static COMMON_SPELLINGS: [(&str, &str); 225] = [
    ("v.cvt.pk.rtz.f16.f32", "v_cvt_pkrtz_f16_f32"),
    // Scalar memory
    ("s.load.b32", "s_load_dword"),
    ("s.load.b64", "s_load_dwordx2"),
    ("s.load.b128", "s_load_dwordx4"),
    ("s.load.b256", "s_load_dwordx8"),
    ("s.load.b512", "s_load_dwordx16"),
    ("s.store.b32", "s_store_dword"),
    ("s.store.b64", "s_store_dwordx2"),
    ("s.store.b128", "s_store_dwordx4"),
    ("s.buffer.load.b32", "s_buffer_load_dword"),
    ("s.buffer.load.b64", "s_buffer_load_dwordx2"),
    ("s.buffer.load.b128", "s_buffer_load_dwordx4"),
    ("s.buffer.load.b256", "s_buffer_load_dwordx8"),
    ("s.buffer.load.b512", "s_buffer_load_dwordx16"),
    ("s.buffer.store.b32", "s_buffer_store_dword"),
    ("s.buffer.store.b64", "s_buffer_store_dwordx2"),
    ("s.buffer.store.b128", "s_buffer_store_dwordx4"),
    // Global loads & stores
    ("global.load.u8", "global_load_ubyte"),
    ("global.load.i8", "global_load_sbyte"),
    ("global.load.u16", "global_load_ushort"),
    ("global.load.i16", "global_load_sshort"),
    ("global.load.b32", "global_load_dword"),
    ("global.load.b64", "global_load_dwordx2"),
    ("global.load.b96", "global_load_dwordx3"),
    ("global.load.b128", "global_load_dwordx4"),
    ("global.load.d16.u8", "global_load_ubyte_d16"),
    ("global.load.d16.hi.u8", "global_load_ubyte_d16_hi"),
    ("global.load.d16.i8", "global_load_sbyte_d16"),
    ("global.load.d16.hi.i8", "global_load_sbyte_d16_hi"),
    ("global.load.d16.b16", "global_load_short_d16"),
    ("global.load.d16.hi.b16", "global_load_short_d16_hi"),
    ("global.load.lds.u8", "global_load_lds_ubyte"),
    ("global.load.lds.i8", "global_load_lds_sbyte"),
    ("global.load.lds.u16", "global_load_lds_ushort"),
    ("global.load.lds.i16", "global_load_lds_sshort"),
    ("global.load.lds.b32", "global_load_lds_dword"),
    ("global.store.b8", "global_store_byte"),
    ("global.store.d16.hi.b8", "global_store_byte_d16_hi"),
    ("global.store.b16", "global_store_short"),
    ("global.store.d16.hi.b16", "global_store_short_d16_hi"),
    ("global.store.b32", "global_store_dword"),
    ("global.store.b64", "global_store_dwordx2"),
    ("global.store.b96", "global_store_dwordx3"),
    ("global.store.b128", "global_store_dwordx4"),
    // Flat loads & stores
    ("flat.load.u8", "flat_load_ubyte"),
    ("flat.load.i8", "flat_load_sbyte"),
    ("flat.load.u16", "flat_load_ushort"),
    ("flat.load.i16", "flat_load_sshort"),
    ("flat.load.b32", "flat_load_dword"),
    ("flat.load.b64", "flat_load_dwordx2"),
    ("flat.load.b96", "flat_load_dwordx3"),
    ("flat.load.b128", "flat_load_dwordx4"),
    ("flat.load.d16.u8", "flat_load_ubyte_d16"),
    ("flat.load.d16.hi.u8", "flat_load_ubyte_d16_hi"),
    ("flat.load.d16.i8", "flat_load_sbyte_d16"),
    ("flat.load.d16.hi.i8", "flat_load_sbyte_d16_hi"),
    ("flat.load.d16.b16", "flat_load_short_d16"),
    ("flat.load.d16.hi.b16", "flat_load_short_d16_hi"),
    ("flat.store.b8", "flat_store_byte"),
    ("flat.store.d16.hi.b8", "flat_store_byte_d16_hi"),
    ("flat.store.b16", "flat_store_short"),
    ("flat.store.d16.hi.b16", "flat_store_short_d16_hi"),
    ("flat.store.b32", "flat_store_dword"),
    ("flat.store.b64", "flat_store_dwordx2"),
    ("flat.store.b96", "flat_store_dwordx3"),
    ("flat.store.b128", "flat_store_dwordx4"),
    // Scratch loads & stores
    ("scratch.load.u8", "scratch_load_ubyte"),
    ("scratch.load.i8", "scratch_load_sbyte"),
    ("scratch.load.u16", "scratch_load_ushort"),
    ("scratch.load.i16", "scratch_load_sshort"),
    ("scratch.load.b32", "scratch_load_dword"),
    ("scratch.load.b64", "scratch_load_dwordx2"),
    ("scratch.load.b96", "scratch_load_dwordx3"),
    ("scratch.load.b128", "scratch_load_dwordx4"),
    ("scratch.load.d16.u8", "scratch_load_ubyte_d16"),
    ("scratch.load.d16.hi.u8", "scratch_load_ubyte_d16_hi"),
    ("scratch.load.d16.i8", "scratch_load_sbyte_d16"),
    ("scratch.load.d16.hi.i8", "scratch_load_sbyte_d16_hi"),
    ("scratch.load.d16.b16", "scratch_load_short_d16"),
    ("scratch.load.d16.hi.b16", "scratch_load_short_d16_hi"),
    ("scratch.load.lds.u8", "scratch_load_lds_ubyte"),
    ("scratch.load.lds.i8", "scratch_load_lds_sbyte"),
    ("scratch.load.lds.u16", "scratch_load_lds_ushort"),
    ("scratch.load.lds.i16", "scratch_load_lds_sshort"),
    ("scratch.load.lds.b32", "scratch_load_lds_dword"),
    ("scratch.store.b8", "scratch_store_byte"),
    ("scratch.store.d16.hi.b8", "scratch_store_byte_d16_hi"),
    ("scratch.store.b16", "scratch_store_short"),
    ("scratch.store.d16.hi.b16", "scratch_store_short_d16_hi"),
    ("scratch.store.b32", "scratch_store_dword"),
    ("scratch.store.b64", "scratch_store_dwordx2"),
    ("scratch.store.b96", "scratch_store_dwordx3"),
    ("scratch.store.b128", "scratch_store_dwordx4"),
    // Buffer loads & stores
    ("buffer.load.u8", "buffer_load_ubyte"),
    ("buffer.load.i8", "buffer_load_sbyte"),
    ("buffer.load.u16", "buffer_load_ushort"),
    ("buffer.load.i16", "buffer_load_sshort"),
    ("buffer.load.b32", "buffer_load_dword"),
    ("buffer.load.b64", "buffer_load_dwordx2"),
    ("buffer.load.b96", "buffer_load_dwordx3"),
    ("buffer.load.b128", "buffer_load_dwordx4"),
    ("buffer.load.d16.u8", "buffer_load_ubyte_d16"),
    ("buffer.load.d16.hi.u8", "buffer_load_ubyte_d16_hi"),
    ("buffer.load.d16.i8", "buffer_load_sbyte_d16"),
    ("buffer.load.d16.hi.i8", "buffer_load_sbyte_d16_hi"),
    ("buffer.load.d16.b16", "buffer_load_short_d16"),
    ("buffer.load.d16.hi.b16", "buffer_load_short_d16_hi"),
    ("buffer.store.b8", "buffer_store_byte"),
    ("buffer.store.d16.hi.b8", "buffer_store_byte_d16_hi"),
    ("buffer.store.b16", "buffer_store_short"),
    ("buffer.store.d16.hi.b16", "buffer_store_short_d16_hi"),
    ("buffer.store.b32", "buffer_store_dword"),
    ("buffer.store.b64", "buffer_store_dwordx2"),
    ("buffer.store.b96", "buffer_store_dwordx3"),
    ("buffer.store.b128", "buffer_store_dwordx4"),
    // Global atomics & L2 cache maintenance
    ("global.atomic.add.u32", "global_atomic_add"),
    ("global.atomic.sub.u32", "global_atomic_sub"),
    ("global.atomic.smin.i32", "global_atomic_smin"),
    ("global.atomic.umin.u32", "global_atomic_umin"),
    ("global.atomic.smax.i32", "global_atomic_smax"),
    ("global.atomic.umax.u32", "global_atomic_umax"),
    ("global.atomic.and.b32", "global_atomic_and"),
    ("global.atomic.or.b32", "global_atomic_or"),
    ("global.atomic.xor.b32", "global_atomic_xor"),
    ("global.atomic.swap.b32", "global_atomic_swap"),
    ("global.atomic.cmpswap.b32", "global_atomic_cmpswap"),
    ("global.atomic.inc.u32", "global_atomic_inc"),
    ("global.atomic.dec.u32", "global_atomic_dec"),
    ("global.atomic.add.u64", "global_atomic_add_x2"),
    ("global.atomic.sub.u64", "global_atomic_sub_x2"),
    ("global.atomic.smin.i64", "global_atomic_smin_x2"),
    ("global.atomic.umin.u64", "global_atomic_umin_x2"),
    ("global.atomic.smax.i64", "global_atomic_smax_x2"),
    ("global.atomic.umax.u64", "global_atomic_umax_x2"),
    ("global.atomic.and.b64", "global_atomic_and_x2"),
    ("global.atomic.or.b64", "global_atomic_or_x2"),
    ("global.atomic.xor.b64", "global_atomic_xor_x2"),
    ("global.atomic.swap.b64", "global_atomic_swap_x2"),
    ("global.atomic.cmpswap.b64", "global_atomic_cmpswap_x2"),
    ("global.atomic.inc.u64", "global_atomic_inc_x2"),
    ("global.atomic.dec.u64", "global_atomic_dec_x2"),
    ("global.wb", "buffer_wbl2"),
    ("global.inv", "buffer_inv"),
    // Flat atomics
    ("flat.atomic.add.u32", "flat_atomic_add"),
    ("flat.atomic.sub.u32", "flat_atomic_sub"),
    ("flat.atomic.smin.i32", "flat_atomic_smin"),
    ("flat.atomic.umin.u32", "flat_atomic_umin"),
    ("flat.atomic.smax.i32", "flat_atomic_smax"),
    ("flat.atomic.umax.u32", "flat_atomic_umax"),
    ("flat.atomic.and.b32", "flat_atomic_and"),
    ("flat.atomic.or.b32", "flat_atomic_or"),
    ("flat.atomic.xor.b32", "flat_atomic_xor"),
    ("flat.atomic.swap.b32", "flat_atomic_swap"),
    ("flat.atomic.cmpswap.b32", "flat_atomic_cmpswap"),
    ("flat.atomic.inc.u32", "flat_atomic_inc"),
    ("flat.atomic.dec.u32", "flat_atomic_dec"),
    ("flat.atomic.add.u64", "flat_atomic_add_x2"),
    ("flat.atomic.sub.u64", "flat_atomic_sub_x2"),
    ("flat.atomic.smin.i64", "flat_atomic_smin_x2"),
    ("flat.atomic.umin.u64", "flat_atomic_umin_x2"),
    ("flat.atomic.smax.i64", "flat_atomic_smax_x2"),
    ("flat.atomic.umax.u64", "flat_atomic_umax_x2"),
    ("flat.atomic.and.b64", "flat_atomic_and_x2"),
    ("flat.atomic.or.b64", "flat_atomic_or_x2"),
    ("flat.atomic.xor.b64", "flat_atomic_xor_x2"),
    ("flat.atomic.swap.b64", "flat_atomic_swap_x2"),
    ("flat.atomic.cmpswap.b64", "flat_atomic_cmpswap_x2"),
    ("flat.atomic.inc.u64", "flat_atomic_inc_x2"),
    ("flat.atomic.dec.u64", "flat_atomic_dec_x2"),
    // Buffer atomics
    ("buffer.atomic.add.u32", "buffer_atomic_add"),
    ("buffer.atomic.sub.u32", "buffer_atomic_sub"),
    ("buffer.atomic.smin.i32", "buffer_atomic_smin"),
    ("buffer.atomic.umin.u32", "buffer_atomic_umin"),
    ("buffer.atomic.smax.i32", "buffer_atomic_smax"),
    ("buffer.atomic.umax.u32", "buffer_atomic_umax"),
    ("buffer.atomic.and.b32", "buffer_atomic_and"),
    ("buffer.atomic.or.b32", "buffer_atomic_or"),
    ("buffer.atomic.xor.b32", "buffer_atomic_xor"),
    ("buffer.atomic.swap.b32", "buffer_atomic_swap"),
    ("buffer.atomic.cmpswap.b32", "buffer_atomic_cmpswap"),
    ("buffer.atomic.inc.u32", "buffer_atomic_inc"),
    ("buffer.atomic.dec.u32", "buffer_atomic_dec"),
    ("buffer.atomic.add.u64", "buffer_atomic_add_x2"),
    ("buffer.atomic.sub.u64", "buffer_atomic_sub_x2"),
    ("buffer.atomic.smin.i64", "buffer_atomic_smin_x2"),
    ("buffer.atomic.umin.u64", "buffer_atomic_umin_x2"),
    ("buffer.atomic.smax.i64", "buffer_atomic_smax_x2"),
    ("buffer.atomic.umax.u64", "buffer_atomic_umax_x2"),
    ("buffer.atomic.and.b64", "buffer_atomic_and_x2"),
    ("buffer.atomic.or.b64", "buffer_atomic_or_x2"),
    ("buffer.atomic.xor.b64", "buffer_atomic_xor_x2"),
    ("buffer.atomic.swap.b64", "buffer_atomic_swap_x2"),
    ("buffer.atomic.cmpswap.b64", "buffer_atomic_cmpswap_x2"),
    ("buffer.atomic.inc.u64", "buffer_atomic_inc_x2"),
    ("buffer.atomic.dec.u64", "buffer_atomic_dec_x2"),
    // LDS reads & writes
    ("ds.load.u8", "ds_read_u8"),
    ("ds.load.i8", "ds_read_i8"),
    ("ds.load.u16", "ds_read_u16"),
    ("ds.load.i16", "ds_read_i16"),
    ("ds.load.b32", "ds_read_b32"),
    ("ds.load.b64", "ds_read_b64"),
    ("ds.load.b96", "ds_read_b96"),
    ("ds.load.b128", "ds_read_b128"),
    ("ds.load.d16.u8", "ds_read_u8_d16"),
    ("ds.load.d16.hi.u8", "ds_read_u8_d16_hi"),
    ("ds.load.d16.i8", "ds_read_i8_d16"),
    ("ds.load.d16.hi.i8", "ds_read_i8_d16_hi"),
    ("ds.load.d16.u16", "ds_read_u16_d16"),
    ("ds.load.d16.hi.u16", "ds_read_u16_d16_hi"),
    ("ds.load.2addr.b32", "ds_read2_b32"),
    ("ds.load.2addr.b64", "ds_read2_b64"),
    ("ds.load.2addr.stride64.b32", "ds_read2st64_b32"),
    ("ds.load.2addr.stride64.b64", "ds_read2st64_b64"),
    ("ds.store.b8", "ds_write_b8"),
    ("ds.store.d16.hi.b8", "ds_write_b8_d16_hi"),
    ("ds.store.b16", "ds_write_b16"),
    ("ds.store.d16.hi.b16", "ds_write_b16_d16_hi"),
    ("ds.store.b32", "ds_write_b32"),
    ("ds.store.b64", "ds_write_b64"),
    ("ds.store.b96", "ds_write_b96"),
    ("ds.store.b128", "ds_write_b128"),
    ("ds.store.2addr.b32", "ds_write2_b32"),
    ("ds.store.2addr.b64", "ds_write2_b64"),
    ("ds.store.2addr.stride64.b32", "ds_write2st64_b32"),
    ("ds.store.2addr.stride64.b64", "ds_write2st64_b64"),
];

pub static SMEM: [&str; 9] = [
    "s_load",
    "s_store",
    "s_buffer_",
    "s_atomic_",
    "s_scratch_",
    "s_dcache_",
    "s_atc_probe",
    "s_memtime",
    "s_memrealtime",
];

pub static TRANS: [&str; 20] = [
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
    "v_exp_legacy_f32",
    "v_log_legacy_f32",
];

/// Parsed ISA and operand tables for one CDNA target instance.
pub(crate) struct Data {
    pub isa: IsaTables,
    pub rows: Vec<Row>,
}

/// Parses a CDNA target's checked-in JSON tables.
pub(crate) fn load_data(isa_json: &str, operand_json: &str) -> Result<Data, Error> {
    Ok(Data {
        isa: tables::isa_tables(isa_json)?,
        rows: tables::operand_table(operand_json)?,
    })
}

/// Cache key `(base_form_id, signs_mask, opsel_len)` for physical forms.
pub(crate) type PhysicalFormKey = (i32, u16, u8);

/// Thread-safe cache of extended physical forms per CDNA target.
pub(crate) type PhysicalFormCache = Mutex<HashMap<PhysicalFormKey, &'static Form>>;

/// LDS bank conflict geometry callback: `(bank_count, lane_phases)`.
pub(crate) type LdsBanksFn = fn(bool, u32) -> Option<(u64, Vec<u32>)>;

/// Per-generation parameters that instantiate a concrete CDNA target (`gfx942`, `gfx950`, ...).
pub(crate) struct Spec {
    /// Target registry name (`"gfx942"`, `"gfx950"`).
    pub name: &'static str,
    /// ELF `e_flags` word (`0x54C` for `gfx942`, `0x54F` for `gfx950`).
    pub e_flags: u32,
    /// Number of XCD chiplets.
    pub xcds: u32,
    /// Total compute units across all XCDs.
    pub cus: u32,
    /// Maximum LDS bytes per workgroup / CU (`65536` on CDNA 3, `163840` on CDNA 4).
    pub lds_max: u32,
    /// Whether `fp8`/`bf8` instructions use CDNA 3 FNUZ encodings (`true`) or OCP (`false`).
    pub fnuz_fp8: bool,
    /// Whether `MUBUF` `lds` DMA supports `buffer_load_dwordx3` and `buffer_load_dwordx4` (CDNA 4+).
    pub wide_lds_dma: bool,
    /// Named scalar registers `(name, code, dwords)`.
    pub specials: &'static [(&'static str, u32, u32)],
    /// `hwreg(...)` `(name, id)` pairs.
    pub hwreg: &'static [(&'static str, u32)],
    /// Mnemonics LLVM spells otherwise on this target.
    pub spelled_otherwise: &'static [&'static str],
    /// Additional `(dotted, mnemonic)` spellings beyond [`COMMON_SPELLINGS`].
    pub extra_spellings: &'static [(&'static str, &'static str)],
    /// Matrix unit mnemonic prefixes/names for `FormRules` and `unit_prefixes`.
    pub matrix: &'static [&'static str],
    /// Authoritative matrix form lookup for this target.
    pub matrix_form: fn(&str) -> Option<MatrixForm>,
    /// Form field builder (`FormRules.fields`).
    pub fields: fn(&str, &str, usize) -> Result<Vec<Field>, Error>,
    /// Form sign-modifier slot builder (`FormRules.signs`).
    pub signs: fn(&str) -> Vec<usize>,
    /// Variable tuple widths per `(spelling, slot)` (`FormRules.widths`).
    pub widths: fn(&str, usize) -> Vec<u32>,
    /// LDS bank conflict geometry (`(bank_count, lane_phases)`).
    pub lds_banks: LdsBanksFn,
    /// Lazy table loader for this target.
    pub data: fn() -> Result<&'static Data, Error>,
    /// Hazard rule count for `Definition.hazards`.
    pub hazard_rules_len: fn() -> usize,
    /// Maximum hazard wait-state window for `Definition.hazards`.
    pub hazard_window: u32,
}

pub(crate) fn waits() -> Vec<WaitInsn> {
    vec![WaitInsn {
        mnemonic: "s_waitcnt",
        fields: &WAITCNT_FIELDS,
        text: WaitText::Named,
    }]
}

pub(crate) fn counters() -> Vec<Counter> {
    vec![
        Counter {
            name: "vm",
            max: 63,
            field: "vmcnt",
            units: Flags::VMEM,
            lands: thin_target::Lands::Wait,
        },
        Counter {
            name: "lgkm",
            max: 15,
            field: "lgkmcnt",
            units: Flags(Flags::DS.0 | Flags::SMEM.0),
            lands: thin_target::Lands::Wait,
        },
    ]
}

pub(crate) fn counter(flags: Flags, _name: &str, _writes: bool) -> Option<usize> {
    if flags.any(Flags::DS | Flags::SMEM) {
        Some(1)
    } else if flags.has(Flags::VMEM) {
        Some(0)
    } else {
        None
    }
}

fn vmem_cache_fields() -> Result<Vec<Field>, Error> {
    let flag = |name| Field::new(name, FieldKind::Flag, 0, false);
    Ok(vec![flag("sc0")?, flag("sc1")?, flag("nt")?])
}

/// Shared CDNA instruction modifier fields by encoding class.
pub(crate) fn fields(spelling: &str, encoding: &str, nsrc: usize) -> Result<Vec<Field>, Error> {
    if spelling == "s_waitcnt" && nsrc == 0 {
        return Ok(vec![
            Field::new("vmcnt", FieldKind::Int, 6, false)?,
            Field::new("expcnt", FieldKind::Int, 3, false)?,
            Field::new("lgkmcnt", FieldKind::Int, 4, false)?,
        ]);
    }
    let int = |name, bits, signed| Field::new(name, FieldKind::Int, bits, signed);
    let flag = |name| Field::new(name, FieldKind::Flag, 0, false);
    let bits = |name| Field::new(name, FieldKind::Bits, nsrc as u32, false);
    Ok(match encoding {
        "ENC_FLAT" => [vec![int("offset", 12, false)?], vmem_cache_fields()?].concat(),
        "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" => {
            [vec![int("offset", 13, true)?], vmem_cache_fields()?].concat()
        }
        "ENC_MUBUF" => [
            vec![flag("offen")?, flag("idxen")?, int("offset", 12, false)?],
            vmem_cache_fields()?,
        ]
        .concat(),
        "ENC_MUBUF lds" => [
            vec![
                flag("offen")?,
                flag("idxen")?,
                int("offset", 12, false)?,
                flag("lds")?,
            ],
            vmem_cache_fields()?,
        ]
        .concat(),
        "ENC_SMEM" => vec![int("offset", 21, true)?, flag("glc")?],
        "ENC_DS" if encoder::is_ds_two_offset(spelling) => {
            vec![
                int("offset0", 8, false)?,
                int("offset1", 8, false)?,
                flag("gds")?,
            ]
        }
        "ENC_DS" => vec![int("offset", 16, false)?, flag("gds")?],
        "ENC_VOP3P" => vec![
            bits("op_sel")?,
            bits("op_sel_hi")?,
            bits("neg_lo")?,
            bits("neg_hi")?,
            flag("clamp")?,
        ],
        "ENC_VOP3PX2" => vec![
            Field::new("op_sel", FieldKind::Bits, 3, false)?,
            Field::new("op_sel_hi", FieldKind::Bits, 3, false)?,
            int("cbsz", 3, false)?,
            int("blgp", 3, false)?,
        ],
        "VOP3P_MFMA" if spelling.starts_with("v_smfmac_") => {
            vec![int("cbsz", 3, false)?, int("abid", 4, false)?]
        }
        "VOP3P_MFMA" if spelling.ends_with("_f8f6f4") => {
            vec![int("cbsz", 3, false)?, int("blgp", 3, false)?]
        }
        "VOP3P_MFMA" if spelling.starts_with("v_mfma_f64_") => {
            vec![int("cbsz", 3, false)?, int("abid", 4, false)?, bits("neg")?]
        }
        "VOP3P_MFMA" => vec![
            int("cbsz", 3, false)?,
            int("abid", 4, false)?,
            int("blgp", 3, false)?,
        ],
        "ENC_VOP1" | "ENC_VOP3"
            if spelling.starts_with("v_permlane") && spelling.contains("_swap_") =>
        {
            vec![flag("bound_ctrl")?, flag("fi")?]
        }
        "ENC_VOP3" if spelling == "v_bitop3_b16" => vec![
            int("bitop3", 8, false)?,
            Field::new("op_sel", FieldKind::Bits, 4, false)?,
        ],
        "ENC_VOP3" if spelling == "v_bitop3_b32" => vec![int("bitop3", 8, false)?],
        "ENC_VOP3" => {
            let mut out = Vec::new();
            if encoder::opsel_bits(spelling) != 0 {
                out.push(Field::new(
                    "op_sel",
                    FieldKind::Bits,
                    encoder::opsel_len(spelling, nsrc) as u32,
                    false,
                )?);
            }
            if encoder::vop3_takes_clamp_omod(spelling) {
                out.push(flag("clamp")?);
                out.push(int("omod", 2, false)?);
            }
            out
        }
        "VOP3_SDST_ENC" if spelling.starts_with("v_div_scale_") => {
            vec![flag("clamp")?, int("omod", 2, false)?]
        }
        "VOP3_SDST_ENC" => vec![flag("clamp")?],
        "VOP1_VOP_DPP" | "VOP2_VOP_DPP" => vec![
            Field::new("dpp", FieldKind::Dpp, 0, false)?,
            int("row_mask", 4, false)?,
            int("bank_mask", 4, false)?,
            flag("bound_ctrl")?,
        ],
        _ => Vec::new(),
    })
}

/// Shared CDNA sign-modifier operand slots for `mnemonic` using `data`.
pub(crate) fn signs_for(data: Result<&'static Data, Error>, mnemonic: &str) -> Vec<usize> {
    let Ok(Data { isa, .. }) = data else {
        return Vec::new();
    };
    if mnemonic.starts_with("v_fma_mix") {
        return vec![1, 2, 3];
    }
    if let Some(base) = mnemonic.strip_suffix("_dpp") {
        let short_name = format!("{base}_e32");
        let e64 = format!("{base}_e64");
        let Some(sop) = isa.ops.get(&short_name) else {
            return Vec::new();
        };
        let shape: &[&str] = match sop.class.as_str() {
            "VOP2" if base.starts_with("v_cndmask") => &["vdst", "src0", "vsrc1", "vcc"],
            "VOP2" if encoder::is_carry_in_vop2(base) => &["vdst", "vcc", "src0", "vsrc1", "vcc"],
            "VOP2" if encoder::is_carry_vop2(base) => &["vdst", "vcc", "src0", "vsrc1"],
            "VOP2" => &["vdst", "src0", "vsrc1"],
            _ => &["vdst", "src0"],
        };
        return (shape.iter().enumerate())
            .filter(|(_, &fname)| {
                if !matches!(fname, "src0" | "vsrc1") {
                    return false;
                }
                if let Some(wop) = isa.ops.get(&e64) {
                    let wf = if fname == "vsrc1" { "src1" } else { "src0" };
                    wop.operand(wf)
                        .is_some_and(|sf| encoder::signs_of(&e64, &wop.class, sf).0)
                } else {
                    sop.operand(fname)
                        .is_some_and(|sf| encoder::is_float_fmt(&sf.format))
                }
            })
            .map(|(k, _)| k)
            .collect();
    }
    let Some(op) = isa.ops.get(mnemonic) else {
        return Vec::new();
    };
    if !matches!(op.class.as_str(), "VOP3" | "VOP3_SDST" | "VOP3SD") {
        return Vec::new();
    }
    let present = ["vdst", "sdst", "src0", "src1", "src2"]
        .iter()
        .filter_map(|n| op.operand(n))
        .filter(|f| f.xml_type != "OPR_EXEC");
    (present.enumerate())
        .filter(|(_, f)| f.field.starts_with("src") && encoder::signs_of(mnemonic, &op.class, f).0)
        .map(|(k, _)| k)
        .collect()
}

/// Default empty `FormRules.widths` callback.
pub(crate) fn no_widths(_: &str, _: usize) -> Vec<u32> {
    Vec::new()
}

/// Shared matrix form table for the 30 dense `v_mfma_*` and 14 sparse `v_smfmac_*`
/// shapes present across CDNA 3 and CDNA 4, parameterized by `fnuz_fp8` and `provisional`.
pub(crate) fn common_mfma_form(
    mnemonic: &str,
    fnuz_fp8: bool,
    provisional: bool,
) -> Option<MatrixForm> {
    use Elem::{Bf16, F16, F32, F64, I32, I8};
    let fp8 = if fnuz_fp8 { Elem::Fp8Fnuz } else { Elem::Fp8 };
    let bf8 = if fnuz_fp8 { Elem::Bf8Fnuz } else { Elem::Bf8 };

    if let Some(rest) = mnemonic.strip_prefix("v_mfma_") {
        let (d, a, b, m, n, k, blocks, passes, cbsz_abid, blgp) = match rest {
            "f32_16x16x16_bf16" => (F32, Bf16, Bf16, 16, 16, 16, 1, 4, false, false),
            "f32_16x16x16_f16" => (F32, F16, F16, 16, 16, 16, 1, 4, false, false),
            "f32_16x16x1_4b_f32" => (F32, F32, F32, 16, 16, 1, 4, 8, true, true),
            "f32_16x16x32_bf8_bf8" => (F32, bf8, bf8, 16, 16, 32, 1, 4, false, false),
            "f32_16x16x32_bf8_fp8" => (F32, bf8, fp8, 16, 16, 32, 1, 4, false, false),
            "f32_16x16x32_fp8_bf8" => (F32, fp8, bf8, 16, 16, 32, 1, 4, false, false),
            "f32_16x16x32_fp8_fp8" => (F32, fp8, fp8, 16, 16, 32, 1, 4, false, false),
            "f32_16x16x4_4b_bf16" => (F32, Bf16, Bf16, 16, 16, 4, 4, 8, true, true),
            "f32_16x16x4_4b_f16" => (F32, F16, F16, 16, 16, 4, 4, 8, true, true),
            "f32_16x16x4_f32" => (F32, F32, F32, 16, 16, 4, 1, 8, false, true),
            "f32_32x32x16_bf8_bf8" => (F32, bf8, bf8, 32, 32, 16, 1, 8, false, false),
            "f32_32x32x16_bf8_fp8" => (F32, bf8, fp8, 32, 32, 16, 1, 8, false, false),
            "f32_32x32x16_fp8_bf8" => (F32, fp8, bf8, 32, 32, 16, 1, 8, false, false),
            "f32_32x32x16_fp8_fp8" => (F32, fp8, fp8, 32, 32, 16, 1, 8, false, false),
            "f32_32x32x1_2b_f32" => (F32, F32, F32, 32, 32, 1, 2, 16, true, true),
            "f32_32x32x2_f32" => (F32, F32, F32, 32, 32, 2, 1, 16, false, true),
            "f32_32x32x4_2b_bf16" => (F32, Bf16, Bf16, 32, 32, 4, 2, 16, true, true),
            "f32_32x32x4_2b_f16" => (F32, F16, F16, 32, 32, 4, 2, 16, true, true),
            "f32_32x32x8_bf16" => (F32, Bf16, Bf16, 32, 32, 8, 1, 8, false, false),
            "f32_32x32x8_f16" => (F32, F16, F16, 32, 32, 8, 1, 8, false, false),
            "f32_4x4x1_16b_f32" => (F32, F32, F32, 4, 4, 1, 16, 2, true, true),
            "f32_4x4x4_16b_bf16" => (F32, Bf16, Bf16, 4, 4, 4, 16, 2, true, true),
            "f32_4x4x4_16b_f16" => (F32, F16, F16, 4, 4, 4, 16, 2, true, true),
            "f64_16x16x4_f64" => (F64, F64, F64, 16, 16, 4, 1, 8, false, true),
            "f64_4x4x4_4b_f64" => (F64, F64, F64, 4, 4, 4, 4, 4, false, true),
            "i32_16x16x32_i8" => (I32, I8, I8, 16, 16, 32, 1, 4, false, false),
            "i32_16x16x4_4b_i8" => (I32, I8, I8, 16, 16, 4, 4, 8, true, true),
            "i32_32x32x16_i8" => (I32, I8, I8, 32, 32, 16, 1, 8, false, false),
            "i32_32x32x4_2b_i8" => (I32, I8, I8, 32, 32, 4, 2, 16, true, true),
            "i32_4x4x4_16b_i8" => (I32, I8, I8, 4, 4, 4, 16, 2, true, true),
            _ => return None,
        };
        let sum = if d.integer() || d == F64 || (a == F32 && b == F32) {
            Summation::Ordered
        } else {
            Summation::Grouped {
                group_half: 8,
                group_byte: 16,
                fraction: 24,
            }
        };
        return Some(MatrixForm {
            a,
            b,
            d,
            m,
            n,
            k,
            blocks,
            wave: 64,
            passes,
            sparse: false,
            cbsz_abid,
            blgp,
            sum,
            provisional,
        });
    }
    let rest = mnemonic.strip_prefix("v_smfmac_")?;
    let (d, a, b, m, n, k, passes) = match rest {
        "f32_16x16x32_bf16" => (F32, Bf16, Bf16, 16, 16, 32, 4),
        "f32_16x16x32_f16" => (F32, F16, F16, 16, 16, 32, 4),
        "f32_16x16x64_bf8_bf8" => (F32, bf8, bf8, 16, 16, 64, 4),
        "f32_16x16x64_bf8_fp8" => (F32, bf8, fp8, 16, 16, 64, 4),
        "f32_16x16x64_fp8_bf8" => (F32, fp8, bf8, 16, 16, 64, 4),
        "f32_16x16x64_fp8_fp8" => (F32, fp8, fp8, 16, 16, 64, 4),
        "f32_32x32x16_bf16" => (F32, Bf16, Bf16, 32, 32, 16, 8),
        "f32_32x32x16_f16" => (F32, F16, F16, 32, 32, 16, 8),
        "f32_32x32x32_bf8_bf8" => (F32, bf8, bf8, 32, 32, 32, 8),
        "f32_32x32x32_bf8_fp8" => (F32, bf8, fp8, 32, 32, 32, 8),
        "f32_32x32x32_fp8_bf8" => (F32, fp8, bf8, 32, 32, 32, 8),
        "f32_32x32x32_fp8_fp8" => (F32, fp8, fp8, 32, 32, 32, 8),
        "i32_16x16x64_i8" => (I32, I8, I8, 16, 16, 64, 4),
        "i32_32x32x32_i8" => (I32, I8, I8, 32, 32, 32, 8),
        _ => return None,
    };
    let sum = if d.integer() {
        Summation::Ordered
    } else {
        Summation::Grouped {
            group_half: 8,
            group_byte: 16,
            fraction: 24,
        }
    };
    Some(MatrixForm {
        a,
        b,
        d,
        m,
        n,
        k,
        blocks: 1,
        wave: 64,
        passes,
        sparse: true,
        cbsz_abid: true,
        blgp: false,
        sum,
        provisional,
    })
}

/// Instantiated CDNA target implementing [`Syntax`] and [`Hooks`].
pub(crate) struct CdnaTarget {
    pub spec: &'static Spec,
    pub isa: &'static IsaTables,
    pub rows: HashSet<&'static str>,
}

impl Syntax for CdnaTarget {
    fn reg_count(&self, file: char) -> u32 {
        match file {
            'v' => 256,
            'a' => 256,
            's' => 102,
            _ => 0,
        }
    }

    fn special(&self, name: &str) -> Option<(u32, u32)> {
        self.spec
            .specials
            .iter()
            .find(|s| s.0 == name)
            .map(|s| (s.1, s.2))
    }

    fn hwreg(&self) -> &[(&'static str, u32)] {
        self.spec.hwreg
    }

    fn mod_flag(&self, token: &str) -> bool {
        matches!(
            token,
            "offen"
                | "idxen"
                | "clamp"
                | "sc0"
                | "sc1"
                | "nt"
                | "glc"
                | "lds"
                | "gds"
                | "row_mirror"
                | "row_half_mirror"
        )
    }

    fn counted(&self, mnemonic: &str) -> Option<Vec<(&'static str, u64)>> {
        (mnemonic == "s_waitcnt").then(|| vec![("vmcnt", 63), ("expcnt", 7), ("lgkmcnt", 15)])
    }

    fn symbol(&self, _modifier: &str, _name: &str) -> Option<i128> {
        None
    }
}

impl Hooks for CdnaTarget {
    fn mem_of(&self, mnemonic: &str) -> Option<Mem> {
        match self.isa.ops.get(mnemonic)?.class.as_str() {
            "DS" => Some(Mem::Ds),
            "FLAT" => Some(Mem::Flat),
            "GLOBAL" => Some(Mem::Global),
            "SCRATCH" => Some(Mem::Scratch),
            "MUBUF" | "MTBUF" => Some(Mem::Buffer),
            "SMEM" => Some(Mem::Smem),
            _ => None,
        }
    }

    fn returning(&self, mnemonic: &str, th: Option<&str>) -> bool {
        mnemonic.starts_with("buffer_atomic") && th == Some("sc0")
    }

    fn nop_word(&self) -> u32 {
        0xBF80_0000
    }

    fn encode(&self, line: &str) -> Result<Encoded, String> {
        encoder::encode(self, line)
    }

    fn e_flags(&self) -> u32 {
        self.spec.e_flags
    }

    fn granule(&self) -> u32 {
        8
    }

    fn directives(&self) -> &[&'static str] {
        &code_object::DIRECTIVES
    }

    fn footer(&self, footer: &Footer) -> Result<String, Error> {
        code_object::footer(self.spec, footer)
    }

    fn resources(&self, block: &Block) -> Result<Resources, Error> {
        code_object::resources(block)
    }

    fn interface_resources(&self, block: &Block) -> Result<Resources, Error> {
        code_object::interface_resources(block)
    }

    fn fnuz_fp8(&self) -> bool {
        self.spec.fnuz_fp8
    }

    fn matrix_form(&self, mnemonic: &str) -> Option<MatrixForm> {
        (self.spec.matrix_form)(mnemonic)
    }

    /// CDNA scope rule (`code = sc0 | (sc1 << 1)`):
    /// - Plain loads/stores: `0` -> Wave, `sc0` (`1`) -> Workgroup, `sc1` (`2`) -> Agent, `sc0 sc1` (`3`) -> System.
    /// - Atomics (`sc0` is the return bit): `0`/`1` -> Agent, `2`/`3` (`sc1`) -> System.
    /// - `buffer_wbl2` / `buffer_inv` reach XCD L2 at `sc1` (`code >= 2`) and count on `vmcnt`.
    fn scope_rule(&self) -> Option<thin_target::ScopeRule> {
        use thin_target::Scope::{Agent, System, Wave, Workgroup};
        Some(thin_target::ScopeRule {
            code: &[("sc0", 0), ("sc1", 1)],
            plain: &[Wave, Workgroup, Agent, System],
            atomic: &[Agent, Agent, System, System],
            writeback_l2: Some(2),
            invalidate_l2: 2,
            maintenance_counted: true,
            nontemporal: ("nt", [0, 0]),
            ipc: Agent,
            ipc_text: "sc1 or nt",
            returns: Some("sc0"),
        })
    }

    fn lds_banks(&self, read: bool, bytes: u32) -> Option<(u64, Vec<u32>)> {
        (self.spec.lds_banks)(read, bytes)
    }

    fn icache_invalidate(&self) -> Option<thin_target::IcacheInvalidate> {
        Some(thin_target::IcacheInvalidate {
            instruction: "s_icache_inv",
            nop: "s_nop",
            nops: 16,
            per_cu: 8,
        })
    }

    fn lib_hooks(&self) -> LibHooks {
        LibHooks {
            lane_granule: Some(64),
            core_clock: Some(ClockRead::Pair {
                op: "s_memtime",
                imm: None,
            }),
            real_clock: Some(ClockRead::Pair {
                op: "s_memrealtime",
                imm: None,
            }),
            real_clock_mhz: Some(100),
            coherent: Some("sc1"),
            system: Some("sc0 sc1"),
            returning_atomic_add: Some("global_atomic_add"),
            atomic_return: Some("sc0"),
            scalar_add: Some("s_add_u32"),
            mask_shr: Some("s_lshr_b64"),
            sleep: Some("s_sleep"),
        }
    }
}

/// Builds a CDNA target from its [`Spec`] and checked-in tables.
pub(crate) fn build(spec: &'static Spec) -> Result<Target, Error> {
    let Data { isa, rows } = (spec.data)()?;
    let form_rules = FormRules {
        matrix: spec.matrix,
        smem: &SMEM,
        trans: &TRANS,
        waits: &["s_waitcnt"],
        barrier: "s_barrier",
        carry_in: &["s_addc_u32", "s_subb_u32"],
        wave_varying: &["s_getreg", "s_memtime", "s_memrealtime"],
        counter,
        fields: spec.fields,
        signs: spec.signs,
        widths: spec.widths,
    };
    let forms = form_rules.forms(rows, &waits())?;
    let encodings: BTreeSet<String> = (isa.ops.keys())
        .filter(|m| !spec.spelled_otherwise.contains(&m.as_str()))
        .cloned()
        .collect();
    let names = rows.iter().map(|r| r.mnemonic.as_str()).collect();
    let counters = counters();
    let mut spellings = COMMON_SPELLINGS.to_vec();
    spellings.extend_from_slice(spec.extra_spellings);
    Target::build(Definition {
        name: spec.name,
        forms,
        files: vec![
            File {
                letter: 'v',
                size: 256,
                granule: 8,
                align64: 2,
            },
            File {
                letter: 's',
                size: 102,
                granule: 8,
                align64: 1,
            },
            File {
                letter: 'a',
                size: 256,
                granule: 4,
                align64: 2,
            },
        ],
        encodings,
        wave: 64,
        lds_max: spec.lds_max,
        hazards: Hazards {
            rules: (spec.hazard_rules_len)(),
            pass: false,
            window: spec.hazard_window,
            delay_hint: None,
        },
        counter_names: counters.iter().map(|c| c.name).collect(),
        counters,
        waits: waits(),
        unit_prefixes: (spec.matrix, &SMEM),
        operands: Operands {
            vop3_literal: false,
            constant_bus: 1,
            valu_sgpr_write_waits_smem: false,
            smem_offset_bits: 21,
        },
        chip: Chip {
            xcds: spec.xcds,
            cus: spec.cus,
            simds: 4,
            lds: spec.lds_max,
            vgprs: 256,
            vgprs_per_lane: 512,
            sgprs: 102,
            user_sgprs: 16,
            preload: 16,
            wave_slots: 8,
            wgids: &[],
        },
        spellings,
        barrier: vec![("s_barrier", vec![])],
        hooks: Box::new(CdnaTarget {
            spec,
            isa,
            rows: names,
        }),
    })
}
