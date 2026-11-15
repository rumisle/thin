//! gfx1201's instruction decoder: the exact inverse of its line encoder
//! driven by the same ISA tables.
//!
//! Decoding is table-driven: the top nine bits of the first dword pick the
//! encoding family (built from the classes' fixed fields), the family's op
//! field indexes a dense opcode table, and each op carries what the
//! encoder's rule for its class reads, resolved once (field slots, operand
//! types and widths, the mnemonic predicates the encoder tests). Each rule
//! below mirrors one of the encoder's, in reverse: it reads the fields the
//! encoder writes and produces the operands and modifiers that make the
//! encoder write those values, or refuses what no line gives. Every other
//! bit must hold what the encoder's fresh word holds (its defaults and
//! opcode), which one comparison checks at the end. Nothing is allocated
//! per instruction but the refusal's text.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

use thin_target::Error;

use super::encoder::{
    dwords, opsel_bits, packed_rules, signs_of, soffset_code, BUFFER_OFFSET_MAX, INLINE_FLOATS,
    LITERAL_CLASSES, SPELLED_OTHERWISE, UNTYPED_FORMAT, VCC_LO,
};
use super::{data, Data, DEPCTR, DEPCTR_NONE, HWREG, SCOPE, SPECIALS, TH_BYPASS, TH_SYMBOLS};
use crate::decode::{label, Decode, DecodeError, Instruction, MVal, Opd, Why};
use crate::tables::{Encoding, Op};

/// The fields the rules read, by name; an index is a field id.
const NAMES: [&str; 45] = [
    "vdst",
    "sdst",
    "src0",
    "src1",
    "src2",
    "vsrc1",
    "ssrc0",
    "ssrc1",
    "simm16",
    "sdata",
    "sbase",
    "soffset",
    "ioffset",
    "th",
    "scope",
    "addr",
    "data0",
    "data1",
    "offset0",
    "offset1",
    "vaddr",
    "vsrc",
    "saddr",
    "opsel",
    "opsel_hi",
    "opsel_hi2",
    "neg",
    "neg_hi",
    "clmp",
    "dpp",
    "row_mask",
    "bank_mask",
    "bc",
    "fi",
    "vsrc0",
    "op",
    "srcx0",
    "srcy0",
    "vdata",
    "rsrc",
    "idxen",
    "offen",
    "format",
    "abs",
    "literal",
];
const NF: usize = NAMES.len();
const VDST: usize = 0;
const SDST: usize = 1;
const SRC0: usize = 2;
const SRC1: usize = 3;
const SRC2: usize = 4;
const VSRC1: usize = 5;
const SSRC0: usize = 6;
const SSRC1: usize = 7;
const SIMM16: usize = 8;
const SDATA: usize = 9;
const SBASE: usize = 10;
const SOFFSET: usize = 11;
const IOFFSET: usize = 12;
const TH: usize = 13;
const SCOPE_F: usize = 14;
const ADDR: usize = 15;
const DATA0: usize = 16;
const DATA1: usize = 17;
const OFFSET0: usize = 18;
const OFFSET1: usize = 19;
const VADDR: usize = 20;
const VSRC: usize = 21;
const SADDR: usize = 22;
const OPSEL: usize = 23;
const OPSEL_HI: usize = 24;
const OPSEL_HI2: usize = 25;
const NEG: usize = 26;
const NEG_HI: usize = 27;
const CLMP: usize = 28;
const DPP: usize = 29;
const ROW_MASK: usize = 30;
const BANK_MASK: usize = 31;
const BC: usize = 32;
const FI: usize = 33;
const VSRC0: usize = 34;
const OP: usize = 35;
const VDATA: usize = 38;
const RSRC: usize = 39;
const IDXEN: usize = 40;
const OFFEN: usize = 41;
const FORMAT: usize = 42;
const ABS: usize = 43;
const LITERAL: usize = 44;

/// The code of `null`, a field's "none".
const NULL: u32 = 124;
/// src0's codes that select DPP16 and DPP8.
const DPP16: u32 = 0xFA;
const DPP8: u32 = 0xE9;

/// Where a field lies in a class's word: `bits` 0 when absent.
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    lo: u8,
    bits: u8,
}

/// An encoding class, resolved.
struct Class {
    name: &'static str,
    bytes: u8,
    slots: [Slot; NF],
    /// Every field (name, mask), for naming what differs.
    fields: Vec<(&'static str, u128)>,
    /// The source fields that may hold the literal code, a bit per field id.
    srcs: u64,
    /// The class has a `_LIT` variant: a 255 source adds a dword.
    lit: bool,
    /// The encoder lets a literal follow (its `LITERAL_CLASSES`).
    literal_ok: bool,
}

/// An operand type the source rule tells apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Xml {
    Src,
    Ssrc,
    Sreg,
    Vgpr,
    SrcVgpr,
    VgprOrInline,
    Barrier,
    /// A lane select: an SGPR, M0 or an inline constant, never a literal.
    Lanesel,
    Simm16,
    /// A hardware-register selector (`hwreg(...)`).
    Hwreg,
    /// A message id, sent (SOPP) or returning (a SOP1 source's byte).
    Sendmsg,
    WaitAlu,
    Label,
    Delay,
    /// A plain 16-bit immediate of its own type (s_clause, s_sleep,
    /// s_wait_event, the combined memory waits, s_version).
    Imm16,
    Other,
}

/// An op's operand in one field, resolved (four bytes: the tables stay
/// small enough for the cache).
#[derive(Clone, Copy, Debug)]
struct Opnd {
    dwords: u8,
    sixteen: bool,
    wide: bool,
    xml: Xml,
}

/// The encoder rule of an op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Salu,
    Sopp,
    Smem,
    Short,
    Vop3,
    Vop3p,
    Global,
    Buffer,
    Setreg,
    Ds,
    Refused(&'static str),
}

/// An op's DPP16 form: its mnemonic, class and fresh word.
#[derive(Clone, Copy, Debug)]
struct DppForm {
    mnemonic: &'static str,
    class: u16,
    base: u128,
}

/// An op, resolved.
struct OpInfo {
    mnemonic: &'static str,
    /// The table entry, for the operand types refusals name.
    op: &'static Op,
    /// The mnemonic decoded lines carry.
    print: &'static str,
    class: u16,
    rule: Rule,
    opd: [Option<Opnd>; NF],
    /// The encoder's fresh word in the op's class.
    base: u128,
    /// The source fields it reads, a bit per field id.
    srcs: u64,
    /// The `th` kind: 0x100 load, 0x200 store, 0x300 atomic.
    cache_kind: u16,
    two_addr: bool,
    /// The VOP3 OPSEL bits it takes (the encoder's `OPSEL_OPS`).
    opsel: u32,
    /// The sign modifiers each of src0..src2 takes: (neg, abs).
    signs: [(bool, bool); 3],
    /// Its short form's fields in line order, for VOP1, VOP2 and VOPC.
    shape: Option<&'static [usize]>,
    packed: ([u64; 4], bool, u64, bool),
    dpp: Result<DppForm, &'static str>,
}

/// A family: the classes sharing fixed bits, and their ops by opcode.
struct Family {
    /// The class whose layout the size rule reads.
    class: u16,
    op: Slot,
    ops: Vec<u16>,
    dpp16: Option<u8>,
    dpp8: Option<u8>,
}

/// The decoder of gfx1201.
pub(crate) struct Decoder {
    lut: Vec<u8>,
    families: Vec<Family>,
    classes: Vec<Class>,
    ops: Vec<OpInfo>,
}

fn mask(s: Slot) -> u128 {
    ((1u128 << s.bits) - 1) << s.lo
}

fn get(raw: u128, s: Slot) -> u32 {
    ((raw >> s.lo) & ((1u128 << s.bits) - 1)) as u32
}

/// The encoder's fresh word (`Word::new`): fixed fields, defaults, opcode.
fn fresh(e: &Encoding, opcode: u64) -> u128 {
    e.fields.iter().fold(0, |w, f| {
        let v = if f.kind == "op" { opcode } else { f.value };
        w | u128::from(v) << f.lo
    })
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn xml_of(t: &str) -> Xml {
    match t {
        "OPR_SRC" => Xml::Src,
        "OPR_SSRC" => Xml::Ssrc,
        "OPR_SREG" => Xml::Sreg,
        "OPR_VGPR" => Xml::Vgpr,
        "OPR_SRC_VGPR" => Xml::SrcVgpr,
        "OPR_SRC_VGPR_OR_INLINE" => Xml::VgprOrInline,
        "OPR_SSRC_BARRIER_ID" => Xml::Barrier,
        "OPR_SSRC_LANESEL" => Xml::Lanesel,
        "OPR_SIMM16" => Xml::Simm16,
        "OPR_HWREG" => Xml::Hwreg,
        "OPR_SENDMSG" | "OPR_SENDMSG_RTN" => Xml::Sendmsg,
        "OPR_WAIT_ALU" => Xml::WaitAlu,
        "OPR_LABEL" => Xml::Label,
        "OPR_DELAY" => Xml::Delay,
        "OPR_CLAUSE" | "OPR_SLEEP" | "OPR_WAIT_EVENT" | "OPR_WAIT_MEM_DS" | "OPR_VERSION" => {
            Xml::Imm16
        }
        _ => Xml::Other,
    }
}

fn rule_of(class: &str) -> Rule {
    match class {
        "SOP1" | "SOP2" | "SOPC" | "SOPK" => Rule::Salu,
        "SOPP" => Rule::Sopp,
        "SMEM" => Rule::Smem,
        "VOP1" | "VOP1_SDST" | "VOP2" | "VOPC" => Rule::Short,
        "VOP3" | "VOP3_SDST" | "VOP3SD" => Rule::Vop3,
        "VOP3P" => Rule::Vop3p,
        "VGLOBAL" => Rule::Global,
        "VBUFFER" => Rule::Buffer,
        "DS" => Rule::Ds,
        c if c.ends_with("_LIT") => Rule::Refused("its class takes a literal of its own"),
        _ => Rule::Refused("thin's encoder has no rule for its class"),
    }
}

fn bad(what: String) -> Error {
    Error::Definition(format!("the decoder: {what}"))
}

impl Decoder {
    fn build(data: &'static Data) -> Result<Decoder, Error> {
        let isa = &data.isa;
        let rows: HashSet<&str> = data.rows.iter().map(|r| r.mnemonic.as_str()).collect();
        let mut class_id: HashMap<&'static str, u16> = HashMap::new();
        let mut classes = Vec::new();
        for (name, e) in &isa.encodings {
            let mut slots = [Slot::default(); NF];
            let mut fields = Vec::new();
            let mut srcs = 0u64;
            for f in &e.fields {
                let slot = Slot {
                    lo: f.lo as u8,
                    bits: f.bits() as u8,
                };
                fields.push((f.name.as_str(), mask(slot)));
                if let Some(id) = NAMES.iter().position(|n| *n == f.name) {
                    slots[id] = slot;
                }
                if matches!(f.kind.as_str(), "src" | "ssrc") && !f.name.contains("dst") {
                    let id = (NAMES.iter().position(|n| *n == f.name))
                        .ok_or_else(|| bad(format!("{name}: a source field it does not name")))?;
                    srcs |= 1 << id;
                }
            }
            class_id.insert(name.as_str(), classes.len() as u16);
            classes.push(Class {
                name: name.as_str(),
                bytes: e.bytes as u8,
                slots,
                fields,
                srcs,
                lit: isa.encodings.contains_key(&format!("{name}_LIT")),
                literal_ok: LITERAL_CLASSES.contains(&name.as_str()),
            });
        }
        // Families: the classes grouped by their fixed field.
        let mut groups: BTreeMap<(u32, u32, u64), Vec<&'static str>> = BTreeMap::new();
        for (name, e) in &isa.encodings {
            let fixed = (e.fields.iter().find(|f| f.kind == "fixed"))
                .ok_or_else(|| bad(format!("{name}: no fixed field")))?;
            if fixed.hi != 31 || fixed.lo < 23 {
                return Err(bad(format!("{name}: fixed bits outside 31..23")));
            }
            groups
                .entry((fixed.hi, fixed.lo, fixed.value))
                .or_default()
                .push(name.as_str());
        }
        let mut families = Vec::new();
        let mut family_of: HashMap<&str, usize> = HashMap::new();
        let mut lut = vec![u8::MAX; 512];
        let mut width = vec![0u32; 512];
        for ((hi, lo, value), names) in &groups {
            let root = (names.iter())
                .filter(|n| !n.ends_with("_LIT") && !n.contains("_DPP"))
                .min_by_key(|n| n.len())
                .ok_or_else(|| bad(format!("{names:?}: no base class")))?;
            let layout = &classes[class_id[root] as usize];
            let op = layout.slots[OP];
            let variant = |suffix: &str| {
                (names.iter())
                    .find(|n| n.ends_with(suffix))
                    .map(|n| classes[class_id[n] as usize].bytes)
            };
            let f = families.len();
            for n in names {
                family_of.insert(*n, f);
                let s = classes[class_id[n] as usize].slots[OP];
                if s.bits != op.bits || s.lo != op.lo {
                    return Err(bad(format!(
                        "{n}: its op field moves within {root}'s family"
                    )));
                }
            }
            families.push(Family {
                class: class_id[root],
                op,
                ops: vec![u16::MAX; 1 << op.bits],
                dpp16: variant("_DPP16"),
                dpp8: variant("_DPP8"),
            });
            let bits = hi - lo + 1;
            for (index, slot) in lut.iter_mut().enumerate() {
                if (index as u64) >> (lo - 23) != *value {
                    continue;
                }
                if bits == width[index] {
                    return Err(bad(format!(
                        "{root}: its fixed bits overlap another family's"
                    )));
                }
                if bits > width[index] {
                    width[index] = bits;
                    *slot = f as u8;
                }
            }
        }
        let mut names: Vec<&String> = isa.ops.keys().collect();
        names.sort();
        // A family without an op field (VOPD pairs two ops in other fields)
        // is decoded only when it has a single op (export).
        let mut per_family = vec![0usize; families.len()];
        for op in isa.ops.values() {
            per_family[family_of[op.class.as_str()]] += 1;
        }
        let mut ops = Vec::new();
        for name in names {
            let op = &isa.ops[name];
            let f = family_of[op.class.as_str()];
            if families[f].op.bits == 0 && per_family[f] > 1 {
                continue;
            }
            let info = op_info(isa, &rows, &classes, &class_id, name, op)?;
            let family = &mut families[f];
            let slot = (family.ops.get_mut(op.opcode as usize))
                .ok_or_else(|| bad(format!("{name}: opcode outside its family")))?;
            if *slot != u16::MAX {
                return Err(bad(format!("{name}: a second op of its opcode")));
            }
            *slot = ops.len() as u16;
            ops.push(info);
        }
        Ok(Decoder {
            lut,
            families,
            classes,
            ops,
        })
    }
}

/// Resolves an op: its rule, operands, fresh word and predicates.
fn op_info(
    isa: &'static crate::tables::IsaTables,
    rows: &HashSet<&str>,
    classes: &[Class],
    class_id: &HashMap<&'static str, u16>,
    name: &'static str,
    op: &'static Op,
) -> Result<OpInfo, Error> {
    let encoding = &isa.encodings[&op.class];
    let mut opd = [None; NF];
    let mut srcs = 0u64;
    let class = class_id[op.class.as_str()];
    for o in &op.operands {
        let Some(id) = NAMES.iter().position(|n| *n == o.field) else {
            continue;
        };
        opd[id] = Some(Opnd {
            dwords: dwords(o) as u8,
            sixteen: o.bits == 16,
            wide: matches!(
                o.format.as_str(),
                "FMT_NUM_F32" | "FMT_NUM_B32" | "FMT_NUM_U32" | "FMT_NUM_I32"
            ),
            xml: xml_of(&o.xml_type),
        });
        // A message id in a source field is its raw byte, never a literal.
        if xml_of(&o.xml_type) != Xml::Sendmsg {
            srcs |= classes[class as usize].srcs & 1 << id;
        }
    }
    let mut rule = rule_of(&op.class);
    if name == "s_setreg_imm32_b32" {
        rule = Rule::Setreg;
    }
    if rule == Rule::Buffer && (name.starts_with("tbuffer") || op.operands.is_empty()) {
        rule = Rule::Refused("thin's encoder takes untyped buffer ops with data only");
    }
    if SPELLED_OTHERWISE.contains(&name) {
        rule = Rule::Refused("LLVM spells it otherwise on gfx1201");
    }
    let print: &'static str = match op.class.as_str() {
        "VOP3SD" if isa.ops.contains_key(&format!("{name}_e32")) => leak(format!("{name}_e64")),
        _ => name,
    };
    let dpp = match (op.class.as_str(), name.strip_suffix("_e32")) {
        ("VOP1" | "VOP2", Some(base)) => {
            let mnemonic = format!("{base}_dpp");
            let dpp_class = format!("{}_DPP16", op.class);
            if !rows.contains(mnemonic.as_str()) {
                Err("no DPP form in the operand table")
            } else if SPELLED_OTHERWISE.contains(&name) {
                Err("LLVM spells it otherwise on gfx1201")
            } else {
                let e = (isa.encodings.get(&dpp_class))
                    .ok_or_else(|| bad(format!("no class {dpp_class}")))?;
                Ok(DppForm {
                    mnemonic: leak(mnemonic),
                    class: class_id[dpp_class.as_str()],
                    base: fresh(e, op.opcode),
                })
            }
        }
        _ => Err("thin's encoder writes DPP16 for VOP1 and VOP2 only"),
    };
    let cache_kind = if name.contains("atomic") {
        0x300
    } else if name.contains("store") {
        0x200
    } else {
        0x100
    };
    Ok(OpInfo {
        mnemonic: name,
        op,
        print,
        class,
        rule,
        opd,
        base: fresh(encoding, op.opcode),
        srcs,
        cache_kind,
        two_addr: name.contains("2addr"),
        shape: short_shape(&op.class, name, &opd),
        opsel: opsel_bits(name),
        signs: ["src0", "src1", "src2"].map(|f| {
            op.operand(f)
                .map_or((false, false), |o| signs_of(name, &op.class, o))
        }),
        packed: packed_rules(print),
        dpp,
    })
}

/// The implicit `vcc_lo` of a short form's shape.
const VCC: usize = usize::MAX;

/// A short form's fields in line order (the encoder's `short_operands`
/// and `fits_shape`), or none when the op's operands are not a short form's.
fn short_shape(class: &str, name: &str, opd: &[Option<Opnd>; NF]) -> Option<&'static [usize]> {
    let n = opd.iter().filter(|o| o.is_some()).count();
    let shape: &'static [usize] = match class {
        _ if n == 0 => &[],
        "VOPC" if name.starts_with("v_cmpx") => &[SRC0, VSRC1],
        "VOPC" => &[VCC, SRC0, VSRC1],
        "VOP2" if name.starts_with("v_cndmask") => &[VDST, SRC0, VSRC1, VCC],
        "VOP2" if name.contains("_co_ci_") => &[VDST, VCC, SRC0, VSRC1, VCC],
        "VOP2" => &[VDST, SRC0, VSRC1],
        _ => &[VDST, SRC0],
    };
    let fields = shape.iter().filter(|&&f| f != VCC).count();
    let all = shape.iter().all(|&f| f == VCC || opd[f].is_some());
    (fields == n && all).then_some(shape)
}

/// The decoder, built on first use.
pub(crate) fn decoder() -> Result<&'static Decoder, Error> {
    static DECODER: OnceLock<Result<Decoder, Error>> = OnceLock::new();
    let built = DECODER.get_or_init(|| Decoder::build(data()?));
    built.as_ref().map_err(Clone::clone)
}

/// A named register of a number and width; NULL is any width (it reads
/// zero and drops writes).
fn special(c: u32, width: u32) -> Option<Opd> {
    (SPECIALS.iter())
        .find(|s| s.1 == c && (s.2 == width || s.0 == "null"))
        .map(Opd::Named)
}

/// A scalar operand of a code and width: numbered and aligned as the
/// encoder requires (a pair at an even SGPR, wider at a multiple of 4), or
/// named.
fn scalar(c: u32, width: u32) -> Option<Opd> {
    if c + width <= VCC_LO {
        if width > 1 && c % super::encoder::scalar_align(width) != 0 {
            return None;
        }
        return Some(Opd::S {
            n: c as u16,
            count: width as u16,
        });
    }
    special(c, width)
}

fn sext(v: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    (i64::from(v) << shift) >> shift
}

/// One instruction being decoded.
struct Cx<'d, 'o> {
    d: &'d Decoder,
    at: usize,
    raw: u128,
    len: u8,
    class: &'d Class,
    info: &'d OpInfo,
    base: u128,
    consumed: u128,
    literal: Option<u32>,
    ins: &'o mut Instruction,
}

/// A rule's result; the refusal is boxed so the happy path moves little.
type R<T> = Result<T, Box<DecodeError>>;

impl Cx<'_, '_> {
    #[cold]
    fn err(&self, why: Why, field: Option<&'static str>, detail: String) -> Box<DecodeError> {
        Box::new(DecodeError {
            offset: self.at,
            bits: self.raw,
            len: self.len,
            size: Some(self.len),
            why,
            class: self.class.name,
            mnemonic: Some(self.info.mnemonic),
            field,
            detail,
        })
    }

    /// The XML type of the op's operand in a field.
    fn xml_name(&self, f: usize) -> &'static str {
        (self.info.op.operands.iter())
            .find(|o| o.field == NAMES[f])
            .map_or("?", |o| o.xml_type.as_str())
    }

    fn get(&self, f: usize) -> u32 {
        get(self.raw, self.class.slots[f])
    }

    fn take(&mut self, f: usize) -> u32 {
        self.consumed |= mask(self.class.slots[f]);
        self.get(f)
    }

    fn opnd(&self, f: usize) -> Opnd {
        self.info.opd[f].expect("the rule reads an operand the op has")
    }

    fn push(&mut self, o: Opd) {
        self.ins.push(o);
    }

    /// A register operand of a file and width (the encoder's `register`).
    fn reg_of(&self, f: usize, c: u32, vector: bool, width: u32) -> R<Opd> {
        if vector {
            if c + width > 256 {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("v{c} +{width}: past v255"),
                ));
            }
            return Ok(Opd::V {
                n: c as u16,
                count: width as u16,
            });
        }
        if c + width <= VCC_LO {
            let align = super::encoder::scalar_align(width);
            if width > 1 && c % align != 0 {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("s{c}: a scalar range starts at a multiple of {align}"),
                ));
            }
            return Ok(Opd::S {
                n: c as u16,
                count: width as u16,
            });
        }
        special(c, width).ok_or_else(|| self.unnamed(f, c, width))
    }

    #[cold]
    fn unnamed(&self, f: usize, c: u32, width: u32) -> Box<DecodeError> {
        self.err(
            Why::Operand,
            Some(NAMES[f]),
            format!("scalar code {c} is no {width}-dword register thin's syntax names"),
        )
    }

    fn register(&mut self, f: usize, vector: bool) -> R<Opd> {
        let width = u32::from(self.opnd(f).dwords);
        let c = self.take(f);
        self.reg_of(f, c, vector, width)
    }

    /// A scalar destination of the VOP3 rule: numbered at any alignment, or named.
    fn sdst(&mut self, f: usize) -> R<Opd> {
        let width = u32::from(self.opnd(f).dwords);
        let c = self.take(f);
        scalar(c, width).ok_or_else(|| self.unnamed(f, c, width))
    }

    /// A source operand (the encoder's `source` and `constant`).
    fn source(&mut self, f: usize) -> R<Opd> {
        let o = self.opnd(f);
        let (vgprs, sgprs, constants, barrier) = match o.xml {
            Xml::Src => (true, true, true, false),
            Xml::Ssrc | Xml::Lanesel => (false, true, true, false),
            Xml::Sreg => (false, true, false, false),
            Xml::Vgpr | Xml::SrcVgpr => (true, false, false, false),
            Xml::VgprOrInline => (true, false, true, false),
            Xml::Barrier => (false, true, true, true),
            Xml::Sendmsg => return Ok(Opd::Int(i64::from(self.take(f)))),
            Xml::Simm16
            | Xml::Hwreg
            | Xml::WaitAlu
            | Xml::Label
            | Xml::Delay
            | Xml::Imm16
            | Xml::Other => {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("operand type {} is not taken", self.xml_name(f)),
                ))
            }
        };
        let width = if barrier { 1 } else { u32::from(o.dwords) };
        let c = self.take(f);
        let refuse = |cx: &Self, what: String| Err(cx.err(Why::Operand, Some(NAMES[f]), what));
        if o.xml == Xml::Lanesel && c == 255 {
            return refuse(self, "a lane select takes no literal".into());
        }
        if c >= 256 {
            if !vgprs {
                return refuse(self, format!("v{}: no VGPR here", c - 256));
            }
            return self.reg_of(f, c - 256, true, width);
        }
        if let Some(s) = scalar(c, width) {
            if !sgprs {
                return refuse(self, format!("scalar code {c}: no scalar register here"));
            }
            if barrier && !matches!(s, Opd::Named(("m0", _, _))) {
                return refuse(
                    self,
                    format!("scalar code {c}: a barrier id is m0 or an inline constant"),
                );
            }
            return Ok(s);
        }
        if !constants {
            return refuse(
                self,
                format!(
                    "code {c}: no {width}-dword register thin's syntax names, and no constant here"
                ),
            );
        }
        match c {
            128..=192 => Ok(Opd::Int(i64::from(c) - 128)),
            193..=208 => Ok(Opd::Int(192 - i64::from(c))),
            240..=247 if o.wide => Ok(Opd::Float(INLINE_FLOATS[c as usize - 240])),
            248 if o.wide && width == 1 => Ok(Opd::Int(0x3E22_F983)),
            240..=248 => refuse(self, format!("inline float code {c} in a {} operand: thin writes inline floats in 32-bit operands only", self.xml_name(f))),
            255 if barrier => refuse(self, "a literal barrier id".into()),
            255 => self.literal(f, o, width),
            _ => refuse(self, format!("code {c}: no {width}-dword operand thin's syntax names")),
        }
    }

    fn literal(&mut self, f: usize, o: Opnd, width: u32) -> R<Opd> {
        let l = self.literal.expect("a 255 source reads the literal");
        let refuse = |cx: &Self, what: String| Err(cx.err(Why::Operand, Some(NAMES[f]), what));
        if !o.wide || width != 1 {
            return refuse(
                self,
                format!("literal 0x{l:x}: thin writes literals in 32-bit operands only"),
            );
        }
        if l <= 64 || INLINE_FLOATS.iter().any(|x| x.to_bits() == l) || l == 0x3E22_F983 {
            return refuse(self, format!("literal 0x{l:x}: thin writes it inline"));
        }
        Ok(Opd::Int(i64::from(l)))
    }

    /// The residual check: every bit no rule read holds the fresh word's value.
    fn finish(self) -> R<()> {
        let bytes = u32::from(self.class.bytes);
        let word = if bytes >= 16 {
            self.raw
        } else {
            self.raw & ((1u128 << (8 * bytes)) - 1)
        };
        let diff = (word ^ self.base) & !self.consumed;
        if diff != 0 {
            let field = (self.class.fields.iter()).find(|(_, m)| m & diff != 0);
            return Err(match field {
                Some(&(name, m)) => {
                    let lo = m.trailing_zeros();
                    self.err(
                        Why::Field,
                        Some(name),
                        format!(
                            "{name} = {}: thin writes {}",
                            (word & m) >> lo,
                            (self.base & m) >> lo
                        ),
                    )
                }
                None => self.err(
                    Why::Field,
                    Some("(no field)"),
                    format!("bits 0x{diff:x} outside every field"),
                ),
            });
        }
        self.ins.size = self.len;
        Ok(())
    }

    fn salu(&mut self) -> R<()> {
        // s_setreg_b32 is written hardware register first (the encoder's
        // `salu`).
        let order = if self.info.mnemonic == "s_setreg_b32" {
            [SIMM16, SDST, SSRC0, SSRC1]
        } else {
            [SDST, SSRC0, SSRC1, SIMM16]
        };
        for f in order {
            let Some(o) = self.info.opd[f] else { continue };
            let opd = match f {
                SDST => self.register(f, false)?,
                SIMM16 if o.xml == Xml::Hwreg => self.hwreg()?,
                SIMM16 if !matches!(o.xml, Xml::Simm16 | Xml::Imm16) => {
                    return Err(self.err(
                        Why::Operand,
                        Some("simm16"),
                        format!(
                            "operand type {}: not a 16-bit immediate here",
                            self.xml_name(f)
                        ),
                    ))
                }
                SIMM16 => Opd::Int(i64::from(self.take(f))),
                _ => self.source(f)?,
            };
            self.push(opd);
        }
        Ok(())
    }

    fn sopp(&mut self) -> R<()> {
        let Some(o) = self.info.opd[SIMM16] else {
            return Ok(());
        };
        let v = self.take(SIMM16);
        match o.xml {
            Xml::WaitAlu => {
                self.ins.counted = true;
                if v & !DEPCTR_NONE != 0 || v == DEPCTR_NONE {
                    self.ins.modify("simm16", MVal::Int(v as i32));
                } else {
                    for &(name, shift, bits) in &DEPCTR {
                        let n = (v >> shift) & ((1 << bits) - 1);
                        if n != (1 << bits) - 1 {
                            self.ins.modify(name, MVal::Int(n as i32));
                        }
                    }
                }
            }
            Xml::Label => {
                let target = self.at as i64 + 4 + 4 * sext(v, 16);
                if target < 0 {
                    return Err(self.err(
                        Why::Branch,
                        Some("simm16"),
                        format!("branch to {target}, before the bytes"),
                    ));
                }
                self.push(Opd::Label(target as u32));
            }
            Xml::Simm16 | Xml::Delay | Xml::Sendmsg | Xml::Imm16 => {
                self.push(Opd::Int(i64::from(v)))
            }
            _ => {
                let what = format!("operand type {} is not taken", self.xml_name(SIMM16));
                return Err(self.err(Why::Operand, Some("simm16"), what));
            }
        }
        Ok(())
    }

    /// The cache fields (the encoder's `cache`).
    fn cache(&mut self, has_dst: Option<bool>) -> R<()> {
        let kind = self.info.cache_kind;
        let th = self.take(TH);
        if kind == 0x300 && has_dst.is_some_and(|d| (th & 1 == 1) != d) {
            return Err(self.err(
                Why::Field,
                Some("th"),
                format!("th = {th}: TH_ATOMIC_RETURN exactly on the returning form"),
            ));
        }
        let scope = self.take(SCOPE_F);
        if th != 0 {
            let v = (u32::from(kind) | th) as i32;
            // Code 3 of a load or store at SCOPE_SYS is spelled BYPASS.
            let bypass = kind != 0x300 && th == 3 && scope == 3;
            let symbol = i128::from(v) | if bypass { TH_BYPASS } else { 0 };
            let value = match TH_SYMBOLS.iter().find(|s| s.1 == symbol) {
                Some(&(name, _)) => MVal::Symbol(name, symbol as i32),
                None => MVal::Int(v),
            };
            self.ins.modify("th", value);
        }
        if scope != 0 {
            self.ins
                .modify("scope", MVal::Symbol(SCOPE[scope as usize], scope as i32));
        }
        Ok(())
    }

    fn smem(&mut self) -> R<()> {
        let (Some(_), Some(base)) = (self.info.opd[SDATA], self.info.opd[SBASE]) else {
            return Err(self.err(
                Why::Form,
                None,
                "not a scalar load: no sdata and sbase".into(),
            ));
        };
        let data = self.register(SDATA, false)?;
        let b = self.take(SBASE) * 2;
        let base = self.reg_of(SBASE, b, false, u32::from(base.dwords))?;
        let soffset = self.take(SOFFSET);
        let offset = sext(self.take(IOFFSET), 24);
        self.push(data);
        self.push(base);
        if soffset == NULL {
            self.push(Opd::Int(offset));
        } else {
            let s = scalar(soffset, 1).ok_or_else(|| self.unnamed(SOFFSET, soffset, 1))?;
            self.push(s);
            if offset != 0 {
                self.ins.modify("offset", MVal::Int(offset as i32));
            }
        }
        self.cache(Some(false))
    }

    fn vop3(&mut self) -> R<()> {
        for f in [VDST, SDST, SRC0, SRC1, SRC2] {
            let Some(o) = self.info.opd[f] else { continue };
            let opd = match f {
                VDST if o.xml == Xml::Vgpr => self.register(f, true)?,
                VDST | SDST => self.sdst(f)?,
                _ => self.source(f)?,
            };
            self.push(opd);
        }
        Ok(())
    }

    fn bits(&self, f: usize, key: &str, v: u32, nsrc: u32, mask: u64) -> R<MVal> {
        if mask == 0 || v >> nsrc != 0 || u64::from(v) & !mask != 0 {
            return Err(self.err(
                Why::Field,
                Some(NAMES[f]),
                format!("{key} 0x{v:x}: not a bit this op takes (it takes 0x{mask:x} of {nsrc})"),
            ));
        }
        let mut items = [0u8; 4];
        for (j, item) in items.iter_mut().enumerate().take(nsrc as usize) {
            *item = (v >> j & 1) as u8;
        }
        Ok(MVal::List(items, nsrc as u8))
    }

    /// The packed modifiers (the encoder's `packed`).
    fn packed(&mut self) -> R<()> {
        let nsrc = [SRC0, SRC1, SRC2]
            .iter()
            .filter(|&&f| self.info.opd[f].is_some())
            .count() as u32;
        let (masks, clamp, hi_default, literals) = self.info.packed;
        let wide_int = (self.ins.operands().iter())
            .any(|o| matches!(o, Opd::Int(v) if !(-16..=64).contains(v)));
        if !literals && (self.literal.is_some() || wide_int) {
            return Err(self.err(Why::Operand, None, "no literal here".into()));
        }
        let opsel = self.take(OPSEL);
        let hi = self.take(OPSEL_HI);
        let hi2 = self.take(OPSEL_HI2);
        let neg = self.take(NEG);
        let neg_hi = self.take(NEG_HI);
        let hi2_default = (self.base & mask(self.class.slots[OPSEL_HI2]) != 0) as u32;
        let default_hi = (
            hi_default as u32 & 3,
            if nsrc == 3 {
                (hi_default as u32 >> 2) & 1
            } else {
                hi2_default
            },
        );
        let mut mods = [None; 4];
        if opsel != 0 {
            mods[0] = Some(("op_sel", self.bits(OPSEL, "op_sel", opsel, nsrc, masks[0])?));
        }
        if (hi, hi2) != default_hi {
            if nsrc < 3 && hi2 != hi2_default {
                return Err(self.err(
                    Why::Field,
                    Some("opsel_hi2"),
                    format!("opsel_hi2 = {hi2}: thin writes {hi2_default}"),
                ));
            }
            let v = hi | if nsrc == 3 { hi2 << 2 } else { 0 };
            mods[1] = Some((
                "op_sel_hi",
                self.bits(OPSEL_HI, "op_sel_hi", v, nsrc, masks[1])?,
            ));
        }
        if neg != 0 {
            mods[2] = Some(("neg_lo", self.bits(NEG, "neg_lo", neg, nsrc, masks[2])?));
        }
        if neg_hi != 0 {
            mods[3] = Some((
                "neg_hi",
                self.bits(NEG_HI, "neg_hi", neg_hi, nsrc, masks[3])?,
            ));
        }
        for (key, value) in mods.into_iter().flatten() {
            self.ins.modify(key, value);
        }
        if self.take(CLMP) == 1 {
            if !clamp {
                return Err(self.err(Why::Field, Some("clmp"), "clamp: not taken".into()));
            }
            self.ins.modify("clamp", MVal::Flag);
        }
        Ok(())
    }

    /// A VOP3 op_sel (the encoder's `vop3_opsel`): the ops that take it, and
    /// only their bits.
    fn vop3_opsel(&mut self) -> R<()> {
        if self.class.slots[OPSEL].bits == 0 || self.info.opsel == 0 {
            return Ok(());
        }
        let v = self.take(OPSEL);
        if v == 0 {
            return Ok(());
        }
        if v & !self.info.opsel != 0 {
            let what = format!(
                "opsel = {v}: {} reads OPSEL bits 0x{:x} only",
                self.info.mnemonic, self.info.opsel
            );
            return Err(self.err(Why::Field, Some("opsel"), what));
        }
        let nsrc = [SRC0, SRC1, SRC2]
            .iter()
            .filter(|&&f| self.info.opd[f].is_some())
            .count();
        let mut items = [0u8; 4];
        for (j, item) in items.iter_mut().enumerate().take(nsrc + 1) {
            let bit = if j == nsrc { 3 } else { j };
            *item = (v >> bit & 1) as u8;
        }
        self.ins.modify("op_sel", MVal::List(items, nsrc as u8 + 1));
        Ok(())
    }

    /// VOP3 sign modifiers (the encoder's `vop3_signs`): NEG[i] and ABS[i]
    /// of source i, on the sources that take them.
    fn vop3_signs(&mut self) -> R<()> {
        let (neg_slot, abs_slot) = (self.class.slots[NEG], self.class.slots[ABS]);
        let neg = if neg_slot.bits > 0 { self.take(NEG) } else { 0 };
        let abs = if abs_slot.bits > 0 { self.take(ABS) } else { 0 };
        if neg | abs == 0 {
            return Ok(());
        }
        let mut k = [VDST, SDST]
            .iter()
            .filter(|&&f| self.info.opd[f].is_some())
            .count();
        for (i, f) in [SRC0, SRC1, SRC2].into_iter().enumerate() {
            let (n, a) = (neg >> i & 1 == 1, abs >> i & 1 == 1);
            if self.info.opd[f].is_none() {
                if n || a {
                    let field = if a { "abs" } else { "neg" };
                    return Err(self.err(
                        Why::Field,
                        Some(field),
                        format!(
                            "{field} bit {i}: {} has no {}",
                            self.info.mnemonic, NAMES[f]
                        ),
                    ));
                }
                continue;
            }
            let (takes_neg, takes_abs) = self.info.signs[i];
            if (n && !takes_neg) || (a && !takes_abs) {
                let field = if a && !takes_abs { "abs" } else { "neg" };
                return Err(self.err(
                    Why::Field,
                    Some(field),
                    format!(
                        "{field} on {}: {} takes none there",
                        NAMES[f], self.info.mnemonic
                    ),
                ));
            }
            // `-5` is the constant -5, not a negated 5: a negated constant is
            // written `-|c|`, so it needs abs too.
            let constant = matches!(self.ins.operands()[k], Opd::Int(_) | Opd::Float(_));
            if n && !a && constant {
                return Err(self.err(
                    Why::Operand,
                    Some("neg"),
                    format!(
                        "neg on the constant in {}: the line syntax writes no negated constant",
                        NAMES[f]
                    ),
                ));
            }
            self.ins.sign(k, n, a);
            k += 1;
        }
        Ok(())
    }

    /// The short forms' fields in line order (`VCC` an implicit vcc_lo).
    fn short_shape(&self) -> R<&'static [usize]> {
        (self.info.shape).ok_or_else(|| {
            self.err(
                Why::Form,
                None,
                "its operands are not a short form's".into(),
            )
        })
    }

    /// The halves rule: a 16-bit VGPR operand of a short form lies below v128.
    fn halves(&self, f: usize, o: &Opd) -> R<()> {
        let sixteen = self.info.opd[f].is_some_and(|o| o.sixteen);
        if let (true, Opd::V { n, count }) = (sixteen, o) {
            if n + count > 128 {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("v{n}: VGPRs from v128 need the e64 form (a 16-bit high half)"),
                ));
            }
        }
        Ok(())
    }

    fn short(&mut self) -> R<()> {
        let sdst = self.class.name == "VOP1_SDST";
        for &f in self.short_shape()? {
            let o = match f {
                VCC => special(VCC_LO, 1).expect("vcc_lo is named"),
                VDST if sdst => self.register(f, false)?,
                VDST | VSRC1 => self.register(f, true)?,
                _ => self.source(f)?,
            };
            if f != VCC {
                self.halves(f, &o)?;
            }
            self.push(o);
        }
        Ok(())
    }

    fn dpp(&mut self) -> R<()> {
        let d = self
            .info
            .dpp
            .map_err(|why| self.err(Why::NotEncoded, Some("src0"), format!("DPP16: {why}")))?;
        self.class = &self.d.classes[d.class as usize];
        self.base = d.base;
        self.ins.mnemonic = d.mnemonic;
        self.take(SRC0);
        for &f in self.short_shape()? {
            let o = match f {
                VCC => special(VCC_LO, 1).expect("vcc_lo is named"),
                _ => {
                    let width = u32::from(self.opnd(f).dwords);
                    let field = if f == SRC0 { VSRC0 } else { f };
                    let c = self.take(field);
                    if width != 1 && self.class.name == "VOP2_DPP16" && f != SRC0 {
                        return Err(self.err(
                            Why::Form,
                            Some(NAMES[f]),
                            format!(
                                "a {width}-dword operand: thin writes DPP16 with one-VGPR operands"
                            ),
                        ));
                    }
                    let o = self.reg_of(field, c, true, 1)?;
                    self.halves(f, &o)?;
                    o
                }
            };
            self.push(o);
        }
        let dpp = self.take(DPP);
        let (key, value) = match dpp {
            0x000..=0x0FF => {
                let q = |i: u32| ((dpp >> (2 * i)) & 3) as u8;
                ("quad_perm", MVal::List([q(0), q(1), q(2), q(3)], 4))
            }
            0x101..=0x10F => ("row_shl", MVal::Int((dpp - 0x100) as i32)),
            0x111..=0x11F => ("row_shr", MVal::Int((dpp - 0x110) as i32)),
            0x121..=0x12F => ("row_ror", MVal::Int((dpp - 0x120) as i32)),
            0x140 => ("row_mirror", MVal::Flag),
            0x141 => ("row_half_mirror", MVal::Flag),
            0x150..=0x15F => ("row_share", MVal::Int((dpp - 0x150) as i32)),
            0x160..=0x16F => ("row_xmask", MVal::Int((dpp - 0x160) as i32)),
            _ => {
                return Err(self.err(
                    Why::Field,
                    Some("dpp"),
                    format!("dpp control 0x{dpp:x}: none thin's syntax names"),
                ))
            }
        };
        self.ins.modify(key, value);
        let row = self.take(ROW_MASK);
        let bank = self.take(BANK_MASK);
        self.ins.modify("row_mask", MVal::Int(row as i32));
        self.ins.modify("bank_mask", MVal::Int(bank as i32));
        if self.take(BC) == 1 {
            self.ins.modify("bound_ctrl", MVal::Int(1));
        }
        if self.take(FI) == 1 {
            self.ins.modify("fi", MVal::Int(1));
        }
        Ok(())
    }

    fn global(&mut self) -> R<()> {
        let (mut dst, data) = (self.info.opd[VDST].is_some(), self.info.opd[VSRC].is_some());
        if self.info.cache_kind == 0x300 && self.get(TH) & 1 == 0 {
            // A non-returning atomic: no destination, its field left 0.
            dst = false;
        }
        if !dst && !data {
            return Err(self.err(
                Why::Form,
                None,
                "neither a load nor a store: no vdst or vsrc".into(),
            ));
        }
        if dst {
            let d = self.register(VDST, true)?;
            self.push(d);
        }
        let saddr = self.take(SADDR);
        let pair = if saddr == NULL {
            None
        } else {
            Some(self.reg_of(SADDR, saddr, false, 2)?)
        };
        let vaddr = self.take(VADDR);
        let addr = self.reg_of(VADDR, vaddr, true, if pair.is_some() { 1 } else { 2 })?;
        self.push(addr);
        if data {
            let v = self.register(VSRC, true)?;
            self.push(v);
        }
        self.push(pair.unwrap_or(Opd::Off));
        let offset = sext(self.take(IOFFSET), 24);
        if offset != 0 {
            self.ins.modify("offset", MVal::Int(offset as i32));
        }
        self.cache(Some(dst))
    }

    /// SIMM16 as a hardware register's bits: by name or, unnamed, by id.
    fn hwreg(&mut self) -> R<Opd> {
        let code = self.take(SIMM16);
        let id = code & 63;
        let name = HWREG.iter().find(|r| r.1 == id).map(|r| r.0);
        let (offset, size) = (code >> 6 & 31, (code >> 11) + 1);
        if offset + size > 32 {
            return Err(self.err(
                Why::Field,
                Some("simm16"),
                format!("hwreg bits {offset}+{size}: past bit 31"),
            ));
        }
        Ok(Opd::Hwreg {
            name,
            offset: offset as u8,
            size: size as u8,
            code: code as u16,
        })
    }

    /// `s_setreg_imm32_b32` (the encoder's `setreg_imm32`): a hardware
    /// register's bits, by name or, unnamed, by id; and the literal dword.
    fn setreg_imm32(&mut self) -> R<()> {
        let reg = self.hwreg()?;
        self.push(reg);
        let imm = self.take(LITERAL);
        self.push(Opd::Int(i64::from(imm)));
        Ok(())
    }

    /// Buffer loads, stores and atomics (the encoder's `buffer`).
    fn buffer(&mut self) -> R<()> {
        let data = self.register(VDATA, true)?;
        let (idxen, offen) = (self.take(IDXEN), self.take(OFFEN));
        let vaddr = self.take(VADDR);
        let addr = match idxen + offen {
            0 if vaddr == 0 => Opd::Off,
            0 => {
                return Err(self.err(
                    Why::Field,
                    Some("vaddr"),
                    format!("vaddr = {vaddr} with neither idxen nor offen: thin writes 0"),
                ))
            }
            n => self.reg_of(VADDR, vaddr, true, n)?,
        };
        let rsrc = self.take(RSRC);
        let rsrc = self.reg_of(RSRC, rsrc, false, 4)?;
        let c = self.take(SOFFSET);
        let soffset = match scalar(c, 1) {
            Some(s) if soffset_code(c) => s,
            _ => {
                return Err(self.err(
                    Why::Operand,
                    Some("soffset"),
                    format!("scalar code {c}: the offset is an SGPR, m0 or null"),
                ))
            }
        };
        for o in [data, addr, rsrc, soffset] {
            self.push(o);
        }
        if idxen == 1 {
            self.ins.modify("idxen", MVal::Flag);
        }
        if offen == 1 {
            self.ins.modify("offen", MVal::Flag);
        }
        let offset = self.take(IOFFSET);
        if i128::from(offset) > BUFFER_OFFSET_MAX {
            return Err(self.err(
                Why::Field,
                Some("ioffset"),
                format!("ioffset = 0x{offset:x}: must be non-negative"),
            ));
        }
        if offset != 0 {
            self.ins.modify("offset", MVal::Int(offset as i32));
        }
        let format = self.take(FORMAT);
        if u64::from(format) != UNTYPED_FORMAT {
            return Err(self.err(
                Why::Field,
                Some("format"),
                format!("format = {format}: thin writes {UNTYPED_FORMAT} on untyped ops"),
            ));
        }
        self.cache(None)
    }

    fn ds(&mut self) -> R<()> {
        for f in [VDST, ADDR, DATA0, DATA1] {
            if self.info.opd[f].is_some() {
                let o = self.register(f, true)?;
                self.push(o);
            }
        }
        let (o0, o1) = (self.take(OFFSET0), self.take(OFFSET1));
        if self.info.two_addr {
            for (key, v) in [("offset0", o0), ("offset1", o1)] {
                if v != 0 {
                    self.ins.modify(key, MVal::Int(v as i32));
                }
            }
        } else if o0 | o1 != 0 {
            self.ins.modify("offset", MVal::Int((o0 | o1 << 8) as i32));
        }
        Ok(())
    }
}

impl Decoder {
    /// Whether a source field among `srcs` holds the literal code.
    fn any_literal(class: &Class, raw: u128, srcs: u64) -> bool {
        let mut rest = srcs;
        while rest != 0 {
            let f = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            if get(raw, class.slots[f]) == 255 {
                return true;
            }
        }
        false
    }

    /// The size of an instruction of a family whose op is not decoded.
    fn size_of(&self, family: &Family, raw: u128) -> u8 {
        let class = &self.classes[family.class as usize];
        let src0 = class.slots[SRC0];
        if src0.bits > 0 {
            match (get(raw, src0), family.dpp16, family.dpp8) {
                (DPP16, Some(bytes), _) | (DPP8, _, Some(bytes)) => return bytes,
                _ => {}
            }
        }
        let lit = class.lit && Self::any_literal(class, raw, class.srcs);
        class.bytes + 4 * lit as u8
    }

    /// Decodes the instruction at `at` into `out`.
    pub(crate) fn decode_into(
        &self,
        bytes: &[u8],
        at: usize,
        out: &mut Instruction,
    ) -> Result<(), DecodeError> {
        self.decode_boxed(bytes, at, out).map_err(|e| *e)
    }

    fn decode_boxed(&self, bytes: &[u8], at: usize, out: &mut Instruction) -> R<()> {
        let rest = &bytes[at.min(bytes.len())..];
        // The next 16 bytes at once (zero past the end); `read(n)` keeps n.
        let all = match rest.get(..16) {
            Some(b) => u128::from_le_bytes(b.try_into().expect("16 bytes")),
            None => {
                let mut b = [0u8; 16];
                b[..rest.len()].copy_from_slice(rest);
                u128::from_le_bytes(b)
            }
        };
        let read = |n: usize| {
            let n = n.min(rest.len()).min(16);
            let keep = if n == 16 {
                u128::MAX
            } else {
                (1u128 << (8 * n)) - 1
            };
            (all & keep, n as u8)
        };
        let refuse = |why, class, size, n: usize, detail: String| {
            let (bits, len) = read(n);
            Box::new(DecodeError {
                offset: at,
                bits,
                len,
                size,
                why,
                class,
                mnemonic: None,
                field: None,
                detail,
            })
        };
        if rest.len() < 4 {
            return Err(refuse(
                Why::Truncated,
                "",
                None,
                4,
                format!("{} bytes left, not a dword", rest.len()),
            ));
        }
        let w0 = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        let Some(family) = self
            .families
            .get(usize::from(self.lut[(w0 >> 23) as usize]))
        else {
            return Err(refuse(
                Why::NoEncoding,
                "",
                None,
                4,
                "no encoding class has these fixed bits".into(),
            ));
        };
        let layout = &self.classes[family.class as usize];
        let (raw, _) = read(16);
        let size = self.size_of(family, raw);
        if rest.len() < usize::from(size) {
            return Err(refuse(
                Why::Truncated,
                layout.name,
                Some(size),
                16,
                format!("{} bytes left of {size}", rest.len()),
            ));
        }
        let opcode = get(raw, family.op);
        let Some(info) = family
            .ops
            .get(opcode as usize)
            .and_then(|&i| self.ops.get(usize::from(i)))
        else {
            let detail = if family.op.bits == 0 {
                format!("thin's encoder has no rule for class {}", layout.name)
            } else {
                format!("no op of opcode {opcode}")
            };
            let why = if family.op.bits == 0 {
                Why::NotEncoded
            } else {
                Why::NoOpcode
            };
            return Err(refuse(
                why,
                layout.name,
                Some(size),
                usize::from(size),
                detail,
            ));
        };
        let class = &self.classes[info.class as usize];
        let src0 = class.slots[SRC0];
        let dpp = family.dpp16.is_some() && src0.bits > 0 && get(raw, src0) == DPP16;
        let dpp8 = family.dpp8.is_some() && src0.bits > 0 && get(raw, src0) == DPP8;
        let lit = !dpp && !dpp8 && class.lit && Self::any_literal(class, raw, info.srcs);
        let len = if dpp || dpp8 {
            size
        } else {
            class.bytes + 4 * lit as u8
        };
        if rest.len() < usize::from(len) {
            return Err(refuse(
                Why::Truncated,
                class.name,
                Some(len),
                16,
                format!("{} bytes left of {len}", rest.len()),
            ));
        }
        let (raw, _) = read(usize::from(len));
        let mut cx = Cx {
            d: self,
            at,
            raw,
            len,
            class,
            info,
            base: info.base,
            consumed: 0,
            literal: lit.then(|| (raw >> (8 * u32::from(class.bytes))) as u32),
            ins: out,
        };
        cx.ins.reset(at as u32, info.print);
        if let Rule::Refused(why) = info.rule {
            return Err(cx.err(Why::NotEncoded, None, why.into()));
        }
        if dpp8 {
            return Err(cx.err(
                Why::NotEncoded,
                Some("src0"),
                format!("{}_DPP8: thin's encoder writes no DPP8", class.name),
            ));
        }
        if dpp {
            if info.rule != Rule::Short || class.name == "VOPC" || class.name == "VOP1_SDST" {
                return Err(cx.err(
                    Why::NotEncoded,
                    Some("src0"),
                    format!(
                        "{}_DPP16: thin's encoder writes DPP16 for VOP1 and VOP2 only",
                        class.name
                    ),
                ));
            }
            cx.dpp()?;
            return cx.finish();
        }
        if cx.literal.is_some() && !class.literal_ok {
            return Err(cx.err(
                Why::Operand,
                None,
                format!("no literal in class {}", class.name),
            ));
        }
        match info.rule {
            Rule::Salu => cx.salu()?,
            Rule::Sopp => cx.sopp()?,
            Rule::Smem => cx.smem()?,
            Rule::Short => cx.short()?,
            Rule::Vop3 => {
                cx.vop3()?;
                cx.vop3_opsel()?;
                cx.vop3_signs()?;
            }
            Rule::Vop3p => {
                cx.vop3()?;
                cx.packed()?;
            }
            Rule::Global => cx.global()?,
            Rule::Buffer => cx.buffer()?,
            Rule::Setreg => cx.setreg_imm32()?,
            Rule::Ds => cx.ds()?,
            Rule::Refused(_) => unreachable!("refused above"),
        }
        if cx.literal.is_some() && !cx.ins.operands().iter().any(|o| matches!(o, Opd::Int(_))) {
            return Err(cx.err(Why::Operand, None, "a literal no operand reads".into()));
        }
        cx.finish()
    }

    /// Decodes a whole word sequence; branches must land on an instruction
    /// or the end.
    pub(crate) fn decode(&self, bytes: &[u8]) -> Result<Vec<Instruction>, DecodeError> {
        let mut out = Vec::with_capacity(bytes.len() / 6);
        let mut at = 0;
        while at < bytes.len() {
            out.push(Instruction::new(0, ""));
            let last = out.last_mut().expect("just pushed");
            self.decode_into(bytes, at, last)?;
            at += usize::from(last.size);
        }
        check_branches(&out, bytes.len())?;
        Ok(out)
    }
}

/// Refuses a branch whose target is no instruction's start nor the end.
pub(crate) fn check_branches(out: &[Instruction], end: usize) -> Result<(), DecodeError> {
    for i in out {
        let Some(t) = i.branch_target() else { continue };
        let ok = t as usize == end || out.binary_search_by_key(&t, |i| i.offset).is_ok();
        if !ok {
            return Err(DecodeError {
                offset: i.offset as usize,
                bits: 0,
                len: 0,
                size: Some(i.size),
                why: Why::Branch,
                class: "SOPP",
                mnemonic: Some(i.mnemonic),
                field: Some("simm16"),
                detail: format!("{i}: {} is no instruction's start", label(t)),
            });
        }
    }
    Ok(())
}

impl Decode for Decoder {
    fn decode(&self, bytes: &[u8]) -> Result<Vec<Instruction>, DecodeError> {
        Decoder::decode(self, bytes)
    }

    fn decode_into(
        &self,
        bytes: &[u8],
        at: usize,
        out: &mut Instruction,
    ) -> Result<(), DecodeError> {
        Decoder::decode_into(self, bytes, at, out)
    }
}

/// The operands, a bit each, of a decoded short-form instruction whose
/// 16-bit VGPR fields end below v128: its
/// encoding is fixed, so a placement must keep them there.
fn short_halves(i: &Instruction) -> Result<u16, Error> {
    let d = decoder()?;
    // The short ops by printed mnemonic, indexed once.
    static SHORT: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let short = SHORT.get_or_init(|| {
        let mut by = HashMap::new();
        for (k, o) in d
            .ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.rule == Rule::Short)
        {
            by.entry(o.print).or_insert(k);
        }
        by
    });
    let info = short.get(i.mnemonic).map(|&k| &d.ops[k]);
    let Some(shape) = info.and_then(|o| o.shape) else {
        return Ok(0);
    };
    let info = info.expect("a short op");
    Ok((shape.iter().enumerate())
        .filter(|&(_, &f)| f != VCC && info.opd[f].is_some_and(|o| o.sixteen))
        .fold(0, |mask, (k, _)| mask | 1 << k))
}

/// The decoded instruction's physical IR form. The DSL's authoring forms
/// intentionally expose fewer fields/modifiers than the line encoder.
/// Preserve the encoder-proved physical shape without changing that DSL
/// contract or bypassing IR verification.
pub(crate) fn physical_form(i: &Instruction) -> Result<&'static thin_target::Form, Error> {
    use std::sync::Mutex;
    use thin_target::{Field, FieldKind, Form};

    type Key = (i32, u16, u8, u16);
    static FORMS: OnceLock<Mutex<HashMap<Key, &'static Form>>> = OnceLock::new();
    let target = crate::get(super::NAME)?;
    let th = i.modifiers().iter().find_map(|m| match m.value {
        MVal::Symbol(name, _) if name.starts_with("TH_") => Some(name),
        _ => None,
    });
    let returning = target.hooks().returning(i.mnemonic, th);
    let stripped = i
        .mnemonic
        .strip_suffix("_e32")
        .or_else(|| i.mnemonic.strip_suffix("_e64"));
    let base = target
        .form(i.mnemonic, i.operands().len(), returning)
        .or_else(|| stripped.and_then(|s| target.form(s, i.operands().len(), returning)))
        .ok_or_else(|| {
            Error::Value(format!(
                "{} with {} operands has no IR form",
                i.mnemonic,
                i.operands().len()
            ))
        })?;
    let signs = (0..i.operands().len())
        .filter(|&k| i.signs(k) != (false, false) && !base.slots[k].modifiers)
        .fold(0u16, |mask, k| mask | 1 << k);
    let opsel = if base.field("op_sel").is_none() {
        i.modifiers()
            .iter()
            .find_map(|m| match (m.key, m.value) {
                ("op_sel", MVal::List(_, n)) => Some(n),
                _ => None,
            })
            .unwrap_or(0)
    } else {
        0
    };
    let halves = short_halves(i)?;
    if signs == 0 && opsel == 0 && halves == 0 {
        return Ok(base);
    }
    if opsel != 0 && opsel_bits(i.mnemonic) == 0 {
        return Err(Error::Value(format!(
            "{} has no physical op_sel rule",
            i.mnemonic
        )));
    }
    let mut forms = FORMS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("physical form cache");
    let form = forms
        .entry((base.id, signs, opsel, halves))
        .or_insert_with(|| {
            let mut form = base.clone();
            for (k, slot) in form.slots.iter_mut().enumerate() {
                if signs & (1 << k) != 0 {
                    slot.modifiers = true;
                }
                if halves & (1 << k) != 0 {
                    slot.limit = Some(128);
                }
            }
            if opsel != 0 {
                form.fields.push(
                    Field::new("op_sel", FieldKind::Bits, opsel.into(), false)
                        .expect("op_sel field"),
                );
            }
            Box::leak(Box::new(form))
        });
    Ok(*form)
}
