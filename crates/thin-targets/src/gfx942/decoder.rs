//! gfx942's instruction decoder: the exact inverse of its line encoder
//! driven by the same ISA tables.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

use thin_target::Error;

use super::encoder::{
    dwords, is_carry_in_vop2, is_carry_vop2, is_ds_two_offset, opsel_bits, packed_rules,
    scalar_align, signs_of, vector_align, EXEC_LO_CODE, INLINE_FLOATS, INV_2PI, INV_2PI_F64,
    LITERAL_CLASSES, M0_CODE, SADDR_OFF, SGPR_COUNT, SPELLED_OTHERWISE, VCC_LO,
};
use super::{data, Data, HWREG, SPECIALS, WAITCNT_NONE};
use crate::decode::{label, Decode, DecodeError, Instruction, MVal, Opd, Why};
use crate::tables::{Encoding, Op};

/// The fields the rules read, by name; an index is a field id (`NF <= 64`).
const NAMES: [&str; 58] = [
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
    "offset",
    "soffset_en",
    "imm",
    "glc",
    "addr",
    "data",
    "data0",
    "data1",
    "offset0",
    "offset1",
    "vaddr",
    "saddr",
    "sve",
    "acc",
    "acc_cd",
    "srsrc",
    "vdata",
    "idxen",
    "offen",
    "lds",
    "sc0",
    "sc1",
    "nt",
    "gds",
    "opsel",
    "opsel_hi",
    "opsel_hi2",
    "neg",
    "neg_hi",
    "clmp",
    "abs",
    "cbsz",
    "abid",
    "blgp",
    "dpp",
    "row_mask",
    "bank_mask",
    "bc",
    "src0_neg",
    "src0_abs",
    "src1_neg",
    "src1_abs",
    "vsrc0",
    "op",
    "literal",
    "omod",
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
const OFFSET: usize = 12;
const SOFFSET_EN: usize = 13;
const IMM: usize = 14;
const GLC: usize = 15;
const ADDR: usize = 16;
const DATA: usize = 17;
const DATA0: usize = 18;
const DATA1: usize = 19;
const OFFSET0: usize = 20;
const OFFSET1: usize = 21;
const VADDR: usize = 22;
const SADDR: usize = 23;
const SVE: usize = 24;
const ACC: usize = 25;
const ACC_CD: usize = 26;
const SRSRC: usize = 27;
const VDATA: usize = 28;
const IDXEN: usize = 29;
const OFFEN: usize = 30;
const LDS: usize = 31;
const SC0: usize = 32;
const SC1: usize = 33;
const NT: usize = 34;
const GDS: usize = 35;
const OPSEL: usize = 36;
const OPSEL_HI: usize = 37;
const OPSEL_HI2: usize = 38;
const NEG: usize = 39;
const NEG_HI: usize = 40;
const CLMP: usize = 41;
const ABS: usize = 42;
const CBSZ: usize = 43;
const ABID: usize = 44;
const BLGP: usize = 45;
const DPP: usize = 46;
const ROW_MASK: usize = 47;
const BANK_MASK: usize = 48;
const BC: usize = 49;
const SRC0_NEG: usize = 50;
const SRC0_ABS: usize = 51;
const SRC1_NEG: usize = 52;
const SRC1_ABS: usize = 53;
const VSRC0: usize = 54;
const OP: usize = 55;
const LITERAL: usize = 56;
const OMOD: usize = 57;

/// `src0` codes that select DPP16 and SDWA.
const DPP16: u32 = 0xFA;
const SDWA: u32 = 0xF9;
/// Sentinel LUT entry for `FLAT`/`SCRATCH`/`GLOBAL` (`seg` in `[15:14]`).
const SEG_DISPATCH: u8 = u8::MAX - 1;

#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    lo: u8,
    bits: u8,
}

struct Class {
    name: &'static str,
    bytes: u8,
    slots: [Slot; NF],
    fields: Vec<(&'static str, u128)>,
    srcs: u64,
    lit: bool,
    literal_ok: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Xml {
    Src,
    SrcNoLit,
    VgprOrInline,
    Ssrc,
    SsrcNoLit,
    Sreg,
    SregNoVcc,
    SregM0,
    Vgpr,
    Agpr,
    VgprOrAgpr,
    Exec,
    Simm8,
    Simm16,
    Hwreg,
    Sendmsg,
    Waitcnt,
    Label,
    Imm16,
    Other,
}

#[derive(Clone, Copy, Debug)]
struct Opnd {
    dwords: u8,
    wide: bool,
    f64_fmt: bool,
    b64_fmt: bool,
    xml: Xml,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Salu,
    Setreg,
    Sopp,
    Smem,
    Short,
    Vop3,
    Vop3p,
    Mfma,
    Flat,
    Mubuf,
    Ds,
    Refused(&'static str),
}

#[derive(Clone, Copy, Debug)]
struct DppForm {
    mnemonic: &'static str,
    class: u16,
    base: u128,
    signs: [(bool, bool); 2],
}

struct OpInfo {
    mnemonic: &'static str,
    op: &'static Op,
    print: &'static str,
    class: u16,
    rule: Rule,
    opd: [Option<Opnd>; NF],
    base: u128,
    srcs: u64,
    two_offset: bool,
    opsel: u32,
    signs: [(bool, bool); 3],
    shape: Option<&'static [usize]>,
    packed: ([u64; 4], bool, u64, bool),
    dpp: Result<DppForm, &'static str>,
}

struct Family {
    class: u16,
    op: Slot,
    ops: Vec<u16>,
    dpp16: Option<u8>,
    sdwa: Option<u8>,
}

pub(crate) struct Decoder {
    lut: Vec<u8>,
    seg_families: [Option<u8>; 4],
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

fn fresh(e: &Encoding, opcode: u64) -> u128 {
    e.fields.iter().fold(0, |w, f| {
        let v = if f.kind == "op" { opcode } else { f.value };
        w | u128::from(v) << f.lo
    })
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn xml_of(t: &str, format: &str) -> Xml {
    if t == "OPR_SIMM8" || format == "FMT_NUM_U8" {
        return Xml::Simm8;
    }
    match t {
        "OPR_SRC" | "OPR_SRC_NOLDS" => Xml::Src,
        "OPR_SRC_NOLIT" | "OPR_SRC_SIMPLE" => Xml::SrcNoLit,
        "OPR_SRC_VGPR_OR_INLINE" => Xml::VgprOrInline,
        "OPR_SSRC" => Xml::Ssrc,
        "OPR_SSRC_NOLIT" | "OPR_SSRC_LANESEL" => Xml::SsrcNoLit,
        "OPR_SREG" | "OPR_SDST" => Xml::Sreg,
        "OPR_SREG_NOVCC" => Xml::SregNoVcc,
        "OPR_SREG_M0" => Xml::SregM0,
        "OPR_VGPR" | "OPR_SRC_VGPR" | "OPR_VGPR_OR_LDS" => Xml::Vgpr,
        "OPR_ACCVGPR" | "OPR_SRC_ACCVGPR" => Xml::Agpr,
        "OPR_VGPR_OR_ACCVGPR" | "OPR_SRC_VGPR_OR_ACCVGPR" | "OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST" => {
            Xml::VgprOrAgpr
        }
        "OPR_EXEC" => Xml::Exec,
        "OPR_SIMM16" => Xml::Simm16,
        "OPR_HWREG" => Xml::Hwreg,
        "OPR_SENDMSG" => Xml::Sendmsg,
        "OPR_WAITCNT" => Xml::Waitcnt,
        "OPR_LABEL" => Xml::Label,
        "OPR_CLAUSE" | "OPR_SLEEP" | "OPR_VERSION" => Xml::Imm16,
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
        "VOP3P_MFMA" => Rule::Mfma,
        "FLAT" | "GLOBAL" | "SCRATCH" => Rule::Flat,
        "MUBUF" => Rule::Mubuf,
        "DS" => Rule::Ds,
        c if c.ends_with("_LIT") => Rule::Refused("its class takes a literal of its own"),
        _ => Rule::Refused("thin's encoder has no rule for its class"),
    }
}

fn bad(what: String) -> Error {
    Error::Definition(format!("the gfx942 decoder: {what}"))
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
                        .ok_or_else(|| bad(format!("{name}: unknown source field {}", f.name)))?;
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

        let mut groups: BTreeMap<(u32, u32, u64, Option<u64>), Vec<&'static str>> = BTreeMap::new();
        for (name, e) in &isa.encodings {
            let fixed = (e
                .fields
                .iter()
                .find(|f| f.kind == "fixed" && f.name == "encoding"))
            .or_else(|| e.fields.iter().find(|f| f.kind == "fixed"))
            .ok_or_else(|| bad(format!("{name}: no fixed encoding field")))?;
            if fixed.hi != 31 || fixed.lo < 23 {
                return Err(bad(format!("{name}: fixed bits outside 31..23")));
            }
            let seg = (e.fields.iter())
                .find(|f| f.kind == "fixed" && f.name == "seg")
                .map(|f| f.value);
            groups
                .entry((fixed.hi, fixed.lo, fixed.value, seg))
                .or_default()
                .push(name.as_str());
        }

        let mut families = Vec::new();
        let mut family_of: HashMap<&str, usize> = HashMap::new();
        let mut lut = vec![u8::MAX; 512];
        let mut width = vec![0u32; 512];
        let mut seg_families = [None; 4];

        for ((hi, lo, value, seg), names) in &groups {
            let root = (names.iter())
                .filter(|n| !n.ends_with("_LIT") && !n.contains("_DPP") && !n.contains("_SDWA"))
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
                sdwa: variant("_SDWA").or_else(|| variant("_SDWA_SDST")),
            });

            if let Some(seg_val) = seg {
                if *seg_val < 4 {
                    seg_families[*seg_val as usize] = Some(f as u8);
                }
            }
            let bits = hi - lo + 1;
            let mark = if seg.is_some() { SEG_DISPATCH } else { f as u8 };
            for (index, slot) in lut.iter_mut().enumerate() {
                if (index as u64) >> (lo - 23) != *value {
                    continue;
                }
                if bits == width[index] && *slot != mark {
                    return Err(bad(format!(
                        "{root}: its fixed bits overlap another family's"
                    )));
                }
                if bits >= width[index] {
                    width[index] = bits;
                    *slot = mark;
                }
            }
        }

        let mut names: Vec<&String> = isa.ops.keys().collect();
        names.sort();
        let mut ops = Vec::new();
        for name in names {
            let op = &isa.ops[name];
            let f = family_of[op.class.as_str()];
            if families[f].op.bits == 0 {
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
            seg_families,
            families,
            classes,
            ops,
        })
    }
}

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
        let xml = xml_of(&o.xml_type, &o.format);
        opd[id] = Some(Opnd {
            dwords: dwords(o) as u8,
            wide: matches!(
                o.format.as_str(),
                "FMT_NUM_F32" | "FMT_NUM_B32" | "FMT_NUM_U32" | "FMT_NUM_I32"
            ),
            f64_fmt: o.format == "FMT_NUM_F64",
            b64_fmt: matches!(
                o.format.as_str(),
                "FMT_NUM_B64" | "FMT_NUM_U64" | "FMT_NUM_I64" | "FMT_NUM_M64"
            ),
            xml,
        });
        if xml != Xml::Sendmsg {
            srcs |= classes[class as usize].srcs & (1 << id);
        }
    }
    let mut rule = rule_of(&op.class);
    if name == "s_setreg_imm32_b32" {
        rule = Rule::Setreg;
    }
    if SPELLED_OTHERWISE.contains(&name) {
        rule = Rule::Refused("LLVM spells it otherwise on gfx942");
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
            } else {
                let e = (isa.encodings.get(&dpp_class))
                    .ok_or_else(|| bad(format!("no class {dpp_class}")))?;
                let e64 = format!("{base}_e64");
                let dpp_signs = [("src0", "src0"), ("vsrc1", "src1")].map(|(short_f, wide_f)| {
                    if let Some(wop) = isa.ops.get(&e64) {
                        wop.operand(wide_f)
                            .map_or((false, false), |sf| signs_of(&e64, &wop.class, sf))
                    } else {
                        op.operand(short_f).map_or((false, false), |sf| {
                            let flt = matches!(
                                sf.format.as_str(),
                                "FMT_NUM_F16" | "FMT_NUM_BF16" | "FMT_NUM_F32" | "FMT_NUM_F64"
                            );
                            (flt, flt)
                        })
                    }
                });
                Ok(DppForm {
                    mnemonic: leak(mnemonic),
                    class: class_id[dpp_class.as_str()],
                    base: fresh(e, op.opcode),
                    signs: dpp_signs,
                })
            }
        }
        _ => Err("thin's encoder writes DPP16 for VOP1 and VOP2 only"),
    };
    let nsrc = ["src0", "src1", "src2"]
        .iter()
        .filter(|&&f| op.operand(f).is_some())
        .count();
    Ok(OpInfo {
        mnemonic: name,
        op,
        print,
        class,
        rule,
        opd,
        base: fresh(encoding, op.opcode),
        srcs,
        two_offset: is_ds_two_offset(name),
        shape: short_shape(&op.class, name, &opd),
        opsel: opsel_bits(name),
        signs: ["src0", "src1", "src2"].map(|f| {
            op.operand(f)
                .map_or((false, false), |o| signs_of(name, &op.class, o))
        }),
        packed: packed_rules(print, nsrc),
        dpp,
    })
}

/// Implicit wave64 `vcc` (`s[106:107]`, 2 dwords) in short/DPP shapes.
const VCC: usize = usize::MAX;

fn short_shape(class: &str, name: &str, opd: &[Option<Opnd>; NF]) -> Option<&'static [usize]> {
    let n = opd.iter().filter(|o| o.is_some()).count();
    let base = name.strip_suffix("_e32").unwrap_or(name);
    let shape: &'static [usize] = match class {
        _ if n == 0 => &[],
        "VOPC" => &[VCC, SRC0, VSRC1],
        "VOP2" if base.starts_with("v_cndmask") => &[VDST, SRC0, VSRC1, VCC],
        "VOP2" if is_carry_in_vop2(base) => &[VDST, VCC, SRC0, VSRC1, VCC],
        "VOP2" if is_carry_vop2(base) => &[VDST, VCC, SRC0, VSRC1],
        "VOP2" => &[VDST, SRC0, VSRC1],
        _ => &[VDST, SRC0],
    };
    let fields = shape.iter().filter(|&&f| f != VCC).count();
    let all = shape.iter().all(|&f| f == VCC || opd[f].is_some());
    (fields == n && all).then_some(shape)
}

pub(crate) fn decoder() -> Result<&'static Decoder, Error> {
    static DECODER: OnceLock<Result<Decoder, Error>> = OnceLock::new();
    let built = DECODER.get_or_init(|| Decoder::build(data()?));
    built.as_ref().map_err(Clone::clone)
}

fn special(c: u32, width: u32) -> Option<Opd> {
    SPECIALS
        .iter()
        .find(|s| s.1 == c && s.2 == width)
        .map(Opd::Named)
}

fn scalar(c: u32, width: u32) -> Option<Opd> {
    if c + width <= SGPR_COUNT {
        if width > 1 && c % scalar_align(width) != 0 {
            return None;
        }
        return Some(Opd::S {
            n: c as u16,
            count: width as u16,
        });
    }
    if c < SGPR_COUNT {
        return None;
    }
    special(c, width)
}

fn sext(v: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    (i64::from(v) << shift) >> shift
}

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

    fn vec_or_acc(&self, f: usize, c: u32, acc: bool, width: u32) -> R<Opd> {
        let letter = if acc { 'a' } else { 'v' };
        if c + width > 256 {
            return Err(self.err(
                Why::Operand,
                Some(NAMES[f]),
                format!("{letter}{c} +{width}: past {letter}255"),
            ));
        }
        let align = vector_align(width);
        if width > 1 && c % align != 0 {
            return Err(self.err(
                Why::Operand,
                Some(NAMES[f]),
                format!("{letter}{c}: multi-dword {letter} range requires even register"),
            ));
        }
        Ok(if acc {
            Opd::A {
                n: c as u16,
                count: width as u16,
            }
        } else {
            Opd::V {
                n: c as u16,
                count: width as u16,
            }
        })
    }

    fn sreg_of(&self, f: usize, c: u32, width: u32) -> R<Opd> {
        if c + width <= SGPR_COUNT {
            let align = scalar_align(width);
            if width > 1 && c % align != 0 {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("s{c}: scalar range starts at a multiple of {align}"),
                ));
            }
            return Ok(Opd::S {
                n: c as u16,
                count: width as u16,
            });
        }
        if c < SGPR_COUNT {
            return Err(self.unnamed(f, c, width));
        }
        special(c, width).ok_or_else(|| self.unnamed(f, c, width))
    }

    #[cold]
    fn unnamed(&self, f: usize, c: u32, width: u32) -> Box<DecodeError> {
        self.err(
            Why::Operand,
            Some(NAMES[f]),
            format!("scalar code {c} is no {width}-dword register gfx942's syntax names"),
        )
    }

    fn register(&mut self, f: usize, file: char) -> R<Opd> {
        let o = self.opnd(f);
        let width = u32::from(o.dwords);
        let c = self.take(f);
        match file {
            'v' => self.vec_or_acc(f, c, false, width),
            'a' => self.vec_or_acc(f, c, true, width),
            _ => {
                if o.xml == Xml::SregNoVcc && (VCC_LO..=VCC_LO + 1).contains(&c) {
                    return Err(self.err(
                        Why::Operand,
                        Some(NAMES[f]),
                        "VCC is not allowed here".into(),
                    ));
                }
                if o.xml == Xml::SregM0 && c != M0_CODE {
                    return Err(self.err(Why::Operand, Some(NAMES[f]), "m0 expected".into()));
                }
                self.sreg_of(f, c, width)
            }
        }
    }

    fn source(&mut self, f: usize) -> R<Opd> {
        let o = self.opnd(f);
        self.source_opnd(f, o)
    }

    fn source_opnd(&mut self, f: usize, o: Opnd) -> R<Opd> {
        let (vgprs, agprs, sgprs, constants, no_lit) = match o.xml {
            Xml::Src => (true, false, true, true, false),
            Xml::SrcNoLit => (true, false, true, true, true),
            Xml::VgprOrInline => (true, false, false, true, true),
            Xml::Ssrc => (false, false, true, true, false),
            Xml::SsrcNoLit => (false, false, true, true, true),
            Xml::Sreg | Xml::SregNoVcc | Xml::SregM0 => (false, false, true, false, true),
            Xml::Vgpr => (true, false, false, false, true),
            Xml::Agpr => (false, true, false, false, true),
            _ => {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("operand type {} is not taken", self.xml_name(f)),
                ))
            }
        };
        let width = u32::from(o.dwords);
        let c = self.take(f);
        let refuse = |cx: &Self, what: String| Err(cx.err(Why::Operand, Some(NAMES[f]), what));
        if no_lit && c == 255 {
            return refuse(self, format!("{} takes no literal", NAMES[f]));
        }
        if c >= 256 {
            if vgprs {
                return self.vec_or_acc(f, c - 256, false, width);
            }
            if agprs {
                return self.vec_or_acc(f, c - 256, true, width);
            }
            return refuse(self, format!("code {c}: no vector register here"));
        }
        if let Some(s) = scalar(c, width) {
            if !sgprs {
                return refuse(self, format!("scalar code {c}: no scalar register here"));
            }
            if o.xml == Xml::SregNoVcc && (VCC_LO..=VCC_LO + 1).contains(&c) {
                return refuse(self, "VCC is not allowed here".into());
            }
            if o.xml == Xml::SregM0 && c != M0_CODE {
                return refuse(self, "m0 expected".into());
            }
            return Ok(s);
        }
        if !constants {
            return refuse(
                self,
                format!("code {c}: no {width}-dword register here, and no constant"),
            );
        }
        match c {
            128..=192 => Ok(Opd::Int(i64::from(c) - 128)),
            193..=208 => Ok(Opd::Int(192 - i64::from(c))),
            240..=247 if o.wide || o.f64_fmt => Ok(Opd::Float(INLINE_FLOATS[c as usize - 240])),
            240..=247 if o.b64_fmt && width == 2 => {
                let f = f64::from(INLINE_FLOATS[c as usize - 240]);
                Ok(Opd::Int(f.to_bits() as i64))
            }
            248 if o.wide && width == 1 => Ok(Opd::Int(i64::from(INV_2PI))),
            248 if (o.f64_fmt || o.b64_fmt) && width == 2 => Ok(Opd::Int(INV_2PI_F64 as i64)),
            240..=248 => refuse(
                self,
                format!("inline float code {c} not encodable in {}", NAMES[f]),
            ),
            255 => self.literal(f, o, width),
            _ => refuse(self, format!("code {c}: no {width}-dword operand named")),
        }
    }

    fn literal(&mut self, f: usize, o: Opnd, width: u32) -> R<Opd> {
        let l = self.literal.expect("a 255 source reads the literal");
        let refuse = |cx: &Self, what: String| Err(cx.err(Why::Operand, Some(NAMES[f]), what));
        if !o.wide || width != 1 {
            return refuse(
                self,
                format!("literal 0x{l:x}: literals are 32-bit operands only"),
            );
        }
        if l <= 64
            || (-16..0).contains(&(l as i32))
            || INLINE_FLOATS.iter().any(|x| x.to_bits() == l)
            || l == INV_2PI
        {
            return refuse(self, format!("literal 0x{l:x}: thin writes it inline"));
        }
        Ok(Opd::Int(i64::from(l)))
    }

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
        let order = if self.info.mnemonic == "s_setreg_b32" {
            [SIMM16, SDST, SSRC0, SSRC1]
        } else {
            [SDST, SSRC0, SSRC1, SIMM16]
        };
        for f in order {
            let Some(o) = self.info.opd[f] else { continue };
            let opd = match f {
                SDST => self.register(f, 's')?,
                SIMM16 if o.xml == Xml::Hwreg => self.hwreg()?,
                SIMM16 if !matches!(o.xml, Xml::Simm16 | Xml::Imm16) => {
                    return Err(self.err(
                        Why::Operand,
                        Some("simm16"),
                        format!("operand type {}: not a 16-bit immediate", self.xml_name(f)),
                    ))
                }
                SIMM16 => Opd::Int(i64::from(self.take(f))),
                _ => self.source(f)?,
            };
            self.push(opd);
        }
        Ok(())
    }

    fn hwreg(&mut self) -> R<Opd> {
        let code = self.take(SIMM16);
        let id = code & 63;
        let name = HWREG.iter().find(|r| r.1 == id).map(|r| r.0);
        let (offset, size) = ((code >> 6) & 31, (code >> 11) + 1);
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

    fn setreg_imm32(&mut self) -> R<()> {
        let reg = self.hwreg()?;
        self.push(reg);
        let imm = self.take(LITERAL);
        self.push(Opd::Int(i64::from(imm)));
        Ok(())
    }

    fn sopp(&mut self) -> R<()> {
        let Some(o) = self.info.opd[SIMM16] else {
            return Ok(());
        };
        let v = self.take(SIMM16);
        match o.xml {
            Xml::Waitcnt => {
                if v & !WAITCNT_NONE != 0 {
                    return Err(self.err(
                        Why::Field,
                        Some("simm16"),
                        format!("simm16 = 0x{v:x}: reserved bits set in s_waitcnt"),
                    ));
                }
                self.ins.counted = true;
                let (vm, exp, lgkm) = super::encoder::unpack_waitcnt(v);
                if v == WAITCNT_NONE {
                    self.ins.modify("vmcnt", MVal::Int(vm as i32));
                    self.ins.modify("expcnt", MVal::Int(exp as i32));
                    self.ins.modify("lgkmcnt", MVal::Int(lgkm as i32));
                } else {
                    if vm < 63 {
                        self.ins.modify("vmcnt", MVal::Int(vm as i32));
                    }
                    if exp < 7 {
                        self.ins.modify("expcnt", MVal::Int(exp as i32));
                    }
                    if lgkm < 15 {
                        self.ins.modify("lgkmcnt", MVal::Int(lgkm as i32));
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
            Xml::Simm16 | Xml::Sendmsg | Xml::Imm16 => self.push(Opd::Int(i64::from(v))),
            _ => {
                return Err(self.err(
                    Why::Operand,
                    Some("simm16"),
                    format!("operand type {} is not taken", self.xml_name(SIMM16)),
                ))
            }
        }
        Ok(())
    }

    fn smem(&mut self) -> R<()> {
        let glc = self.take(GLC);
        let has_data = self.info.opd[SDATA].is_some();
        let has_base = self.info.opd[SBASE].is_some();
        let has_off = self.info.opd[OFFSET].is_some() || self.info.opd[SOFFSET].is_some();

        match (has_data, has_base, has_off) {
            (false, false, false) => {}
            (true, false, false) => {
                let d = self.register(SDATA, 's')?;
                self.push(d);
            }
            (false, true, true) => {
                let bw = u32::from(self.opnd(SBASE).dwords);
                let b = self.take(SBASE) * 2;
                let base = self.sreg_of(SBASE, b, bw)?;
                self.push(base);
                self.smem_offset()?;
            }
            (true, true, true) => {
                let d_opnd = self.opnd(SDATA);
                let data = if d_opnd.xml == Xml::Simm8 {
                    let v = self.take(SDATA);
                    if v >= 128 {
                        return Err(self.err(
                            Why::Operand,
                            Some("sdata"),
                            "0..127 expected".into(),
                        ));
                    }
                    Opd::Int(i64::from(v))
                } else {
                    self.register(SDATA, 's')?
                };
                let bw = u32::from(self.opnd(SBASE).dwords);
                let b = self.take(SBASE) * 2;
                let base = self.sreg_of(SBASE, b, bw)?;
                self.push(data);
                self.push(base);
                self.smem_offset()?;
            }
            _ => {
                return Err(self.err(Why::Form, None, "unsupported SMEM operand shape".into()));
            }
        }
        if glc != 0 {
            self.ins.modify("glc", MVal::Flag);
        }
        Ok(())
    }

    fn smem_offset(&mut self) -> R<()> {
        let imm = self.take(IMM);
        let soff_en = self.take(SOFFSET_EN);
        match (imm, soff_en) {
            (0, 0) => {
                let c = self.take(OFFSET);
                if c > M0_CODE {
                    return Err(self.unnamed(OFFSET, c, 1));
                }
                let s = scalar(c, 1).ok_or_else(|| self.unnamed(OFFSET, c, 1))?;
                self.push(s);
                Ok(())
            }
            (1, 0) => {
                let off = sext(self.take(OFFSET), 21);
                self.push(Opd::Int(off));
                Ok(())
            }
            (1, 1) => {
                let c = self.take(SOFFSET);
                if c > M0_CODE {
                    return Err(self.unnamed(SOFFSET, c, 1));
                }
                let s = scalar(c, 1).ok_or_else(|| self.unnamed(SOFFSET, c, 1))?;
                let off = sext(self.take(OFFSET), 21);
                self.push(s);
                self.ins.modify("offset", MVal::Int(off as i32));
                Ok(())
            }
            _ => Err(self.err(
                Why::Field,
                Some("soffset_en"),
                "imm=0 with soffset_en=1 is invalid on gfx942 SMEM".into(),
            )),
        }
    }

    fn short(&mut self) -> R<()> {
        let sdst = self.class.name == "VOP1_SDST";
        let shape = self
            .info
            .shape
            .ok_or_else(|| self.err(Why::Form, None, "operands do not match short shape".into()))?;
        for &f in shape {
            let o = match f {
                VCC => special(VCC_LO, 2).expect("vcc is named"),
                VDST if sdst => self.register(f, 's')?,
                VDST if self.opnd(f).xml == Xml::Agpr => self.register(f, 'a')?,
                VDST | VSRC1 => self.register(f, 'v')?,
                _ => self.source(f)?,
            };
            self.push(o);
        }
        Ok(())
    }

    fn vop3(&mut self) -> R<()> {
        for f in [VDST, SDST, SRC0, SRC1, SRC2] {
            let Some(o) = self.info.opd[f] else { continue };
            if o.xml == Xml::Exec {
                let c = self.take(f);
                if c != EXEC_LO_CODE {
                    return Err(self.err(
                        Why::Operand,
                        Some(NAMES[f]),
                        format!("code {c}: exec expected"),
                    ));
                }
                continue;
            }
            let opd = match f {
                VDST if o.xml == Xml::Vgpr => self.register(f, 'v')?,
                VDST if o.xml == Xml::Agpr => self.register(f, 'a')?,
                VDST | SDST => self.register(f, 's')?,
                _ => self.source(f)?,
            };
            self.push(opd);
        }
        Ok(())
    }

    fn vop3_opsel(&mut self) -> R<()> {
        if self.class.slots[OPSEL].bits == 0 || self.info.opsel == 0 {
            return Ok(());
        }
        let v = self.take(OPSEL);
        if v == 0 {
            return Ok(());
        }
        if v & !self.info.opsel != 0 {
            return Err(self.err(
                Why::Field,
                Some("opsel"),
                format!(
                    "opsel = {v}: {} reads OPSEL bits 0x{:x} only",
                    self.info.mnemonic, self.info.opsel
                ),
            ));
        }
        let nsrc = [SRC0, SRC1, SRC2]
            .iter()
            .filter(|&&f| self.info.opd[f].is_some())
            .count();
        let len = super::encoder::opsel_len(self.info.mnemonic, nsrc);
        let mut items = [0u8; 4];
        for (j, item) in items.iter_mut().enumerate().take(len) {
            let bit = if j + 1 == len && len < 4 { 3 } else { j };
            *item = ((v >> bit) & 1) as u8;
        }
        self.ins.modify("op_sel", MVal::List(items, len as u8));
        Ok(())
    }

    fn vop3_signs_and_clamp(&mut self) -> R<()> {
        let (neg_slot, abs_slot) = (self.class.slots[NEG], self.class.slots[ABS]);
        let neg = if neg_slot.bits > 0 { self.take(NEG) } else { 0 };
        let abs = if abs_slot.bits > 0 { self.take(ABS) } else { 0 };
        if neg | abs != 0 {
            let mut k = [VDST, SDST]
                .iter()
                .filter(|&&f| self.info.opd[f].is_some_and(|o| o.xml != Xml::Exec))
                .count();
            for (i, f) in [SRC0, SRC1, SRC2].into_iter().enumerate() {
                let (n, a) = ((neg >> i) & 1 == 1, (abs >> i) & 1 == 1);
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
                        format!("{field} on {}: {} takes none", NAMES[f], self.info.mnemonic),
                    ));
                }
                let constant = matches!(self.ins.operands()[k], Opd::Int(_) | Opd::Float(_));
                if n && !a && constant {
                    return Err(self.err(
                        Why::Operand,
                        Some("neg"),
                        format!(
                            "neg on constant in {}: line syntax writes no negated constant",
                            NAMES[f]
                        ),
                    ));
                }
                self.ins.sign(k, n, a);
                k += 1;
            }
        }
        if self.class.slots[CLMP].bits > 0 && self.take(CLMP) == 1 {
            self.ins.modify("clamp", MVal::Flag);
        }
        if self.class.slots[OMOD].bits > 0 {
            match self.take(OMOD) {
                0 => {}
                1 => self.ins.modify("mul", MVal::Int(2)),
                2 => self.ins.modify("mul", MVal::Int(4)),
                3 => self.ins.modify("div", MVal::Int(2)),
                _ => unreachable!("2-bit omod"),
            }
        }
        Ok(())
    }

    fn bits(&self, f: usize, key: &str, v: u32, nsrc: u32, mask: u64) -> R<MVal> {
        if mask == 0 || v >> nsrc != 0 || u64::from(v) & !mask != 0 {
            return Err(self.err(
                Why::Field,
                Some(NAMES[f]),
                format!("{key} 0x{v:x}: not taken (mask 0x{mask:x} of {nsrc})"),
            ));
        }
        let mut items = [0u8; 4];
        for (j, item) in items.iter_mut().enumerate().take(nsrc as usize) {
            *item = ((v >> j) & 1) as u8;
        }
        Ok(MVal::List(items, nsrc as u8))
    }

    fn packed(&mut self) -> R<()> {
        let nsrc = [SRC0, SRC1, SRC2]
            .iter()
            .filter(|&&f| self.info.opd[f].is_some())
            .count() as u32;
        let (masks, clamp, hi_default, takes_signs) = self.info.packed;
        let opsel = self.take(OPSEL);
        let hi = self.take(OPSEL_HI);
        let hi2 = self.take(OPSEL_HI2);
        let neg = self.take(NEG);
        let neg_hi = self.take(NEG_HI);

        let hi2_expected = ((hi_default as u32) >> 2) & 1;
        let default_hi = ((hi_default as u32) & 3, hi2_expected);

        if opsel != 0 {
            let val = self.bits(OPSEL, "op_sel", opsel, nsrc, masks[0])?;
            self.ins.modify("op_sel", val);
        }
        if (hi, hi2) != default_hi {
            if nsrc < 3 && hi2 != hi2_expected {
                return Err(self.err(
                    Why::Field,
                    Some("opsel_hi2"),
                    format!("opsel_hi2 = {hi2}: thin writes {hi2_expected}"),
                ));
            }
            let v = hi | if nsrc == 3 { hi2 << 2 } else { 0 };
            let val = self.bits(OPSEL_HI, "op_sel_hi", v, nsrc, masks[1])?;
            self.ins.modify("op_sel_hi", val);
        }
        if takes_signs {
            if (neg >> nsrc) != 0 || (neg_hi >> nsrc) != 0 {
                return Err(self.err(Why::Field, Some("neg"), "sign bit past nsrc".into()));
            }
            for i in 0..nsrc as usize {
                let (n, a) = ((neg >> i) & 1 == 1, (neg_hi >> i) & 1 == 1);
                if n || a {
                    let k = 1 + i;
                    let constant = matches!(self.ins.operands()[k], Opd::Int(_) | Opd::Float(_));
                    if n && !a && constant {
                        return Err(self.err(
                            Why::Operand,
                            Some("neg"),
                            "neg on constant in v_fma_mix".into(),
                        ));
                    }
                    self.ins.sign(k, n, a);
                }
            }
        } else {
            if neg != 0 {
                let val = self.bits(NEG, "neg_lo", neg, nsrc, masks[2])?;
                self.ins.modify("neg_lo", val);
            }
            if neg_hi != 0 {
                let val = self.bits(NEG_HI, "neg_hi", neg_hi, nsrc, masks[3])?;
                self.ins.modify("neg_hi", val);
            }
        }
        if self.take(CLMP) == 1 {
            if !clamp {
                return Err(self.err(Why::Field, Some("clmp"), "clamp: not taken".into()));
            }
            self.ins.modify("clamp", MVal::Flag);
        }
        Ok(())
    }

    fn mfma(&mut self) -> R<()> {
        let smfmac = self.info.mnemonic.starts_with("v_smfmac_");
        let cd_acc = self.take(ACC_CD) != 0;
        let acc_ab = self.take(ACC);

        let w_dst = u32::from(self.opnd(VDST).dwords);
        let vdst = self.take(VDST);
        self.push(self.vec_or_acc(VDST, vdst, cd_acc, w_dst)?);

        for (idx, f) in [SRC0, SRC1].into_iter().enumerate() {
            let w = u32::from(self.opnd(f).dwords);
            let code = self.take(f);
            if code < 256 {
                return Err(self.err(
                    Why::Operand,
                    Some(NAMES[f]),
                    format!("code {code}: MFMA matrix A/B requires VGPR or AGPR"),
                ));
            }
            let is_acc = ((acc_ab >> idx) & 1) != 0;
            self.push(self.vec_or_acc(f, code - 256, is_acc, w)?);
        }

        let o2 = self.opnd(SRC2);
        let w2 = u32::from(o2.dwords);
        let c2 = self.take(SRC2);
        let op2 = if smfmac {
            if c2 < 256 {
                return Err(self.err(
                    Why::Operand,
                    Some("src2"),
                    "v_smfmac src2 requires VGPR".into(),
                ));
            }
            self.vec_or_acc(SRC2, c2 - 256, false, w2)?
        } else if c2 >= 256 {
            self.vec_or_acc(SRC2, c2 - 256, cd_acc, w2)?
        } else {
            match c2 {
                128..=192 => Opd::Int(i64::from(c2) - 128),
                193..=208 => Opd::Int(192 - i64::from(c2)),
                240..=247 if o2.wide || o2.f64_fmt => Opd::Float(INLINE_FLOATS[c2 as usize - 240]),
                240..=247 if o2.b64_fmt && w2 == 2 => {
                    Opd::Int(f64::from(INLINE_FLOATS[c2 as usize - 240]).to_bits() as i64)
                }
                248 if o2.wide && w2 == 1 => Opd::Int(i64::from(INV_2PI)),
                248 if (o2.f64_fmt || o2.b64_fmt) && w2 == 2 => Opd::Int(INV_2PI_F64 as i64),
                _ => {
                    return Err(self.err(
                        Why::Operand,
                        Some("src2"),
                        format!("code {c2}: invalid MFMA src2 constant"),
                    ))
                }
            }
        };
        self.push(op2);

        let cbsz = self.take(CBSZ);
        let abid = self.take(ABID);
        let blgp = self.take(BLGP);
        if smfmac && blgp != 0 {
            return Err(self.err(
                Why::Field,
                Some("blgp"),
                "blgp not taken on v_smfmac".into(),
            ));
        }
        if cbsz != 0 {
            self.ins.modify("cbsz", MVal::Int(cbsz as i32));
        }
        if abid != 0 {
            self.ins.modify("abid", MVal::Int(abid as i32));
        }
        if blgp != 0 {
            self.ins.modify("blgp", MVal::Int(blgp as i32));
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
        let shape = self
            .info
            .shape
            .ok_or_else(|| self.err(Why::Form, None, "operands do not match DPP shape".into()))?;
        let mut src0_idx = None;
        let mut src1_idx = None;
        for (idx, &f) in shape.iter().enumerate() {
            let o = match f {
                VCC => special(VCC_LO, 2).expect("vcc is named"),
                _ => {
                    let opnd = self.opnd(f);
                    let width = u32::from(opnd.dwords);
                    let field = if f == SRC0 { VSRC0 } else { f };
                    if f == SRC0 {
                        src0_idx = Some(idx);
                    } else if f == VSRC1 {
                        src1_idx = Some(idx);
                    }
                    let c = self.take(field);
                    let is_acc = opnd.xml == Xml::Agpr;
                    self.vec_or_acc(field, c, is_acc, width)?
                }
            };
            self.push(o);
        }

        let s0_neg = self.take(SRC0_NEG) == 1;
        let s0_abs = self.take(SRC0_ABS) == 1;
        if s0_neg || s0_abs {
            let (ok_n, ok_a) = d.signs[0];
            let Some(idx) = src0_idx.filter(|_| (!s0_neg || ok_n) && (!s0_abs || ok_a)) else {
                return Err(self.err(
                    Why::Field,
                    Some("src0_neg"),
                    "sign not taken on src0".into(),
                ));
            };
            self.ins.sign(idx, s0_neg, s0_abs);
        }
        if self.class.slots[SRC1_NEG].bits > 0 {
            let s1_neg = self.take(SRC1_NEG) == 1;
            let s1_abs = self.take(SRC1_ABS) == 1;
            if s1_neg || s1_abs {
                let (ok_n, ok_a) = d.signs[1];
                let Some(idx) = src1_idx.filter(|_| (!s1_neg || ok_n) && (!s1_abs || ok_a)) else {
                    return Err(self.err(
                        Why::Field,
                        Some("src1_neg"),
                        "sign not taken on vsrc1".into(),
                    ));
                };
                self.ins.sign(idx, s1_neg, s1_abs);
            }
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
            0x130 => ("wave_shl", MVal::Int(1)),
            0x134 => ("wave_rol", MVal::Int(1)),
            0x138 => ("wave_shr", MVal::Int(1)),
            0x13C => ("wave_ror", MVal::Int(1)),
            0x140 => ("row_mirror", MVal::Flag),
            0x141 => ("row_half_mirror", MVal::Flag),
            0x142 => ("row_bcast", MVal::Int(15)),
            0x143 => ("row_bcast", MVal::Int(31)),
            0x150..=0x15F => ("row_newbcast", MVal::Int((dpp - 0x150) as i32)),
            _ => {
                return Err(self.err(
                    Why::Field,
                    Some("dpp"),
                    format!("dpp control 0x{dpp:x}: none gfx942's syntax names"),
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
        Ok(())
    }

    fn cache_flags(&mut self, sc0: u32, sc1: u32, nt: u32) {
        if sc0 != 0 {
            self.ins.modify("sc0", MVal::Flag);
        }
        if sc1 != 0 {
            self.ins.modify("sc1", MVal::Flag);
        }
        if nt != 0 {
            self.ins.modify("nt", MVal::Flag);
        }
    }

    fn flat_family(&mut self) -> R<()> {
        let cname = self.class.name;
        let is_atomic = self.info.mnemonic.contains("atomic");
        let sc0 = self.take(SC0);
        let sc1 = self.take(SC1);
        let nt = self.take(NT);
        let acc = self.take(ACC) != 0;

        let mut has_dst = self.info.opd[VDST].is_some();
        let has_data = self.info.opd[DATA].is_some();
        if is_atomic && sc0 == 0 {
            has_dst = false;
        }
        if !has_dst && !has_data && acc {
            return Err(self.err(Why::Field, Some("acc"), "acc=1 without vdst or data".into()));
        }

        let dst_opd = if has_dst {
            let w = u32::from(self.opnd(VDST).dwords);
            let c = self.take(VDST);
            Some(self.vec_or_acc(VDST, c, acc, w)?)
        } else {
            None
        };
        let data_opd = if has_data {
            let w = u32::from(self.opnd(DATA).dwords);
            let c = self.take(DATA);
            Some(self.vec_or_acc(DATA, c, acc, w)?)
        } else {
            None
        };

        if cname == "FLAT" {
            let has_addr = self.info.opd[ADDR].is_some();
            if let Some(d) = dst_opd {
                self.push(d);
            }
            if has_addr {
                let a = self.take(ADDR);
                self.push(self.vec_or_acc(ADDR, a, false, 2)?);
            }
            if let Some(v) = data_opd {
                self.push(v);
            }
            let raw_off = self.take(OFFSET);
            if raw_off >= 4096 {
                return Err(self.err(
                    Why::Field,
                    Some("offset"),
                    format!("flat offset 0x{raw_off:x}: bit 12 must be 0"),
                ));
            }
            if raw_off != 0 {
                self.ins.modify("offset", MVal::Int(raw_off as i32));
            }
            self.cache_flags(sc0, sc1, nt);
            return Ok(());
        }

        // GLOBAL and SCRATCH
        let has_addr = self.info.opd[ADDR].is_some();
        if let Some(d) = dst_opd {
            self.push(d);
        }
        if has_addr {
            let saddr = self.take(SADDR);
            if cname == "GLOBAL" {
                let a = self.take(ADDR);
                if saddr == SADDR_OFF {
                    self.push(self.vec_or_acc(ADDR, a, false, 2)?);
                    if let Some(v) = data_opd {
                        self.push(v);
                    }
                    self.push(Opd::Off);
                } else {
                    if saddr > SADDR_OFF {
                        return Err(self.unnamed(SADDR, saddr, 2));
                    }
                    let s = self.sreg_of(SADDR, saddr, 2)?;
                    self.push(self.vec_or_acc(ADDR, a, false, 1)?);
                    if let Some(v) = data_opd {
                        self.push(v);
                    }
                    self.push(s);
                }
            } else {
                // SCRATCH
                let sve = self.take(SVE);
                let a = self.take(ADDR);
                let vaddr_op = if sve == 0 {
                    if a != 0 {
                        return Err(self.err(
                            Why::Field,
                            Some("addr"),
                            format!("scratch addr = {a} with sve=0: thin writes 0"),
                        ));
                    }
                    Opd::Off
                } else {
                    self.vec_or_acc(ADDR, a, false, 1)?
                };
                let saddr_op = if saddr == SADDR_OFF {
                    Opd::Off
                } else {
                    if saddr > SADDR_OFF {
                        return Err(self.unnamed(SADDR, saddr, 1));
                    }
                    self.sreg_of(SADDR, saddr, 1)?
                };
                self.push(vaddr_op);
                if let Some(v) = data_opd {
                    self.push(v);
                }
                self.push(saddr_op);
            }
        } else {
            let saddr = self.take(SADDR);
            if saddr != 0 {
                return Err(self.err(
                    Why::Field,
                    Some("saddr"),
                    format!("saddr = {saddr}: thin writes 0"),
                ));
            }
        }
        let off = sext(self.take(OFFSET), 13);
        if off != 0 {
            self.ins.modify("offset", MVal::Int(off as i32));
        }
        self.cache_flags(sc0, sc1, nt);
        Ok(())
    }

    fn mubuf(&mut self) -> R<()> {
        let sc0 = self.take(SC0);
        let sc1 = self.take(SC1);
        let nt = self.take(NT);
        if self.info.op.operands.is_empty() {
            let soff = self.take(SOFFSET);
            if soff != 0 {
                return Err(self.err(
                    Why::Field,
                    Some("soffset"),
                    format!("soffset = {soff}: thin writes 0 on {0}", self.info.mnemonic),
                ));
            }
            self.cache_flags(sc0, sc1, nt);
            return Ok(());
        }

        let lds = self.take(LDS);
        let acc = self.take(ACC) != 0;
        if lds != 0 {
            if !self.info.mnemonic.starts_with("buffer_load_") || acc {
                return Err(self.err(
                    Why::Field,
                    Some("lds"),
                    "lds=1 is valid on buffer_load_* without acc only".into(),
                ));
            }
        } else {
            let w = u32::from(self.opnd(VDATA).dwords);
            let vd = self.take(VDATA);
            self.push(self.vec_or_acc(VDATA, vd, acc, w)?);
        }

        let idxen = self.take(IDXEN);
        let offen = self.take(OFFEN);
        let va = self.take(VADDR);
        let vaddr = match idxen + offen {
            0 if va == 0 => Opd::Off,
            0 => {
                return Err(self.err(
                    Why::Field,
                    Some("vaddr"),
                    format!("vaddr = {va} without idxen or offen: thin writes 0"),
                ))
            }
            n => self.vec_or_acc(VADDR, va, false, n)?,
        };
        let srsrc = self.take(SRSRC) * 4;
        let rsrc = self.sreg_of(SRSRC, srsrc, 4)?;
        let soff_opnd = Opnd {
            dwords: 1,
            wide: true,
            f64_fmt: false,
            b64_fmt: false,
            xml: Xml::SsrcNoLit,
        };
        let soff = self.source_opnd(SOFFSET, soff_opnd)?;
        for o in [vaddr, rsrc, soff] {
            self.push(o);
        }
        if idxen == 1 {
            self.ins.modify("idxen", MVal::Flag);
        }
        if offen == 1 {
            self.ins.modify("offen", MVal::Flag);
        }
        let offset = self.take(OFFSET);
        if offset != 0 {
            self.ins.modify("offset", MVal::Int(offset as i32));
        }
        self.cache_flags(sc0, sc1, nt);
        if lds != 0 {
            self.ins.modify("lds", MVal::Flag);
        }
        Ok(())
    }

    fn ds(&mut self) -> R<()> {
        let acc = self.take(ACC) != 0;
        let mut any_av = false;
        for f in [VDST, ADDR, DATA0, DATA1] {
            let Some(o) = self.info.opd[f] else { continue };
            let c = self.take(f);
            let width = u32::from(o.dwords);
            let use_acc = if f != ADDR && o.xml == Xml::VgprOrAgpr {
                any_av = true;
                acc
            } else {
                false
            };
            self.push(self.vec_or_acc(f, c, use_acc, width)?);
        }
        if acc && !any_av {
            return Err(self.err(
                Why::Field,
                Some("acc"),
                "acc=1 on DS op without data/vdst".into(),
            ));
        }
        let gds = self.take(GDS);
        if self.info.mnemonic.starts_with("ds_gws_") && gds == 0 {
            return Err(self.err(Why::Field, Some("gds"), "ds_gws_* requires gds".into()));
        }
        let (o0, o1) = (self.take(OFFSET0), self.take(OFFSET1));
        if self.info.two_offset {
            for (key, v) in [("offset0", o0), ("offset1", o1)] {
                if v != 0 {
                    self.ins.modify(key, MVal::Int(v as i32));
                }
            }
        } else if o0 | o1 != 0 {
            self.ins
                .modify("offset", MVal::Int((o0 | (o1 << 8)) as i32));
        }
        if gds != 0 {
            self.ins.modify("gds", MVal::Flag);
        }
        Ok(())
    }
}

impl Decoder {
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

    fn size_of(&self, family: &Family, raw: u128) -> u8 {
        let class = &self.classes[family.class as usize];
        let src0 = class.slots[SRC0];
        if src0.bits > 0 {
            match (get(raw, src0), family.dpp16, family.sdwa) {
                (DPP16, Some(bytes), _) | (SDWA, _, Some(bytes)) => return bytes,
                _ => {}
            }
        }
        let lit = class.lit && Self::any_literal(class, raw, class.srcs);
        class.bytes + 4 * lit as u8
    }

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
        let lut_entry = self.lut[(w0 >> 23) as usize];
        let fam_idx = if lut_entry == SEG_DISPATCH {
            self.seg_families[((w0 >> 14) & 3) as usize]
        } else {
            (lut_entry != u8::MAX).then_some(lut_entry)
        };
        let Some(family) = fam_idx.and_then(|idx| self.families.get(usize::from(idx))) else {
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
            return Err(refuse(
                Why::NoOpcode,
                layout.name,
                Some(size),
                usize::from(size),
                format!("no op of opcode {opcode}"),
            ));
        };
        let class = &self.classes[info.class as usize];
        let src0 = class.slots[SRC0];
        let dpp = family.dpp16.is_some() && src0.bits > 0 && get(raw, src0) == DPP16;
        let sdwa = family.sdwa.is_some() && src0.bits > 0 && get(raw, src0) == SDWA;
        let lit = !dpp && !sdwa && class.lit && Self::any_literal(class, raw, info.srcs);
        let len = if dpp || sdwa {
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
        if sdwa {
            return Err(cx.err(
                Why::NotEncoded,
                Some("src0"),
                format!("{}_SDWA: thin's encoder writes no SDWA", class.name),
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
            Rule::Setreg => cx.setreg_imm32()?,
            Rule::Sopp => cx.sopp()?,
            Rule::Smem => cx.smem()?,
            Rule::Short => cx.short()?,
            Rule::Vop3 => {
                cx.vop3()?;
                cx.vop3_opsel()?;
                cx.vop3_signs_and_clamp()?;
            }
            Rule::Vop3p => {
                cx.vop3()?;
                cx.packed()?;
            }
            Rule::Mfma => cx.mfma()?,
            Rule::Flat => cx.flat_family()?,
            Rule::Mubuf => cx.mubuf()?,
            Rule::Ds => cx.ds()?,
            Rule::Refused(_) => unreachable!("refused above"),
        }
        if cx.literal.is_some() && !cx.ins.operands().iter().any(|o| matches!(o, Opd::Int(_))) {
            return Err(cx.err(Why::Operand, None, "a literal no operand reads".into()));
        }
        cx.finish()
    }

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

/// Decoded instruction's physical IR form on gfx942.
pub(crate) fn physical_form(i: &Instruction) -> Result<&'static thin_target::Form, Error> {
    use std::sync::Mutex;
    use thin_target::{Field, FieldKind, Form};

    type Key = (i32, u16, u8);
    static FORMS: OnceLock<Mutex<HashMap<Key, &'static Form>>> = OnceLock::new();
    let target = crate::get(super::NAME)?;
    let th = i
        .modifiers()
        .iter()
        .find(|m| m.key == "sc0" && matches!(m.value, MVal::Flag))
        .map(|_| "sc0");
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
    if signs == 0 && opsel == 0 {
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
    let form = forms.entry((base.id, signs, opsel)).or_insert_with(|| {
        let mut form = base.clone();
        for (k, slot) in form.slots.iter_mut().enumerate() {
            if signs & (1 << k) != 0 {
                slot.modifiers = true;
            }
        }
        if opsel != 0 {
            form.fields.push(
                Field::new("op_sel", FieldKind::Bits, opsel.into(), false).expect("op_sel field"),
            );
        }
        Box::leak(Box::new(form))
    });
    Ok(*form)
}
