//! gfx942's line encoder: encodes .s lines as `llvm-mc -mcpu=gfx942` does;
//! any other line is refused.

use thin_target::{
    dpp16_control, parse_line, Encoded, Line, ModValue, Operand, OperandKind, Value,
};

use super::{Gfx942, WAITCNT_NONE};
use crate::tables::{Encoding, Op, OpOperand};

type Refusal = Result<(), String>;

/// Mnemonics LLVM spells otherwise on gfx942.
pub(super) const SPELLED_OTHERWISE: [&str; 0] = [];

/// The f32 values with inline codes 240..247, in code order.
pub(super) const INLINE_FLOATS: [f32; 8] = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];

/// The bits of 1/(2π) as an f32, the inline constant 248.
pub(super) const INV_2PI: u32 = 0x3E22_F983;

/// The f64 high-word bits for inline float codes 240..247 (`(f as f64).to_bits()`),
/// and 1/(2π) in f64 (`0x3FC4_5F30_6DC9_C882`).
pub(super) const INV_2PI_F64: u64 = 0x3FC4_5F30_6DC9_C882;

/// Numbered SGPR count (`s0..s101`), followed by `flat_scratch` (102), `xnack_mask` (104), `vcc_lo` (106).
pub(super) const SGPR_COUNT: u32 = 102;
pub(super) const VCC_LO: u32 = 106;
pub(super) const M0_CODE: u32 = 124;
pub(super) const EXEC_LO_CODE: u32 = 126;
pub(super) const SADDR_OFF: u32 = 0x7F;

/// An instruction word being built in one encoding class.
struct Word<'a> {
    class: &'a str,
    encoding: &'a Encoding,
    bits: u128,
    literal: Option<u32>,
    scalar_bus: Option<(u32, u32)>,
}

impl<'a> Word<'a> {
    fn new(class: &'a str, encoding: &'a Encoding, opcode: u64) -> Result<Word<'a>, String> {
        let mut word = Word {
            class,
            encoding,
            bits: 0,
            literal: None,
            scalar_bus: None,
        };
        for f in &encoding.fields {
            let value = match f.kind.as_str() {
                "op" => opcode,
                _ => f.value,
            };
            if value != 0 {
                word.put(&f.name, value)?;
            }
        }
        Ok(word)
    }

    fn put(&mut self, name: &str, value: u64) -> Refusal {
        let field = (self.encoding.field(name))
            .ok_or_else(|| format!("{}: no field {name}", self.class))?;
        let bits = field.bits();
        if bits < 64 && value >> bits != 0 {
            return Err(format!("{name} {value}: does not fit its {bits} bits"));
        }
        let mask = ((1u128 << bits) - 1) << field.lo;
        self.bits = (self.bits & !mask) | (u128::from(value) << field.lo);
        Ok(())
    }

    fn literal(&mut self, value: u32) -> Refusal {
        match self.literal {
            Some(v) if v != value => Err("a second, different literal".into()),
            _ => {
                self.literal = Some(value);
                Ok(())
            }
        }
    }

    fn use_scalar(&mut self, code: u32, width: u32) -> Refusal {
        if !self.class.starts_with("VOP") || code >= 128 {
            return Ok(());
        }
        match self.scalar_bus {
            Some((prev, pw)) if (prev, pw) != (code, width) => {
                Err("constant bus limit (1 scalar source on gfx942) exceeded".into())
            }
            _ => {
                self.scalar_bus = Some((code, width));
                Ok(())
            }
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = self.bits.to_le_bytes()[..self.encoding.bytes].to_vec();
        if let Some(literal) = self.literal {
            out.extend(literal.to_le_bytes());
        }
        out
    }
}

/// The dwords of an operand field on wave64 (`FMT_NUM_M64` is 2 dwords = 64 bits).
pub(super) fn dwords(o: &OpOperand) -> u32 {
    (o.bits / 32).max(1)
}

/// Where a scalar range of `width` dwords may start: a pair at an even SGPR,
/// three dwords or more at a multiple of 4.
pub(crate) fn scalar_align(width: u32) -> u32 {
    if width > 2 {
        4
    } else {
        width.max(1)
    }
}

/// CDNA3 requires multi-dword VGPR and AGPR tuples to start at an even register.
pub(crate) fn vector_align(width: u32) -> u32 {
    if width > 1 {
        2
    } else {
        1
    }
}

fn aligned(o: &Operand) -> Refusal {
    match o.kind {
        OperandKind::S if o.count > 1 && o.n < SGPR_COUNT => {
            let align = scalar_align(o.count);
            if o.n % align != 0 {
                return Err(format!(
                    "`{}`: a scalar range starts at a multiple of {align}",
                    o.text
                ));
            }
        }
        OperandKind::V | OperandKind::A if o.count > 1 => {
            let align = vector_align(o.count);
            if o.n % align != 0 {
                return Err(format!(
                    "`{}`: a multi-dword {} range starts at an even register",
                    o.text,
                    o.kind.name()
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

/// A register operand of a fixed file and width.
fn register(o: &Operand, file: OperandKind, width: u32) -> Result<u32, String> {
    if o.kind != file || o.count != width {
        return Err(format!(
            "`{}`: a {}-register {} operand",
            o.text,
            width,
            file.name()
        ));
    }
    if file == OperandKind::S && o.n == 125 {
        return Err("`null` is not supported on gfx942".into());
    }
    aligned(o)?;
    Ok(o.n)
}

/// A VGPR-or-AGPR operand of `width` dwords, returning `(reg, is_acc)`.
fn av_register(o: &Operand, width: u32) -> Result<(u32, bool), String> {
    match o.kind {
        OperandKind::V if o.count == width => {
            aligned(o)?;
            Ok((o.n, false))
        }
        OperandKind::A if o.count == width => {
            aligned(o)?;
            Ok((o.n, true))
        }
        _ => Err(format!(
            "`{}`: a {width}-dword VGPR or AGPR operand",
            o.text
        )),
    }
}

/// Source code for an operand field (`OPR_SRC*`, `OPR_SSRC*`, `OPR_SREG*`, `OPR_VGPR*`, `OPR_ACCVGPR*`).
fn source(word: &mut Word<'_>, field: &OpOperand, o: &Operand) -> Result<u32, String> {
    let t = field.xml_type.as_str();
    let (vgprs, agprs, sgprs, constants, no_lit) = match t {
        "OPR_SRC" | "OPR_SRC_NOLDS" => (true, false, true, true, false),
        "OPR_SRC_NOLIT" | "OPR_SRC_SIMPLE" => (true, false, true, true, true),
        "OPR_SRC_VGPR_OR_INLINE" => (true, false, false, true, true),
        "OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST" => (true, true, false, true, true),
        "OPR_VGPR_OR_ACCVGPR" | "OPR_SRC_VGPR_OR_ACCVGPR" => (true, true, false, false, true),
        "OPR_SSRC" => (false, false, true, true, false),
        "OPR_SSRC_NOLIT" => (false, false, true, true, true),
        "OPR_SSRC_LANESEL" => (false, false, true, true, true),
        "OPR_SREG" | "OPR_SREG_NOVCC" | "OPR_SREG_M0" => (false, false, true, false, true),
        "OPR_VGPR" | "OPR_SRC_VGPR" | "OPR_VGPR_OR_LDS" => (true, false, false, false, true),
        "OPR_ACCVGPR" | "OPR_SRC_ACCVGPR" => (false, true, false, false, true),
        other => return Err(format!("`{}`: operand type {other} is not taken", o.text)),
    };
    let width = dwords(field);
    let text = &o.text;
    let code = match o.kind {
        OperandKind::V if vgprs && o.count == width => {
            aligned(o)?;
            256 + o.n
        }
        OperandKind::V if !vgprs => return Err(format!("`{text}`: no VGPR here")),
        OperandKind::A if agprs && o.count == width => {
            aligned(o)?;
            256 + o.n
        }
        OperandKind::A if !agprs => return Err(format!("`{text}`: no AGPR here")),
        OperandKind::S if sgprs && o.count == width => {
            if o.n == 125 {
                return Err("`null` is not supported on gfx942".into());
            }
            if t == "OPR_SREG_NOVCC" && (VCC_LO..=VCC_LO + 1).contains(&o.n) {
                return Err(format!("`{text}`: VCC is not allowed here"));
            }
            if t == "OPR_SREG_M0" && o.n != M0_CODE {
                return Err(format!("`{text}`: m0 expected"));
            }
            aligned(o)?;
            if !(t == "OPR_SSRC_LANESEL" && o.n == M0_CODE) {
                word.use_scalar(o.n, width)?;
            }
            o.n
        }
        OperandKind::S if !sgprs => return Err(format!("`{text}`: no scalar register here")),
        OperandKind::V | OperandKind::A | OperandKind::S => {
            return Err(format!("`{text}`: a {width}-dword operand"))
        }
        _ if !constants => return Err(format!("`{text}`: no constant here")),
        _ => constant(word, field, o, width, !no_lit)?,
    };
    Ok(code)
}

fn constant(
    word: &mut Word<'_>,
    field: &OpOperand,
    o: &Operand,
    width: u32,
    allow_literal: bool,
) -> Result<u32, String> {
    let text = &o.text;
    let wide = matches!(
        field.format.as_str(),
        "FMT_NUM_F32" | "FMT_NUM_B32" | "FMT_NUM_U32" | "FMT_NUM_I32"
    );
    let f64_fmt = field.format == "FMT_NUM_F64";
    let b64_fmt = matches!(
        field.format.as_str(),
        "FMT_NUM_B64" | "FMT_NUM_U64" | "FMT_NUM_I64" | "FMT_NUM_M64"
    );
    match (o.kind, &o.value) {
        (OperandKind::Int, Some(Value::Int(v))) => {
            if (0..=64).contains(v) {
                return Ok(128 + *v as u32);
            }
            if (-16..0).contains(v) {
                return Ok((192 - v) as u32);
            }
            if b64_fmt && width == 2 {
                // 64-bit inline floats in hex or 32-bit zero-extended literal if allowed
                let u = *v as u64;
                if let Some(i) = INLINE_FLOATS
                    .iter()
                    .position(|f| f64::from(*f).to_bits() == u)
                {
                    return Ok(240 + i as u32);
                }
                if u == INV_2PI_F64 {
                    return Ok(248);
                }
            }
            if !allow_literal || !wide || width != 1 {
                return Err(format!("`{text}`: only an inline constant here"));
            }
            if !(-(1i128 << 31)..(1i128 << 32)).contains(v) {
                return Err(format!("`{text}`: does not fit 32 bits"));
            }
            let bits = *v as u32;
            if let Some(i) = INLINE_FLOATS.iter().position(|f| f.to_bits() == bits) {
                return Ok(240 + i as u32);
            }
            if bits == INV_2PI {
                return Ok(248);
            }
            word.literal(bits)?;
            Ok(255)
        }
        (OperandKind::Float, Some(Value::Float(f))) if wide || f64_fmt => {
            if *f == 0.0 && f.is_sign_positive() {
                return Ok(128);
            }
            if f64_fmt {
                if f.to_bits() == INV_2PI_F64 || (*f as f32).to_bits() == INV_2PI {
                    return Ok(248);
                }
                if let Some(i) = INLINE_FLOATS.iter().position(|x| f64::from(*x) == *f) {
                    return Ok(240 + i as u32);
                }
                return Err(format!("`{text}`: only inline f64 constants here"));
            }
            if (*f as f32).to_bits() == INV_2PI {
                return Ok(248);
            }
            if let Some(i) = INLINE_FLOATS.iter().position(|x| f64::from(*x) == *f) {
                return Ok(240 + i as u32);
            }
            let single = *f as f32;
            if allow_literal && width == 1 && f64::from(single) == *f {
                word.literal(single.to_bits())?;
                return Ok(255);
            }
            Err(format!("`{text}`: not an encodable f32 constant here"))
        }
        _ => Err(format!("`{text}`: operand not taken")),
    }
}

fn fields<'a>(op: &'a Op, names: &[&str]) -> Vec<&'a OpOperand> {
    names.iter().filter_map(|n| op.operand(n)).collect()
}

fn count(line: &Line, n: usize) -> Refusal {
    if line.operands.len() != n {
        return Err(format!("{} operands, not {n}", line.operands.len()));
    }
    Ok(())
}

fn int_modifier(line: &Line, key: &str) -> Result<Option<i128>, String> {
    match line.modifiers.take(key) {
        None => Ok(None),
        Some(ModValue::Int(v)) => Ok(Some(*v)),
        Some(ModValue::List(_)) => Err(format!("{key}: an integer")),
    }
}

fn flag_modifier(line: &Line, key: &str) -> Result<u64, String> {
    Ok(u64::from(int_modifier(line, key)?.is_some_and(|v| v != 0)))
}

/// Scalar ALU ops (`SOP1`, `SOP2`, `SOPC`, `SOPK`).
fn salu(word: &mut Word<'_>, op: &Op, line: &Line) -> Refusal {
    let order: &[&str] = if line.mnemonic == "s_setreg_b32" {
        &["simm16", "sdst"]
    } else {
        &["sdst", "ssrc0", "ssrc1", "simm16"]
    };
    let present = fields(op, order);
    count(line, present.len())?;
    for (f, o) in present.into_iter().zip(&line.operands) {
        let value = match f.field.as_str() {
            "sdst" => register(o, OperandKind::S, dwords(f))?,
            "simm16" => {
                let taken = match f.xml_type.as_str() {
                    "OPR_SIMM16" | "OPR_VERSION" => !o.text.starts_with("hwreg("),
                    "OPR_HWREG" => true,
                    _ => false,
                };
                let v = o.int().filter(|_| taken);
                match v {
                    Some(v) if (-32768..65536).contains(&v) => (v & 0xFFFF) as u32,
                    _ => return Err(format!("`{}`: not a 16-bit immediate here", o.text)),
                }
            }
            _ => source(word, f, o)?,
        };
        word.put(&f.field, u64::from(value))?;
    }
    Ok(())
}

/// `s_setreg_imm32_b32 hwreg(...), imm32` (`SOPK_LIT`).
fn setreg_imm32(word: &mut Word<'_>, line: &Line) -> Refusal {
    count(line, 2)?;
    let (reg, imm) = (&line.operands[0], &line.operands[1]);
    let Some(code) = reg.int().filter(|_| reg.text.starts_with("hwreg(")) else {
        return Err(format!("`{}`: a hwreg(...) operand", reg.text));
    };
    word.put("simm16", code as u64)?;
    match imm.int().filter(|_| imm.kind == OperandKind::Int) {
        Some(v) if (-(1i128 << 31)..1i128 << 32).contains(&v) => {
            word.put("literal", u64::from(v as u32))
        }
        _ => Err(format!("`{}`: a 32-bit integer", imm.text)),
    }
}

/// Packs `vmcnt`, `expcnt`, `lgkmcnt` into `s_waitcnt`'s 16-bit `simm16`.
/// GFX9 layout: `vmcnt[3:0]` at `[3:0]`, `expcnt[2:0]` at `[6:4]`,
/// `lgkmcnt[3:0]` at `[11:8]`, `vmcnt[5:4]` at `[15:14]`.
pub(super) fn pack_waitcnt(vm: u32, exp: u32, lgkm: u32) -> u32 {
    (vm & 0xF) | ((exp & 0x7) << 4) | ((lgkm & 0xF) << 8) | (((vm >> 4) & 0x3) << 14)
}

pub(super) fn unpack_waitcnt(simm16: u32) -> (u32, u32, u32) {
    let vm = (simm16 & 0xF) | (((simm16 >> 14) & 0x3) << 4);
    let exp = (simm16 >> 4) & 0x7;
    let lgkm = (simm16 >> 8) & 0xF;
    (vm, exp, lgkm)
}

fn waitcnt(line: &Line) -> Result<u32, String> {
    if let Some(&ModValue::Int(v)) = line.modifiers.take("simm16") {
        if (0..65536).contains(&v) && (v as u32) & !WAITCNT_NONE == 0 {
            return Ok(v as u32);
        }
        return Err(format!("`{v}`: not a canonical s_waitcnt mask"));
    }
    let mut vm = 63u32;
    let mut exp = 7u32;
    let mut lgkm = 15u32;
    let mut any = false;
    for (name, max, dst) in [
        ("vmcnt", 63i128, &mut vm),
        ("expcnt", 7i128, &mut exp),
        ("lgkmcnt", 15i128, &mut lgkm),
    ] {
        if let Some(given) = line.modifiers.take(name) {
            let ModValue::Int(v) = given else {
                return Err(format!("{name}: a count"));
            };
            if !(0..=max).contains(v) {
                return Err(format!("{name}({v}): 0..{max}"));
            }
            *dst = *v as u32;
            any = true;
        }
    }
    if !any {
        return Err("s_waitcnt: no count".into());
    }
    Ok(pack_waitcnt(vm, exp, lgkm))
}

/// Program control (`SOPP`).
fn sopp(word: &mut Word<'_>, op: &Op, line: &Line) -> Result<Option<Encoded>, String> {
    let simm = op.operand("simm16");
    let kind = simm.map(|s| s.xml_type.as_str());
    match (kind, line.operands.as_slice()) {
        (Some("OPR_WAITCNT"), []) => {
            word.put("simm16", u64::from(waitcnt(line)?))?;
        }
        (Some("OPR_WAITCNT"), [o]) => match o.int().filter(|_| o.kind == OperandKind::Int) {
            Some(v) if (0..65536).contains(&v) && (v as u32) & !WAITCNT_NONE == 0 => {
                word.put("simm16", v as u64)?;
            }
            _ => return Err(format!("`{}`: not a canonical s_waitcnt mask", o.text)),
        },
        (None, []) => {}
        (Some(kind), [o]) => match (kind, o.kind, &o.value) {
            ("OPR_LABEL", OperandKind::Label, Some(Value::Label(label))) => {
                return Ok(Some(Encoded::Branch {
                    raw: word.bits as u32,
                    label: label.clone(),
                }))
            }
            (
                "OPR_SIMM16" | "OPR_SENDMSG" | "OPR_CLAUSE" | "OPR_SLEEP",
                OperandKind::Int,
                Some(Value::Int(v)),
            ) if (-32768..65536).contains(v) => {
                word.put("simm16", (*v & 0xFFFF) as u64)?;
            }
            _ => return Err(format!("`{}`: not taken here", o.text)),
        },
        _ => return Err(format!("{} operands", line.operands.len())),
    }
    Ok(None)
}

/// Signed 21-bit offset for GFX9 SMEM.
fn offset21(v: i128) -> Result<u64, String> {
    if !(-(1 << 20)..(1 << 20)).contains(&v) {
        return Err(format!("offset {v}: does not fit 21 bits signed"));
    }
    Ok((v & 0x1F_FFFF) as u64)
}

/// Scalar memory (`SMEM`):
/// - 0-operand (`s_dcache_inv`, `s_dcache_wb`, `s_gl1_inv`, ...),
/// - 1-operand (`s_memtime s[0:1]`, `s_memrealtime s[0:1]`, `s_dcache_discard_x2 ...`? No, discard takes base+off),
/// - 2-operand (`s_dcache_discard`, `s_atc_probe*`: `[sdata|imm, sbase, off]` or `[sbase, off]`),
/// - 3-operand (`s_load_*`, `s_store_*`, `s_buffer_*`, `s_atomic_*`).
fn smem(word: &mut Word<'_>, op: &Op, line: &Line) -> Refusal {
    let glc = flag_modifier(line, "glc")?;
    if glc != 0 {
        word.put("glc", 1)?;
    }
    let modifier = int_modifier(line, "offset")?;
    let has_data = op.operand("sdata").is_some();
    let has_base = op.operand("sbase").is_some();
    let has_off = op.operand("offset").is_some() || op.operand("soffset").is_some();
    let ops = &line.operands;

    match (has_data, has_base, has_off) {
        (false, false, false) => {
            count(line, 0)?;
            if modifier.is_some() {
                return Err("offset: not taken here".into());
            }
            Ok(())
        }
        (true, false, false) => {
            count(line, 1)?;
            if modifier.is_some() {
                return Err("offset: not taken here".into());
            }
            let d = op.operand("sdata").expect("sdata");
            word.put(
                "sdata",
                u64::from(register(&ops[0], OperandKind::S, dwords(d))?),
            )
        }
        (false, true, true) => {
            count(line, 2)?;
            let b = op.operand("sbase").expect("sbase");
            let breg = register(&ops[0], OperandKind::S, dwords(b))?;
            word.put("sbase", u64::from(breg / 2))?;
            smem_offset(word, &ops[1], modifier)
        }
        (true, true, true) => {
            count(line, 3)?;
            let d = op.operand("sdata").expect("sdata");
            let b = op.operand("sbase").expect("sbase");
            let data_val = if d.xml_type == "OPR_SIMM8" || d.format == "FMT_NUM_U8" {
                match ops[0].int().filter(|_| ops[0].kind == OperandKind::Int) {
                    Some(v) if (0..128).contains(&v) => v as u32,
                    _ => return Err(format!("`{}`: 0..127 expected", ops[0].text)),
                }
            } else {
                register(&ops[0], OperandKind::S, dwords(d))?
            };
            word.put("sdata", u64::from(data_val))?;
            let breg = register(&ops[1], OperandKind::S, dwords(b))?;
            word.put("sbase", u64::from(breg / 2))?;
            smem_offset(word, &ops[2], modifier)
        }
        _ => Err(format!("{}: unsupported SMEM shape", line.mnemonic)),
    }
}

fn smem_offset(word: &mut Word<'_>, off_op: &Operand, modifier: Option<i128>) -> Refusal {
    match (off_op, modifier) {
        (o, None) if o.kind == OperandKind::S && o.count == 1 && o.n <= M0_CODE => {
            // Register offset only: imm=0, soffset_en=0, register number in `offset[6:0]`.
            word.put("offset", u64::from(o.n))
        }
        (o, Some(m)) if o.kind == OperandKind::S && o.count == 1 && o.n <= M0_CODE => {
            // Register + immediate offset: imm=1, soffset_en=1, reg in `soffset`, imm in `offset`.
            word.put("imm", 1)?;
            word.put("soffset_en", 1)?;
            word.put("soffset", u64::from(o.n))?;
            word.put("offset", offset21(m)?)
        }
        (o, None) if o.kind == OperandKind::Int => {
            word.put("imm", 1)?;
            word.put("offset", offset21(o.int().unwrap_or(0))?)
        }
        _ => Err(format!("`{}`: not a valid SMEM offset", off_op.text)),
    }
}

/// Whether an operand is wave64 `vcc` (`s[106:107]`, 2 dwords).
fn is_vcc64(o: &Operand) -> bool {
    o.kind == OperandKind::S && o.n == VCC_LO && o.count == 2
}

fn is_vgpr(o: &Operand, width: u32) -> bool {
    o.kind == OperandKind::V && o.count == width && (width == 1 || o.n % 2 == 0)
}

pub(super) fn is_carry_vop2(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "v_add_co_u32"
            | "v_sub_co_u32"
            | "v_subrev_co_u32"
            | "v_addc_co_u32"
            | "v_subb_co_u32"
            | "v_subbrev_co_u32"
    )
}

pub(super) fn is_carry_in_vop2(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32"
    )
}

fn fits_short(op: &Op, mnemonic: &str, ops: &[Operand]) -> bool {
    let unsigned = ops.iter().all(|o| !o.neg && !o.abs);
    unsigned && short_operands(&op.class, mnemonic, ops).is_ok() && fits_shape(op, mnemonic, ops)
}

fn fits_shape(op: &Op, mnemonic: &str, ops: &[Operand]) -> bool {
    let v1 = op.operand("vsrc1").map_or(1, dwords);
    let vd = op.operand("vdst").map_or(1, dwords);
    match op.class.as_str() {
        "VOPC" => match ops {
            [d, _, b] => is_vcc64(d) && is_vgpr(b, v1),
            [_, b] if mnemonic.starts_with("v_cmpx") => is_vgpr(b, v1),
            _ => false,
        },
        "VOP2" if mnemonic.starts_with("v_cndmask") => {
            matches!(ops, [d, _, b, c] if is_vgpr(d, vd) && is_vgpr(b, v1) && is_vcc64(c))
        }
        "VOP2" if is_carry_in_vop2(mnemonic) => {
            matches!(ops, [d, c, _, b, ci] if is_vgpr(d, vd) && is_vcc64(c) && is_vgpr(b, v1) && is_vcc64(ci))
        }
        "VOP2" if is_carry_vop2(mnemonic) => {
            matches!(ops, [d, c, _, b] if is_vgpr(d, vd) && is_vcc64(c) && is_vgpr(b, v1))
        }
        "VOP2" => matches!(ops, [d, _, b] if is_vgpr(d, vd) && is_vgpr(b, v1)),
        "VOP1" if mnemonic == "v_accvgpr_mov_b32" => {
            matches!(ops, [d, a] if d.kind == OperandKind::A && d.count == 1 && a.kind == OperandKind::A && a.count == 1)
        }
        "VOP1" => match ops {
            [] => op.operands.is_empty(),
            [d, _] => is_vgpr(d, vd),
            _ => false,
        },
        _ => true,
    }
}

fn short_operands<'o>(
    class: &str,
    mnemonic: &str,
    ops: &'o [Operand],
) -> Result<Vec<(&'static str, &'o Operand)>, String> {
    let base = mnemonic.strip_suffix("_e32").unwrap_or(mnemonic);
    Ok(match (class, ops) {
        ("VOPC", [_, a, b]) | ("VOPC", [a, b]) => vec![("src0", a), ("vsrc1", b)],
        ("VOP2", [d, _, a, b, _]) if is_carry_in_vop2(base) => {
            vec![("vdst", d), ("src0", a), ("vsrc1", b)]
        }
        ("VOP2", [d, _, a, b]) if is_carry_vop2(base) => {
            vec![("vdst", d), ("src0", a), ("vsrc1", b)]
        }
        ("VOP2", [d, a, b, _]) if base.starts_with("v_cndmask") => {
            vec![("vdst", d), ("src0", a), ("vsrc1", b)]
        }
        ("VOP2", [d, a, b]) => vec![("vdst", d), ("src0", a), ("vsrc1", b)],
        ("VOP1" | "VOP1_SDST", [d, a]) => vec![("vdst", d), ("src0", a)],
        ("VOP1" | "VOP1_SDST", []) => vec![],
        _ => return Err(format!("{} operands", ops.len())),
    })
}

fn short(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let class = op.class.as_str();
    if op.operands.is_empty() {
        return count(line, 0);
    }
    let base = mnemonic.strip_suffix("_e32").unwrap_or(mnemonic);
    if class != "VOP1_SDST" && !fits_shape(op, base, &line.operands) {
        return Err("the operands do not fit the short form".into());
    }
    if class == "VOP2" && (is_carry_in_vop2(base) || base.starts_with("v_cndmask")) {
        word.use_scalar(VCC_LO, 2)?;
    }
    let by_field = short_operands(class, base, &line.operands)?;
    if by_field.len() != op.operands.len() {
        return Err(format!("{} operands", line.operands.len()));
    }
    for (name, o) in by_field {
        let f = op
            .operand(name)
            .ok_or_else(|| format!("no operand {name}"))?;
        let value = match name {
            "vdst" if class == "VOP1_SDST" => register(o, OperandKind::S, dwords(f))?,
            "vdst" if f.xml_type == "OPR_ACCVGPR" => register(o, OperandKind::A, dwords(f))?,
            "vdst" | "vsrc1" => register(o, OperandKind::V, dwords(f))?,
            _ => source(word, f, o)?,
        };
        word.put(name, u64::from(value))?;
    }
    Ok(())
}

/// VOP3, VOP3_SDST, VOP3SD, and VOP3P operands.
fn vop3(word: &mut Word<'_>, op: &Op, line: &Line) -> Refusal {
    let mut present = fields(op, &["vdst", "sdst", "src0", "src1", "src2"]);
    if let Some(at) = present.iter().position(|f| f.xml_type == "OPR_EXEC") {
        if line.operands.len() + 1 == present.len() {
            word.put(&present[at].field, u64::from(EXEC_LO_CODE))?;
            present.remove(at);
        }
    }
    if line.operands.len() != present.len() {
        return Err(format!(
            "{} operands, not {}",
            line.operands.len(),
            present.len()
        ));
    }
    for (f, o) in present.iter().zip(&line.operands) {
        let value = match f.field.as_str() {
            "vdst" if f.xml_type == "OPR_VGPR" => register(o, OperandKind::V, dwords(f))?,
            "vdst" if f.xml_type == "OPR_ACCVGPR" => register(o, OperandKind::A, dwords(f))?,
            "vdst" | "sdst" => {
                let s = register(o, OperandKind::S, dwords(f))?;
                if f.xml_type == "OPR_SREG_NOVCC" && (VCC_LO..=VCC_LO + 1).contains(&s) {
                    return Err(format!("`{}`: VCC is not allowed here", o.text));
                }
                s
            }
            _ => source(word, f, o)?,
        };
        word.put(&f.field, u64::from(value))?;
    }
    Ok(())
}

/// VOP3 ops accepting `op_sel` on gfx942 (verified exhaustively against `llvm-mc -mcpu=gfx942`).
pub(super) const OPSEL_OPS: [(&str, u32); 28] = [
    ("v_add_i16", 0b1011),
    ("v_alignbit_b32", 0b1111),
    ("v_alignbyte_b32", 0b1111),
    ("v_cvt_pk_bf8_f32", 0b1011),
    ("v_cvt_pk_fp8_f32", 0b1011),
    ("v_cvt_pknorm_i16_f16", 0b1011),
    ("v_cvt_pknorm_u16_f16", 0b1011),
    ("v_cvt_sr_bf8_f32", 0b1111),
    ("v_cvt_sr_fp8_f32", 0b1111),
    ("v_div_fixup_f16", 0b1111),
    ("v_fma_f16", 0b1111),
    ("v_mad_f16", 0b1111),
    ("v_mad_i16", 0b1111),
    ("v_mad_i32_i16", 0b1111),
    ("v_mad_u16", 0b1111),
    ("v_mad_u32_u16", 0b1111),
    ("v_max3_f16", 0b1111),
    ("v_max3_i16", 0b1111),
    ("v_max3_u16", 0b1111),
    ("v_med3_f16", 0b1111),
    ("v_med3_i16", 0b1111),
    ("v_med3_u16", 0b1111),
    ("v_min3_f16", 0b1111),
    ("v_min3_i16", 0b1111),
    ("v_min3_u16", 0b1111),
    ("v_pack_b32_f16", 0b1011),
    ("v_pk_fmac_f16_e64", 0b1011),
    ("v_sub_i16", 0b1011),
];

pub(super) fn opsel_bits(name: &str) -> u32 {
    OPSEL_OPS
        .iter()
        .find(|(m, _)| *m == name)
        .map_or(0, |o| o.1)
}

pub(super) fn opsel_len(name: &str, nsrc: usize) -> usize {
    if matches!(name, "v_cvt_sr_fp8_f32" | "v_cvt_sr_bf8_f32") {
        4
    } else {
        nsrc + 1
    }
}

fn vop3_opsel(word: &mut Word<'_>, op: &Op, name: &str, line: &Line) -> Refusal {
    let allowed = opsel_bits(name);
    if allowed == 0 {
        return Ok(());
    }
    let Some(given) = line.modifiers.take("op_sel") else {
        return Ok(());
    };
    let nsrc = fields(op, &["src0", "src1", "src2"]).len();
    let len = opsel_len(name, nsrc);
    let ModValue::List(bits) = given else {
        return Err(format!("op_sel: a list of {len} bits"));
    };
    if bits.len() != len || bits.iter().any(|b| !(0..=1).contains(b)) {
        return Err(format!("op_sel: a list of {len} bits"));
    }
    let value = bits.iter().enumerate().fold(0u32, |v, (j, b)| {
        let bit = if j + 1 == len && len < 4 { 3 } else { j };
        v | (*b as u32) << bit
    });
    if value & !allowed != 0 {
        return Err(format!(
            "op_sel: {name} reads OPSEL bits 0x{allowed:x} only"
        ));
    }
    word.put("opsel", u64::from(value))
}

/// Sign modifiers `(neg, abs)` for VOP3/VOP3_SDST/VOP3SD/DPP16 on gfx942
/// (verified exhaustively against `llvm-mc -mcpu=gfx942`).
pub(super) fn signs_of(name: &str, class: &str, field: &OpOperand) -> (bool, bool) {
    if matches!(
        name,
        "v_cmp_class_f16_e64" | "v_cmpx_class_f16_e64" | "v_ldexp_f16_e64"
    ) && field.field != "src0"
    {
        return (false, false);
    }
    let float = matches!(
        field.format.as_str(),
        "FMT_NUM_F16" | "FMT_NUM_BF16" | "FMT_NUM_F32" | "FMT_NUM_F64"
    );
    let cndmask = name == "v_cndmask_b32_e64" && (field.field == "src0" || field.field == "src1");
    let neg = float || cndmask;
    let abs = neg && class != "VOP3SD";
    (neg, abs)
}

fn vop3_signs_and_clamp(word: &mut Word<'_>, op: &Op, name: &str, line: &Line) -> Refusal {
    let mut present = fields(op, &["vdst", "sdst", "src0", "src1", "src2"]);
    if let Some(at) = present.iter().position(|f| f.xml_type == "OPR_EXEC") {
        if line.operands.len() + 1 == present.len() {
            present.remove(at);
        }
    }
    let (mut neg, mut abs) = (0u64, 0u64);
    for (f, o) in present.into_iter().zip(&line.operands) {
        if !o.neg && !o.abs {
            continue;
        }
        let (takes_neg, takes_abs) = signs_of(name, &op.class, f);
        let i = match f.field.as_str() {
            "src0" => 0,
            "src1" => 1,
            "src2" => 2,
            _ => return Err(format!("`{}`: no neg or abs on a destination", o.text)),
        };
        if (o.neg && !takes_neg) || (o.abs && !takes_abs) {
            let what = if o.abs && !takes_abs { "abs" } else { "neg" };
            return Err(format!(
                "`{}`: {name} takes no {what} on {}",
                o.text, f.field
            ));
        }
        neg |= u64::from(o.neg) << i;
        abs |= u64::from(o.abs) << i;
    }
    if neg != 0 {
        word.put("neg", neg)?;
    }
    if abs != 0 {
        word.put("abs", abs)?;
    }
    if flag_modifier(line, "clamp")? != 0 {
        word.put("clmp", 1)?;
    }
    if word.encoding.field("omod").is_some() {
        let mul = int_modifier(line, "mul")?;
        let div = int_modifier(line, "div")?;
        let omod = match (mul, div) {
            (None, None) => 0,
            (Some(2), None) => 1,
            (Some(4), None) => 2,
            (None, Some(2)) => 3,
            (Some(_), Some(_)) => return Err("cannot combine mul and div".into()),
            (Some(v), None) => return Err(format!("mul:{v}: expected mul:2 or mul:4")),
            (None, Some(v)) => return Err(format!("div:{v}: expected div:2")),
        };
        if omod != 0 {
            word.put("omod", omod)?;
        }
    }
    Ok(())
}

/// VOP3P packed rules on gfx942: `(masks[op_sel, op_sel_hi, neg_lo, neg_hi], clamp, hi_default, signs)`.
pub(super) fn packed_rules(m: &str, nsrc: usize) -> ([u64; 4], bool, u64, bool) {
    let src_mask = (1u64 << nsrc) - 1;
    if m.starts_with("v_accvgpr_") {
        ([0, 0, 0, 0], false, 7, false)
    } else if m.starts_with("v_fma_mix") {
        ([7, 7, 0, 0], true, 0, true)
    } else if m.starts_with("v_dot") {
        ([0, 0, 7, 7], true, 7, false)
    } else {
        // v_pk_* (2-src -> mask 3, 3-src -> mask 7)
        ([src_mask, src_mask, src_mask, src_mask], true, 7, false)
    }
}

fn packed(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let nsrc = fields(op, &["src0", "src1", "src2"]).len();
    let (masks, clamp, hi_default, takes_signs) = packed_rules(mnemonic, nsrc);
    if word.literal.is_some() {
        return Err("no literal in VOP3P on gfx942".into());
    }
    let mut sign_neg = 0u64;
    let mut sign_abs = 0u64;
    let present = fields(op, &["vdst", "src0", "src1", "src2"]);
    for (f, o) in present.into_iter().zip(&line.operands) {
        if !o.neg && !o.abs {
            continue;
        }
        if !takes_signs {
            return Err(format!("`{}`: no neg or abs here", o.text));
        }
        let i = match f.field.as_str() {
            "src0" => 0,
            "src1" => 1,
            "src2" => 2,
            _ => return Err(format!("`{}`: no neg or abs on a destination", o.text)),
        };
        sign_neg |= u64::from(o.neg) << i;
        sign_abs |= u64::from(o.abs) << i;
    }

    let mut values = [None; 4];
    for (k, key) in ["op_sel", "op_sel_hi", "neg_lo", "neg_hi"]
        .iter()
        .enumerate()
    {
        let Some(given) = line.modifiers.take(key) else {
            continue;
        };
        let ModValue::List(bits) = given else {
            return Err(format!("{key}: a list of {nsrc} bits"));
        };
        let past = bits.iter().skip(nsrc).any(|b| *b != 0);
        let shape = (1..=4).contains(&bits.len()) && bits.iter().all(|b| (0..=1).contains(b));
        if masks[k] == 0 || !shape || past {
            return Err(format!("{key}: not taken as given"));
        }
        let v = bits
            .iter()
            .enumerate()
            .fold(0u64, |v, (j, b)| v | (*b as u64) << j);
        if v & !masks[k] != 0 {
            return Err(format!("{key}: a bit this op does not take"));
        }
        values[k] = Some(v);
    }
    let hi = values[1].unwrap_or(hi_default);
    let hi2 = if nsrc == 3 {
        (hi >> 2) & 1
    } else {
        (hi_default >> 2) & 1
    };
    word.put("opsel_hi", hi & 3)?;
    word.put("opsel_hi2", hi2)?;
    if let Some(v) = values[0] {
        word.put("opsel", v)?;
    }
    let neg = values[2].unwrap_or(0) | sign_neg;
    let neg_hi = values[3].unwrap_or(0) | sign_abs;
    if neg != 0 {
        word.put("neg", neg)?;
    }
    if neg_hi != 0 {
        word.put("neg_hi", neg_hi)?;
    }
    if flag_modifier(line, "clamp")? != 0 {
        if !clamp {
            return Err("clamp: not taken".into());
        }
        word.put("clmp", 1)?;
    }
    Ok(())
}

/// Matrix ops (`VOP3P_MFMA`): `v_mfma_*` and `v_smfmac_*`.
fn mfma(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let present = fields(op, &["vdst", "src0", "src1", "src2"]);
    count(line, present.len())?;
    let smfmac = mnemonic.starts_with("v_smfmac_");
    let (vdst, cd_acc) = av_register(&line.operands[0], dwords(present[0]))?;
    word.put("vdst", u64::from(vdst))?;
    word.put("acc_cd", u64::from(cd_acc))?;

    let mut acc_ab = 0u64;
    for (idx, (&f, o)) in present[1..3].iter().zip(&line.operands[1..3]).enumerate() {
        let (reg, is_acc) = av_register(o, dwords(f))?;
        word.put(&f.field, 256 + u64::from(reg))?;
        acc_ab |= u64::from(is_acc) << idx;
    }
    word.put("acc", acc_ab)?;

    let f2 = present[3];
    let o2 = &line.operands[3];
    let w2 = dwords(f2);
    let src2_code = if smfmac {
        // v_smfmac_* src2 is always a VGPR index vector (OPR_SRC_VGPR).
        256 + register(o2, OperandKind::V, w2)?
    } else {
        match o2.kind {
            OperandKind::V | OperandKind::A => {
                let (reg, is_acc) = av_register(o2, w2)?;
                if is_acc != cd_acc {
                    return Err(format!(
                        "`{}`: MFMA src2 register file must match vdst",
                        o2.text
                    ));
                }
                256 + reg
            }
            OperandKind::Int | OperandKind::Float => constant(word, f2, o2, w2, false)?,
            _ => return Err(format!("`{}`: invalid MFMA src2 operand", o2.text)),
        }
    };
    word.put("src2", u64::from(src2_code))?;

    if let Some(cbsz) = int_modifier(line, "cbsz")? {
        if !(0..8).contains(&cbsz) {
            return Err(format!("cbsz {cbsz}: 0..7"));
        }
        word.put("cbsz", cbsz as u64)?;
    }
    if let Some(abid) = int_modifier(line, "abid")? {
        if !(0..16).contains(&abid) {
            return Err(format!("abid {abid}: 0..15"));
        }
        word.put("abid", abid as u64)?;
    }
    if let Some(blgp) = int_modifier(line, "blgp")? {
        if smfmac || !(0..8).contains(&blgp) {
            return Err(format!("blgp {blgp}: not valid here"));
        }
        word.put("blgp", blgp as u64)?;
    }
    Ok(())
}

/// DPP16 (`VOP1_DPP16`, `VOP2_DPP16`).
fn dpp(t: &Gfx942, mnemonic: &str, line: &Line) -> Result<Vec<u8>, String> {
    let base = mnemonic.strip_suffix("_dpp").expect("a _dpp mnemonic");
    if !t.rows.contains(mnemonic) {
        return Err(format!("{mnemonic}: not a DPP form"));
    }
    let short_name = format!("{base}_e32");
    let op = (t.isa.ops.get(&short_name))
        .filter(|op| matches!(op.class.as_str(), "VOP1" | "VOP2"))
        .ok_or_else(|| format!("{mnemonic}: DPP is VOP1 and VOP2 only"))?;
    let by_field = short_operands(&op.class, base, &line.operands)?;
    // Verify shape ignoring neg/abs on sources
    let unsigned_ops: Vec<Operand> = line
        .operands
        .iter()
        .cloned()
        .map(|mut o| {
            o.neg = false;
            o.abs = false;
            o
        })
        .collect();
    if !fits_shape(op, base, &unsigned_ops) {
        return Err("the operands do not fit the DPP form".into());
    }
    let class = format!("{}_DPP16", op.class);
    let encoding = t
        .isa
        .encodings
        .get(&class)
        .ok_or_else(|| format!("no class {class}"))?;
    let mut word = Word::new(&class, encoding, op.opcode)?;
    word.put("src0", 0xFA)?;
    for (name, o) in by_field {
        let f = op.operand(name).ok_or_else(|| format!("no field {name}"))?;
        let field = if name == "src0" { "vsrc0" } else { name };
        let file = if f.xml_type.contains("ACCVGPR") {
            OperandKind::A
        } else {
            OperandKind::V
        };
        if (o.neg || o.abs) && name == "vdst" {
            return Err(format!("`{}`: no neg or abs on vdst", o.text));
        }
        if o.neg || o.abs {
            let e64 = format!("{base}_e64");
            let sign_ok = if let Some(wop) = t.isa.ops.get(&e64) {
                let sname = if name == "vsrc1" { "src1" } else { "src0" };
                wop.operand(sname)
                    .map_or((false, false), |sf| signs_of(&e64, &wop.class, sf))
            } else {
                let float = matches!(
                    f.format.as_str(),
                    "FMT_NUM_F16" | "FMT_NUM_BF16" | "FMT_NUM_F32" | "FMT_NUM_F64"
                );
                (float, float)
            };
            if (o.neg && !sign_ok.0) || (o.abs && !sign_ok.1) {
                return Err(format!("`{}`: no neg or abs here", o.text));
            }
            let prefix = if name == "vsrc1" { "src1" } else { "src0" };
            if o.neg {
                word.put(&format!("{prefix}_neg"), 1)?;
            }
            if o.abs {
                word.put(&format!("{prefix}_abs"), 1)?;
            }
        }
        word.put(field, u64::from(register(o, file, dwords(f))?))?;
    }
    word.put("dpp", u64::from(dpp_code(line)?))?;
    let mask = |key| -> Result<u64, String> {
        let v = int_modifier(line, key)?.unwrap_or(0xF);
        if !(0..16).contains(&v) {
            return Err(format!("{key} {v}: 0..15"));
        }
        Ok(v as u64)
    };
    word.put("row_mask", mask("row_mask")?)?;
    word.put("bank_mask", mask("bank_mask")?)?;
    word.put("bc", flag_modifier(line, "bound_ctrl")?)?;
    Ok(word.bytes())
}

fn dpp_code(line: &Line) -> Result<u32, String> {
    let mut controls = Vec::new();
    if let Some(given) = line.modifiers.take("quad_perm") {
        match given {
            ModValue::List(l) if l.len() == 4 && l.iter().all(|x| (0..4).contains(x)) => {
                controls.push(
                    l.iter()
                        .enumerate()
                        .fold(0u32, |v, (i, x)| v | (*x as u32) << (2 * i)),
                );
            }
            _ => return Err("quad_perm: four selectors 0..3".into()),
        }
    }
    for key in [
        "row_shl",
        "row_shr",
        "row_ror",
        "row_newbcast",
        "row_bcast",
        "wave_shl",
        "wave_rol",
        "wave_shr",
        "wave_ror",
    ] {
        if let Some(given) = line.modifiers.take(key) {
            let ModValue::Int(v) = given else {
                return Err(format!("{key}: an integer"));
            };
            let code = u32::from(dpp16_control(&format!("{key}:{v}"))?);
            controls.push(code);
        }
    }
    for (key, code) in [("row_mirror", 0x140u32), ("row_half_mirror", 0x141u32)] {
        if line.modifiers.take(key).is_some() {
            controls.push(code);
        }
    }
    match controls.as_slice() {
        [one] => Ok(*one),
        _ => Err("exactly one DPP control".into()),
    }
}

fn cache_flags(word: &mut Word<'_>, line: &Line) -> Refusal {
    for key in ["sc0", "sc1", "nt"] {
        if flag_modifier(line, key)? != 0 {
            word.put(key, 1)?;
        }
    }
    Ok(())
}

/// `FLAT`, `GLOBAL`, and `SCRATCH` memory ops.
fn flat_family(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let class = op.class.as_str();
    let (dst, data) = (op.operand("vdst"), op.operand("data"));
    let ops = &line.operands;

    let mut acc_seen: Option<bool> = None;
    let mut check_acc = |is_acc: bool| -> Refusal {
        if let Some(prev) = acc_seen {
            if prev != is_acc {
                return Err("vdst and data must use the same register file (v or a)".into());
            }
        }
        acc_seen = Some(is_acc);
        Ok(())
    };

    if class == "FLAT" {
        let (d, addr, v) = match (dst, data, ops.as_slice()) {
            (Some(_), Some(_), [d, a, v]) => (Some(d), Some(a), Some(v)),
            (Some(_), Some(_), [a, v]) if mnemonic.contains("atomic") => (None, Some(a), Some(v)),
            (Some(_), None, [d, a]) => (Some(d), Some(a), None),
            (None, Some(_), [a, v]) => (None, Some(a), Some(v)),
            (None, None, []) => (None, None, None),
            _ => return Err(format!("{} operands", ops.len())),
        };
        let has_sc0 = line.modifiers.iter().any(|(k, _)| k == "sc0");
        if d.is_some() && mnemonic.contains("atomic") && !has_sc0 {
            return Err("returning flat atomic requires sc0".into());
        }
        if d.is_none() && mnemonic.contains("atomic") && has_sc0 {
            return Err("non-returning flat atomic cannot set sc0".into());
        }
        if let (Some(d), Some(f)) = (d, dst) {
            let (r, is_acc) = av_register(d, dwords(f))?;
            check_acc(is_acc)?;
            word.put("vdst", u64::from(r))?;
        }
        if let (Some(v), Some(f)) = (v, data) {
            let (r, is_acc) = av_register(v, dwords(f))?;
            check_acc(is_acc)?;
            word.put("data", u64::from(r))?;
        }
        if let Some(a) = addr {
            word.put("addr", u64::from(register(a, OperandKind::V, 2)?))?;
        }
        if acc_seen == Some(true) {
            word.put("acc", 1)?;
        }
        let off = int_modifier(line, "offset")?.unwrap_or(0);
        if !(0..4096).contains(&off) {
            return Err(format!("offset {off}: 0..4095 for flat"));
        }
        word.put("offset", off as u64)?;
        return cache_flags(word, line);
    }

    // GLOBAL and SCRATCH
    let (d, addr, v, saddr) = match (dst, data, ops.as_slice()) {
        (Some(_), Some(_), [d, a, v, s]) => (Some(d), Some(a), Some(v), Some(s)),
        (Some(_), Some(_), [a, v, s]) if mnemonic.contains("atomic") => {
            (None, Some(a), Some(v), Some(s))
        }
        (Some(_), None, [d, a, s]) => (Some(d), Some(a), None, Some(s)),
        (None, Some(_), [a, v, s]) => (None, Some(a), Some(v), Some(s)),
        // global_load_lds_* / scratch_load_lds_* (2 operands: addr, saddr)
        (None, None, [a, s]) => (None, Some(a), None, Some(s)),
        // global_wb* / global_inv* (0 operands)
        (None, None, []) => (None, None, None, None),
        _ => return Err(format!("{} operands", ops.len())),
    };
    let has_sc0 = line.modifiers.iter().any(|(k, _)| k == "sc0");
    if d.is_some() && mnemonic.contains("atomic") && !has_sc0 {
        return Err(format!("returning {mnemonic} requires sc0"));
    }
    if d.is_none() && mnemonic.contains("atomic") && has_sc0 {
        return Err(format!("non-returning {mnemonic} cannot set sc0"));
    }
    if let (Some(d), Some(f)) = (d, dst) {
        let (r, is_acc) = av_register(d, dwords(f))?;
        check_acc(is_acc)?;
        word.put("vdst", u64::from(r))?;
    }
    if let (Some(v), Some(f)) = (v, data) {
        let (r, is_acc) = av_register(v, dwords(f))?;
        check_acc(is_acc)?;
        word.put("data", u64::from(r))?;
    }
    if acc_seen == Some(true) {
        word.put("acc", 1)?;
    }
    if let (Some(addr), Some(saddr)) = (addr, saddr) {
        if class == "GLOBAL" {
            if saddr.kind == OperandKind::Off {
                word.put("saddr", u64::from(SADDR_OFF))?;
                word.put("addr", u64::from(register(addr, OperandKind::V, 2)?))?;
            } else {
                let s = register(saddr, OperandKind::S, 2)?;
                if s >= SADDR_OFF {
                    return Err(format!("`{}`: invalid global saddr", saddr.text));
                }
                word.put("saddr", u64::from(s))?;
                word.put("addr", u64::from(register(addr, OperandKind::V, 1)?))?;
            }
        } else {
            // SCRATCH: vaddr is `off` (sve=0) or 1-dword VGPR (sve=1); saddr is `off` (0x7f) or 1-dword SGPR.
            if addr.kind == OperandKind::Off {
                word.put("sve", 0)?;
                word.put("addr", 0)?;
            } else {
                word.put("sve", 1)?;
                word.put("addr", u64::from(register(addr, OperandKind::V, 1)?))?;
            }
            if saddr.kind == OperandKind::Off {
                word.put("saddr", u64::from(SADDR_OFF))?;
            } else {
                let s = register(saddr, OperandKind::S, 1)?;
                if s >= SADDR_OFF {
                    return Err(format!("`{}`: invalid scratch saddr", saddr.text));
                }
                word.put("saddr", u64::from(s))?;
            }
        }
    } else if saddr.is_none() {
        word.put("saddr", 0)?;
    }
    let off = int_modifier(line, "offset")?.unwrap_or(0);
    if !(-4096..4096).contains(&off) {
        return Err(format!("offset {off}: -4096..4095"));
    }
    word.put("offset", (off & 0x1FFF) as u64)?;
    cache_flags(word, line)
}

/// `MUBUF` buffer loads, stores, atomics, LDS DMA, and cache maintenance (`buffer_wbl2`, `buffer_inv`).
fn mubuf(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let ops = &line.operands;
    if op.operands.is_empty() {
        // buffer_wbl2 / buffer_inv
        count(line, 0)?;
        word.put("soffset", 0)?;
        return cache_flags(word, line);
    }
    let Some(data_f) = op.operand("vdata") else {
        return Err(format!("{mnemonic}: buffer op without vdata"));
    };
    let lds = flag_modifier(line, "lds")?;
    let idxen = flag_modifier(line, "idxen")?;
    let offen = flag_modifier(line, "offen")?;
    let (vdata_opt, vaddr_op, rsrc_op, soff_op) = match (lds != 0, ops.as_slice()) {
        (true, [va, rs, so]) if mnemonic.starts_with("buffer_load_") => (None, va, rs, so),
        (false, [vd, va, rs, so]) => (Some(vd), va, rs, so),
        _ => return Err(format!("{} operands", ops.len())),
    };
    if let Some(vd) = vdata_opt {
        let (reg, is_acc) = av_register(vd, dwords(data_f))?;
        word.put("vdata", u64::from(reg))?;
        if is_acc {
            word.put("acc", 1)?;
        }
    } else {
        word.put("lds", 1)?;
    }
    match (idxen + offen, vaddr_op.kind) {
        (0, OperandKind::Off) => {}
        (0, _) => return Err(format!("`{}`: `off` without idxen or offen", vaddr_op.text)),
        (n, _) => {
            word.put(
                "vaddr",
                u64::from(register(vaddr_op, OperandKind::V, n as u32)?),
            )?;
        }
    }
    word.put("idxen", idxen)?;
    word.put("offen", offen)?;
    let rsrc = register(rsrc_op, OperandKind::S, 4)?;
    word.put("srsrc", u64::from(rsrc / 4))?;

    let soff_dummy = OpOperand {
        field: "soffset".into(),
        xml_type: "OPR_SSRC_NOLIT".into(),
        format: "FMT_NUM_U32".into(),
        bits: 32,
    };
    let soff = source(word, &soff_dummy, soff_op)?;
    if soff >= 256 || soff == 255 {
        return Err(format!(
            "`{}`: invalid buffer soffset on gfx942",
            soff_op.text
        ));
    }
    word.put("soffset", u64::from(soff))?;
    let offset = int_modifier(line, "offset")?.unwrap_or(0);
    if !(0..4096).contains(&offset) {
        return Err(format!("offset {offset}: 0..4095"));
    }
    word.put("offset", offset as u64)?;
    cache_flags(word, line)
}

pub(super) fn is_ds_two_offset(mnemonic: &str) -> bool {
    mnemonic.starts_with("ds_read2")
        || mnemonic.starts_with("ds_write2")
        || mnemonic.contains("2addr")
}

/// `DS` (LDS and GDS).
fn ds(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let present = fields(op, &["vdst", "addr", "data0", "data1"]);
    count(line, present.len())?;
    let mut acc_seen: Option<bool> = None;
    for (f, o) in present.into_iter().zip(&line.operands) {
        if f.field == "addr" || f.xml_type == "OPR_VGPR" {
            word.put(&f.field, u64::from(register(o, OperandKind::V, dwords(f))?))?;
        } else {
            let (reg, is_acc) = av_register(o, dwords(f))?;
            if let Some(prev) = acc_seen {
                if prev != is_acc {
                    return Err("DS data/vdst operands must share register file (v or a)".into());
                }
            }
            acc_seen = Some(is_acc);
            word.put(&f.field, u64::from(reg))?;
        }
    }
    if acc_seen == Some(true) {
        word.put("acc", 1)?;
    }
    let gds = flag_modifier(line, "gds")?;
    if mnemonic.starts_with("ds_gws_") && gds == 0 {
        return Err(format!("{mnemonic} requires gds"));
    }
    if gds != 0 {
        word.put("gds", 1)?;
    }
    if is_ds_two_offset(mnemonic) {
        for key in ["offset0", "offset1"] {
            let v = int_modifier(line, key)?.unwrap_or(0);
            if !(0..256).contains(&v) {
                return Err(format!("{key} {v}: 0..255"));
            }
            word.put(key, v as u64)?;
        }
        return Ok(());
    }
    let offset = int_modifier(line, "offset")?.unwrap_or(0);
    if !(0..65536).contains(&offset) {
        return Err(format!("offset {offset}: 0..65535"));
    }
    word.put("offset0", (offset & 0xFF) as u64)?;
    word.put("offset1", (offset >> 8) as u64)
}

fn select<'t>(t: &'t Gfx942, line: &Line) -> Result<(String, &'t Op), String> {
    let m = line.mnemonic.as_str();
    let ops = &t.isa.ops;
    let pick = |name: &str| ops.get(name).map(|op| (name.to_owned(), op));
    let found = if let Some(base) = m.strip_suffix("_e64") {
        pick(m).or_else(|| pick(base).filter(|(_, op)| op.class == "VOP3SD"))
    } else if m.ends_with("_e32") {
        pick(m)
    } else if let Some((short_name, short_op)) = pick(&format!("{m}_e32")) {
        if line.modifiers.is_empty() && fits_short(short_op, m, &line.operands) {
            Some((short_name, short_op))
        } else if let Some(long) = pick(&format!("{m}_e64")) {
            Some(long)
        } else if let Some(bare) = pick(m).filter(|(_, op)| op.class.starts_with("VOP3")) {
            Some(bare)
        } else {
            Some((short_name, short_op))
        }
    } else {
        pick(m)
    };
    found.ok_or_else(|| format!("{m}: not a gfx942 mnemonic"))
}

/// Classes on gfx942 that allow a 32-bit literal constant (`VOP3*` does NOT on CDNA3).
pub(super) const LITERAL_CLASSES: [&str; 7] =
    ["SOP1", "SOP2", "SOPC", "VOP1", "VOP1_SDST", "VOP2", "VOPC"];

/// Encodes one line.
pub fn encode(t: &Gfx942, text: &str) -> Result<Encoded, String> {
    let line = parse_line(t, text)?;
    let signed = line.operands.iter().find(|o| o.neg || o.abs);
    let no_signs = |o: &Operand| Err(format!("`{}`: no neg or abs here", o.text));
    let bytes = if line.mnemonic.ends_with("_dpp") {
        dpp(t, &line.mnemonic, &line)?
    } else {
        let (name, op) = select(t, &line)?;
        if op.class.ends_with("_LIT") && name != "s_setreg_imm32_b32" {
            return Err(format!(
                "{name}: its class {} takes a literal of its own",
                op.class
            ));
        }
        if let Some(o) = signed
            .filter(|_| !matches!(op.class.as_str(), "VOP3" | "VOP3_SDST" | "VOP3SD" | "VOP3P"))
        {
            return no_signs(o);
        }
        let encoding = (t.isa.encodings.get(&op.class))
            .ok_or_else(|| format!("{name}: no class {}", op.class))?;
        let mut word = Word::new(&op.class, encoding, op.opcode)?;
        let m = line.mnemonic.as_str();
        match op.class.as_str() {
            "SOP1" | "SOP2" | "SOPC" | "SOPK" => salu(&mut word, op, &line)?,
            "SOPK_LIT" => setreg_imm32(&mut word, &line)?,
            "SOPP" => {
                if let Some(branch) = sopp(&mut word, op, &line)? {
                    return unread(&line).map(|()| branch);
                }
            }
            "SMEM" => smem(&mut word, op, &line)?,
            "VOP1" | "VOP1_SDST" | "VOP2" | "VOPC" => short(&mut word, op, m, &line)?,
            "VOP3" | "VOP3_SDST" | "VOP3SD" => {
                vop3(&mut word, op, &line)?;
                vop3_opsel(&mut word, op, &name, &line)?;
                vop3_signs_and_clamp(&mut word, op, &name, &line)?;
            }
            "VOP3P" => {
                vop3(&mut word, op, &line)?;
                packed(&mut word, op, m, &line)?;
            }
            "VOP3P_MFMA" => mfma(&mut word, op, m, &line)?,
            "FLAT" | "GLOBAL" | "SCRATCH" => flat_family(&mut word, op, m, &line)?,
            "MUBUF" => mubuf(&mut word, op, m, &line)?,
            "DS" => ds(&mut word, op, m, &line)?,
            other => return Err(format!("{name}: class {other} is not encoded")),
        }
        if word.literal.is_some() && !LITERAL_CLASSES.contains(&word.class) {
            return Err(format!("{name}: no literal in class {}", word.class));
        }
        word.bytes()
    };
    unread(&line)?;
    Ok(Encoded::Bytes(bytes))
}

fn unread(line: &Line) -> Refusal {
    match line.modifiers.unread() {
        Some(key) => Err(format!("modifier {key} is not taken")),
        None => Ok(()),
    }
}
