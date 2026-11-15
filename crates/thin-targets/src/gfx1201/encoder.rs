//! gfx1201's line encoder: the .s lines the compiler
//! emits, as llvm-mc encodes them; any other line is refused.

use thin_target::{parse_line, Encoded, Line, ModValue, Operand, OperandKind, Value};

use super::{Gfx1201, DEPCTR, DEPCTR_NONE, TH_BYPASS};
use crate::tables::{Encoding, Op, OpOperand};

type Refusal = Result<(), String>;

/// Mnemonics LLVM spells otherwise on gfx1201.
pub(super) const SPELLED_OTHERWISE: [&str; 7] = [
    "v_mov_b16_e32",
    "v_mov_b16_e64",
    "v_swap_b16_e32",
    "s_waitcnt",
    "s_barrier_init",
    "s_barrier_join",
    "s_get_barrier_state",
];

/// The f32 values with inline codes 240..247, in code order.
pub(super) const INLINE_FLOATS: [f32; 8] = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];

/// The bits of 1/(2π) as an f32, the inline constant 248.
const INV_2PI: u32 = 0x3E22_F983;

/// The scalar code of `vcc_lo` and the first named register.
pub(super) const VCC_LO: u32 = 106;

/// An instruction word being built in one encoding class.
struct Word<'a> {
    class: &'a str,
    encoding: &'a Encoding,
    bits: u128,
    literal: Option<u32>,
}

impl<'a> Word<'a> {
    fn new(class: &'a str, encoding: &'a Encoding, opcode: u64) -> Result<Word<'a>, String> {
        let mut word = Word {
            class,
            encoding,
            bits: 0,
            literal: None,
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

    /// Writes a field, refusing a value it cannot hold or a field the class lacks.
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

    /// Takes a literal, refusing a second different one.
    fn literal(&mut self, value: u32) -> Refusal {
        match self.literal {
            Some(v) if v != value => Err("a second, different literal".into()),
            _ => {
                self.literal = Some(value);
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

/// The dwords of an operand field: 1 for a lane mask, else its bits / 32.
pub(super) fn dwords(o: &OpOperand) -> u32 {
    if o.format == "FMT_NUM_M64" {
        1
    } else {
        (o.bits / 32).max(1)
    }
}

/// A register operand of a file and width.
fn register(o: &Operand, file: OperandKind, width: u32) -> Result<u32, String> {
    // NULL is a scalar operand of any width.
    if file == OperandKind::S && o.kind == file && o.n == NULL_CODE && o.count == 1 {
        return Ok(NULL_CODE);
    }
    if o.kind != file || o.count != width {
        return Err(format!(
            "`{}`: a {}-register {} operand",
            o.text,
            width,
            file.name()
        ));
    }
    aligned(o)?;
    Ok(o.n)
}

/// Refuses a misaligned scalar range: a pair starts at an even SGPR and a
/// wider range at a multiple of 4, as sources and
/// destinations alike.
fn aligned(o: &Operand) -> Refusal {
    let align = scalar_align(o.count);
    if o.kind == OperandKind::S && o.count > 1 && o.n < VCC_LO && o.n % align != 0 {
        return Err(format!(
            "`{}`: a scalar range starts at a multiple of {align}",
            o.text
        ));
    }
    Ok(())
}

/// Where a scalar range of `width` dwords may start: a pair at an even
/// SGPR, three dwords or more at a multiple of 4.
pub(crate) fn scalar_align(width: u32) -> u32 {
    if width > 2 {
        4
    } else {
        width.max(1)
    }
}

/// The source code of an operand for an operand field.
fn source(word: &mut Word<'_>, field: &OpOperand, o: &Operand) -> Result<u32, String> {
    let (vgprs, sgprs, constants, barrier) = match field.xml_type.as_str() {
        "OPR_SRC" => (true, true, true, false),
        "OPR_SSRC" => (false, true, true, false),
        "OPR_SREG" => (false, true, false, false),
        "OPR_VGPR" | "OPR_SRC_VGPR" => (true, false, false, false),
        "OPR_SRC_VGPR_OR_INLINE" => (true, false, true, false),
        "OPR_SSRC_BARRIER_ID" => (false, true, true, true),
        "OPR_SSRC_LANESEL" => (false, true, true, false),
        other => return Err(format!("`{}`: operand type {other} is not taken", o.text)),
    };
    let width = if barrier { 1 } else { dwords(field) };
    let text = &o.text;
    let code = match o.kind {
        OperandKind::V if vgprs && o.count == width => 256 + o.n,
        OperandKind::V if !vgprs => return Err(format!("`{text}`: no VGPR here")),
        OperandKind::S if sgprs && o.count == width => {
            aligned(o)?;
            o.n
        }
        OperandKind::S if !sgprs => return Err(format!("`{text}`: no scalar register here")),
        // NULL reads zero at any width.
        OperandKind::S if o.n == NULL_CODE && o.count == 1 => NULL_CODE,
        OperandKind::V | OperandKind::S => {
            return Err(format!("`{text}`: a {width}-dword operand"))
        }
        _ if !constants => return Err(format!("`{text}`: no constant here")),
        _ => constant(word, field, o, width)?,
    };
    if field.xml_type == "OPR_SSRC_LANESEL" && code == 255 {
        return Err(format!("`{text}`: a lane select takes no literal"));
    }
    if barrier && (code == 255 || (o.kind == OperandKind::S && o.name.as_deref() != Some("m0"))) {
        return Err(format!(
            "`{text}`: a barrier id is m0 or an inline constant"
        ));
    }
    Ok(code)
}

/// The code of a constant operand: inline, or a literal it takes.
fn constant(
    word: &mut Word<'_>,
    field: &OpOperand,
    o: &Operand,
    width: u32,
) -> Result<u32, String> {
    let text = &o.text;
    let wide = matches!(
        field.format.as_str(),
        "FMT_NUM_F32" | "FMT_NUM_B32" | "FMT_NUM_U32" | "FMT_NUM_I32"
    );
    match (o.kind, &o.value) {
        (OperandKind::Int, Some(Value::Int(v))) => {
            if (0..=64).contains(v) {
                return Ok(128 + *v as u32);
            }
            if (-16..0).contains(v) {
                return Ok((192 - v) as u32);
            }
            if !wide || width != 1 {
                return Err(format!("`{text}`: only an integer -16..64 here"));
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
        (OperandKind::Float, Some(Value::Float(f))) if wide => {
            // +0.0 is the integer 0's code; -0.0 is not inline (its bits
            // are 0x80000000), so it falls through to a literal.
            if *f == 0.0 && f.is_sign_positive() {
                return Ok(128);
            }
            // 1/(2π) has no short decimal: its f32 rounding names it.
            if (*f as f32).to_bits() == INV_2PI {
                return Ok(248);
            }
            if let Some(i) = INLINE_FLOATS.iter().position(|x| f64::from(*x) == *f) {
                return Ok(240 + i as u32);
            }
            let single = *f as f32;
            if width == 1 && f64::from(single) == *f {
                word.literal(single.to_bits())?;
                return Ok(255);
            }
            Err(format!("`{text}`: not exactly an f32"))
        }
        _ => Err(format!("`{text}`: operand not taken")),
    }
}

/// The operand fields of an op that are among `names`, in that order.
fn fields<'a>(op: &'a Op, names: &[&str]) -> Vec<&'a OpOperand> {
    names.iter().filter_map(|n| op.operand(n)).collect()
}

fn count(line: &Line, n: usize) -> Refusal {
    if line.operands.len() != n {
        return Err(format!("{} operands, not {n}", line.operands.len()));
    }
    Ok(())
}

/// Scalar ALU ops.
fn salu(word: &mut Word<'_>, op: &Op, line: &Line) -> Refusal {
    // s_setreg_b32 writes the hardware register its SIMM16 names from the
    // SGPR in its SDST field, and is written in that order.
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
            // A hardware-register selector is hwreg(...) or its code.
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
            // A returning message's id is the source field's raw byte.
            "ssrc0" if f.xml_type == "OPR_SENDMSG_RTN" => match o.int() {
                Some(v) if o.kind == OperandKind::Int && (0..256).contains(&v) => v as u32,
                _ => return Err(format!("`{}`: a message id, 0..255", o.text)),
            },
            _ => source(word, f, o)?,
        };
        word.put(&f.field, u64::from(value))?;
    }
    Ok(())
}

/// `s_setreg_imm32_b32 hwreg(NAME, offset, size), imm32`: writes `size`
/// bits of the 32-bit literal into a hardware register at `offset` (RDNA4
/// ISA guide, S_SETREG_IMM32_B32: "this instruction requires a
/// 32-bit literal constant"). SIMM16 holds hwreg(...)'s code; the literal
/// is the SOPK_LIT class's own dword, always present.
fn setreg_imm32(word: &mut Word<'_>, line: &Line) -> Refusal {
    count(line, 2)?;
    let (reg, imm) = (&line.operands[0], &line.operands[1]);
    let code = reg.int().filter(|_| reg.text.starts_with("hwreg("));
    let Some(code) = code else {
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

/// Program control.
fn sopp(word: &mut Word<'_>, op: &Op, line: &Line) -> Result<Option<Encoded>, String> {
    let simm = op.operand("simm16");
    let kind = simm.map(|s| s.xml_type.as_str());
    match (kind, line.operands.as_slice()) {
        (Some("OPR_WAIT_ALU"), []) => {
            let value = match line.modifiers.take("simm16") {
                Some(ModValue::Int(v)) if (0..65536).contains(v) => *v as u32,
                Some(_) => return Err("s_wait_alu takes 0..65535".into()),
                None => depctr(line)?,
            };
            word.put("simm16", u64::from(value))?;
        }
        (None, []) => {}
        (Some(kind), [o]) => match (kind, o.kind, &o.value) {
            ("OPR_LABEL", OperandKind::Label, Some(Value::Label(label))) => {
                return Ok(Some(Encoded::Branch {
                    raw: word.bits as u32,
                    label: label.clone(),
                }))
            }
            ("OPR_SIMM16" | "OPR_DELAY" | "OPR_SENDMSG", OperandKind::Int, Some(Value::Int(v)))
                if (0..65536).contains(v) =>
            {
                word.put("simm16", *v as u64)?;
            }
            // The 16-bit immediates of s_clause, s_sleep, s_wait_event and
            // the combined memory waits.
            (
                "OPR_CLAUSE" | "OPR_SLEEP" | "OPR_WAIT_EVENT" | "OPR_WAIT_MEM_DS",
                OperandKind::Int,
                Some(Value::Int(v)),
            ) if (0..65536).contains(v) => {
                word.put("simm16", *v as u64)?;
            }
            _ => return Err(format!("`{}`: not taken here", o.text)),
        },
        _ => return Err(format!("{} operands", line.operands.len())),
    }
    Ok(None)
}

/// The packed `s_wait_alu` value of its named counts.
fn depctr(line: &Line) -> Result<u32, String> {
    let mut value = DEPCTR_NONE;
    let mut any = false;
    for &(name, shift, bits) in &DEPCTR {
        if let Some(given) = line.modifiers.take(name) {
            let ModValue::Int(v) = given else {
                return Err(format!("{name}: a count"));
            };
            if !(0..1 << bits).contains(v) {
                return Err(format!("{name}({v}): does not fit {bits} bits"));
            }
            value = (value & !(((1 << bits) - 1) << shift)) | (*v as u32) << shift;
            any = true;
        }
    }
    if !any {
        return Err("s_wait_alu: no count".into());
    }
    Ok(value)
}

/// Whether an operand is exactly `vcc_lo`.
fn is_vcc(o: &Operand) -> bool {
    o.kind == OperandKind::S && o.n == VCC_LO && o.count == 1
}

fn is_vgpr(o: &Operand, width: u32) -> bool {
    o.kind == OperandKind::V && o.count == width
}

/// Whether a line's operands fit an op's short form, the halves rule
/// included.
fn fits_short(op: &Op, mnemonic: &str, ops: &[Operand]) -> bool {
    let halves_ok =
        short_operands(&op.class, mnemonic, ops).is_ok_and(|by| halves(op, &by).is_ok());
    let unsigned = ops.iter().all(|o| !o.neg && !o.abs);
    halves_ok && unsigned && fits_shape(op, mnemonic, ops)
}

/// Whether a line's operands have the shape of an op's short form.
fn fits_shape(op: &Op, mnemonic: &str, ops: &[Operand]) -> bool {
    let v1 = op.operand("vsrc1").map_or(1, dwords);
    let vd = op.operand("vdst").map_or(1, dwords);
    match op.class.as_str() {
        "VOPC" => match ops {
            [d, _, b] => is_vcc(d) && is_vgpr(b, v1),
            [_, b] if mnemonic.starts_with("v_cmpx") => is_vgpr(b, v1),
            _ => false,
        },
        "VOP2" if mnemonic.starts_with("v_cndmask") => {
            matches!(ops, [d, _, b, c] if is_vgpr(d, vd) && is_vgpr(b, v1) && is_vcc(c))
        }
        "VOP2" if mnemonic.contains("_co_ci_") => {
            matches!(ops, [d, c, _, b, ci] if is_vgpr(d, vd) && is_vcc(c) && is_vgpr(b, v1) && is_vcc(ci))
        }
        "VOP2" => matches!(ops, [d, _, b] if is_vgpr(d, vd) && is_vgpr(b, v1)),
        _ => true,
    }
}

/// The 16-bit halves rule: a 16-bit VGPR operand of a short form lies
/// below v128.
fn halves(op: &Op, by_field: &[(&str, &Operand)]) -> Refusal {
    for (field, o) in by_field {
        let sixteen = op.operand(field).is_some_and(|f| f.bits == 16);
        if sixteen && o.kind == OperandKind::V && o.n + o.count > 128 {
            return Err(format!("`{}`: VGPRs from v128 need the e64 form", o.text));
        }
    }
    Ok(())
}

/// The operands of a short form by field: (vdst, src0, vsrc1).
fn short_operands<'o>(
    class: &str,
    mnemonic: &str,
    ops: &'o [Operand],
) -> Result<Vec<(&'static str, &'o Operand)>, String> {
    Ok(match (class, ops) {
        ("VOPC", [_, a, b]) | ("VOPC", [a, b]) => vec![("src0", a), ("vsrc1", b)],
        ("VOP2", [d, _, a, b, _]) if mnemonic.contains("_co_ci_") => {
            vec![("vdst", d), ("src0", a), ("vsrc1", b)]
        }
        ("VOP2", [d, a, b, _]) if mnemonic.starts_with("v_cndmask") => {
            vec![("vdst", d), ("src0", a), ("vsrc1", b)]
        }
        ("VOP2", [d, a, b]) => vec![("vdst", d), ("src0", a), ("vsrc1", b)],
        ("VOP1" | "VOP1_SDST", [d, a]) => vec![("vdst", d), ("src0", a)],
        ("VOP1" | "VOP1_SDST", []) => vec![],
        _ => return Err(format!("{} operands", ops.len())),
    })
}

/// Short VALU forms.
fn short(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let class = op.class.as_str();
    if op.operands.is_empty() {
        return count(line, 0);
    }
    let vop1 = matches!(class, "VOP1" | "VOP1_SDST");
    if !vop1 && !fits_shape(op, mnemonic, &line.operands) {
        return Err("the operands do not fit the short form".into());
    }
    let by_field = short_operands(class, mnemonic, &line.operands)?;
    if by_field.len() != op.operands.len() {
        return Err(format!("{} operands", line.operands.len()));
    }
    halves(op, &by_field)?;
    for (name, o) in by_field {
        let f = op
            .operand(name)
            .ok_or_else(|| format!("no operand {name}"))?;
        let value = match name {
            "vdst" if class == "VOP1_SDST" => register(o, OperandKind::S, dwords(f))?,
            "vdst" | "vsrc1" => register(o, OperandKind::V, dwords(f))?,
            _ => source(word, f, o)?,
        };
        word.put(name, u64::from(value))?;
    }
    Ok(())
}

/// The cache fields of a memory op.
/// `has_dst`: whether the line has a returning form's destination; none
/// for buffer atomics, whose return is the `th` bit alone (RDNA4 ISA
/// guide, Table 47: "For atomics, indicates whether or not to return
/// the pre-op value").
fn cache(word: &mut Word<'_>, mnemonic: &str, line: &Line, has_dst: Option<bool>) -> Refusal {
    let kind = if mnemonic.contains("atomic") {
        0x300
    } else if mnemonic.contains("store") {
        0x200
    } else {
        0x100
    };
    let th = match line.modifiers.take("th") {
        None => None,
        Some(ModValue::Int(v)) if v & !(TH_BYPASS | 7) == kind => Some(*v),
        Some(_) => return Err("th: not a hint of this op's kind".into()),
    };
    let scope = match line.modifiers.take("scope") {
        None => 0,
        Some(ModValue::Int(v)) if (0..4).contains(v) => *v as u64,
        Some(_) => return Err("scope: 0..3".into()),
    };
    // Code 3 of a load or store is BYPASS at SCOPE_SYS and LU or WB below
    // it: each name only at its scopes.
    let bypass = th.is_some_and(|v| v & TH_BYPASS != 0);
    if kind != 0x300 && th.is_some_and(|v| v & 7 == 3) && bypass != (scope == SCOPE_SYS) {
        return Err("th: BYPASS exactly at SCOPE_SYS".into());
    }
    let th = th.map_or(0, |v| (v & 7) as u64);
    if kind == 0x100 && th == 7 {
        return Err("th: load hint 7 is reserved".into());
    }
    if kind == 0x300 && has_dst.is_some_and(|d| (th & 1 == 1) != d) {
        return Err("th: TH_ATOMIC_RETURN exactly on the returning form".into());
    }
    word.put("th", th)?;
    if scope != 0 {
        word.put("scope", scope)?;
    }
    Ok(())
}

/// SCOPE_SYS, the widest scope: a BYPASS hint's.
const SCOPE_SYS: u64 = 3;

/// A 24-bit signed offset.
fn offset24(v: i128) -> Result<u64, String> {
    if !(-(1 << 23)..1 << 23).contains(&v) {
        return Err(format!("offset {v}: does not fit 24 bits signed"));
    }
    Ok((v & 0xFF_FFFF) as u64)
}

fn int_modifier(line: &Line, key: &str) -> Result<Option<i128>, String> {
    match line.modifiers.take(key) {
        None => Ok(None),
        Some(ModValue::Int(v)) => Ok(Some(*v)),
        Some(ModValue::List(_)) => Err(format!("{key}: an integer")),
    }
}

/// Scalar memory.
fn smem(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let (Some(data), Some(base)) = (op.operand("sdata"), op.operand("sbase")) else {
        return Err("not a scalar load".into());
    };
    count(line, 3)?;
    let ops = &line.operands;
    word.put(
        "sdata",
        u64::from(register(&ops[0], OperandKind::S, dwords(data))?),
    )?;
    let b = register(&ops[1], OperandKind::S, dwords(base))?;
    if b % 2 != 0 {
        return Err(format!("`{}`: the base is an even pair", ops[1].text));
    }
    word.put("sbase", u64::from(b / 2))?;
    let modifier = int_modifier(line, "offset")?;
    match (&ops[2], modifier) {
        (o, m) if o.kind == OperandKind::S && o.count == 1 => {
            word.put("soffset", u64::from(o.n))?;
            word.put("ioffset", offset24(m.unwrap_or(0))?)?;
        }
        (o, None) if o.kind == OperandKind::Int => {
            word.put("ioffset", offset24(o.int().unwrap_or(0))?)?;
        }
        (o, _) => return Err(format!("`{}`: not an offset", o.text)),
    }
    cache(word, mnemonic, line, Some(false))
}

/// VOP3, VOP3_SDST and VOP3SD, and VOP3P's operands.
fn vop3(word: &mut Word<'_>, op: &Op, line: &Line) -> Refusal {
    let mut present = fields(op, &["vdst", "sdst", "src0", "src1", "src2"]);
    // v_cmpx writes EXEC: thin's line names its field's register first
    // (`exec_lo`); a line may leave it out as LLVM's does, and the field
    // then holds EXEC_LO's code.
    if let Some(at) = present.iter().position(|f| f.xml_type == "OPR_EXEC") {
        // Named, the field holds the register given (the decoder prints the
        // field as it finds it).
        if line.operands.len() + 1 == present.len() {
            word.put(&present[at].field, u64::from(EXEC_LO_CODE))?;
            present.remove(at);
        }
    }
    vop3_operands(word, &present, &line.operands)
}

fn vop3_operands(word: &mut Word<'_>, present: &[&OpOperand], operands: &[Operand]) -> Refusal {
    if operands.len() != present.len() {
        return Err(format!(
            "{} operands, not {}",
            operands.len(),
            present.len()
        ));
    }
    for (f, o) in present.iter().zip(operands) {
        let value = match f.field.as_str() {
            "vdst" if f.xml_type == "OPR_VGPR" => register(o, OperandKind::V, dwords(f))?,
            "vdst" | "sdst" => {
                if o.kind != OperandKind::S || o.count != dwords(f) {
                    return Err(format!(
                        "`{}`: a {}-dword scalar operand",
                        o.text,
                        dwords(f)
                    ));
                }
                o.n
            }
            _ => source(word, f, o)?,
        };
        word.put(&f.field, u64::from(value))?;
    }
    Ok(())
}

/// The VOP3 ops whose OPSEL selects a half (RDNA4 ISA guide, "OPSEL
/// with FP8 and BF8 Conversions"), each with the OPSEL bits it reads: the
/// packs write DST[31:16] with OPSEL[3]; the unpacks read SRC0[31:16] with
/// OPSEL[0].
pub(super) const OPSEL_OPS: [(&str, u32); 4] = [
    ("v_cvt_pk_fp8_f32", 0b1000),
    ("v_cvt_pk_bf8_f32", 0b1000),
    ("v_cvt_pk_f32_fp8_e64", 0b0001),
    ("v_cvt_pk_f32_bf8_e64", 0b0001),
];

/// The OPSEL bits an op takes, none for most.
pub(super) fn opsel_bits(name: &str) -> u32 {
    OPSEL_OPS
        .iter()
        .find(|(m, _)| *m == name)
        .map_or(0, |o| o.1)
}

/// A VOP3 `op_sel:[s0,..,dst]`: one bit per source, then the
/// destination's, OPSEL[3] (VOP3 OPSEL: "[0..2] = src,
/// [3] = dst"). Only the ops of [`OPSEL_OPS`] take it, and only their bits.
fn vop3_opsel(word: &mut Word<'_>, op: &Op, name: &str, line: &Line) -> Refusal {
    let allowed = opsel_bits(name);
    if allowed == 0 {
        return Ok(());
    }
    let Some(given) = line.modifiers.take("op_sel") else {
        return Ok(());
    };
    let nsrc = fields(op, &["src0", "src1", "src2"]).len();
    let ModValue::List(bits) = given else {
        return Err(format!("op_sel: a list of {} bits", nsrc + 1));
    };
    if bits.len() != nsrc + 1 || bits.iter().any(|b| !(0..=1).contains(b)) {
        return Err(format!("op_sel: a list of {} bits", nsrc + 1));
    }
    let value = bits.iter().enumerate().fold(0u32, |v, (j, b)| {
        let bit = if j == nsrc { 3 } else { j };
        v | (*b as u32) << bit
    });
    if value & !allowed != 0 {
        return Err(format!(
            "op_sel: {name} reads OPSEL bits 0x{allowed:x} only"
        ));
    }
    word.put("opsel", u64::from(value))
}

/// The ops whose untyped sources take sign modifiers: the moves and the
/// select (RDNA4 ISA guide: "input modifiers are supported for:
/// V_MOV_B32, V_MOV_B16, V_MOVREL*_B32 and V_CNDMASK"), their data sources.
const SIGNED_MOVES: [&str; 7] = [
    "v_mov_b32_e64",
    "v_movreld_b32_e64",
    "v_movrels_b32_e64",
    "v_movrelsd_b32_e64",
    "v_movrelsd_2_b32_e64",
    "v_cndmask_b32_e64",
    "v_cndmask_b16",
];

/// The F32 sources the guide denies sign modifiers: the fp8 and bf8
/// conversions (RDNA4 ISA guide: "These instructions do not support ABS,
/// NEG, OMOD or CLAMP").
const UNSIGNED_CONVERTS: [&str; 4] = [
    "v_cvt_pk_fp8_f32",
    "v_cvt_pk_bf8_f32",
    "v_cvt_sr_fp8_f32",
    "v_cvt_sr_bf8_f32",
];

/// The sign modifiers a VOP3 source takes: (neg, abs). The RDNA4 ISA guide
/// ABS and NEG "apply to floating point inputs and are undefined
/// for any other type of input", plus the moves and V_CNDMASK; not
/// integer or bitwise ops, lane reads and writes, PERMLANE, QSAD, SWAP or
/// the fp8 / bf8 conversions; DOT2 takes NEG but not ABS. A float input is
/// one the XML types F16, BF16, F32 or F64. VOP3SD has no ABS field.
pub(super) fn signs_of(name: &str, class: &str, field: &OpOperand) -> (bool, bool) {
    let float = matches!(
        field.format.as_str(),
        "FMT_NUM_F16" | "FMT_NUM_BF16" | "FMT_NUM_F32" | "FMT_NUM_F64"
    );
    let data = field.field == "src0" || field.field == "src1";
    let neg =
        (float && !UNSIGNED_CONVERTS.contains(&name)) || (SIGNED_MOVES.contains(&name) && data);
    let abs = neg && !name.contains("dot2") && class != "VOP3SD";
    (neg, abs)
}

/// Sign modifiers on VOP3 sources: NEG[i] and ABS[i] of source i (VOP3 NEG and ABS fields).
fn vop3_signs(word: &mut Word<'_>, op: &Op, name: &str, line: &Line) -> Refusal {
    let present = fields(op, &["vdst", "sdst", "src0", "src1", "src2"]);
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
    Ok(())
}

/// What a packed op takes: masks of op_sel, op_sel_hi, neg_lo, neg_hi;
/// clamp; the op_sel_hi default; literals.
pub(super) fn packed_rules(m: &str) -> ([u64; 4], bool, u64, bool) {
    let integer = ["_iu4", "_iu8", "_u4", "_u8"].iter().any(|s| m.contains(s));
    let matrix = m.starts_with("v_wmma") || m.starts_with("v_swmmac");
    let swmmac = m.starts_with("v_swmmac");
    if m.starts_with("v_pk_") || m.starts_with("v_dot2_f32_") {
        ([7, 7, 7, 7], true, 7, true)
    } else if m.starts_with("v_fma_mix") {
        ([7, 7, 0, 0], true, 0, false)
    } else if matrix && integer {
        ([0, 0, 3, 0], true, 7, true)
    } else if matrix && (m.contains("fp8") || m.contains("bf8")) {
        let neg = if swmmac { 0 } else { 4 };
        ([0, 0, neg, neg], false, 7, true)
    } else if matrix {
        let neg = if swmmac { 3 } else { 7 };
        ([0, 0, neg, neg], false, 7, true)
    } else if m.starts_with("v_dot4_f32_") {
        ([0, 0, 4, 4], false, 7, true)
    } else if integer {
        ([0, 0, 7, 7], true, 7, true)
    } else {
        ([0, 0, 0, 0], false, 7, true)
    }
}

/// Packed ops' modifiers.
fn packed(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let nsrc = fields(op, &["src0", "src1", "src2"]).len();
    let (masks, clamp, hi_default, literals) = packed_rules(mnemonic);
    if !literals {
        let wide_int = line
            .operands
            .iter()
            .any(|o| o.int().is_some_and(|v| !(-16..=64).contains(&v)));
        if word.literal.is_some() || wide_int {
            return Err("no literal here".into());
        }
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
        // A list of fewer bits than sources leaves the rest 0; bits past the
        // sources (up to four) must be 0.
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
    word.put("opsel_hi", hi & 3)?;
    if nsrc == 3 {
        word.put("opsel_hi2", (hi >> 2) & 1)?;
    }
    for (k, field) in [(0, "opsel"), (2, "neg"), (3, "neg_hi")] {
        if let Some(v) = values[k] {
            word.put(field, v)?;
        }
    }
    if line.modifiers.take("clamp").is_some() {
        if !clamp {
            return Err("clamp: not taken".into());
        }
        word.put("clmp", 1)?;
    }
    Ok(())
}

/// A DPP control's code.
fn dpp_control(line: &Line) -> Result<u64, String> {
    let mut controls = Vec::new();
    if let Some(given) = line.modifiers.take("quad_perm") {
        match given {
            ModValue::List(l) if l.len() == 4 && l.iter().all(|x| (0..4).contains(x)) => {
                controls.push(
                    l.iter()
                        .enumerate()
                        .fold(0u64, |v, (i, x)| v | (*x as u64) << (2 * i)),
                );
            }
            _ => return Err("quad_perm: four selectors 0..3".into()),
        }
    }
    for (key, base, low) in [
        ("row_shl", 0x100, 1),
        ("row_shr", 0x110, 1),
        ("row_ror", 0x120, 1),
        ("row_share", 0x150, 0),
        ("row_xmask", 0x160, 0),
    ] {
        if let Some(given) = line.modifiers.take(key) {
            match given {
                ModValue::Int(k) if (low..16).contains(k) => controls.push(base + *k as u64),
                _ => return Err(format!("{key}: out of range")),
            }
        }
    }
    for (key, code) in [("row_mirror", 0x140), ("row_half_mirror", 0x141)] {
        if line.modifiers.take(key).is_some() {
            controls.push(code);
        }
    }
    match controls.as_slice() {
        [one] => Ok(*one),
        _ => Err("exactly one DPP control".into()),
    }
}

fn truthy(line: &Line, key: &str) -> Result<u64, String> {
    Ok(u64::from(int_modifier(line, key)?.is_some_and(|v| v != 0)))
}

/// DPP16.
fn dpp(t: &Gfx1201, mnemonic: &str, line: &Line) -> Result<Vec<u8>, String> {
    let base = mnemonic.strip_suffix("_dpp").expect("a _dpp mnemonic");
    if !t.rows.contains(mnemonic) {
        return Err(format!("{mnemonic}: not a DPP form"));
    }
    let short_name = format!("{base}_e32");
    let op = (t.isa.ops.get(&short_name))
        .filter(|op| matches!(op.class.as_str(), "VOP1" | "VOP2"))
        .ok_or_else(|| format!("{mnemonic}: DPP is VOP1 and VOP2 only"))?;
    if SPELLED_OTHERWISE.contains(&short_name.as_str()) {
        return Err(format!("{short_name}: LLVM spells it otherwise"));
    }
    if op.class == "VOP2" && !fits_shape(op, base, &line.operands) {
        return Err("the operands do not fit the short form".into());
    }
    let by_field = short_operands(&op.class, base, &line.operands)?;
    halves(op, &by_field)?;
    let class = format!("{}_DPP16", op.class);
    let encoding = t
        .isa
        .encodings
        .get(&class)
        .ok_or_else(|| format!("no class {class}"))?;
    let mut word = Word::new(&class, encoding, op.opcode)?;
    word.put("src0", 0xFA)?;
    for (name, o) in by_field {
        let field = if name == "src0" { "vsrc0" } else { name };
        word.put(field, u64::from(register(o, OperandKind::V, 1)?))?;
    }
    word.put("dpp", dpp_control(line)?)?;
    let mask =
        |key| -> Result<u64, String> { Ok(int_modifier(line, key)?.map_or(0xF, |v| v as u64)) };
    word.put("row_mask", mask("row_mask")?)?;
    word.put("bank_mask", mask("bank_mask")?)?;
    word.put("bc", truthy(line, "bound_ctrl")?)?;
    word.put("fi", truthy(line, "fi")?)?;
    Ok(word.bytes())
}

/// Global memory.
fn global(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let (dst, data) = (op.operand("vdst"), op.operand("vsrc"));
    let ops = &line.operands;
    let (d, addr, v, saddr) = match (dst, data, ops.as_slice()) {
        (Some(_), Some(_), [d, a, v, s]) => (Some(d), a, Some(v), s),
        // A non-returning atomic: the returning form's operands less vdst.
        (Some(_), Some(_), [a, v, s]) if mnemonic.contains("atomic") => (None, a, Some(v), s),
        (Some(_), None, [d, a, s]) => (Some(d), a, None, s),
        (None, Some(_), [a, v, s]) => (None, a, Some(v), s),
        _ => return Err(format!("{} operands", ops.len())),
    };
    if let (Some(d), Some(f)) = (d, dst) {
        word.put("vdst", u64::from(register(d, OperandKind::V, dwords(f))?))?;
    }
    if let (Some(v), Some(f)) = (v, data) {
        word.put("vsrc", u64::from(register(v, OperandKind::V, dwords(f))?))?;
    }
    if saddr.kind == OperandKind::Off {
        word.put("vaddr", u64::from(register(addr, OperandKind::V, 2)?))?;
    } else {
        word.put("saddr", u64::from(register(saddr, OperandKind::S, 2)?))?;
        word.put("vaddr", u64::from(register(addr, OperandKind::V, 1)?))?;
    }
    word.put(
        "ioffset",
        offset24(int_modifier(line, "offset")?.unwrap_or(0))?,
    )?;
    cache(word, mnemonic, line, Some(d.is_some()))
}

/// The FORMAT an untyped buffer op carries: 1 (BUF_FMT_8_UNORM), as
/// llvm-mc writes it. Untyped ops take their data format from the opcode
/// or the resource, never this field.
pub(super) const UNTYPED_FORMAT: u64 = 1;

/// The largest buffer offset: IOFFSET is a 24-bit field that "must be
/// non-negative", so 23 bits.
pub(super) const BUFFER_OFFSET_MAX: i128 = (1 << 23) - 1;

/// Whether a scalar code is a buffer op's SOFFSET: an SGPR (VCC and the
/// TTMPs included, per the RDNA4 ISA guide), M0 or NULL.
pub(super) fn soffset_code(n: u32) -> bool {
    n <= NULL_CODE + 1
}

/// The scalar code of `null`; `m0` follows it.
const NULL_CODE: u32 = 124;

/// EXEC_LO's operand code (Table 47).
const EXEC_LO_CODE: u32 = 126;

/// Buffer loads, stores and atomics:
/// `op vdata, vaddr|off, s[rsrc:rsrc+3], soffset [idxen] [offen]
/// [offset:n] [th:..] [scope:..]`. The address VGPRs are one per enabled
/// IDXEN and OFFEN, index first (Table 50), `off` when neither is; the
/// resource is four SGPRs at a multiple of 4; SOFFSET is an SGPR, M0 or
/// NULL (Table 47). A returning atomic is `th:TH_ATOMIC_RETURN` on the
/// same operands, its data VGPRs receiving the pre-op value.
/// Typed buffer ops (TBUFFER, a FORMAT of their own) are refused.
fn buffer(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    if mnemonic.starts_with("tbuffer") {
        return Err(format!("{mnemonic}: typed buffer ops are not encoded"));
    }
    let Some(data) = op.operand("vdata") else {
        return Err(format!("{mnemonic}: a buffer op without data"));
    };
    count(line, 4)?;
    let ops = &line.operands;
    let flag = |key| -> Result<u64, String> {
        Ok(int_modifier(line, key)?.map_or(0, |v| u64::from(v != 0)))
    };
    let (idxen, offen) = (flag("idxen")?, flag("offen")?);
    word.put(
        "vdata",
        u64::from(register(&ops[0], OperandKind::V, dwords(data))?),
    )?;
    match (idxen + offen, ops[1].kind) {
        (0, OperandKind::Off) => {}
        (0, _) => return Err(format!("`{}`: `off` without idxen or offen", ops[1].text)),
        (n, _) => word.put(
            "vaddr",
            u64::from(register(&ops[1], OperandKind::V, n as u32)?),
        )?,
    }
    word.put("idxen", idxen)?;
    word.put("offen", offen)?;
    word.put("rsrc", u64::from(register(&ops[2], OperandKind::S, 4)?))?;
    let soffset = &ops[3];
    if soffset.kind != OperandKind::S || soffset.count != 1 || !soffset_code(soffset.n) {
        return Err(format!(
            "`{}`: the offset is an SGPR, m0 or null",
            soffset.text
        ));
    }
    word.put("soffset", u64::from(soffset.n))?;
    let offset = int_modifier(line, "offset")?.unwrap_or(0);
    if !(0..=BUFFER_OFFSET_MAX).contains(&offset) {
        return Err(format!("offset {offset}: 0..{BUFFER_OFFSET_MAX}"));
    }
    word.put("ioffset", offset as u64)?;
    word.put("format", UNTYPED_FORMAT)?;
    cache(word, mnemonic, line, None)
}

/// LDS.
fn ds(word: &mut Word<'_>, op: &Op, mnemonic: &str, line: &Line) -> Refusal {
    let present = fields(op, &["vdst", "addr", "data0", "data1"]);
    count(line, present.len())?;
    for (f, o) in present.into_iter().zip(&line.operands) {
        word.put(&f.field, u64::from(register(o, OperandKind::V, dwords(f))?))?;
    }
    if mnemonic.contains("2addr") {
        for key in ["offset0", "offset1"] {
            word.put(key, int_modifier(line, key)?.unwrap_or(0) as u64)?;
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

/// The op a mnemonic selects.
fn select<'t>(t: &'t Gfx1201, line: &Line) -> Result<(String, &'t Op), String> {
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
    found.ok_or_else(|| format!("{m}: not a gfx1201 mnemonic"))
}

/// The classes a literal may follow.
pub(super) const LITERAL_CLASSES: [&str; 11] = [
    "SOP1",
    "SOP2",
    "SOPC",
    "VOP1",
    "VOP1_SDST",
    "VOP2",
    "VOPC",
    "VOP3",
    "VOP3_SDST",
    "VOP3SD",
    "VOP3P",
];

/// Encodes one line.
pub fn encode(t: &Gfx1201, text: &str) -> Result<Encoded, String> {
    let line = parse_line(t, text)?;
    let signed = line.operands.iter().find(|o| o.neg || o.abs);
    let no_signs = |o: &Operand| Err(format!("`{}`: no neg or abs here", o.text));
    let bytes = if line.mnemonic.ends_with("_dpp") {
        if let Some(o) = signed {
            return no_signs(o);
        }
        dpp(t, &line.mnemonic, &line)?
    } else {
        let (name, op) = select(t, &line)?;
        if op.class.ends_with("_LIT") && name != "s_setreg_imm32_b32" {
            return Err(format!(
                "{name}: its class {} takes a literal of its own",
                op.class
            ));
        }
        if SPELLED_OTHERWISE.contains(&name.as_str()) {
            return Err(format!("{name}: LLVM spells it otherwise on gfx1201"));
        }
        if let Some(o) =
            signed.filter(|_| !matches!(op.class.as_str(), "VOP3" | "VOP3_SDST" | "VOP3SD"))
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
            "SMEM" => smem(&mut word, op, m, &line)?,
            "VOP1" | "VOP1_SDST" | "VOP2" | "VOPC" => short(&mut word, op, m, &line)?,
            "VOP3" | "VOP3_SDST" | "VOP3SD" => {
                vop3(&mut word, op, &line)?;
                vop3_opsel(&mut word, op, &name, &line)?;
                vop3_signs(&mut word, op, &name, &line)?;
            }
            "VOP3P" => {
                vop3(&mut word, op, &line)?;
                packed(&mut word, op, m, &line)?;
            }
            "VGLOBAL" => global(&mut word, op, m, &line)?,
            "VBUFFER" => buffer(&mut word, op, m, &line)?,
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

/// Refuses a line with a modifier no consumer read.
fn unread(line: &Line) -> Refusal {
    match line.modifiers.unread() {
        Some(key) => Err(format!("modifier {key} is not taken")),
        None => Ok(()),
    }
}
