//! Decoded instructions: what a target's decoder returns, the exact inverse
//! of its line encoder.
//!
//! An [`Instruction`] is the encoder's own line in a compact, `Copy` form:
//! a mnemonic from the target's tables, at most five operands and seven
//! modifiers, nothing on the heap. [`Instruction::line`] gives the
//! [`Line`] the `.s` reader parses, and its `Display` the text that parses
//! to that line, so `parse_line(i.to_string()) == i.line()` and the
//! encoder's bytes for that text are the bytes decoded.

use std::fmt::{self, Write as _};

use thin_target::{Line, ModValue, Modifiers, Operand, OperandKind, Value};

/// The most operands a line of an encoder has (`v_mfma_scale_f32_*_f8f6f4`).
pub const MAX_OPERANDS: usize = 6;
/// The most modifiers (`s_wait_alu` with its seven counters).
pub const MAX_MODIFIERS: usize = 7;

/// A decoded operand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Opd {
    /// VGPRs `v[n: n+count-1]`.
    V {
        /// The first.
        n: u16,
        /// How many.
        count: u16,
    },
    /// Numbered SGPRs `s[n: n+count-1]`.
    S {
        /// The first.
        n: u16,
        /// How many.
        count: u16,
    },
    /// AGPRs `a[n: n+count-1]`.
    A {
        /// The first.
        n: u16,
        /// How many.
        count: u16,
    },
    /// A named register of the scalar source space (`vcc_lo`, `m0`,...):
    /// the target's entry (name, number, dwords).
    Named(&'static (&'static str, u32, u32)),
    /// An integer: an inline constant, a literal or an immediate field.
    Int(i64),
    /// An inline float constant.
    Float(f32),
    /// `off`.
    Off,
    /// `hwreg(NAME, offset, size)`, or `hwreg(id, offset, size)` for an id
    /// the target names not: a hardware register's bits, its SIMM16 `code`
    /// (id, offset and size-1 packed).
    Hwreg {
        /// The register's name, when it has one.
        name: Option<&'static str>,
        /// The lowest bit.
        offset: u8,
        /// How many bits.
        size: u8,
        /// The packed SIMM16.
        code: u16,
    },
    /// A branch target: the byte offset of the label, from the start of
    /// the decoded bytes; its label is `.L<offset>`.
    Label(u32),
}

/// A modifier's value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MVal {
    /// A bare flag (`clamp`).
    Flag,
    /// `key:int`.
    Int(i32),
    /// `key:NAME`, a name the target's syntax maps to the value.
    Symbol(&'static str, i32),
    /// `key:[a,b,...]`, at most four small items.
    List([u8; 4], u8),
}

/// A modifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mod {
    /// Its key.
    pub key: &'static str,
    /// Its value.
    pub value: MVal,
}

/// A decoded instruction.
#[derive(Clone, Copy)]
pub struct Instruction {
    /// Its byte offset in the decoded bytes.
    pub offset: u32,
    /// Its bytes, literal included.
    pub size: u8,
    /// Its mnemonic, as the encoder selects it.
    pub mnemonic: &'static str,
    /// Whether it is a wait line with named counts:
    /// its modifiers print as `name(n)`, or a lone `simm16` as the
    /// integer.
    pub counted: bool,
    nops: u8,
    ops: [Opd; MAX_OPERANDS],
    /// Each operand's sign modifiers: bit 0 neg (`-x`), bit 1 abs (`|x|`).
    signs: [u8; MAX_OPERANDS],
    nmods: u8,
    mods: [Mod; MAX_MODIFIERS],
}

impl Instruction {
    /// An instruction with no operands or modifiers yet.
    pub fn new(offset: u32, mnemonic: &'static str) -> Instruction {
        Instruction {
            offset,
            size: 0,
            mnemonic,
            counted: false,
            nops: 0,
            ops: [Opd::Off; MAX_OPERANDS],
            signs: [0; MAX_OPERANDS],
            nmods: 0,
            mods: [Mod {
                key: "",
                value: MVal::Flag,
            }; MAX_MODIFIERS],
        }
    }

    /// Makes it a fresh instruction again, keeping its storage.
    pub fn reset(&mut self, offset: u32, mnemonic: &'static str) {
        self.offset = offset;
        self.size = 0;
        self.mnemonic = mnemonic;
        self.counted = false;
        self.nops = 0;
        self.nmods = 0;
    }

    /// Appends an operand.
    pub fn push(&mut self, o: Opd) {
        self.ops[self.nops as usize] = o;
        self.signs[self.nops as usize] = 0;
        self.nops += 1;
    }

    /// Gives operand `k` its sign modifiers: `-x`, `|x|` or `-|x|`.
    pub fn sign(&mut self, k: usize, neg: bool, abs: bool) {
        self.signs[k] = u8::from(neg) | u8::from(abs) << 1;
    }

    /// Operand `k`'s sign modifiers: (neg, abs).
    pub fn signs(&self, k: usize) -> (bool, bool) {
        (self.signs[k] & 1 != 0, self.signs[k] & 2 != 0)
    }

    /// Appends a modifier.
    pub fn modify(&mut self, key: &'static str, value: MVal) {
        self.mods[self.nmods as usize] = Mod { key, value };
        self.nmods += 1;
    }

    /// The operands.
    pub fn operands(&self) -> &[Opd] {
        &self.ops[..self.nops as usize]
    }

    /// The modifiers, in print order.
    pub fn modifiers(&self) -> &[Mod] {
        &self.mods[..self.nmods as usize]
    }

    /// The byte offset a branch goes to.
    pub fn branch_target(&self) -> Option<u32> {
        self.operands().iter().find_map(|o| match o {
            Opd::Label(t) => Some(*t),
            _ => None,
        })
    }

    /// The line the `.s` reader parses from this instruction's text.
    pub fn line(&self) -> Line {
        let mut modifiers = Modifiers::default();
        for m in self.modifiers() {
            let value = match m.value {
                MVal::Flag => ModValue::Int(1),
                MVal::Int(v) | MVal::Symbol(_, v) => ModValue::Int(i128::from(v)),
                MVal::List(items, n) => {
                    ModValue::List(items[..n as usize].iter().map(|&b| i128::from(b)).collect())
                }
            };
            modifiers.insert(m.key, value);
        }
        Line {
            mnemonic: self.mnemonic.to_owned(),
            operands: (self.operands().iter().enumerate())
                .map(|(k, o)| operand(o, self.signs(k)))
                .collect(),
            modifiers,
        }
    }
}

/// The label of a branch target.
pub fn label(offset: u32) -> String {
    format!(".L{offset}")
}

fn int_text(out: &mut impl fmt::Write, v: i64) -> fmt::Result {
    if v < 0 || v <= 64 {
        write!(out, "{v}")
    } else {
        write!(out, "0x{v:x}")
    }
}

fn opd_text(out: &mut impl fmt::Write, o: &Opd) -> fmt::Result {
    match *o {
        Opd::V { n, count: 1 } => write!(out, "v{n}"),
        Opd::V { n, count } => write!(out, "v[{n}:{}]", n + count - 1),
        Opd::S { n, count: 1 } => write!(out, "s{n}"),
        Opd::S { n, count } => write!(out, "s[{n}:{}]", n + count - 1),
        Opd::A { n, count: 1 } => write!(out, "a{n}"),
        Opd::A { n, count } => write!(out, "a[{n}:{}]", n + count - 1),
        Opd::Named(&(name, _, _)) => out.write_str(name),
        Opd::Int(v) => int_text(out, v),
        Opd::Float(f) => write!(out, "{f:?}"),
        Opd::Off => out.write_str("off"),
        Opd::Hwreg {
            name: Some(name),
            offset,
            size,
            ..
        } => write!(out, "hwreg({name}, {offset}, {size})"),
        Opd::Hwreg {
            name: None,
            offset,
            size,
            code,
        } => write!(out, "hwreg({}, {offset}, {size})", code & 63),
        Opd::Label(t) => write!(out, ".L{t}"),
    }
}

/// An operand's text with its sign modifiers.
fn signed_text(out: &mut impl fmt::Write, o: &Opd, (neg, abs): (bool, bool)) -> fmt::Result {
    if neg {
        out.write_char('-')?;
    }
    if abs {
        out.write_char('|')?;
    }
    opd_text(out, o)?;
    if abs {
        out.write_char('|')?;
    }
    Ok(())
}

/// The parsed operand of a decoded one, as `parse_operand` makes it.
fn operand(o: &Opd, signs: (bool, bool)) -> Operand {
    let mut text = String::new();
    signed_text(&mut text, o, signs).expect("a String takes any text");
    let mut bare = String::new();
    opd_text(&mut bare, o).expect("a String takes any text");
    let (kind, n, count, value, name) = match *o {
        Opd::V { n, count } => (OperandKind::V, n, count, None, None),
        Opd::S { n, count } => (OperandKind::S, n, count, None, None),
        Opd::A { n, count } => (OperandKind::A, n, count, None, None),
        Opd::Named(&(name, n, count)) => (
            OperandKind::S,
            n as u16,
            count as u16,
            None,
            Some(name.to_owned()),
        ),
        Opd::Int(v) => (
            OperandKind::Int,
            0,
            1,
            Some(Value::Int(i128::from(v))),
            None,
        ),
        Opd::Float(f) => (
            OperandKind::Float,
            0,
            1,
            Some(Value::Float(f64::from(f))),
            None,
        ),
        Opd::Off => (OperandKind::Off, 0, 1, None, None),
        Opd::Hwreg { code, .. } => (
            OperandKind::Simm,
            0,
            1,
            Some(Value::Int(i128::from(code))),
            None,
        ),
        Opd::Label(_) => (OperandKind::Label, 0, 1, Some(Value::Label(bare)), None),
    };
    Operand {
        kind,
        n: u32::from(n),
        count: u32::from(count),
        neg: signs.0,
        abs: signs.1,
        value,
        text,
        name,
    }
}

/// Instructions are equal when their offsets, sizes, mnemonics, operands
/// and modifiers are.
impl PartialEq for Instruction {
    fn eq(&self, o: &Instruction) -> bool {
        (self.offset, self.size, self.mnemonic, self.counted)
            == (o.offset, o.size, o.mnemonic, o.counted)
            && self.operands() == o.operands()
            && self.signs[..self.nops as usize] == o.signs[..o.nops as usize]
            && self.modifiers() == o.modifiers()
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}+{}: {self}", self.offset, self.size)
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mnemonic)?;
        if self.counted {
            for m in self.modifiers() {
                let v = match m.value {
                    MVal::Int(v) => v,
                    _ => 0,
                };
                f.write_char(' ')?;
                if m.key == "simm16" {
                    int_text(f, i64::from(v))?;
                } else {
                    write!(f, "{}({v})", m.key)?;
                }
            }
            return Ok(());
        }
        for (i, o) in self.operands().iter().enumerate() {
            f.write_str(if i == 0 { " " } else { ", " })?;
            signed_text(f, o, self.signs(i))?;
        }
        for m in self.modifiers() {
            write!(f, " {}", m.key)?;
            match m.value {
                MVal::Flag => {}
                MVal::Int(v) => {
                    f.write_char(':')?;
                    int_text(f, i64::from(v))?;
                }
                MVal::Symbol(name, _) => write!(f, ":{name}")?,
                MVal::List(items, n) => {
                    f.write_str(":[")?;
                    for (i, b) in items[..n as usize].iter().enumerate() {
                        if i > 0 {
                            f.write_char(',')?;
                        }
                        write!(f, "{b}")?;
                    }
                    f.write_char(']')?;
                }
            }
        }
        Ok(())
    }
}

/// Why a decoder refuses bytes; the classes of the coverage report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Why {
    /// The bytes end inside an instruction.
    Truncated,
    /// No encoding class has these fixed bits.
    NoEncoding,
    /// The class has no op of this opcode.
    NoOpcode,
    /// The encoder takes no line of this class or op.
    NotEncoded,
    /// A field holds a value no line the encoder takes gives it.
    Field,
    /// An operand's code is nothing the line syntax spells, or the encoder
    /// does not take it in that place.
    Operand,
    /// The op's operands are not a shape the encoder takes.
    Form,
    /// A branch goes outside the bytes or into an instruction.
    Branch,
}

impl Why {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Why::Truncated => "truncated",
            Why::NoEncoding => "no encoding",
            Why::NoOpcode => "no opcode",
            Why::NotEncoded => "not encoded",
            Why::Field => "field",
            Why::Operand => "operand",
            Why::Form => "form",
            Why::Branch => "branch",
        }
    }
}

/// A refusal: where, the bits, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError {
    /// The byte offset of the instruction.
    pub offset: usize,
    /// The bytes read of it, little-endian (the first byte lowest).
    pub bits: u128,
    /// How many bytes `bits` holds.
    pub len: u8,
    /// The instruction's whole size when its class is known, so a reader
    /// may step past it.
    pub size: Option<u8>,
    /// The class of refusal.
    pub why: Why,
    /// The encoding class, or `""`.
    pub class: &'static str,
    /// The op's mnemonic, when known.
    pub mnemonic: Option<&'static str>,
    /// The field at fault, when one is.
    pub field: Option<&'static str>,
    /// The specifics.
    pub detail: String,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}: ", self.offset)?;
        for b in &self.bits.to_le_bytes()[..self.len as usize] {
            write!(f, "{b:02x}")?;
        }
        let what = self.mnemonic.unwrap_or(self.class);
        if what.is_empty() {
            write!(f, ": {}", self.detail)
        } else {
            write!(f, ": {what}: {}", self.detail)
        }
    }
}

impl std::error::Error for DecodeError {}

/// A target's decoder.
pub trait Decode: Send + Sync {
    /// Decodes a word sequence whole: every instruction, each branch landing
    /// on an instruction's start or the end; or the first refusal.
    fn decode(&self, bytes: &[u8]) -> Result<Vec<Instruction>, DecodeError>;
    /// Decodes the one instruction at a byte offset; its branch target is
    /// not checked. A refusal's `size`, when known, steps past it.
    fn decode_at(&self, bytes: &[u8], at: usize) -> Result<Instruction, DecodeError> {
        let mut out = Instruction::new(0, "");
        self.decode_into(bytes, at, &mut out).map(|()| out)
    }
    /// [`Decode::decode_at`] into an instruction's storage, which a walk
    /// over many reuses.
    fn decode_into(
        &self,
        bytes: &[u8],
        at: usize,
        out: &mut Instruction,
    ) -> Result<(), DecodeError>;
}
