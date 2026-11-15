//! Decoded machine execution and byte provenance.
//!
//! Owns waves, workgroups, global memory, LDS, in-flight loads, scope checks,
//! canonical instruction semantics and execution reports. The target reader
//! supplies decoded operations; the machine has no mnemonic parser of its own.
//!
//! The tags module depends on no other module here. The machine uses tags but
//! never GEMM internals. The gemm module uses tags and implements the checker
//! interface. This layering is checked by tests. Checkers execute natively with
//! plain-data problems and never call back into Python.
//!
//! Tag tables are caller-owned Rust values. The Python binding keeps process
//! scoped tables and offers a scoped fresh-table context for tests. The GEMM
//! checker is experimental, test-only tooling, not a correctness certificate.
//! See docs/dev/emulator.md and docs/dev/limitations.md for model boundaries.

use std::fmt;

pub mod api;
pub mod gemm;
pub mod machine;
pub mod tags;

use thin_target::Target;

/// The class of an emulator error; normative, apart from the message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The kernel's fault (`EmuError`).
    Fault,
    /// Something the emulator does not model (`EmuLimit`).
    Emulator,
    /// No wave can run or none makes progress (`Deadlock`).
    Deadlock,
    /// An IPC buffer accessed without the policy bits (`IpcPolicy`).
    Ipc,
    /// A read the scope model refuses (`ScopeModel`).
    Scope,
    /// A register touched while a load into it is pending (`RegisterInFlight`,
    /// cause `fault`).
    InFlight,
}

/// A refusal or fault of a run, or a reading error of a `.s` (raised as
/// `ValueError`) when `cause` is none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmuError {
    /// The class; none for a reading error.
    pub cause: Option<Cause>,
    /// The message (which wave, instruction, byte, buffer).
    pub message: String,
}

impl fmt::Display for EmuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EmuError {}

/// What an operand of a decoded instruction is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Base {
    /// Registers of `v`, `s` or `a` as (file, first, count), or a modelled
    /// named register as (name, part, count).
    Reg(String, u32, u32),
    /// A 32-bit pattern and its text.
    Literal(u32, String),
    /// A label reference.
    Label(String),
}

/// An operand with its source modifiers (a literal's abs is always false).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operand {
    /// What it is.
    pub base: Base,
    /// Negated.
    pub neg: bool,
    /// Absolute value.
    pub abs: bool,
}

/// A field's value: a bare flag, an integer or a list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Field {
    /// A bare flag.
    True,
    /// An integer.
    Int(i128),
    /// A list.
    List(Vec<i128>),
}

/// A decoded instruction, the machine's input.
#[derive(Clone, Debug, PartialEq)]
pub struct Ins {
    /// The canonical op name the semantics are keyed by.
    pub canon: String,
    /// The form kind's name (`valu`, `salu`, `ds`, …).
    pub kind: String,
    /// The operands in order.
    pub operands: Vec<Operand>,
    /// The fields by name, in order.
    pub fields: Vec<(String, Field)>,
    /// The matrix form's name for a matrix op; the machine routes by it.
    pub matrix: Option<String>,
    /// The text for messages.
    pub text: String,
    /// The markers' source.
    pub comment: String,
    /// The source line, or 0.
    pub line: usize,
}

impl Ins {
    /// An instruction from decoded JSON: `{op, kind,
    /// operands, fields}`; a lane mask's count is rescaled to W / 32 (the
    /// input represents masks at W = 64).
    pub fn of(json: &serde_json::Value, w: usize) -> Ins {
        let s = |v: &serde_json::Value| v.as_str().unwrap_or("").to_owned();
        let int = |v: &serde_json::Value| v.as_u64().unwrap_or(0) as u32;
        let operands: Vec<Operand> = json["operands"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|o| {
                let base = if let Some(r) = o.get("reg") {
                    let file = s(&r[0]);
                    let mut count = int(&r[2]);
                    if matches!(file.as_str(), "exec" | "vcc") && count == 2 {
                        count = (w / 32).max(1) as u32;
                    }
                    Base::Reg(file, int(&r[1]), count)
                } else if let Some(l) = o.get("lit") {
                    let hex = s(&l[0]);
                    let v = u32::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap_or(0);
                    Base::Literal(v, s(&l[1]))
                } else {
                    Base::Label(s(&o["label"]))
                };
                Operand {
                    base,
                    neg: o["neg"].as_bool().unwrap_or(false),
                    abs: o["abs"].as_bool().unwrap_or(false),
                }
            })
            .collect();
        let fields = json["fields"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| {
                let f = match v {
                    serde_json::Value::Bool(_) => Field::True,
                    serde_json::Value::Array(a) => Field::List(
                        a.iter()
                            .map(|x| i128::from(x.as_i64().unwrap_or(0)))
                            .collect(),
                    ),
                    other => Field::Int(i128::from(other.as_i64().unwrap_or(0))),
                };
                (k.clone(), f)
            })
            .collect();
        let canon = s(&json["op"]);
        let mut ins = Ins {
            text: String::new(),
            canon,
            kind: s(&json["kind"]),
            operands,
            fields,
            matrix: json["matrix"].as_str().map(str::to_owned),
            comment: json["comment"].as_str().unwrap_or("").to_owned(),
            line: json["line"].as_u64().unwrap_or(0) as usize,
        };
        ins.text = json["text"]
            .as_str()
            .map_or_else(|| ins.default_text(), str::to_owned);
        ins
    }

    /// `canon a, b, c`: the default text.
    pub fn default_text(&self) -> String {
        let ops: Vec<String> = self.operands.iter().map(Operand::spelling).collect();
        if ops.is_empty() {
            self.canon.clone()
        } else {
            format!("{} {}", self.canon, ops.join(", "))
        }
    }

    /// A field's value, if present.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }
}

impl Operand {
    /// The operand's spelling: `v5`, `s[4:7]`, a named register's name, a
    /// literal's text, a label.
    pub fn spelling(&self) -> String {
        let core = match &self.base {
            Base::Reg(f, a, n) if f.len() == 1 => {
                if *n == 1 {
                    format!("{f}{a}")
                } else {
                    format!("{f}[{a}:{}]", a + n - 1)
                }
            }
            Base::Reg(f, _, _) => f.clone(),
            Base::Literal(v, t) => {
                if t.is_empty() {
                    format!("{v:#x}")
                } else {
                    t.clone()
                }
            }
            Base::Label(l) => l.clone(),
        };
        let core = if self.abs { format!("|{core}|") } else { core };
        if self.neg {
            format!("-{core}")
        } else {
            core
        }
    }
}

/// A workgroup's coordinates in the dispatch grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkgroupId(pub [u32; 3]);

impl From<(u32, u32)> for WorkgroupId {
    fn from((x, y): (u32, u32)) -> Self {
        Self([x, y, 0])
    }
}

impl From<(u32, u32, u32)> for WorkgroupId {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self([x, y, z])
    }
}

impl TryFrom<Vec<u32>> for WorkgroupId {
    type Error = EmuError;

    fn try_from(coordinates: Vec<u32>) -> Result<Self, Self::Error> {
        match *coordinates.as_slice() {
            [x, y] => Ok(Self([x, y, 0])),
            [x, y, z] => Ok(Self([x, y, z])),
            _ => Err(EmuError {
                cause: None,
                message: "workgroup id: expected two or three coordinates".into(),
            }),
        }
    }
}

/// A decoded program.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
    /// The instructions in order.
    pub ins: Vec<Ins>,
    /// Byte-address ranges needed only by PC-relative instructions.
    pub pc_layout: Option<machine::pc::Layout>,
    /// Label (leading dots stripped) → index of the instruction after it.
    pub labels: Vec<(String, usize)>,
    /// Section lines: (index of the next instruction, text, line).
    pub sections: Vec<(usize, String, usize)>,
    /// The descriptor's user SGPR count; 2 when absent.
    pub user_sgprs: u32,
    /// Descriptor workgroup-id enables in x, y, z order; absent for a body.
    pub workgroup_ids: Option<[bool; 3]>,
    /// SGPR pair receiving the kernarg pointer; absent for a body.
    pub kernarg_sgpr: Option<u32>,
    /// (index of the next instruction, line) of each label item, for the
    /// claims' label rule; empty for a program not read off a `.s`.
    pub label_lines: Vec<(usize, usize)>,
}

/// Reads a `.s` with `read_s(text, body = true)` and the target's decoding
/// map.
pub fn read_program(text: &str, target: &Target) -> Result<Program, EmuError> {
    machine::program::read_program(text, target)
}

pub use tags::Tables;
