//! The instruction maker `O.op`.

mod checks;
pub mod families;
pub mod num;

use std::sync::Arc;

use thin_ir::{Atom, Attr, FieldId, FieldValue, Imm, ImmValue, Instr, Operand, Reg, Tag};
use thin_target::{phys_named, Flags, Phys, SlotKind, Target};

use crate::error::{value, Result};
use crate::lds::Address;
use crate::values::{reg_repr, ValueRec};

pub use num::{int0, is_literal32, py_float};

/// The field names the maker takes: every field a target's line can spell
/// (`thin_target::FIELD_NAMES`); whether a form has one is the verifier's
/// check.
pub fn fields() -> &'static [&'static str] {
    &thin_target::FIELD_NAMES[..thin_target::LINE_FIELDS]
}

/// An operand as the author gives it, after `as_reg`.
#[derive(Clone, Debug)]
pub enum Arg {
    /// None.
    None,
    /// A bool.
    Bool(bool),
    /// An int.
    Int(i128),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
    /// A register and its modifier byte.
    Reg(Reg, u8),
    /// A named register and its modifier byte.
    Phys(&'static Phys, u8),
    /// An LDS address.
    Addr(Address),
}

/// A field's value as the author gives it.
#[derive(Clone, Debug, PartialEq)]
pub enum FArg {
    /// None.
    None,
    /// A bool.
    Bool(bool),
    /// An int.
    Int(i128),
    /// A string.
    Str(String),
    /// A list or tuple.
    List(Vec<FArg>),
}

impl FArg {
    /// Whether the field is set: not None, False or "".
    pub fn set(&self) -> bool {
        !matches!(self, FArg::None | FArg::Bool(false))
            && !matches!(self, FArg::Str(s) if s.is_empty())
    }

    /// Its text in a line's tail.
    fn text(&self) -> String {
        match self {
            FArg::None => "None".into(),
            FArg::Bool(b) => if *b { "True" } else { "False" }.into(),
            FArg::Int(v) => v.to_string(),
            FArg::Str(s) => s.clone(),
            FArg::List(xs) => format!(
                "[{}]",
                xs.iter().map(FArg::text).collect::<Vec<_>>().join(",")
            ),
        }
    }

    fn int(&self) -> Option<i128> {
        match self {
            FArg::Int(v) => Some(*v),
            FArg::Bool(b) => Some(i128::from(*b)),
            FArg::Str(s) => int0(s),
            _ => None,
        }
    }
}

/// The notes an instruction carries.
#[derive(Clone, Debug, Default)]
pub struct Notes {
    /// Its comment.
    pub comment: String,
    /// Its tag.
    pub tag: Option<Tag>,
    /// Its ablation classes.
    pub abl: Vec<String>,
    /// What it stages.
    pub stage: Option<Stage>,
}

/// An operand staged into the scratch SGPR before the instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `s.add.u32 _lit, soffset, offset`.
    Add(Reg, i128),
    /// `s.mov.b32 _lit, offset`.
    Mov(i128),
}

/// What the maker made: an instruction, or an LDS access to make later
///
#[derive(Clone, Debug)]
pub enum Made {
    /// An instruction.
    Instr(Box<Instr>),
    /// An LDS access.
    Lds(Box<crate::lds::Access>),
}

/// The maker's view of the values, for the registers' texts.
pub struct Ctx<'a> {
    /// The target.
    pub t: &'static Target,
    /// The program's values.
    pub values: &'a [ValueRec],
}

impl Ctx<'_> {
    fn repr(&self, r: &Reg) -> String {
        reg_repr(self.values, r)
    }
}

/// The value name a register reads, `[index]` when nonzero.
fn value_name(c: &Ctx<'_>, r: &Reg) -> String {
    match usize::try_from(r.value).ok().and_then(|v| c.values.get(v)) {
        Some(v) if v.index != 0 => format!("{}[{}]", v.name, v.index),
        Some(v) => v.name.to_string(),
        None => "_lit".into(),
    }
}

/// `O.op(name, *operands, **fields)`.
pub fn op(
    c: &Ctx<'_>,
    name: &str,
    args: Vec<Arg>,
    fields: Vec<(String, FArg)>,
    notes: Notes,
) -> Result<Made> {
    let t = c.t;
    let name = t.spell(name).ok_or_else(|| crate::error::Error {
        class: crate::error::Class::Value,
        msg: format!("{name}: {} has no such op", t.name()),
    })?;
    if name.starts_with("ds_") {
        if let Some(k) = args.iter().position(|a| matches!(a, Arg::Addr(_))) {
            return crate::lds::access(&name, args, k, fields, notes)
                .map(|a| Made::Lds(Box::new(a)));
        }
    }
    if let Some(k) = args.iter().position(|a| matches!(a, Arg::None)) {
        return value(format!(
            "{name}: operand {k} is None (a register, a value or a constant expected)"
        ));
    }
    let fields = check_fields(&name, fields)?;
    let args: Vec<Arg> = args.into_iter().map(|a| masked(t, a)).collect();
    checks::before_form(c, &name, &args, &fields)?;
    let th = fields.iter().find_map(|(k, v)| match (k.as_str(), v) {
        ("th", FArg::Str(s)) => Some(s.as_str()),
        _ => None,
    });
    let returning = t.hooks().returning(&name, th);
    let Some(form) = t.form(&name, args.len(), returning) else {
        return Err(no_form(c, &name, &args, &fields));
    };
    let mut ops = Vec::with_capacity(args.len());
    let mut mods = Vec::with_capacity(args.len());
    for (k, a) in args.iter().enumerate() {
        let (o, m) = operand(&name, form, k, a)?;
        ops.push(o);
        mods.push(m);
    }
    let fv = field_values(&fields)?;
    let mods = mods
        .iter()
        .any(|&m| m != 0)
        .then(|| mods.into_boxed_slice());
    let mut i = Instr::new(form, ops, fv, mods);
    i = with_notes(i, notes, c)?;
    if t.unit_of(&name) == Some(Flags::VALU) {
        checks::constant_bus(c, &name, &i, &args)?;
    }
    Ok(Made::Instr(Box::new(i)))
}

/// Replaces a named register given as an object or a string by its lane
/// mask register.
fn masked(t: &Target, a: Arg) -> Arg {
    match a {
        Arg::Phys(p, m) => Arg::Phys(t.mask_reg(p), m),
        Arg::Str(s) => match phys_named(&s) {
            Some(p) => Arg::Phys(t.mask_reg(p), 0),
            None => Arg::Str(s),
        },
        a => a,
    }
}

/// Unknown names refused; the set fields kept in the given order.
fn check_fields(name: &str, fields: Vec<(String, FArg)>) -> Result<Vec<(String, FArg)>> {
    let known = self::fields();
    let mut unknown: Vec<&str> = fields
        .iter()
        .map(|f| f.0.as_str())
        .filter(|f| !known.contains(f))
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return value(format!(
            "{name}: no field {} (the fields: {})",
            unknown.join(", "),
            known.join(", ")
        ));
    }
    Ok(fields.into_iter().filter(|f| f.1.set()).collect())
}

/// A string's Python repr.
pub fn py_repr(s: &str) -> String {
    match (s.contains('\''), s.contains('"')) {
        (true, false) => format!("\"{s}\""),
        _ => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
    }
}

/// The no-form refusal: the instruction as written.
fn no_form(
    c: &Ctx<'_>,
    name: &str,
    args: &[Arg],
    fields: &[(String, FArg)],
) -> crate::error::Error {
    let t = c.t;
    crate::error::Error {
        class: crate::error::Class::Value,
        msg: format!(
            "`{}`: no {} form has this mnemonic and operand count (the target's operand table)",
            written(c, name, args, fields),
            t.name()
        ),
    }
}

/// The instruction as written: registers by their reprs, modifiers as
/// `-x` and `|x|`, the set fields as a line's tail.
pub fn written(c: &Ctx<'_>, name: &str, args: &[Arg], fields: &[(String, FArg)]) -> String {
    let ops: Vec<String> = (args.iter())
        .map(|a| {
            let (text, m) = match a {
                Arg::Reg(r, m) => (crate::values::reg_text(c.values, r), *m),
                Arg::Phys(p, m) => (p.name.to_owned(), *m),
                Arg::Int(v) => (v.to_string(), 0),
                Arg::Float(x) => (py_float(*x), 0),
                Arg::Str(s) => (s.clone(), 0),
                Arg::Bool(b) => (if *b { "True" } else { "False" }.to_owned(), 0),
                Arg::None => ("None".to_owned(), 0),
                Arg::Addr(a) => (a.text(), 0),
            };
            modified(text, m)
        })
        .collect();
    let mut out = name.to_owned();
    if !ops.is_empty() {
        out = format!("{out} {}", ops.join(", "));
    }
    for (f, v) in fields {
        let tail = match (f.as_str(), v) {
            (_, FArg::Bool(false)) => continue,
            ("bound_ctrl", FArg::Bool(true)) => "bound_ctrl:1".into(),
            (_, FArg::Bool(true)) => f.clone(),
            ("row_mask" | "bank_mask", FArg::Int(x)) => format!("{f}:0x{x:x}"),
            ("dpp", v) => v.text(),
            (_, v) => format!("{f}:{}", v.text()),
        };
        out = format!("{out} {tail}");
    }
    out
}

fn modified(text: String, m: u8) -> String {
    let t = if m & 2 != 0 {
        format!("|{text}|")
    } else {
        text
    };
    if m & 1 != 0 {
        format!("-{t}")
    } else {
        t
    }
}

/// An IR operand.
fn operand(name: &str, form: &thin_target::Form, k: usize, a: &Arg) -> Result<(Operand, u8)> {
    Ok(match a {
        Arg::Reg(r, m) => (Operand::Reg(*r), *m),
        Arg::Phys(p, m) => (Operand::Phys(p), *m),
        Arg::Str(s) => match phys_named(s) {
            Some(p) => (Operand::Phys(p), 0),
            None if form.slot(k).is_some_and(|sl| sl.kind == SlotKind::Label) => {
                (Operand::Block(s.as_str().into()), 0)
            }
            None => match imm_of_text(s) {
                Ok(imm) => (Operand::Imm(imm), 0),
                Err(_) => {
                    return value(format!(
                        "{name}: operand {k} {}: not a register's name, a label or a number's text (an \
                         immediate such as hwreg(..) is given as its encoded int)",
                        py_repr(s)
                    ))
                }
            },
        },
        Arg::Bool(_) => return value(format!("{name}: operand {k} is a bool")),
        Arg::Int(v) => (Operand::Imm(Imm::int(*v)), 0),
        Arg::Float(x) => (
            Operand::Imm(Imm {
                value: ImmValue::Float(*x),
                text: "".into(),
            }),
            0,
        ),
        Arg::None | Arg::Addr(_) => return value(format!("{name}: operand {k} is not an operand")),
    })
}

/// `Imm(v, s)` with v = `int(s, 0)`, else `float(s)`; the text kept only
/// when `str(v) != s`.
pub fn imm_of_text(s: &str) -> Result<Imm> {
    if let Some(v) = int0(s) {
        let text = if v.to_string() == s { "" } else { s };
        return Ok(Imm {
            value: ImmValue::Int(v),
            text: text.into(),
        });
    }
    match s.trim().parse::<f64>() {
        Ok(x) => {
            let text = if py_float(x) == s { "" } else { s };
            Ok(Imm {
                value: ImmValue::Float(x),
                text: text.into(),
            })
        }
        Err(_) => value(format!("could not convert string to float: {}", py_repr(s))),
    }
}

/// The field values: a list as a tuple.
fn field_values(fields: &[(String, FArg)]) -> Result<Vec<(FieldId, FieldValue)>> {
    let mut out = Vec::new();
    for (k, v) in fields {
        let id = FieldId::named(k)?;
        let fv = match v {
            FArg::Bool(b) => FieldValue::Bool(*b),
            FArg::Int(x) => FieldValue::Int(*x),
            FArg::Str(s) => FieldValue::Text(s.as_str().into()),
            FArg::List(xs) if xs.iter().all(|x| matches!(x, FArg::Str(_))) && !xs.is_empty() => {
                FieldValue::Names(xs.iter().map(|x| x.text().into()).collect())
            }
            FArg::List(xs) => FieldValue::Ints(xs.iter().filter_map(FArg::int).collect()),
            FArg::None => continue,
        };
        out.push((id, fv));
    }
    Ok(out)
}

/// The notes as attributes.
pub fn with_notes(mut i: Instr, notes: Notes, c: &Ctx<'_>) -> Result<Instr> {
    if !notes.comment.is_empty() {
        i = i.with_attr(Attr::Comment(notes.comment.as_str().into()))?;
    }
    if let Some(tag) = notes.tag.filter(|t| !t.is_empty()) {
        i = i.with_attr(Attr::Tag(tag))?;
    }
    if !notes.abl.is_empty() {
        let mut abl = notes.abl.clone();
        abl.sort();
        abl.dedup();
        i = i.with_attr(Attr::Ablation(
            abl.iter().map(|a| a.as_str().into()).collect(),
        ))?;
    }
    if let Some(stage) = notes.stage {
        i = i.with_attr(Attr::Staged(stage_object(&stage, c)))?;
    }
    Ok(i)
}

/// The staged operation as a host object: `('add', '<reg>', offset)` or
/// `('mov', offset)`.
pub fn stage_object(stage: &Stage, c: &Ctx<'_>) -> thin_ir::HostObject {
    use thin_ir::Datum;
    let (d, repr) = match stage {
        Stage::Add(r, off) => (
            Datum::Tuple(vec![
                Datum::Str("add".into()),
                Datum::Reg(*r),
                Datum::Int(*off),
            ]),
            format!("('add', '{}', {off})", c.repr(r)),
        ),
        Stage::Mov(off) => (
            Datum::Tuple(vec![Datum::Str("mov".into()), Datum::Int(*off)]),
            format!("('mov', {off})"),
        ),
    };
    thin_ir::HostObject::of_datum(d, repr)
}

/// A tag of atoms.
pub fn tag(atoms: Vec<Atom>) -> Tag {
    Arc::from(atoms)
}
