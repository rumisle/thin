//! The verifier: the first instruction, in layout order, that
//! breaks an invariant, with the first check it fails.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use thin_target::{dpp16_control, phys_named, FieldKind, Kind, Phys, Slot, SlotKind, Target};

use crate::{pyfmt, AttrClass, FieldId, FieldValue, Instr, Operand, Reg};

/// What the verifier reads of a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueFacts {
    /// Its `repr()`, `%<file><id>:<name>`.
    pub repr: String,
    /// Its dwords.
    pub n: u32,
    /// Its alignment.
    pub align: u32,
    /// Whether it is pinned.
    pub pinned: bool,
}

/// The index of the first instruction that breaks an invariant and its
/// message, `` `<form name>`: <msg> ``; `values` has every value a register
/// names, by number. With `printable`, a wait marker fails too.
pub fn verify(
    target: &Target,
    instrs: &[Arc<Instr>],
    values: &HashMap<i32, ValueFacts>,
    printable: bool,
) -> Option<(usize, String)> {
    let name = FieldId::of("name");
    let labels: HashSet<&str> = (instrs.iter())
        .filter(|i| i.form().kind == Kind::Label)
        .filter_map(|i| i.field(name).and_then(FieldValue::text))
        .collect();
    let mut defined = HashSet::new();
    instrs.iter().enumerate().find_map(|(k, i)| {
        let twice = (i.form().kind == Kind::Label)
            .then(|| i.field(name).and_then(FieldValue::text))
            .flatten()
            .filter(|l| !defined.insert(*l))
            .map(|l| format!("label {l} is defined twice"));
        let msg = twice
            .or_else(|| operand_count(i))
            .or_else(|| (0..i.ops().len()).find_map(|n| operand(target, i, n, values, &labels)))
            .or_else(|| fields(i))
            .or_else(|| attributes(i))
            .or_else(|| {
                (printable && i.is_wait()).then(|| "a wait request left at the printer".into())
            })?;
        Some((k, format!("`{}`: {msg}", i.form().name)))
    })
}

/// A fixed form takes one operand per slot; a variadic form with no slot
/// takes none.
fn operand_count(i: &Instr) -> Option<String> {
    let (n, f) = (i.ops().len(), i.form());
    let ok = if f.variadic {
        !f.slots.is_empty() || n == 0
    } else {
        n == f.slots.len()
    };
    (!ok).then(|| format!("{n} operands; the form takes {}", f.slots.len()))
}

/// Operand n against its slot.
fn operand(
    target: &Target,
    i: &Instr,
    n: usize,
    values: &HashMap<i32, ValueFacts>,
    labels: &HashSet<&str>,
) -> Option<String> {
    let slot = i.form().slot(n)?;
    let sname = if slot.xml_type.is_empty() {
        n.to_string()
    } else {
        slot.xml_type.clone()
    };
    let kind = slot.kind.name();
    let takes = |what: String| Some(format!("operand {n} is {what}; slot {sname} takes {kind}"));
    let op = &i.ops()[n];
    let refused = match op {
        Operand::Reg(r) => register(target, slot, n, r, &values[&r.value], &sname),
        Operand::Phys(p) => {
            let off = p.name == "off" && !slot.widths.is_empty();
            let named = [
                SlotKind::Ssrc,
                SlotKind::Vsrc,
                SlotKind::Sgpr,
                SlotKind::Any,
            ];
            if off || named.contains(&slot.kind) {
                None
            } else if slot.kind == SlotKind::Special {
                special(target, slot, p)
                    .map(|want| format!("operand {n} is {}; slot {sname} takes {want}", p.name))
            } else {
                takes(p.name.to_owned())
            }
        }
        Operand::Imm(_) => match slot.kind {
            SlotKind::Ssrc | SlotKind::Vsrc | SlotKind::Avc | SlotKind::Imm => None,
            _ => takes("an immediate".into()),
        },
        Operand::Block(_) if slot.kind == SlotKind::Label => None,
        Operand::Block(_) => takes("a block".into()),
    };
    refused
        .or_else(|| {
            let modified = i.modified(n) && !slot.modifiers;
            modified.then(|| format!("operand {n}: modifiers on slot {sname}, which takes none"))
        })
        .or_else(|| match op {
            Operand::Block(b) if !labels.contains(&**b) => {
                Some(format!("branches to {b}, not a block of this program"))
            }
            _ => None,
        })
}

/// A named register against a special slot, which takes the one register
/// its XML type names (`OPR_VCC`: VCC, as the target's mask register):
/// none if it fits, else the register it takes.
fn special(target: &Target, slot: &Slot, p: &Phys) -> Option<&'static str> {
    let name = slot.xml_type.strip_prefix("OPR_")?.to_ascii_lowercase();
    let want = target.mask_reg(phys_named(&name)?);
    (want.name != p.name).then_some(want.name)
}

/// A register operand against its slot: file, widths, its value's extent
/// and alignment.
fn register(
    target: &Target,
    slot: &Slot,
    k: usize,
    r: &Reg,
    v: &ValueFacts,
    sname: &str,
) -> Option<String> {
    let kind = slot.kind.name();
    if !slot.kind.files().contains(r.file) {
        return Some(format!(
            "operand {k} is a {} register; slot {sname} takes {kind}",
            r.file
        ));
    }
    if !slot.widths.is_empty() && !slot.widths.contains(&r.n) {
        let widths: Vec<String> = slot.widths.iter().map(u32::to_string).collect();
        return Some(format!(
            "operand {k} is {} dwords; slot {sname} takes {}",
            r.n,
            pyfmt::tuple(&widths)
        ));
    }
    if slot.widths.is_empty() && slot.dwords != 0 && r.n != slot.dwords {
        return Some(format!(
            "operand {k} is {} dwords; slot {sname} takes {}",
            r.n, slot.dwords
        ));
    }
    if r.lo + r.n > v.n {
        return Some(format!(
            "operand {k}: dwords {}..{} outside {} ({})",
            r.lo,
            (r.lo + r.n).saturating_sub(1),
            v.repr,
            v.n
        ));
    }
    let a64 = target.file(r.file).map_or(0, |f| f.align64);
    (r.n >= 2 && a64 != 0 && v.align % a64 != 0 && !v.pinned).then(|| {
        format!(
            "operand {k}: a {}-dword range of {}, aligned {}",
            r.n, v.repr, v.align
        )
    })
}

/// Each field against the form's declaration, in the instruction's order;
/// then the required fields.
fn fields(i: &Instr) -> Option<String> {
    let form = i.form();
    let mut seen = HashSet::new();
    for (id, value) in i.fields() {
        let name = id.name();
        if !seen.insert(name) {
            return Some(format!("field {name} given twice"));
        }
        let Some(field) = form.field(name) else {
            let takes: Vec<&str> = form.fields.iter().map(|f| f.name).collect();
            let takes = if takes.is_empty() {
                "none".into()
            } else {
                takes.join(", ")
            };
            return Some(format!("no field {name} (it takes {takes})"));
        };
        let refused = match field.kind {
            // A bool is no int here: `True` would read as 1, a mask of
            // one bit, where a writer likely meant all of them.
            FieldKind::Int if matches!(value, FieldValue::Bool(_)) => {
                Some(format!("{name}: an int, not a bool"))
            }
            FieldKind::Int => {
                let (lo, hi) = if field.signed {
                    (
                        -(1i128 << (field.bits - 1)),
                        (1i128 << (field.bits - 1)) - 1,
                    )
                } else {
                    (0, (1i128 << field.bits) - 1)
                };
                let v = match value {
                    FieldValue::Int(_) => value.int(),
                    FieldValue::Object(o) => o.facts.int,
                    _ => None,
                };
                (!v.is_some_and(|v| (lo..=hi).contains(&v)))
                    .then(|| format!("{name}:{} outside {lo}..{hi}", value.py_str()))
            }
            FieldKind::Bits => {
                let ok = matches!(value, FieldValue::Ints(v)
                    if !v.is_empty() && v.len() <= field.bits as usize && v.iter().all(|&x| x == 0 || x == 1));
                (!ok).then(|| format!("{name}:{}: {} bits at most", value.py_str(), field.bits))
            }
            FieldKind::Flag => {
                (!matches!(value, FieldValue::Bool(_))).then(|| format!("{name}: a flag"))
            }
            FieldKind::Text => value.text().is_none().then(|| format!("{name}: a text")),
            FieldKind::Choice => (!value
                .text()
                .is_some_and(|t| field.choices.iter().any(|c| c == t)))
            .then(|| {
                format!(
                    "{name}: {}, not one of {}",
                    value.py_repr(),
                    field.choices.join(", ")
                )
            }),
            FieldKind::Tuple => {
                let ok = match value {
                    FieldValue::Ints(_) | FieldValue::Names(_) | FieldValue::Tags(_) => true,
                    FieldValue::Object(o) => o.facts.is_tuple,
                    _ => false,
                };
                (!ok).then(|| format!("{name}: a tuple"))
            }
            FieldKind::Dpp => match value.text() {
                Some(t) => dpp16_control(t).err(),
                None => Some(format!("{name}: a DPP control's text")),
            },
        };
        if refused.is_some() {
            return refused;
        }
    }
    (form.fields.iter())
        .find(|f| f.required && i.field_named(f.name).is_none())
        .map(|f| format!("field {} missing", f.name))
}

/// The first present attribute on a form whose flags meet none of its
/// class's `on`.
fn attributes(i: &Instr) -> Option<String> {
    let flags = i.form().flags;
    AttrClass::ALL.into_iter().find_map(|c| {
        let misplaced = c.on().0 != 0 && !flags.any(c.on()) && i.attrs().has(c);
        misplaced.then(|| {
            format!(
                "attribute {} on a form that is none of {:?}",
                c.name(),
                c.on()
            )
        })
    })
}
