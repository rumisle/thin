//! The IR's own ops as every target carries them.

use crate::form::{Field, Form, Pseudo, Slot};
use crate::vocab::{FieldKind, Flags, Kind, SlotKind};
use crate::Error;

/// The own ops' names, in the order of their ids −1, −2, …
pub static OWN_OPS: [(&str, Pseudo); 11] = [
    ("ir.wait_regs", Pseudo::WaitRegs),
    ("ir.wait_tags", Pseudo::WaitTags),
    ("ir.wait_all", Pseudo::WaitAll),
    ("ir.wait_left", Pseudo::WaitLeft),
    ("ir.wait_smem", Pseudo::WaitSmem),
    ("ir.discard", Pseudo::Discard),
    ("ir.assume", Pseudo::Assume),
    ("ir.uniform", Pseudo::Uniform),
    ("ir.undefined", Pseudo::Undefined),
    ("ir.label", Pseudo::Label),
    ("ir.comment", Pseudo::Comment),
];

/// An own op's `any` slot of a role.
fn any(role: u8) -> Slot {
    Slot {
        kind: SlotKind::Any,
        dwords: 0,
        role,
        xml_type: String::new(),
        sgpr_widths: Vec::new(),
        modifiers: false,
        literal: false,
        widths: Vec::new(),
        limit: None,
    }
}

fn text(name: &str) -> Result<Field, Error> {
    Field::new(name, FieldKind::Text, 0, false)
}

fn int(name: &str, bits: u32) -> Result<Field, Error> {
    Ok(Field::new(name, FieldKind::Int, bits, false)?.required())
}

/// A wait op's fields after its own: the counter, said and ablated.
fn wait_fields(counters: &[String]) -> Result<Vec<Field>, Error> {
    Ok(vec![
        Field::new("counter", FieldKind::Choice, 0, false)?
            .required()
            .choices(counters),
        text("said")?,
        Field::new("ablated", FieldKind::Flag, 0, false)?,
    ])
}

/// The eleven own ops of a target whose counters have these names.
pub fn own_ops(counters: &[String]) -> Result<Vec<Form>, Error> {
    let mut forms = Vec::with_capacity(OWN_OPS.len());
    for (k, &(name, pseudo)) in OWN_OPS.iter().enumerate() {
        let kind = if pseudo == Pseudo::Label {
            Kind::Label
        } else {
            Kind::Comment
        };
        let mut form = Form::plain(-1 - k as i32, name, kind, Flags::PSEUDO);
        form.pseudo = Some(pseudo);
        let (slots, variadic, fields) = match pseudo {
            Pseudo::WaitRegs => (
                vec![any(0)],
                true,
                [
                    vec![Field::new("regions", FieldKind::Tuple, 0, false)?],
                    wait_fields(counters)?,
                ]
                .concat(),
            ),
            Pseudo::WaitTags => (
                vec![],
                false,
                [
                    vec![Field::new("tags", FieldKind::Tuple, 0, false)?.required()],
                    wait_fields(counters)?,
                ]
                .concat(),
            ),
            Pseudo::WaitAll | Pseudo::WaitSmem => (vec![], false, wait_fields(counters)?),
            Pseudo::WaitLeft => (
                vec![],
                false,
                [vec![int("left", 8)?], wait_fields(counters)?].concat(),
            ),
            Pseudo::Discard | Pseudo::Uniform => {
                (vec![any(0)], true, vec![text("why")?.required()])
            }
            Pseudo::Undefined => (
                vec![any(0)],
                true,
                vec![
                    text("why")?.required(),
                    Field::new("on", FieldKind::Choice, 0, false)?.choices(&[
                        "next".to_owned(),
                        "taken".to_owned(),
                        "fall".to_owned(),
                    ]),
                ],
            ),
            Pseudo::Assume => (
                vec![any(1)],
                false,
                vec![int("lo", 32)?, int("hi", 32)?, text("why")?.required()],
            ),
            Pseudo::Label => (vec![], false, vec![text("name")?.required()]),
            Pseudo::Comment => (vec![], false, vec![text("text")?.required()]),
        };
        (form.slots, form.variadic, form.fields) = (slots, variadic, fields);
        forms.push(form);
    }
    Ok(forms)
}
