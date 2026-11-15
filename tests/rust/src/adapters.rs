//! Shared decoded-instruction and assembly adapters for golden tests.

use serde_json::Value;

/// A decoded operand in the golden' encoding.
pub fn operand_json(o: &thin_emu::Operand) -> Value {
    let mut m = serde_json::Map::new();
    match &o.base {
        thin_emu::Base::Reg(f, a, n) => {
            m.insert("reg".into(), serde_json::json!([f, a, n]));
        }
        thin_emu::Base::Literal(v, t) => {
            m.insert("lit".into(), serde_json::json!([format!("{v:#010x}"), t]));
        }
        thin_emu::Base::Label(l) => {
            m.insert("label".into(), Value::from(l.as_str()));
        }
    }
    if o.neg {
        m.insert("neg".into(), Value::Bool(true));
    }
    if o.abs {
        m.insert("abs".into(), Value::Bool(true));
    }
    Value::Object(m)
}

/// A decoded instruction's fields in the golden' encoding.
pub fn fields_json(fields: &[(String, thin_emu::Field)]) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in fields {
        let v = match v {
            thin_emu::Field::True => Value::Bool(true),
            thin_emu::Field::Int(i) => Value::from(*i as i64),
            thin_emu::Field::List(l) => l.iter().map(|x| Value::from(*x as i64)).collect(),
        };
        m.insert(k.clone(), v);
    }
    Value::Object(m)
}

/// A decoded instruction from the emu golden' encoding,
/// rescaled to a W = 32 lane mask when `wave` is 32.
pub fn decoded_ins(ins: &Value, wave: u32) -> thin_emu::Ins {
    thin_emu::Ins::of(ins, wave as usize)
}

/// A decoded instruction in `reading.json`'s row notation: [mnemonic,
/// operands, fields, line, text, comment] (the mnemonic is the text's
/// first word).
pub fn reading_row(ins: &thin_emu::Ins) -> Value {
    let mnemonic = ins.text.split(' ').next().unwrap_or("");
    serde_json::json!([
        mnemonic,
        ins.operands.iter().map(operand_json).collect::<Vec<_>>(),
        fields_json(&ins.fields),
        ins.line,
        ins.text,
        ins.comment,
    ])
}

/// The gfx1201 target.
pub fn default_target() -> &'static thin_target::Target {
    thin_targets::get(thin_targets::default_name()).expect("default target is registered")
}
