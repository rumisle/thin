//! Read golden IR notation: register values, labels, fields and abstract CFG streams.

use std::collections::HashMap;
use std::sync::Arc;

use thin_ir::{Attr, FieldId, FieldValue, Imm, Instr, Operand, Reg, ValueInfo};
use thin_target::{phys_named, Pseudo, Target};

/// Instructions read from gfx1201 lines, with one value table across them.
pub struct Lines {
    target: &'static Target,
    spellings: HashMap<String, i32>,
    /// The values, numbered by first appearance.
    pub values: Vec<ValueInfo>,
}

impl Lines {
    /// An empty table on `target`.
    pub fn new(target: &'static Target) -> Lines {
        Lines {
            target,
            spellings: HashMap::new(),
            values: Vec::new(),
        }
    }

    /// The instructions of lines, in order.
    pub fn read(&mut self, lines: &[&str]) -> Result<Vec<Arc<Instr>>, String> {
        lines.iter().map(|l| self.line(l).map(Arc::new)).collect()
    }

    /// The instruction of one line.
    pub fn line(&mut self, line: &str) -> Result<Instr, String> {
        let (body, attrs) = line.split_once(" ; ").unwrap_or((line, ""));
        if let Some(name) = body.strip_suffix(':') {
            let label = self.target.pseudo(Pseudo::Label);
            return Ok(Instr::new(
                label,
                Vec::new(),
                vec![(FieldId::named("name").unwrap(), text(name))],
                None,
            ));
        }
        let (body, fields) = match body.split_once(" {") {
            Some((b, f)) => (b, f.strip_suffix('}').ok_or("an unclosed {")?),
            None => (body, ""),
        };
        let (mnemonic, ops) = body.split_once(' ').unwrap_or((body, ""));
        let ops: Vec<Operand> = (ops.split(", ").filter(|o| !o.is_empty()))
            .map(|o| self.operand(o))
            .collect::<Result<_, _>>()?;
        let form = (self.target.form(mnemonic, ops.len(), false))
            .ok_or_else(|| format!("no form {mnemonic}/{}", ops.len()))?;
        let fields = (fields.split('|').filter(|f| !f.is_empty()))
            .map(|f| {
                let (k, v) = f.split_once('=').ok_or_else(|| format!("a field {f}"))?;
                Ok((
                    FieldId::named(k).map_err(|e| e.to_string())?,
                    field_value(v),
                ))
            })
            .collect::<Result<_, String>>()?;
        let mut instr = Instr::new(form, ops, fields, None);
        for attr in attrs.split(" ; ").filter(|a| !a.is_empty()) {
            instr = instr
                .with_attr(attribute(attr)?)
                .map_err(|e| e.to_string())?;
        }
        Ok(instr)
    }

    /// One operand.
    fn operand(&mut self, o: &str) -> Result<Operand, String> {
        if let Some(label) = o.strip_prefix('@') {
            return Ok(Operand::Block(label.into()));
        }
        if let Some(p) = phys_named(o) {
            return Ok(Operand::Phys(p));
        }
        if let Some(v) = int(o) {
            return Ok(Operand::Imm(Imm::int(v)));
        }
        let file = o.chars().next().ok_or("an empty operand")?;
        let range = &o[1..];
        let (lo, hi) = match range.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            Some(r) => r.split_once(':').ok_or_else(|| format!("a register {o}"))?,
            None => (range, range),
        };
        let (lo, hi): (u32, u32) = (
            lo.parse().map_err(|_| format!("a register {o}"))?,
            hi.parse().map_err(|_| format!("a register {o}"))?,
        );
        let n = hi - lo + 1;
        let next = self.values.len() as i32;
        let value = *self.spellings.entry(o.to_owned()).or_insert(next);
        if value == next {
            self.values.push(ValueInfo {
                file,
                n,
                phys: Some(lo),
                align: 1,
                ..ValueInfo::default()
            });
        }
        Ok(Operand::Reg(Reg::new(value, file, 0, n)))
    }
}

/// An integer in decimal or `0x` hex, signed.
fn int(o: &str) -> Option<i128> {
    let (neg, body) = o.strip_prefix('-').map_or((false, o), |b| (true, b));
    let v = match body.strip_prefix("0x") {
        Some(hex) => i128::from_str_radix(hex, 16).ok()?,
        None => body.parse().ok()?,
    };
    Some(if neg { -v } else { v })
}

/// A text field value.
fn text(t: &str) -> FieldValue {
    FieldValue::Text(t.into())
}

/// A field value: an int, true/false, else a text.
fn field_value(v: &str) -> FieldValue {
    match v {
        "true" => FieldValue::Bool(true),
        "false" => FieldValue::Bool(false),
        _ => int(v).map_or_else(|| text(v), FieldValue::Int),
    }
}

/// `m0_set=C` or `waves=0bMASK`.
fn attribute(a: &str) -> Result<Attr, String> {
    match a.split_once('=') {
        Some(("m0_set", c)) => Ok(Attr::M0Set(int(c).ok_or("an M0Set constant")?)),
        Some(("waves", m)) => {
            let mask =
                u64::from_str_radix(m.trim_start_matches("0b"), 2).map_err(|e| e.to_string())?;
            Ok(Attr::WavesTaken(
                (0..64)
                    .filter(|w| mask >> w & 1 == 1)
                    .map(i128::from)
                    .collect(),
            ))
        }
        _ => Err(format!("an attribute {a}")),
    }
}

/// The gfx1201 line an abstract stream item stands for.
pub fn abstract_line(item: &str) -> String {
    match item.split_once(' ') {
        Some(("br" | "brw", label)) => format!("s_cbranch_scc1 @{label}"),
        Some(("jmp", label)) => format!("s_branch @{label}"),
        _ if item == "end" => "s_endpgm".into(),
        _ if item == "op" => "v_mov_b32 v0, 0".into(),
        _ => item.to_owned(),
    }
}
