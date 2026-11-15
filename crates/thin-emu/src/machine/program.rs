//! Reading a decoded program off a `.s`:
//! reader, then the target's decoding map. The emulator has no
//! reader of its own.

use thin_target::{
    parse_line, read_s, CodeItem, ModValue, OperandKind, Target, Value as LineValue,
};

use super::facts::Facts;
use crate::{Base, EmuError, Field, Ins, Operand, Program};

/// Python's `repr` of a string (single quotes unless it holds one and no
/// double quote), for messages that quote a line.
pub fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

fn reading_error(message: impl Into<String>) -> EmuError {
    EmuError {
        cause: None,
        message: message.into(),
    }
}

/// The lines of a text as Python's `splitlines(keepends=True)` gives them.
fn raw_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let end = match b[i] {
            b'\n' => Some(i + 1),
            b'\r' if b.get(i + 1) == Some(&b'\n') => Some(i + 2),
            b'\r' | 0x0B | 0x0C | 0x1C..=0x1E => Some(i + 1),
            _ => None,
        };
        if let Some(e) = end {
            out.push(&text[start..e]);
            start = e;
            i = e;
        } else {
            i += 1;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// One operand of the line reader through the decoding map.
fn operand(facts: &Facts, op: &thin_target::Operand) -> Result<Operand, String> {
    let (neg, abs) = (op.neg, op.abs);
    let base = match op.kind {
        OperandKind::V | OperandKind::S | OperandKind::A => match &op.name {
            Some(name) => match facts.named_reg(name) {
                Some((file, part, count)) => Base::Reg(file.to_owned(), part, count),
                None => {
                    return Err(format!(
                    "operand not understood: {} (a named register the emulator has no model of)",
                    py_repr(name)
                ))
                }
            },
            None => {
                let file = match op.kind {
                    OperandKind::V => "v",
                    OperandKind::S => "s",
                    _ => "a",
                };
                Base::Reg(file.to_owned(), op.n, op.count)
            }
        },
        OperandKind::Off => Base::Reg("off".to_owned(), 0, 1),
        OperandKind::Int | OperandKind::Simm => {
            let v = op.int().unwrap_or(0);
            let text = op
                .text
                .trim_start_matches('|')
                .trim_end_matches('|')
                .to_owned();
            return Ok(Operand {
                base: Base::Literal(v.rem_euclid(1 << 32) as u32, text),
                neg,
                abs: false,
            });
        }
        OperandKind::Float => {
            let v = match op.value {
                Some(LineValue::Float(f)) => f,
                _ => 0.0,
            };
            return Ok(Operand {
                base: Base::Literal((v as f32).to_bits(), op.text.clone()),
                neg,
                abs: false,
            });
        }
        OperandKind::Label => {
            let name = match &op.value {
                Some(LineValue::Label(l)) => l.clone(),
                _ => op.text.clone(),
            };
            Base::Label(name)
        }
    };
    Ok(Operand { base, neg, abs })
}

/// The decoded instruction of one instruction item.
fn decode(
    target: &Target,
    facts: &Facts,
    text: &str,
    comment: &str,
    line: usize,
    raw: &str,
) -> Result<Ins, EmuError> {
    let parsed = parse_line(target.hooks(), text)
        .map_err(|m| reading_error(format!("line {line}: {m} in {}", py_repr(raw))))?;
    let m = parsed.mnemonic.as_str();
    let stripped = m
        .strip_suffix("_e32")
        .or_else(|| m.strip_suffix("_e64"))
        .unwrap_or(m);
    let mut operands = Vec::with_capacity(parsed.operands.len());
    for o in &parsed.operands {
        operands.push(operand(facts, o).map_err(reading_error)?);
    }
    let n = parsed.operands.len();
    let form = target
        .form(m, n, false)
        .or_else(|| target.form(m, n, true))
        .or_else(|| target.named(m));
    let kind = match form {
        Some(f) => f.kind.name().to_owned(),
        None => match target.unit_of(m) {
            Some(flags) if flags.has(thin_target::Flags::SALU) => "salu".to_owned(),
            Some(flags) if flags.has(thin_target::Flags::BRANCH) => "branch".to_owned(),
            Some(flags) if flags.has(thin_target::Flags::SMEM) => "smem".to_owned(),
            _ if m.starts_with("s_") => "salu".to_owned(),
            _ => "valu".to_owned(),
        },
    };
    let matrix = target
        .hooks()
        .matrix_form(m)
        .is_some()
        .then(|| m.to_owned());
    let is_wait = form.is_some_and(|f| f.is_wait()) || kind == "wait";
    let mut fields = Vec::new();
    for (key, value) in parsed.modifiers.iter() {
        if is_wait && key == "simm16" {
            continue;
        }
        let v = match value {
            ModValue::Int(1) if target.hooks().mod_flag(key) => Field::True,
            ModValue::Int(i) => Field::Int(*i),
            ModValue::List(l) => Field::List(l.clone()),
        };
        fields.push((key.to_owned(), v));
    }
    Ok(Ins {
        canon: target.canon(stripped),
        kind,
        operands,
        fields,
        matrix,
        text: text.to_owned(),
        comment: comment.to_owned(),
        line,
    })
}

/// Reads a `.s` with `read_s(text, body = true)` and the target's decoding
/// map.
pub fn read_program(text: &str, target: &Target) -> Result<Program, EmuError> {
    let facts = Facts::of(target);
    let file = read_s(text, true).map_err(|e| reading_error(e.to_string()))?;
    let raw = raw_lines(text);
    let mut program = Program {
        user_sgprs: 2,
        ..Program::default()
    };
    for item in &file.code {
        let n = program.ins.len();
        match item {
            CodeItem::Note { text, line } => program.sections.push((n, text.clone(), *line)),
            CodeItem::Label { name, line } => {
                program.label_lines.push((n, *line));
                let name = name.to_owned();
                match program.labels.iter_mut().find(|(l, _)| *l == name) {
                    Some(entry) => entry.1 = n,
                    None => program.labels.push((name, n)),
                }
            }
            CodeItem::Align { .. } => {}
            CodeItem::Ins {
                text,
                comment,
                line,
            } => {
                let raw_line = raw.get(line.saturating_sub(1)).copied().unwrap_or("");
                program
                    .ins
                    .push(decode(target, &facts, text, comment, *line, raw_line)?);
            }
        }
    }
    if let Some((v, _)) = file.block.get("user_sgpr_count") {
        if let Some(n) = thin_target::int_value(v) {
            program.user_sgprs = n as u32;
        }
    }
    if !file.block.is_empty() {
        let flag = |key: &str| -> Result<bool, EmuError> {
            match file.block.get(key) {
                Some((v, _)) => match thin_target::int_value(v) {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(reading_error(format!(".amdhsa_{key}: expected 0 or 1"))),
                },
                None => Ok(false),
            }
        };
        program.workgroup_ids = Some([
            flag("system_sgpr_workgroup_id_x")?,
            flag("system_sgpr_workgroup_id_y")?,
            flag("system_sgpr_workgroup_id_z")?,
        ]);
        if flag("user_sgpr_kernarg_segment_ptr")? {
            program.kernarg_sgpr = Some(
                2 * (u32::from(flag("user_sgpr_dispatch_ptr")?)
                    + u32::from(flag("user_sgpr_queue_ptr")?)),
            );
        }
    }
    if program.ins.iter().any(|i| super::pc::is_pc(&i.canon)) {
        let assembly = thin_target::assemble_with_layout(&file.code, target.hooks())
            .map_err(|e| reading_error(format!("PC byte layout: {e}")))?;
        program.pc_layout = Some(super::pc::Layout {
            ranges: assembly
                .instructions
                .iter()
                .map(|r| (r.start as u64, r.end as u64))
                .collect(),
            end: assembly.bytes.len() as u64,
        });
    }
    Ok(program)
}

/// The descriptor's user SGPR count of a `.s`.
pub fn user_sgprs(text: &str) -> Result<u32, EmuError> {
    let file = read_s(text, true).map_err(|e| reading_error(e.to_string()))?;
    Ok(file
        .block
        .get("user_sgpr_count")
        .and_then(|(v, _)| thin_target::int_value(v))
        .map_or(2, |n| n as u32))
}
