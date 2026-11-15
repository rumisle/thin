//! The printer: the one place an operand, a modifier or a field becomes
//! text, and register spelling.

use thin_target::{Pseudo, WaitText, LINE_FIELDS};

use crate::{pyfmt, Error, FieldValue, ImmValue, Instr, Operand};

/// The modifier bit for negation.
pub const NEG: u8 = 1;
/// The modifier bit for absolute value.
pub const ABS: u8 = 2;

/// The column a commented line's comment starts after.
const COMMENT_COLUMN: usize = 60;

/// An instruction's line without its comment; `regs[k]`
/// spells its k-th register operand. Only a malformed field (names or tags
/// in a line field) fails.
pub fn line(i: &Instr, regs: &[String]) -> Result<String, Error> {
    let form = i.form();
    let mut next = regs.iter();
    let ops: Vec<String> = (i.ops().iter().enumerate())
        .map(|(k, op)| operand(i, k, op, &mut next))
        .collect();
    let field = |name: &str| {
        i.field_named(name)
            .map_or(String::new(), FieldValue::py_str)
    };
    if let Some(p) = form.pseudo {
        let all = ops.join(", ");
        return Ok(match p {
            Pseudo::Label => format!("{}:", field("name")),
            Pseudo::Comment => field("text"),
            Pseudo::Discard => format!("; discard {all}: {}", field("why")),
            Pseudo::Uniform => format!("; uniform: {all} ({})", field("why")),
            Pseudo::Undefined => match field("on").as_str() {
                // an edge claim: a comment only
                on @ ("taken" | "fall") => format!("; undefined on {on}: {all} ({})", field("why")),
                _ => format!("; undefined: {all} ({})", field("why")),
            },
            Pseudo::Assume => format!(
                "; assume: {all} in [{}, {}] ({})",
                field("lo"),
                field("hi"),
                field("why")
            ),
            _ => "; wait request".into(),
        });
    }
    let mut out = form.printed().to_owned();
    if let Some(text) = form.wait_text {
        for f in &form.fields {
            let Some(v) = i.field_named(f.name) else {
                continue;
            };
            match text {
                WaitText::Named => out.push_str(&format!(" {}({})", f.name, v.py_str())),
                WaitText::Bare => out.push_str(&format!(" {}", v.py_str())),
            }
        }
        return Ok(out);
    }
    if !ops.is_empty() {
        out.push(' ');
        out.push_str(&ops.join(", "));
    }
    let mut fields: Vec<_> = i
        .fields()
        .iter()
        .filter(|(id, v)| (id.0 as usize) < LINE_FIELDS && !v.is_none())
        .collect();
    fields.sort_by_key(|(id, _)| *id);
    for (id, v) in fields {
        if let Some(text) = suffix(id.name(), v)? {
            out.push(' ');
            out.push_str(&text);
        }
    }
    Ok(out)
}

/// Operand k's spelling, with its modifiers; registers take the next of
/// `regs`.
fn operand<'a>(
    i: &Instr,
    k: usize,
    op: &Operand,
    regs: &mut impl Iterator<Item = &'a String>,
) -> String {
    let text = match op {
        Operand::Reg(_) => regs.next().cloned().unwrap_or_default(),
        Operand::Phys(p) => p.name.to_owned(),
        Operand::Block(name) => name.to_string(),
        Operand::Imm(imm) if !imm.text.is_empty() => imm.text.to_string(),
        Operand::Imm(imm) => match imm.value {
            ImmValue::Int(v) => v.to_string(),
            ImmValue::Float(x) => pyfmt::repr_float(x),
        },
    };
    let m = i.mods().and_then(|m| m.get(k).copied()).unwrap_or(0);
    let neg = if m & NEG != 0 { "-" } else { "" };
    if m & ABS != 0 {
        format!("{neg}|{text}|")
    } else {
        format!("{neg}{text}")
    }
}

/// A line field's spelling, or none: a flag holding
/// false is not spelled.
fn suffix(key: &str, v: &FieldValue) -> Result<Option<String>, Error> {
    Ok(Some(match (key, v) {
        (_, FieldValue::Bool(false)) => return Ok(None),
        ("dpp", v) => v.py_str(),
        // The bound control bit, as llvm-mc spells it on gfx12.
        ("bound_ctrl", FieldValue::Bool(true)) => "bound_ctrl:1".into(),
        ("row_mask" | "bank_mask", FieldValue::Int(m)) => format!("{key}:{m:#x}"),
        (_, FieldValue::Bool(true)) => key.into(),
        (_, FieldValue::Ints(v)) => format!("{key}:[{}]", join(v.iter().map(i128::to_string))),
        (_, FieldValue::Names(_) | FieldValue::Tags(_)) => {
            return Err(Error::Type(format!(
                "{key}: a line field holds an int, a flag or ints, not names"
            )))
        }
        (_, v) => format!("{key}:{}", v.py_str()),
    }))
}

/// Items joined by commas with no spaces.
fn join(items: impl Iterator<Item = String>) -> String {
    items.collect::<Vec<_>>().join(",")
}

/// A physical register: `<f><p>` for one dword, else `<f>[<p>:<p+n−1>]`.
pub fn phys_text(file: char, p: u32, n: u32) -> String {
    if n == 1 {
        format!("{file}{p}")
    } else {
        format!("{file}[{p}:{}]", p + n - 1)
    }
}

/// A register operand spelled physically once its value is placed at
/// `base`: an SGPR range must start at a multiple of 2
/// (a pair) or 4 (wider), and the range must lie in the value's `value_n`
/// dwords; `repr` is the operand's repr, for the messages.
pub fn placed_text(
    repr: &str,
    file: char,
    base: u32,
    lo: u32,
    n: u32,
    value_n: u32,
) -> Result<String, Error> {
    let p = base + lo;
    let align = if n == 2 { 2 } else { 4 };
    if file == 's' && n > 1 && p % align != 0 {
        let what = if n == 2 { "pair" } else { "tuple" };
        return Err(Error::Value(format!(
            "{repr} at s{p}: an SGPR {what} starts at a multiple of {align} (allocate it aligned)"
        )));
    }
    if lo + n > value_n {
        return Err(Error::Value(format!(
            "{repr}: sub {lo} + width {n} exceeds declared width {value_n}"
        )));
    }
    Ok(phys_text(file, p, n))
}

/// What the by-name spelling reads of a register operand and its value.
#[derive(Clone, Copy, Debug)]
pub struct Named<'a> {
    /// The file letter.
    pub file: char,
    /// The value's name.
    pub name: &'a str,
    /// The value's position in its group.
    pub index: u32,
    /// How many values the group has.
    pub group: usize,
    /// The first dword.
    pub lo: u32,
    /// The dwords.
    pub n: u32,
    /// The value's dwords.
    pub value_n: u32,
    /// `short(loc)` of the value's loc, when known.
    pub short: Option<&'a str>,
}

/// A register operand by name, placement-independent:
/// `v:x#1`, `s:p#1x2`, `v:z#1+0x2`, `v:q[1]`, `v:t#1@k.py:50`.
pub fn name_text(r: &Named<'_>) -> String {
    let mut out = format!("{}:{}", r.file, r.name);
    if r.group > 1 {
        out.push_str(&format!("[{}]", r.index));
    }
    if r.lo != 0 || r.n < r.value_n {
        out.push_str(&format!("+{}", r.lo));
    }
    if r.n > 1 {
        out.push_str(&format!("x{}", r.n));
    }
    if let (true, Some(short)) = (r.name.starts_with("t#"), r.short) {
        out.push_str(&format!("@{short}"));
    }
    out
}

/// The .s line of an instruction that is not its own line: tab-indented,
/// with a comment after column 60.
pub fn commented(line: &str, comment: Option<&str>) -> String {
    match comment {
        Some(c) => format!("\t{line:<COMMENT_COLUMN$} // {c}"),
        None => format!("\t{line}"),
    }
}

/// The line with each label serial masked: `.L<w>_<digits>` at the end of
/// a word becomes `.L<w>_#`.
pub fn mask_serials(line: &str) -> String {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find(".L") {
        out.push_str(&rest[..at + 2]);
        rest = &rest[at + 2..];
        let end = rest.find(|c| !word(c)).unwrap_or(rest.len());
        let w = &rest[..end];
        match w.rfind('_') {
            Some(u)
                if u > 0 && u + 1 < w.len() && w[u + 1..].bytes().all(|b| b.is_ascii_digit()) =>
            {
                out.push_str(&w[..u]);
                out.push_str("_#");
                rest = &rest[end..];
            }
            _ => {}
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serials_are_masked_at_word_ends() {
        assert_eq!(mask_serials(".Lloop_12:"), ".Lloop_#:");
        assert_eq!(mask_serials("s_branch .Lend_3_4"), "s_branch .Lend_3_#");
        assert_eq!(mask_serials(".Lx_1a .L_5 .Lab_ ok"), ".Lx_1a .L_5 .Lab_ ok");
        assert_eq!(mask_serials("a.Lp_7,.Lq_8"), "a.Lp_#,.Lq_#");
    }

    #[test]
    fn registers_spell_by_name() {
        let r = Named {
            file: 'v',
            name: "z#1",
            index: 0,
            group: 1,
            lo: 0,
            n: 2,
            value_n: 4,
            short: Some("k.py:5"),
        };
        assert_eq!(name_text(&r), "v:z#1+0x2");
        assert_eq!(
            name_text(&Named {
                name: "t#1",
                n: 4,
                ..r
            }),
            "v:t#1x4@k.py:5"
        );
        assert_eq!(
            name_text(&Named {
                name: "q",
                group: 3,
                index: 1,
                n: 4,
                ..r
            }),
            "v:q[1]x4"
        );
    }

    #[test]
    fn placed_sgpr_ranges_must_be_aligned() {
        assert_eq!(placed_text("r", 's', 4, 0, 2, 2).unwrap(), "s[4:5]");
        let Err(Error::Value(msg)) = placed_text("r", 's', 5, 0, 2, 2) else {
            panic!()
        };
        assert_eq!(
            msg,
            "r at s5: an SGPR pair starts at a multiple of 2 (allocate it aligned)"
        );
        let Err(Error::Value(msg)) = placed_text("r", 's', 6, 0, 4, 4) else {
            panic!()
        };
        assert_eq!(
            msg,
            "r at s6: an SGPR tuple starts at a multiple of 4 (allocate it aligned)"
        );
        let Err(Error::Value(msg)) = placed_text("r", 'v', 0, 3, 2, 4) else {
            panic!()
        };
        assert_eq!(msg, "r: sub 3 + width 2 exceeds declared width 4");
    }

    #[test]
    fn comments_start_after_column_60() {
        let line = commented("v_mov_b32 v0, v1", Some("x"));
        assert_eq!(line.find("//"), Some(62));
        assert_eq!(
            commented(&"y".repeat(70), Some("c")),
            format!("\t{} // c", "y".repeat(70))
        );
    }
}
