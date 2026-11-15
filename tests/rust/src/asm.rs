//! Assembly construction and snapshot helpers shared by integration tests.

use std::sync::Arc;

use thin_ir::{Attr, FieldId, FieldValue, Imm, ImmValue, Instr, Operand, Reg, Snapshot};
use thin_target::{phys_named, FieldKind, Form, Pseudo, Target, WaitText};

/// The value numbers of the two files.
const V: i32 = 0;
const S: i32 = 1;

/// A free-list snapshot of parsed instructions with gathered wait requests.
pub fn snapshot(t: &'static Target, instrs: Vec<Arc<Instr>>) -> Snapshot {
    Snapshot::of_list(t, instrs.into()).expect("the lines' registers and markers")
}

/// The instructions of lines.
pub fn read(t: &'static Target, lines: &[&str]) -> Result<Vec<Arc<Instr>>, String> {
    lines.iter().map(|l| line(t, l).map(Arc::new)).collect()
}

/// An integer token: decimal or `0x` hex, signed.
fn int(t: &str) -> Option<i128> {
    let (neg, body) = t.strip_prefix('-').map_or((false, t), |b| (true, b));
    let v = match body.strip_prefix("0x") {
        Some(h) => i128::from_str_radix(h, 16).ok()?,
        None if body.bytes().all(|b| b.is_ascii_digit()) && !body.is_empty() => {
            body.parse().ok()?
        }
        None => return None,
    };
    Some(if neg { -v } else { v })
}

/// A register token: `v5`, `s[0:1]`.
pub fn register(t: &str) -> Option<Reg> {
    let (file, value) = match t.as_bytes().first()? {
        b'v' => ('v', V),
        b's' => ('s', S),
        _ => return None,
    };
    let rest = &t[1..];
    let (lo, hi) = match rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        Some(r) => r.split_once(':')?,
        None => (rest, rest),
    };
    let (lo, hi): (u32, u32) = (lo.parse().ok()?, hi.parse().ok()?);
    Some(Reg::new(value, file, lo, hi - lo + 1))
}

/// How a line's register tokens resolve: to a register, or none for the
/// physical reading.
pub type Resolve<'r> = &'r dyn Fn(&str) -> Option<Reg>;

/// An operand token and its modifier byte.
fn operand(t: &str) -> (Operand, u8) {
    operand_with(t, &register)
}

/// An operand token and its modifier byte, registers by `resolve`.
fn operand_with(t: &str, resolve: Resolve<'_>) -> (Operand, u8) {
    let (neg, rest) = match t.strip_prefix('-') {
        Some(r) if int(t).is_none() && t.parse::<f64>().is_err() => (1, r),
        _ => (0, t),
    };
    let (abs, rest) = match rest.strip_prefix('|').and_then(|r| r.strip_suffix('|')) {
        Some(r) => (2, r),
        None => (0, rest),
    };
    let op = if let Some(r) = resolve(rest) {
        Operand::Reg(r)
    } else if let Some(p) = phys_named(rest) {
        Operand::Phys(p)
    } else if let Some(v) = int(rest) {
        Operand::Imm(Imm::int(v))
    } else if let Ok(x) = rest.parse::<f64>() {
        Operand::Imm(Imm {
            value: ImmValue::Float(x),
            text: "".into(),
        })
    } else {
        Operand::Block(rest.into())
    };
    (op, neg | abs)
}

/// Splits at top-level commas.
fn pieces(rest: &str) -> Vec<&str> {
    let (mut depth, mut start, mut out) = (0, 0, Vec::new());
    for (i, c) in rest.char_indices() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(rest[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(rest[start..].trim());
    out.into_iter().filter(|p| !p.is_empty()).collect()
}

/// A field token of a form.
fn field(form: &Form, token: &str) -> Result<(FieldId, FieldValue), String> {
    let (key, value) = token.split_once(':').unwrap_or((token, ""));
    let id = |name: &str| FieldId::named(name).map_err(|e| e.to_string());
    let Some(f) = form.field(key) else {
        if form.field("dpp").is_some() && FieldId::named(key).is_err() {
            return Ok((id("dpp")?, FieldValue::Text(token.into())));
        }
        return Ok((id(key)?, undeclared(value)));
    };
    let v = match f.kind {
        FieldKind::Flag => FieldValue::Bool(true),
        FieldKind::Int => {
            FieldValue::Int(int(value).ok_or_else(|| format!("{token}: not an int"))?)
        }
        FieldKind::Bits => {
            let inner = value.trim_start_matches('[').trim_end_matches(']');
            let ints = inner
                .split(',')
                .map(int)
                .collect::<Option<Vec<_>>>()
                .ok_or("a bit list")?;
            FieldValue::Ints(ints.into())
        }
        FieldKind::Dpp => FieldValue::Text(token.into()),
        _ => FieldValue::Text(value.into()),
    };
    Ok((id(key)?, v))
}

/// The value of a field the form does not declare: an int, a list of
/// ints, a flag when bare, else a text.
fn undeclared(value: &str) -> FieldValue {
    let list = value.strip_prefix('[').and_then(|v| v.strip_suffix(']'));
    if let Some(ints) = list.and_then(|l| l.split(',').map(int).collect::<Option<Vec<_>>>()) {
        return FieldValue::Ints(ints.into());
    }
    match (value, int(value)) {
        ("", _) => FieldValue::Bool(true),
        (_, Some(v)) => FieldValue::Int(v),
        _ => FieldValue::Text(value.into()),
    }
}

/// A count-less wait's fields from its text, if the text is its counts.
fn counts(form: &Form, rest: &str) -> Option<Vec<(FieldId, FieldValue)>> {
    let int_field = |name: &str, v: i128| {
        (
            FieldId::named(name).expect("a count field"),
            FieldValue::Int(v),
        )
    };
    match form.wait_text? {
        WaitText::Bare => Some(vec![int_field(form.fields.first()?.name, int(rest)?)]),
        WaitText::Named => rest
            .split_whitespace()
            .map(|p| {
                let (name, n) = p.strip_suffix(')')?.split_once('(')?;
                Some(int_field(name, int(n)?))
            })
            .collect(),
    }
}

/// The instruction of one line.
pub fn line(t: &'static Target, text: &str) -> Result<Instr, String> {
    line_with(t, text, &register)
}

/// The instruction of one line, its registers by `resolve`.
pub fn line_with(t: &'static Target, text: &str, resolve: Resolve<'_>) -> Result<Instr, String> {
    let text = text.trim();
    if let Some(marker) = text.strip_prefix("@wait ") {
        return wait_marker(t, marker);
    }
    if let Some(claim) = text.strip_prefix("; ").and_then(|c| own_op(t, c, resolve)) {
        return claim;
    }
    if let Some(comment) = text.strip_prefix(';') {
        let field = (
            FieldId::named("text").unwrap(),
            FieldValue::Text(format!(";{comment}").into()),
        );
        return Ok(Instr::new(
            t.pseudo(Pseudo::Comment),
            Vec::new(),
            vec![field],
            None,
        ));
    }
    if let Some(name) = text.strip_suffix(':') {
        let field = (
            FieldId::named("name").unwrap(),
            FieldValue::Text(name.into()),
        );
        return Ok(Instr::new(
            t.pseudo(Pseudo::Label),
            Vec::new(),
            vec![field],
            None,
        ));
    }
    let (body, comment) = match text.split_once("//") {
        Some((b, c)) => (b.trim(), Some(c.trim())),
        None => (text, None),
    };
    let (mnemonic, rest) = body
        .split_once(' ')
        .map_or((body, ""), |(m, r)| (m, r.trim()));
    let countless = t
        .form(mnemonic, 0, false)
        .and_then(|f| Some((f, counts(f, rest)?)));
    let mut instr = if let Some((form, fields)) = countless.filter(|_| !rest.is_empty()) {
        Instr::new(form, Vec::new(), fields, None)
    } else {
        operands_line(t, mnemonic, rest, resolve)?
    };
    if let Some(c) = comment {
        instr = instr
            .with_attr(Attr::Comment(c.into()))
            .map_err(|e| e.to_string())?;
    }
    Ok(instr)
}

/// A register an own op's line names by its repr, `v:x[0]x1+1`: read as
/// the dwords from register 0 that the repr's `+lo` and `xn` give.
fn repr_register(t: &str) -> Option<Reg> {
    let (file, rest) = t.split_once(':')?;
    let (_, size) = rest.split_once("]x")?;
    let (n, lo) = size.split_once('+').unwrap_or((size, "0"));
    let value = if file == "v" { V } else { S };
    Some(Reg::new(
        value,
        file.chars().next()?,
        lo.parse().ok()?,
        n.parse().ok()?,
    ))
}

/// A claim's line as its own op: `discard OPS: WHY`,
/// `assume: OPS in [LO, HI] (WHY)`, `uniform: OPS (WHY)`, `undefined: OPS
/// (WHY)`; none for another comment.
fn own_op(t: &'static Target, claim: &str, resolve: Resolve<'_>) -> Option<Result<Instr, String>> {
    let text = |k: &str, v: &str| {
        (
            FieldId::named(k).expect("a field"),
            FieldValue::Text(v.into()),
        )
    };
    let ops = |list: &str| {
        list.split(", ")
            .map(|r| repr_register(r).or_else(|| resolve(r)).map(Operand::Reg))
            .collect::<Option<Vec<_>>>()
    };
    let (kind, ops, fields) = if let Some(rest) = claim.strip_prefix("discard ") {
        let (list, why) = rest.split_once(": ")?;
        (Pseudo::Discard, ops(list)?, vec![text("why", why)])
    } else if let Some(rest) = claim.strip_prefix("assume: ") {
        let (list, rest) = rest.split_once(" in [")?;
        let (bounds, why) = rest.split_once("] (")?;
        let (lo, hi) = bounds.split_once(", ")?;
        let int_field =
            |k: &str, v: &str| Some((FieldId::named(k).ok()?, FieldValue::Int(v.parse().ok()?)));
        let fields = vec![
            int_field("lo", lo)?,
            int_field("hi", hi)?,
            text("why", why.strip_suffix(')')?),
        ];
        (Pseudo::Assume, ops(list)?, fields)
    } else {
        let (kind, rest) = (claim
            .strip_prefix("uniform: ")
            .map(|r| (Pseudo::Uniform, r)))
        .or_else(|| {
            claim
                .strip_prefix("undefined: ")
                .map(|r| (Pseudo::Undefined, r))
        })?;
        let (list, why) = rest.rsplit_once(" (")?;
        (kind, ops(list)?, vec![text("why", why.strip_suffix(')')?)])
    };
    Some(Ok(Instr::new(t.pseudo(kind), ops, fields, None)))
}

/// A wait marker, `@wait <counter> <kind> <args>`: `regs R…`, `all`,
/// `left N`, `smem` or `tags T…`.
fn wait_marker(t: &'static Target, text: &str) -> Result<Instr, String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let [counter, kind, args @ ..] = words.as_slice() else {
        return Err(format!("@wait {text}: a counter and a kind"));
    };
    let id = |k: &str| FieldId::named(k).expect("a wait field");
    let mut fields = vec![(id("counter"), FieldValue::Text((*counter).into()))];
    let mut ops = Vec::new();
    let pseudo = match *kind {
        "regs" => {
            ops = args.iter().map(|a| operand(a).0).collect();
            Pseudo::WaitRegs
        }
        "all" => Pseudo::WaitAll,
        "smem" => Pseudo::WaitSmem,
        "left" => {
            let n = args.first().and_then(|a| int(a)).ok_or("@wait left N")?;
            fields.push((id("left"), FieldValue::Int(n)));
            Pseudo::WaitLeft
        }
        "tags" => {
            let tags: Vec<thin_ir::Tag> = args
                .iter()
                .map(|a| vec![thin_ir::Atom::Str((*a).into())].into())
                .collect();
            fields.push((id("tags"), FieldValue::Tags(tags.into())));
            Pseudo::WaitTags
        }
        other => return Err(format!("@wait: no kind {other}")),
    };
    Ok(Instr::new(t.pseudo(pseudo), ops, fields, None))
}

/// A line of operands and trailing fields.
fn operands_line(
    t: &'static Target,
    mnemonic: &str,
    rest: &str,
    resolve: Resolve<'_>,
) -> Result<Instr, String> {
    let mut tokens: Vec<&str> = pieces(rest);
    let mut trailing: Vec<&str> = Vec::new();
    if let Some(last) = tokens.pop() {
        let mut words = last.split_whitespace();
        tokens.extend(words.next());
        trailing = words.collect();
    }
    let (ops, mods): (Vec<Operand>, Vec<u8>) =
        tokens.iter().map(|t| operand_with(t, resolve)).unzip();
    let form = (t.form(mnemonic, ops.len(), false))
        .ok_or_else(|| format!("no form {mnemonic}/{}", ops.len()))?;
    let fields = trailing
        .iter()
        .map(|tok| field(form, tok))
        .collect::<Result<Vec<_>, _>>()?;
    let mods = mods
        .iter()
        .any(|&m| m != 0)
        .then(|| mods.into_boxed_slice());
    Ok(Instr::new(form, ops, fields, mods))
}

/// An instruction's line with physical registers.
pub fn spell(s: &Snapshot, i: &Instr) -> String {
    let regs: Vec<String> = i
        .ops()
        .iter()
        .filter_map(Operand::reg)
        .map(|r| s.reg_text(r))
        .collect();
    thin_ir::print::line(i, &regs).expect("a well-formed line")
}

/// A native message spelled as the golden write it: instructions as
/// their lines, no source locations.
pub fn message(s: &Snapshot, parts: &[thin_ir::Part]) -> String {
    thin_ir::Diag::spell(parts, &|k| spell(s, &s.instrs()[k]), None)
}
