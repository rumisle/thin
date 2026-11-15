//! The line syntax every encoder reads.

use std::cell::Cell;

/// What a line encoder reads from its target.
pub trait Syntax {
    /// Numbered registers of file `v`, `s` or `a`; 0 for one it lacks.
    fn reg_count(&self, file: char) -> u32;
    /// A named register of the scalar source space: (number, dwords).
    fn special(&self, name: &str) -> Option<(u32, u32)>;
    /// The hardware-register ids of `hwreg(...)`, each with its code.
    fn hwreg(&self) -> &[(&'static str, u32)];
    /// Whether a modifier is written bare.
    fn mod_flag(&self, token: &str) -> bool;
    /// For a wait line written with named counts: each name and its largest value.
    fn counted(&self, mnemonic: &str) -> Option<Vec<(&'static str, u64)>>;
    /// The value a modifier spells by name.
    fn symbol(&self, modifier: &str, name: &str) -> Option<i128>;
}

/// Whether text is integer text: `-?(0x[0-9a-fA-F]+|[0-9]+)`.
pub fn int_text(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    match s.strip_prefix("0x") {
        Some(hex) => !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()),
        None => !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()),
    }
}

/// Whether text is float text: `-?([0-9]+\.[0-9]*|\.[0-9]+)(e[-+]?[0-9]+)?`.
pub fn float_text(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    let digits = |t: &str| t.bytes().all(|b| b.is_ascii_digit());
    let (mantissa, exponent) = match s.split_once('e') {
        Some((m, e)) => (m, Some(e)),
        None => (s, None),
    };
    let mantissa_ok = mantissa
        .split_once('.')
        .is_some_and(|(a, b)| digits(a) && digits(b) && !(a.is_empty() && b.is_empty()));
    let exponent_ok = exponent.is_none_or(|e| {
        let e = e.strip_prefix(['-', '+']).unwrap_or(e);
        !e.is_empty() && digits(e)
    });
    mantissa_ok && exponent_ok
}

/// Whether text is label text.
pub fn label_text(s: &str) -> bool {
    let word = |t: &str| t.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if let Some(rest) = s.strip_prefix('.') {
        return rest.strip_prefix('L').is_some_and(word);
    }
    s.strip_prefix('L').is_some_and(word)
        || s.bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
            && word(s)
}

/// Register text's file, first register and last: `v5`, `s[2:3]`.
pub fn reg_text(s: &str) -> Option<(char, String, String)> {
    let file = s.chars().next().filter(|c| matches!(c, 'v' | 's' | 'a'))?;
    let rest = &s[1..];
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    if digits(rest) {
        return Some((file, rest.to_owned(), rest.to_owned()));
    }
    let (a, b) = rest.strip_prefix('[')?.strip_suffix(']')?.split_once(':')?;
    (digits(a) && digits(b)).then(|| (file, a.to_owned(), b.to_owned()))
}

/// Python's `int(s, 0)`, as restricts it, within a signed 128-bit integer.
pub fn int_value(s: &str) -> Option<i128> {
    let s = s.trim();
    let (negative, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let lower = s.get(..2).map(str::to_ascii_lowercase);
    let (radix, digits) = match lower.as_deref() {
        Some("0x") => (16, &s[2..]),
        Some("0o") => (8, &s[2..]),
        Some("0b") => (2, &s[2..]),
        _ => (10, s),
    };
    let digits = if radix != 10 {
        digits.strip_prefix('_').unwrap_or(digits)
    } else {
        digits
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    if radix == 10 && clean.len() > 1 && clean.starts_with('0') && clean.bytes().any(|b| b != b'0')
    {
        return None;
    }
    let magnitude = u128::from_str_radix(&clean, radix).ok()?;
    if negative {
        (magnitude <= 1 << 127).then(|| (magnitude as i128).wrapping_neg())
    } else {
        i128::try_from(magnitude).ok()
    }
}

/// An operand's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandKind {
    /// A vector register.
    V,
    /// A scalar register, or a named one of the scalar source space.
    S,
    /// An accumulator register.
    A,
    /// An integer.
    Int,
    /// A float.
    Float,
    /// `off`.
    Off,
    /// A label.
    Label,
    /// An immediate a function spells (`gpr_idx(...)`, `hwreg(...)`).
    Simm,
}

impl OperandKind {
    /// The memory kind's stable JSON name.
    pub fn name(self) -> &'static str {
        match self {
            OperandKind::V => "v",
            OperandKind::S => "s",
            OperandKind::A => "a",
            OperandKind::Int => "int",
            OperandKind::Float => "float",
            OperandKind::Off => "off",
            OperandKind::Label => "label",
            OperandKind::Simm => "simm",
        }
    }
}

/// An operand's value, for the kinds that have one.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// An integer (also a simm).
    Int(i128),
    /// A float.
    Float(f64),
    /// A label's name.
    Label(String),
}

/// A parsed operand.
#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    /// Its kind.
    pub kind: OperandKind,
    /// Its first register number.
    pub n: u32,
    /// Its register count.
    pub count: u32,
    /// Whether it is negated.
    pub neg: bool,
    /// Whether it has absolute value bars.
    pub abs: bool,
    /// Its value.
    pub value: Option<Value>,
    /// The token as written.
    pub text: String,
    /// A named register's name, so a reader can tell it from a numbered one.
    pub name: Option<String>,
}

impl Operand {
    fn of(kind: OperandKind, text: &str) -> Operand {
        Operand {
            kind,
            n: 0,
            count: 1,
            neg: false,
            abs: false,
            value: None,
            text: text.to_owned(),
            name: None,
        }
    }

    /// The integer value of an int or simm operand.
    pub fn int(&self) -> Option<i128> {
        match (self.kind, &self.value) {
            (OperandKind::Int | OperandKind::Simm, Some(Value::Int(v))) => Some(*v),
            _ => None,
        }
    }
}

fn hwreg(syntax: &dyn Syntax, t: &str) -> Result<i128, String> {
    let ids: Vec<&str> = syntax.hwreg().iter().map(|(id, _)| *id).collect();
    let refuse = || {
        format!(
            "operand not understood: `{t}` (hwreg ids: {})",
            ids.join(", ")
        )
    };
    let inner = t
        .strip_prefix("hwreg(")
        .and_then(|r| r.strip_suffix(')'))
        .ok_or_else(refuse)?;
    let parts: Vec<&str> = inner.split(',').map(|p| p.trim_start()).collect();
    let number = |p: &str| {
        p.parse::<u32>()
            .ok()
            .filter(|_| p.bytes().all(|b| b.is_ascii_digit()))
    };
    // A register by name, or by its id: a decimal number below 64, the
    // SIMM16 id field's six bits (as LLVM's syntax also takes `hwreg(26)`).
    let code = (syntax.hwreg().iter())
        .find(|(id, _)| *id == parts[0])
        .map(|(_, c)| *c)
        .or_else(|| number(parts[0]).filter(|&n| n < 64))
        .ok_or_else(refuse)?;
    let (off, size) = match parts.as_slice() {
        [_] => (0, 32),
        [_, off, size] => (
            number(off).ok_or_else(refuse)?,
            number(size).ok_or_else(refuse)?,
        ),
        _ => return Err(refuse()),
    };
    if off >= 32 || size < 1 || size > 32 - off {
        return Err(refuse());
    }
    Ok(i128::from(code | off << 6 | (size - 1) << 11))
}

fn gpr_idx(t: &str) -> Result<i128, String> {
    let inner = t
        .strip_prefix("gpr_idx(")
        .and_then(|r| r.strip_suffix(')'))
        .ok_or_else(|| format!("`{t}`: gpr_idx(...) ends with `)`"))?;
    let mut value = 0;
    for item in inner.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        value |= match item {
            "SRC0" => 1,
            "SRC1" => 2,
            "SRC2" => 4,
            "DST" => 8,
            _ => return Err(format!("gpr_idx `{t}`: SRC0, SRC1, SRC2, DST")),
        };
    }
    Ok(value)
}

/// Parses one operand token.
pub fn parse_operand(syntax: &dyn Syntax, token: &str) -> Result<Operand, String> {
    let mut t = token;
    let mut neg = false;
    if let Some(rest) = t.strip_prefix('-') {
        // A sign before a number is the number's own (`-1`, `-.5`); before
        // anything else it is the neg modifier.
        if !int_text(t) && !float_text(t) {
            if rest.starts_with('-') {
                return Err(format!("`{token}`: one sign at most (neg is written once)"));
            }
            neg = true;
            t = rest;
        }
    }
    let mut abs = false;
    if (t.len() >= 2 && t.starts_with('|') && t.ends_with('|')) || t == "|" {
        abs = true;
        t = t.trim_start_matches('|').trim_end_matches('|');
        if t.is_empty() {
            return Err(format!("operand not understood: `{token}`"));
        }
    }
    let with = |kind| Operand {
        neg,
        abs,
        ..Operand::of(kind, token)
    };
    if let Some((file, a, b)) = reg_text(t) {
        let past = || {
            let last = i64::from(syntax.reg_count(file)) - 1;
            format!("`{token}`: past {file}{last} (special registers by name only)")
        };
        let (Ok(a), Ok(b)) = (a.parse::<u64>(), b.parse::<u64>()) else {
            return Err(past());
        };
        if b < a {
            return Err(format!("register range `{token}` is empty"));
        }
        if a + (b - a + 1) > u64::from(syntax.reg_count(file)) {
            return Err(past());
        }
        let kind = match file {
            'v' => OperandKind::V,
            's' => OperandKind::S,
            _ => OperandKind::A,
        };
        return Ok(Operand {
            n: a as u32,
            count: (b - a + 1) as u32,
            ..with(kind)
        });
    }
    if let Some((n, dwords)) = syntax.special(t) {
        return Ok(Operand {
            n,
            count: dwords,
            name: Some(t.to_owned()),
            ..with(OperandKind::S)
        });
    }
    if t == "off" {
        return Ok(Operand::of(OperandKind::Off, token));
    }
    if t.starts_with("hwreg(") {
        return Ok(Operand {
            value: Some(Value::Int(hwreg(syntax, t)?)),
            ..Operand::of(OperandKind::Simm, token)
        });
    }
    if t.starts_with("gpr_idx(") {
        return Ok(Operand {
            value: Some(Value::Int(gpr_idx(t)?)),
            ..Operand::of(OperandKind::Simm, token)
        });
    }
    if int_text(t) {
        let v = int_value(t).ok_or_else(|| format!("`{token}`: not an integer"))?;
        return Ok(Operand {
            value: Some(Value::Int(v)),
            ..with(OperandKind::Int)
        });
    }
    if float_text(t) {
        let v: f64 = t
            .parse()
            .map_err(|_| format!("operand not understood: `{token}`"))?;
        return Ok(Operand {
            value: Some(Value::Float(v)),
            ..with(OperandKind::Float)
        });
    }
    if label_text(t) {
        if neg || abs {
            return Err(format!("`{token}`: a label takes no neg or abs"));
        }
        return Ok(Operand {
            value: Some(Value::Label(t.to_owned())),
            ..Operand::of(OperandKind::Label, token)
        });
    }
    Err(format!("operand not understood: `{token}`"))
}

/// A modifier's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModValue {
    /// An integer (1 for a bare flag).
    Int(i128),
    /// A bracketed list.
    List(Vec<i128>),
}

/// A line's modifiers in order, each remembering whether it was read.
#[derive(Debug, Default)]
pub struct Modifiers {
    items: Vec<(String, ModValue, Cell<bool>)>,
}

impl Modifiers {
    /// Appends a modifier, unread (a decoder builds lines with it).
    pub fn insert(&mut self, key: &str, value: ModValue) {
        self.items.push((key.to_owned(), value, Cell::new(false)));
    }

    fn has(&self, key: &str) -> bool {
        self.items.iter().any(|(k, _, _)| k == key)
    }

    /// The value of a modifier, marking it read.
    pub fn take(&self, key: &str) -> Option<&ModValue> {
        let (_, value, read) = self.items.iter().find(|(k, _, _)| k == key)?;
        read.set(true);
        Some(value)
    }

    /// The modifiers in order: (key, value).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ModValue)> {
        self.items.iter().map(|(k, v, _)| (k.as_str(), v))
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The first modifier no consumer read.
    pub fn unread(&self) -> Option<&str> {
        self.items
            .iter()
            .find(|(_, _, r)| !r.get())
            .map(|(k, _, _)| k.as_str())
    }
}

/// Modifiers are equal when their keys and values are, in order; what was
/// read is not compared.
impl PartialEq for Modifiers {
    fn eq(&self, other: &Modifiers) -> bool {
        self.iter().eq(other.iter())
    }
}

/// A parsed line.
#[derive(Debug, PartialEq)]
pub struct Line {
    /// The mnemonic.
    pub mnemonic: String,
    /// The operands.
    pub operands: Vec<Operand>,
    /// The modifiers.
    pub modifiers: Modifiers,
}

/// Splits at top-level commas, counting depth over brackets and parentheses.
fn top_level(rest: &str) -> Vec<&str> {
    let (mut depth, mut start, mut pieces) = (0i32, 0, Vec::new());
    for (i, c) in rest.char_indices() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                pieces.push(&rest[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    pieces.push(&rest[start..]);
    pieces
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

/// Splits on runs of whitespace, `&` and `,`, keeping an empty leading or
/// trailing piece.
fn count_pieces(rest: &str) -> Vec<&str> {
    let separator = |c: char| c.is_whitespace() || c == '&' || c == ',';
    let mut pieces = Vec::new();
    let mut piece_start = 0;
    let mut in_run = false;
    for (i, c) in rest.char_indices() {
        if separator(c) {
            if !in_run {
                pieces.push(&rest[piece_start..i]);
                in_run = true;
            }
        } else if in_run {
            piece_start = i;
            in_run = false;
        }
    }
    pieces.push(if in_run { "" } else { &rest[piece_start..] });
    pieces
}

fn counted_line(
    mnemonic: &str,
    rest: &str,
    counts: &[(&str, u64)],
    line: &mut Line,
) -> Result<(), String> {
    if int_text(rest) {
        let v = int_value(rest).ok_or_else(|| format!("{mnemonic}: `{rest}` not an integer"))?;
        line.modifiers.insert("simm16", ModValue::Int(v));
        return Ok(());
    }
    if rest.is_empty() {
        return Ok(());
    }
    let names: Vec<String> = counts.iter().map(|(n, _)| format!("{n}(n)")).collect();
    for piece in count_pieces(rest) {
        let parsed = piece
            .strip_suffix(')')
            .and_then(|p| p.split_once('('))
            .filter(|(name, num)| {
                counts.iter().any(|(n, _)| n == name) && int_text(num) && !num.starts_with('-')
            })
            .and_then(|(name, num)| Some((name, int_value(num)?)));
        let Some((name, value)) = parsed else {
            return Err(format!(
                "{mnemonic}: `{piece}` is not one of {}",
                names.join(", ")
            ));
        };
        if line.modifiers.has(name) {
            return Err(format!("{mnemonic}: {name} given twice"));
        }
        line.modifiers.insert(name, ModValue::Int(value));
    }
    Ok(())
}

fn modifier_value(
    syntax: &dyn Syntax,
    key: &str,
    value: &str,
    line: &str,
) -> Result<ModValue, String> {
    let refuse = || format!("{key}:{value}: not an integer (in `{line}`)");
    if let Some(v) = syntax.symbol(key, value) {
        return Ok(ModValue::Int(v));
    }
    if value.starts_with('[') {
        let inner = value.trim_start_matches('[').trim_end_matches(']');
        let items = inner.split(',').map(int_value).collect::<Option<Vec<_>>>();
        return items.map(ModValue::List).ok_or_else(refuse);
    }
    int_value(value).map(ModValue::Int).ok_or_else(refuse)
}

/// Whether a token is a modifier when it starts the last piece.
fn starts_modifiers(syntax: &dyn Syntax, token: &str) -> bool {
    let head = token.split_once(':').map(|(h, _)| h);
    let keyed = head.is_some_and(|h| {
        !h.is_empty()
            && h.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    });
    syntax.mod_flag(token) || (keyed && reg_text(token).is_none())
}

/// Parses a line into mnemonic, operands and modifiers.
pub fn parse_line(syntax: &dyn Syntax, text: &str) -> Result<Line, String> {
    let (mnemonic, rest) = match text.split_once(' ') {
        Some((m, r)) => (m, r.trim()),
        None => (text, ""),
    };
    let mut line = Line {
        mnemonic: mnemonic.to_owned(),
        operands: Vec::new(),
        modifiers: Modifiers::default(),
    };
    if let Some(counts) = syntax.counted(mnemonic) {
        counted_line(mnemonic, rest, &counts, &mut line)?;
        return Ok(line);
    }
    let pieces = top_level(rest);
    let Some((last, init)) = pieces.split_last() else {
        return Ok(line);
    };
    for piece in init {
        line.operands.push(parse_operand(syntax, piece)?);
    }
    let tokens: Vec<&str> = if last.starts_with("hwreg(") {
        vec![last]
    } else {
        last.split_whitespace().collect()
    };
    let mut rest = tokens.as_slice();
    if let Some((first, after)) = rest.split_first() {
        if !starts_modifiers(syntax, first) {
            line.operands.push(parse_operand(syntax, first)?);
            rest = after;
        }
    }
    for &token in rest {
        if token.starts_with("gpr_idx(") {
            line.operands.push(parse_operand(syntax, token)?);
            continue;
        }
        let key = token.split_once(':').map_or(token, |(k, _)| k);
        if line.modifiers.has(key) {
            return Err(format!("modifier {key} given twice in `{text}`"));
        }
        let value = match token.split_once(':') {
            None if syntax.mod_flag(token) => ModValue::Int(1),
            Some((key, value)) if !key.is_empty() && !syntax.mod_flag(key) => {
                modifier_value(syntax, key, value, text)?
            }
            _ => return Err(format!("modifier not understood `{token}` in `{text}`")),
        };
        line.modifiers.insert(key, value);
    }
    Ok(line)
}

/// The code of one DPP16 control as a `dpp` field holds it
/// (`quad_perm:[0,1,2,3]`, `row_shr:1`, `row_mirror`): the RDNA DPP_CTRL
/// value, or why the text is no control.
pub fn dpp16_control(text: &str) -> Result<u16, String> {
    let bad = || format!("dpp `{text}`: not a DPP16 control");
    match text {
        "row_mirror" => return Ok(0x140),
        "row_half_mirror" => return Ok(0x141),
        _ => {}
    }
    let (key, arg) = text.split_once(':').ok_or_else(bad)?;
    if key == "quad_perm" {
        let inner = arg.strip_prefix('[').and_then(|a| a.strip_suffix(']'));
        let lanes: Vec<u16> = (inner.ok_or_else(bad)?.split(','))
            .map(|x| x.trim().parse::<u16>().ok().filter(|&x| x < 4))
            .collect::<Option<_>>()
            .ok_or_else(bad)?;
        let [a, b, c, d] = lanes[..] else {
            return Err(bad());
        };
        return Ok(a | b << 2 | c << 4 | d << 6);
    }
    let (base, low) = match (key, arg) {
        ("wave_shl", "1") => return Ok(0x130),
        ("wave_rol", "1") => return Ok(0x134),
        ("wave_shr", "1") => return Ok(0x138),
        ("wave_ror", "1") => return Ok(0x13C),
        ("row_bcast", "15") => return Ok(0x142),
        ("row_bcast", "31") => return Ok(0x143),
        ("row_shl", _) => (0x100, 1),
        ("row_shr", _) => (0x110, 1),
        ("row_ror", _) => (0x120, 1),
        ("row_share" | "row_newbcast", _) => (0x150, 0),
        ("row_xmask", _) => (0x160, 0),
        _ => return Err(bad()),
    };
    let k = int_value(arg)
        .filter(|k| (low..16).contains(k))
        .ok_or_else(bad)?;
    Ok(base + k as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_follow_python() {
        assert_eq!(int_value("0x_ff"), Some(255));
        assert_eq!(int_value(" 5 "), Some(5));
        assert_eq!(int_value("000"), Some(0));
        assert_eq!(int_value("012"), None);
        assert_eq!(int_value("1__0"), None);
        assert_eq!(int_value("0X1f"), Some(31));
        assert_eq!(int_value("-"), None);
    }

    #[test]
    fn text_classes() {
        assert!(float_text("-.5e1") && float_text("1.") && !float_text("1e5") && !float_text("."));
        assert!(!float_text("1.0e"));
        assert!(label_text("e5") && label_text(".L1") && !label_text(".x") && !label_text("9a"));
        assert!(int_text("-0x1F") && !int_text("0X1f"));
    }
}
