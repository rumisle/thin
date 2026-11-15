//! The writer's claims, read from a program's section lines.

use std::collections::BTreeMap;

use super::facts::Facts;
use crate::{Cause, EmuError, Program};

/// A register a claim names: (file, first, count), file `v`, `s`, `a` or
/// `scc`.
pub type ClaimReg = (String, u32, u32);

/// A uniform claim before one instruction: its registers and its text
/// (several claims' texts joined by `"; "`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Uniform {
    /// The registers, in order.
    pub regs: Vec<ClaimReg>,
    /// The claim's text.
    pub text: String,
}

/// An assume claim: a register range must lie in [lo, hi].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assume {
    /// The register.
    pub reg: ClaimReg,
    /// The lowest value.
    pub lo: u64,
    /// The highest value.
    pub hi: u64,
    /// The claim's text.
    pub text: String,
}

/// The claims of a program, by the index of the instruction they precede.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Claims {
    /// Uniform claims.
    pub uniform: BTreeMap<usize, Uniform>,
    /// Assume claims, in order at each index.
    pub assume: BTreeMap<usize, Vec<Assume>>,
    /// Undefined claims' registers at each index.
    pub undefined: BTreeMap<usize, Vec<ClaimReg>>,
    /// Edge claims' registers by edge: (the source instruction, the
    /// first instruction the edge reaches).
    pub edges: BTreeMap<(usize, usize), Vec<ClaimReg>>,
}

/// A placed register's name: `vN`, `v[a:b]`, `sN`, `s[a:b]`, `aN`, `a[a:b]`.
fn placed(item: &str, files: &str) -> Option<ClaimReg> {
    let mut chars = item.chars();
    let file = chars.next()?;
    if !files.contains(file) {
        return None;
    }
    let rest = chars.as_str();
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if digits(rest) {
        return Some((file.to_string(), rest.parse().ok()?, 1));
    }
    let inner = rest.strip_prefix('[')?.strip_suffix(']')?;
    let (a, b) = inner.split_once(':')?;
    if !digits(a) || !digits(b) {
        return None;
    }
    let (a, b): (u32, u32) = (a.parse().ok()?, b.parse().ok()?);
    (b >= a).then(|| (file.to_string(), a, b - a + 1))
}

/// The list of a claim line: after the prefix, up to its ` (why)`, split
/// on `, `; none without the reason.
fn items<'a>(text: &'a str, prefix: &str) -> Option<Vec<&'a str>> {
    let rest = text.strip_prefix(prefix)?;
    let (list, why) = rest.split_once(" (")?;
    why.ends_with(')').then(|| list.split(", ").collect())
}

/// A line that starts as a claim but does not read as one.
fn malformed(line: usize, text: &str, want: &str) -> EmuError {
    fault(format!(
        "line {line}: `; {text}` is not a claim line: {want}"
    ))
}

/// The registers of a claim's list, each of `files` or, with `facts`, a
/// named register it models; none if any item is neither. An item the
/// printer spelled as a value's repr (`s:x#1[0]x1`: a value never
/// placed, so holding nothing to check) is left out.
fn registers(list: &[&str], files: &str, facts: Option<&Facts>) -> Option<Vec<ClaimReg>> {
    (list.iter())
        .filter(|i| !(i.contains(':') && i.contains('#')))
        .map(|i| {
            placed(i, files).or_else(|| {
                let (f, part, n) = facts?.named_reg(i)?;
                Some((f.to_owned(), part, n))
            })
        })
        .collect()
}

fn decimal(s: &str) -> Option<u64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}

/// `assume: R in [LO, HI] (why)`, exactly.
fn assume_line(text: &str) -> Option<Assume> {
    let rest = text.strip_prefix("assume: ")?;
    let (reg, rest) = rest.split_once(" in [")?;
    let reg = placed(reg, "vsa")?;
    let (lo, rest) = rest.split_once(", ")?;
    let (hi, rest) = rest.split_once("] (")?;
    if !rest.ends_with(')') {
        return None;
    }
    Some(Assume {
        reg,
        lo: decimal(lo)?,
        hi: decimal(hi)?,
        text: text.to_owned(),
    })
}

fn fault(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Fault),
        message,
    }
}

/// Whether the next code item after a note at `line` before instruction
/// `index` (other than notes) is a label.
fn before_label(program: &Program, index: usize, line: usize) -> bool {
    let next_ins = program.ins.get(index).map(|i| i.line).unwrap_or(usize::MAX);
    program
        .label_lines
        .iter()
        .any(|(i, l)| *i == index && *l > line && *l < next_ins)
}

/// One kind's claims: uniform, assume or undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `uniform:` lines.
    Uniform,
    /// `assume:` lines.
    Assume,
    /// `undefined:` lines.
    Undefined,
}

/// Reads one kind of claims off a program's section lines.
pub fn read(
    program: &Program,
    facts: &Facts,
    kind: Kind,
    claims: &mut Claims,
) -> Result<(), EmuError> {
    for (index, text, line) in &program.sections {
        match kind {
            Kind::Uniform if text.starts_with("uniform:") => {
                let regs = items(text, "uniform: ").and_then(|l| registers(&l, "s", Some(facts)));
                let Some(regs) = regs else {
                    return Err(malformed(
                        *line,
                        text,
                        "`uniform: R, ... (why)`, R an SGPR or a named scalar register",
                    ));
                };
                if regs.is_empty() {
                    continue; // only values never placed: nothing to check
                }
                if before_label(program, *index, *line) {
                    return Err(fault(format!(
                        "line {line}: `; {text}` is followed by a label: the claim would be \
                         checked on every path into it (place Builder.uniform after an \
                         instruction of its own)"
                    )));
                }
                let entry = claims.uniform.entry(*index).or_default();
                entry.regs.extend(regs);
                if !entry.text.is_empty() {
                    entry.text.push_str("; ");
                }
                entry.text.push_str(text);
            }
            Kind::Uniform => {}
            Kind::Assume if text.starts_with("assume:") => {
                let Some(a) = assume_line(text) else {
                    return Err(malformed(
                        *line,
                        text,
                        "`assume: R in [LO, HI] (why)`, LO and HI decimal",
                    ));
                };
                if before_label(program, *index, *line) {
                    return Err(fault(format!(
                        "line {line}: `; {text}` is followed by a label: the claim would be \
                         checked on every path into it (place Builder.assume after an \
                         instruction of its own)"
                    )));
                }
                claims.assume.entry(*index).or_default().push(a);
            }
            Kind::Assume => {}
            Kind::Undefined if text.starts_with("undefined on ") => {
                let want = "`undefined on taken|fall: R, ... (why)`, R a register";
                let (on, list) = if let Some(l) = items(text, "undefined on taken: ") {
                    ("taken", l)
                } else if let Some(l) = items(text, "undefined on fall: ") {
                    ("fall", l)
                } else {
                    return Err(malformed(*line, text, want));
                };
                let Some(regs) = registers(&list, "vsa", None) else {
                    return Err(malformed(*line, text, want));
                };
                if regs.is_empty() {
                    continue; // only values never placed: nothing to check
                }
                let label = before_label(program, *index, *line);
                let edge = if on == "taken" {
                    let branch = program.ins.get(*index).filter(|i| {
                        !label && (i.canon.starts_with("s.cbranch") || i.canon == "s.branch")
                    });
                    let target = branch.and_then(|i| {
                        i.operands.iter().find_map(|o| match &o.base {
                            crate::Base::Label(n) => {
                                program.labels.iter().find(|(m, _)| m == n).map(|(_, k)| *k)
                            }
                            _ => None,
                        })
                    });
                    let Some(t) = target else {
                        return Err(fault(format!(
                            "line {line}: `; {text}` is not followed by a branch: an edge claim \
                             on taken stands just before its branch"
                        )));
                    };
                    (*index, t)
                } else {
                    if !label || *index == 0 {
                        return Err(fault(format!(
                            "line {line}: `; {text}` is not followed by a label: an edge claim on \
                             fall stands just before the label it falls into"
                        )));
                    }
                    (*index - 1, *index)
                };
                claims.edges.entry(edge).or_default().extend(regs);
            }
            Kind::Undefined if text.starts_with("undefined:") => {
                let regs = items(text, "undefined: ").and_then(|l| registers(&l, "vsa", None));
                let Some(regs) = regs else {
                    return Err(malformed(
                        *line,
                        text,
                        "`undefined: R, ... (why)`, R a register",
                    ));
                };
                if regs.is_empty() {
                    continue; // only values never placed: nothing to check
                }
                if before_label(program, *index, *line) {
                    return Err(fault(format!(
                        "line {line}: `; {text}` is followed by a label, not by an instruction \
                         of its own: the claim would poison every path through that label \
                         (Builder.undefined emits its own s_nop)"
                    )));
                }
                claims.undefined.entry(*index).or_default().extend(regs);
            }
            Kind::Undefined => {}
        }
    }
    Ok(())
}

/// Every claim of a program.
pub fn all(program: &Program, facts: &Facts) -> Result<Claims, EmuError> {
    let mut claims = Claims::default();
    for kind in [Kind::Uniform, Kind::Assume, Kind::Undefined] {
        read(program, facts, kind, &mut claims)?;
    }
    Ok(claims)
}

/// A held value's spelling in a uniform claim's message: one value as
/// `0x…`, a range as a list.
pub fn held_text(values: &[Vec<u32>]) -> String {
    let parts: Vec<String> = values
        .iter()
        .map(|v| match v.as_slice() {
            [one] => format!("{one:#x}"),
            many => format!(
                "[{}]",
                many.iter()
                    .map(|x| format!("{x:#x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// The message of a false assume claim.
pub fn assume_false(text: &str, pc: usize, wave: usize, lane: Option<usize>, value: u64) -> String {
    let held = if value == u64::from(super::POISON) {
        "POISON (never written, or its load not landed)".to_owned()
    } else {
        format!("{value:#x}")
    };
    let lane = lane.map_or(String::new(), |l| format!(" lane {l}"));
    format!("`; {text}` before instruction {pc} is false: wave {wave}{lane} holds {held} (Builder.assume's claim)")
}
