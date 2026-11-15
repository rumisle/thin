//! Native messages: a refusal as parts the host spells.

use std::fmt;

/// The class of a refusal; the host raises the class of that name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagClass {
    /// A plain `Diag` (the verifier).
    Diag,
    /// `ScheduleError`: a kernel the compiler refuses to build.
    Schedule,
    /// `ExprError`.
    Expr,
    /// `FlowError`.
    Flow,
    /// `MemError`.
    Mem,
    /// `AssignError`.
    Assign,
    /// `RangeError`.
    Range,
    /// `NoFixpoint`, a plain `ValueError`.
    NoFixpoint,
    /// `AsmError`, a plain `ValueError`.
    Asm,
}

impl DiagClass {
    /// The name the host maps to its class.
    pub fn name(self) -> &'static str {
        match self {
            DiagClass::Diag => "Diag",
            DiagClass::Schedule => "Schedule",
            DiagClass::Expr => "Expr",
            DiagClass::Flow => "Flow",
            DiagClass::Mem => "Mem",
            DiagClass::Assign => "Assign",
            DiagClass::Range => "Range",
            DiagClass::NoFixpoint => "NoFixpoint",
            DiagClass::Asm => "Asm",
        }
    }
}

/// A part of a message, so that the core needs no printer and no location
/// table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Text.
    Text(String),
    /// The line of instruction k.
    Instr(usize),
    /// `" (file:line)"` of the source that emitted instruction k, or "".
    Where(usize),
    /// The first n characters of instruction k's line.
    Clip(usize, usize),
}

/// A user-facing refusal: its class, the instruction it is at (by its
/// loc, 0 when unknown), its message and its notes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diag {
    /// The class.
    pub class: DiagClass,
    /// Where it was raised.
    pub loc: u32,
    /// The message.
    pub msg: Vec<Part>,
    /// Notes, each with its location.
    pub notes: Vec<(u32, Vec<Part>)>,
}

impl Diag {
    /// The message with lines spelled by `line` and sources by `where_`
    /// (dropped when none), as the host spells it.
    pub fn spell(
        parts: &[Part],
        line: &dyn Fn(usize) -> String,
        where_: Option<&dyn Fn(usize) -> String>,
    ) -> String {
        let mut out = String::new();
        for part in parts {
            match part {
                Part::Text(t) => out.push_str(t),
                Part::Instr(k) => out.push_str(&line(*k)),
                Part::Where(k) => out.push_str(&where_.map_or(String::new(), |w| w(*k))),
                Part::Clip(k, n) => out.extend(line(*k).chars().take(*n)),
            }
        }
        out
    }
}

impl fmt::Display for Diag {
    /// The message with instructions shown by index and no sources, for
    /// hosts without a printer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let line = |k: usize| format!("<instruction {k}>");
        f.write_str(&Diag::spell(&self.msg, &line, None))
    }
}

impl std::error::Error for Diag {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_spell_through_the_callbacks() {
        let parts = [
            Part::Text("at ".into()),
            Part::Instr(1),
            Part::Where(1),
            Part::Text(": ".into()),
            Part::Clip(2, 3),
        ];
        let line = |k: usize| format!("line{k}");
        let where_ = |k: usize| format!(" (k.py:{k})");
        assert_eq!(
            Diag::spell(&parts, &line, Some(&where_)),
            "at line1 (k.py:1): lin"
        );
        assert_eq!(Diag::spell(&parts, &line, None), "at line1: lin");
    }
}
