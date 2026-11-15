//! Register spelling and the printer's forms over a built kernel
//! and refusals spelled with them.

use std::sync::Arc;

use thin_dsl::Builder;
use thin_ir::print::{commented, line, placed_text};
use thin_ir::{Diag, DiagClass, Instr, Operand, Part, Reg};
use thin_target::{Flags, Kind, Pseudo};

/// A refusal fragment: plain text or a host-rendered short location.
/// a location ("" for an unknown one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    /// Text.
    Text(String),
    /// A location.
    Where(u32),
}

/// A refusal of the pipeline: its class (a thin.ir error class name, or
/// `Value`, `Type`, `Assertion`), its location, its text and its notes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The class.
    pub class: &'static str,
    /// The location, 0 for none.
    pub loc: u32,
    /// The text.
    pub msg: Vec<Piece>,
    /// The notes: each a location and a text.
    pub notes: Vec<(u32, Vec<Piece>)>,
}

impl Refusal {
    /// A refusal of one text.
    pub fn new(class: &'static str, msg: impl Into<String>) -> Refusal {
        Refusal {
            class,
            loc: 0,
            msg: vec![Piece::Text(msg.into())],
            notes: Vec::new(),
        }
    }

    /// The text with its locations dropped (tests and the binding's tail).
    pub fn text(&self) -> String {
        self.msg
            .iter()
            .map(|p| match p {
                Piece::Text(t) => t.as_str(),
                Piece::Where(_) => "",
            })
            .collect()
    }
}

/// The pipeline's result.
pub type Result<T> = std::result::Result<T, Refusal>;

/// A Builder refusal as the pipeline's.
pub fn of_dsl(e: thin_dsl::Error) -> Refusal {
    // The pipeline's classes drop the Python name's "Error" (ScheduleError
    // is "Schedule").
    let name = e.class.name();
    Refusal::new(name.strip_suffix("Error").unwrap_or(name), e.msg)
}

/// How a refusal spells instruction k.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Physical registers once placed.
    Bare,
    /// Registers by name.
    Named,
    /// Placeholders.
    Text,
}

/// The spellings of a kernel's registers.
pub struct Spell<'a> {
    b: &'a Builder,
}

impl<'a> Spell<'a> {
    /// The spellings of `b`.
    pub fn new(b: &'a Builder) -> Spell<'a> {
        Spell { b }
    }

    fn named_of(&self, r: &Reg) -> String {
        self.b.name_text(r)
    }

    /// A register by name.
    pub fn name(&self, r: &Reg) -> String {
        self.named_of(r)
    }

    /// A register physically once its file is placed, else by name.
    pub fn text(&self, r: &Reg) -> Result<String> {
        if !self.b.file(r.file).placed {
            return Ok(self.named_of(r));
        }
        let repr = self.b.repr(r);
        let n = self.b.number(r).map_err(of_dsl)?;
        let value_n = usize::try_from(r.value)
            .ok()
            .and_then(|k| self.b.values.get(k))
            .map_or(0, |v| v.n);
        placed_text(&repr, r.file, n - r.lo, r.lo, r.n, value_n)
            .map_err(|e| Refusal::new("Value", e.to_string()))
    }

    fn line_with(&self, i: &Instr, spell: &dyn Fn(&Reg) -> Result<String>) -> Result<String> {
        let regs = i
            .ops()
            .iter()
            .filter_map(Operand::reg)
            .map(spell)
            .collect::<Result<Vec<_>>>()?;
        line(i, &regs).map_err(|e| Refusal::new("Value", e.to_string()))
    }

    fn own(i: &Instr) -> bool {
        i.form().flags.has(Flags::PSEUDO)
    }

    /// `text(i)`: placeholders.
    pub fn placeholders(&self, i: &Instr) -> String {
        self.b.text(i)
    }

    /// `bare(i)`.
    pub fn bare(&self, i: &Instr) -> Result<String> {
        match Self::own(i) {
            true => Ok(self.b.text(i)),
            false => self.line_with(i, &|r| self.text(r)),
        }
    }

    /// `named(i)`.
    pub fn named(&self, i: &Instr) -> String {
        match Self::own(i) {
            true => self.b.text(i),
            false => self.line_with(i, &|r| Ok(self.name(r))).unwrap_or_default(),
        }
    }

    /// `annotation(i)`: a claim's line, its registers as placed (a repr
    /// where one cannot be spelled), else none.
    fn annotation(&self, i: &Instr) -> Option<String> {
        let claim = matches!(
            i.form().pseudo,
            Some(Pseudo::Discard | Pseudo::Uniform | Pseudo::Undefined | Pseudo::Assume)
        );
        claim.then(|| {
            self.line_with(i, &|r| Ok(self.text(r).unwrap_or_else(|_| self.b.repr(r))))
                .unwrap_or_default()
        })
    }

    /// `render(i)`.
    pub fn render(&self, i: &Instr) -> Result<String> {
        if let Some(a) = self.annotation(i) {
            return Ok(a);
        }
        if matches!(i.form().kind, Kind::Label | Kind::Comment) {
            return Ok(self.b.text(i));
        }
        Ok(commented(&self.bare(i)?, i.attrs().comment.as_deref()))
    }

    /// The body's lines.
    pub fn body(&self, instrs: &[Arc<Instr>]) -> Result<Vec<String>> {
        instrs.iter().map(|i| self.render(i)).collect()
    }

    /// The form a refusal spells instruction k of `instrs` in.
    pub fn form(&self, i: &Instr, f: Form) -> String {
        match f {
            Form::Bare => self.bare(i).unwrap_or_else(|_| self.named(i)),
            Form::Named => self.named(i),
            Form::Text => self.b.text(i),
        }
    }

    /// A pass's Diag spelled over the instructions it read.
    pub fn diag(&self, d: Diag, instrs: &[Arc<Instr>], locs: &[u32], f: Form) -> Refusal {
        let pieces = |parts: &[Part]| -> Vec<Piece> {
            (parts.iter())
                .map(|p| match p {
                    Part::Text(t) => Piece::Text(t.clone()),
                    Part::Instr(k) => Piece::Text(self.form(&instrs[*k], f)),
                    Part::Clip(k, n) => {
                        Piece::Text(self.form(&instrs[*k], f).chars().take(*n).collect())
                    }
                    Part::Where(k) => Piece::Where(locs.get(*k).copied().unwrap_or(0)),
                })
                .collect()
        };
        Refusal {
            class: class_name(d.class),
            loc: d.loc,
            msg: pieces(&d.msg),
            notes: d.notes.iter().map(|(loc, p)| (*loc, pieces(p))).collect(),
        }
    }
}

/// A Diag class as the thin.ir class name.
pub fn class_name(c: DiagClass) -> &'static str {
    match c {
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
