//! Operands: registers of values, named registers, immediates and blocks
//!

use std::sync::Arc;

use thin_target::{Phys, PHYS};

/// Dwords `lo..= lo+n−1` of value number `value` in file `file`, reached
/// through the value's hand-laid part number `part` (0: its own name).
///
/// The part is not identity: two registers are equal when their value, lo
/// and n are.
#[derive(Clone, Copy, Debug)]
pub struct Reg {
    /// The value's number in its program (negative only until a snapshot refuses it).
    pub value: i32,
    /// The first dword.
    pub lo: u32,
    /// The number of dwords.
    pub n: u32,
    /// The hand-laid part it goes through, numbered from 1; 0 for none.
    pub part: u32,
    /// The value's register file letter.
    pub file: char,
}

impl PartialEq for Reg {
    fn eq(&self, other: &Reg) -> bool {
        (self.value, self.lo, self.n) == (other.value, other.lo, other.n)
    }
}

impl Eq for Reg {}

impl std::hash::Hash for Reg {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        (self.value, self.lo, self.n).hash(h);
    }
}

impl Reg {
    /// Dwords `lo.. lo+n` of a value, through no part.
    pub fn new(value: i32, file: char, lo: u32, n: u32) -> Reg {
        Reg {
            value,
            lo,
            n,
            part: 0,
            file,
        }
    }

    /// Dword k of it, keeping the part.
    pub fn dword(self, k: u32) -> Reg {
        Reg {
            lo: self.lo + k,
            n: 1,
            ..self
        }
    }
}

/// An immediate's value: the bits as the slot reads them (any 64-bit
/// pattern, signed or unsigned), or a float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImmValue {
    /// An integer.
    Int(i128),
    /// An f32 or f64 value.
    Float(f64),
}

/// An immediate and the writer's spelling, kept only where the value's own
/// spelling would not give it back.
#[derive(Clone, Debug, PartialEq)]
pub struct Imm {
    /// The value.
    pub value: ImmValue,
    /// The writer's text, or empty.
    pub text: Arc<str>,
}

impl Imm {
    /// An integer immediate with no text.
    pub fn int(v: i128) -> Imm {
        Imm {
            value: ImmValue::Int(v),
            text: "".into(),
        }
    }
}

/// An explicit operand.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    /// Dwords of a value.
    Reg(Reg),
    /// A named register; its whole is `PHYS[p.whole]`.
    Phys(&'static Phys),
    /// An immediate.
    Imm(Imm),
    /// A branch target: the label that starts the block.
    Block(Arc<str>),
}

impl Operand {
    /// The register, if it is one.
    pub fn reg(&self) -> Option<&Reg> {
        match self {
            Operand::Reg(r) => Some(r),
            _ => None,
        }
    }

    /// The named register, if it is one.
    pub fn phys(&self) -> Option<&'static Phys> {
        match self {
            Operand::Phys(p) => Some(p),
            _ => None,
        }
    }

    /// The whole register of a named-register operand.
    pub fn whole(&self) -> Option<&'static Phys> {
        self.phys().map(|p| &PHYS[p.whole])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_ignores_the_part() {
        let a = Reg::new(1, 'v', 0, 2);
        let b = Reg { part: 3, ..a };
        assert_eq!(a, b);
        assert_ne!(a, a.dword(1));
        assert_eq!(b.dword(1).part, 3);
    }
}
