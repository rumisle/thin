//! What an instruction needs right before it.

use thin_target::WaitCounts;

/// The VALU op a need asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Valu {
    /// None.
    #[default]
    None,
    /// A `v_nop`.
    Nop,
    /// A move of the instruction's operand k onto itself.
    Copy(usize),
}

/// What must come right before an instruction: wait states, wait counts,
/// a VALU op and a delay hint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Need {
    /// Wait states (an `s_nop N` gives N + 1).
    pub states: u32,
    /// A count per wait field, in the target's `wait_fields` order.
    pub waits: WaitCounts,
    /// The VALU op.
    pub valu: Valu,
    /// An `s_delay_alu` operand; 0 is none. A hint, not a hazard.
    pub delay: u32,
    /// The delay hint of the VALU op itself, placed before it: a copy
    /// reads the register it copies, which may still be in flight; 0 is
    /// none.
    pub valu_delay: u32,
}

impl Need {
    /// Nothing, over `fields` wait fields.
    pub fn empty(fields: usize) -> Need {
        Need {
            states: 0,
            waits: WaitCounts::empty(fields),
            valu: Valu::None,
            delay: 0,
            valu_delay: 0,
        }
    }

    /// Whether it asks for nothing.
    pub fn is_empty(&self) -> bool {
        self.states == 0
            && self.waits.is_empty()
            && self.valu == Valu::None
            && self.delay == 0
            && self.valu_delay == 0
    }

    /// Whether it asks for more than `b` (delays never count).
    pub fn exceeds(&self, b: &Need) -> bool {
        let valu = matches!(
            (self.valu, b.valu),
            (Valu::Nop | Valu::Copy(_), Valu::None) | (Valu::Copy(_), Valu::Nop)
        );
        self.states > b.states || self.waits.exceeds(&b.waits) || valu
    }

    /// Both needs at once.
    pub fn join(&self, b: &Need) -> Need {
        let valu = match (self.valu, b.valu) {
            (Valu::Copy(k), _) | (_, Valu::Copy(k)) => Valu::Copy(k),
            (Valu::Nop, _) | (_, Valu::Nop) => Valu::Nop,
            _ => Valu::None,
        };
        Need {
            states: self.states.max(b.states),
            waits: self.waits.min(&b.waits),
            valu,
            delay: self.delay.max(b.delay),
            valu_delay: self.valu_delay.max(b.valu_delay),
        }
    }
}

/// The text of an `s_delay_alu` operand, in LLVM's syntax.
pub fn delay_text(delay: u32) -> String {
    let id = |x: u32| match x {
        1..=4 => format!("VALU_DEP_{x}"),
        5..=7 => format!("TRANS32_DEP_{}", x - 4),
        9..=11 => format!("SALU_CYCLE_{}", x - 8),
        _ => x.to_string(),
    };
    const SKIPS: [&str; 6] = ["SAME", "NEXT", "SKIP_1", "SKIP_2", "SKIP_3", "SKIP_4"];
    let (id0, skip, id1) = (delay & 15, delay >> 4 & 7, delay >> 7 & 15);
    let mut out = format!("instid0({})", id(id0));
    if skip != 0 {
        out.push_str(&format!(
            " | instskip({})",
            SKIPS.get(skip as usize).unwrap_or(&"?")
        ));
    }
    if id1 != 0 {
        out.push_str(&format!(" | instid1({})", id(id1)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn need(states: u32, waits: Vec<Option<u32>>, valu: Valu) -> Need {
        Need {
            states,
            waits: WaitCounts(waits),
            valu,
            delay: 0,
            valu_delay: 0,
        }
    }

    #[test]
    fn join_and_exceeds_follow_section_3_1() {
        let a = need(2, vec![Some(3), None], Valu::Nop);
        let b = need(1, vec![Some(1), Some(0)], Valu::Copy(2));
        let j = a.join(&b);
        assert_eq!(j, need(2, vec![Some(1), Some(0)], Valu::Copy(2)));
        assert!(b.exceeds(&a) && a.exceeds(&b));
        assert!(!need(0, vec![None, None], Valu::Nop).exceeds(&need(
            0,
            vec![None, None],
            Valu::Copy(1)
        )));
        let delayed = Need {
            delay: 5,
            ..Need::empty(2)
        };
        assert!(!delayed.exceeds(&Need::empty(2)) && !delayed.is_empty());
    }

    #[test]
    fn delays_spell_as_llvm_does() {
        assert_eq!(delay_text(1), "instid0(VALU_DEP_1)");
        assert_eq!(
            delay_text(641),
            "instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)"
        );
        assert_eq!(
            delay_text(145),
            "instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)"
        );
        assert_eq!(delay_text(9), "instid0(SALU_CYCLE_1)");
        assert_eq!(delay_text(0x70), "instid0(0) | instskip(?)");
    }
}
