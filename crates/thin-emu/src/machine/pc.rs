//! Byte-address semantics of GETPC, SETPC and SWAPPC.

use super::wave::Wave;
use crate::{Cause, EmuError, Ins};

/// The assembler's explicit instruction ranges in a zero-based text image.
/// Gaps are alignment nops; literal extension words stay inside a range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Start and exclusive end, in decoded instruction order.
    pub ranges: Vec<(u64, u64)>,
    /// Exclusive text end, including any trailing alignment nops.
    pub end: u64,
}

fn invalid(message: &str) -> EmuError {
    EmuError {
        cause: None,
        message: format!("pc_layout: {message}"),
    }
}

impl Layout {
    /// Validates range order, alignment and agreement with the decoded rows.
    pub fn validate(&self, instructions: usize) -> Result<(), EmuError> {
        if self.ranges.len() != instructions || self.end % 4 != 0 {
            return Err(invalid(
                "range count or text end does not match the program",
            ));
        }
        let mut previous = 0;
        for &(start, end) in &self.ranges {
            if start < previous || start % 4 != 0 || end % 4 != 0 || end <= start || end > self.end
            {
                return Err(invalid(
                    "instruction ranges overlap, are unaligned or leave the text",
                ));
            }
            previous = end;
        }
        Ok(())
    }

    /// Reads the host encoding of ranges and text end, without inferring
    /// widths from canonical names or instruction indices.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, EmuError> {
        let ranges = value
            .get("ranges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| invalid("expected instruction ranges"))?
            .iter()
            .map(|range| {
                let pair = range
                    .as_array()
                    .filter(|r| r.len() == 2)
                    .ok_or_else(|| invalid("expected a start/end pair"))?;
                let number = |k: usize| {
                    pair[k]
                        .as_u64()
                        .ok_or_else(|| invalid("range addresses must be unsigned integers"))
                };
                Ok((number(0)?, number(1)?))
            })
            .collect::<Result<Vec<_>, EmuError>>()?;
        let end = value
            .get("end")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| invalid("expected an unsigned text end"))?;
        Ok(Self { ranges, end })
    }

    fn target(&self, address: u64) -> Result<usize, EmuError> {
        if address % 4 == 0 && address <= self.end {
            let next = self.ranges.partition_point(|&(start, _)| start < address);
            if self
                .ranges
                .get(next)
                .is_some_and(|&(start, _)| start == address)
                || next == 0
                || self.ranges[next - 1].1 <= address
            {
                // Alignment nops have no decoded rows. Entering their gap
                // reaches the following explicit instruction, or text end.
                return Ok(next);
            }
        }
        Err(EmuError {
            cause: Some(Cause::Fault),
            message: format!(
                "PC target {address:#x} is not an instruction or alignment-nop boundary"
            ),
        })
    }
}

pub(super) fn is_pc(canon: &str) -> bool {
    matches!(canon, "s.getpc.b64" | "s.setpc.b64" | "s.swappc.b64")
}

pub(super) fn execute(
    canon: &str,
    wave: &mut Wave,
    ins: &Ins,
    layout: Option<&Layout>,
    row: usize,
) -> Result<Option<usize>, EmuError> {
    let layout = layout.ok_or_else(|| EmuError {
        cause: Some(Cause::Emulator),
        message: "PC instructions require an explicit byte-address layout".into(),
    })?;
    let &(address, _) = layout
        .ranges
        .get(row)
        .ok_or_else(|| invalid("missing current instruction"))?;
    let operand = |index| {
        ins.operands.get(index).ok_or_else(|| EmuError {
            cause: Some(Cause::Fault),
            message: format!("{}: missing PC operand", ins.text),
        })
    };
    if canon == "s.getpc.b64" {
        wave.swrite64(operand(0)?, address + 4)?;
        return Ok(None);
    }
    // Capture the source before changing the link, including aliased pairs.
    let address = wave.sread64(operand(usize::from(canon == "s.swappc.b64"))?)?;
    let target = layout.target(address)?;
    if canon == "s.swappc.b64" {
        wave.swrite64(operand(0)?, layout.ranges[row].0 + 4)?;
    }
    wave.pc = target;
    Ok(Some(target))
}
