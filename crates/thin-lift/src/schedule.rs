//! Removing scheduling instructions, the inverse of inserting them.
//!
//! Instruction offsets and branch targets remain in the input address
//! space: this is a structural view, not a relocated executable program.
//! Re-inserting the residue restores every decoded instruction exactly.

use thin_targets::Instruction;

/// An instruction removed from the input scheduled sequence.
#[derive(Clone, Copy)]
pub struct Scheduled {
    /// Its instruction index in the input sequence, not a byte offset.
    pub position: usize,
    /// The complete decoded instruction, including its original offset.
    pub instruction: Instruction,
}

/// Scheduling instructions removed by [`unschedule`], in original order.
#[derive(Clone, Default)]
pub struct Schedule {
    /// Entries ordered by strictly increasing original instruction index.
    pub entries: Vec<Scheduled>,
}

/// A malformed schedule cannot be inserted into a sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// An entry points past the reconstructed sequence.
    Position {
        /// The invalid original instruction index.
        position: usize,
        /// Number of instructions in the reconstructed sequence.
        len: usize,
    },
    /// Original positions must be unique and strictly increasing.
    Order,
    /// The reconstructed sequence length overflows usize.
    Length,
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Position { position, len } => {
                write!(f, "schedule position {position} outside {len} instructions")
            }
            Self::Order => f.write_str("schedule positions are not strictly increasing"),
            Self::Length => f.write_str("scheduled sequence length overflows usize"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Whether an instruction belongs to the schedule removed by this step.
/// All `s_wait_*` instructions are included, not just memory counters.
pub fn is_scheduling(instruction: &Instruction) -> bool {
    instruction.mnemonic.starts_with("s_wait_") || instruction.mnemonic == "s_delay_alu"
}

/// Separates decoded instructions from their schedule without changing
/// either one's operands, modifiers, offsets or relative order.
pub fn unschedule(instructions: &[Instruction]) -> (Vec<Instruction>, Schedule) {
    let mut body = Vec::new();
    let mut entries = Vec::new();
    for (position, &instruction) in instructions.iter().enumerate() {
        if is_scheduling(&instruction) {
            entries.push(Scheduled {
                position,
                instruction,
            });
        } else {
            body.push(instruction);
        }
    }
    (body, Schedule { entries })
}

impl Schedule {
    /// Re-inserts the saved scheduling instructions at their original
    /// indices. Adjacent, leading and trailing entries need no special
    /// representation. Refuses duplicate, unsorted or out-of-range indices.
    pub fn insert(&self, body: &[Instruction]) -> Result<Vec<Instruction>, ScheduleError> {
        let len = body
            .len()
            .checked_add(self.entries.len())
            .ok_or(ScheduleError::Length)?;
        let mut previous = None;
        for entry in &self.entries {
            if entry.position >= len {
                return Err(ScheduleError::Position {
                    position: entry.position,
                    len,
                });
            }
            if previous.is_some_and(|p| p >= entry.position) {
                return Err(ScheduleError::Order);
            }
            previous = Some(entry.position);
        }
        let mut entries = self.entries.iter().peekable();
        let mut body = body.iter();
        let mut out = Vec::with_capacity(len);
        for position in 0..len {
            if entries.peek().is_some_and(|e| e.position == position) {
                out.push(entries.next().expect("entry at this position").instruction);
            } else {
                out.push(
                    *body
                        .next()
                        .expect("validated schedule leaves a body position"),
                );
            }
        }
        Ok(out)
    }
}
