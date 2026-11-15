//! Snapshot-to-result compiler passes.
//!
//! Owns register assignment, wait counts, hazards, ranges, alias checking and
//! LDS proofs. Each pass is a pure function of its snapshot and explicit options;
//! the pipeline applies its returned changes. A pass never edits its input.
//!
//! Hardware hazards are supplied through hooks, so this crate names no target
//! instance. Refusals are structured thin_ir diagnostics. Dependencies are
//! thin-ir and thin-target.

pub mod alias;
pub mod checks;
pub mod hazards;
pub mod lds;
mod need;
pub mod ranges;
pub mod regalloc;
pub mod waits;

pub use need::{delay_text, Need, Valu};
