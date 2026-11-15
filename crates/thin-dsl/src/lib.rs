//! Instruction Builder state and expression lowering.
//!
//! Owns typed values, register files, instructions, expression trees, lane and
//! control-flow regions, waits, entry arguments and LDS layouts. Python provides
//! operators and context managers while Rust makes lowering decisions; the core
//! never calls back into Python.
//!
//! The Builder emits without reordering, assigning registers or inserting
//! scheduling instructions. All target queries use the generic interface. Host
//! source locations are explicitly supplied through Builder::loc rather than
//! captured here. Dependencies are thin-ir and thin-target.

pub mod args;
pub mod builder;
pub mod entry;
pub mod error;
pub mod expr;
pub mod flow;
pub mod gather;
pub mod lanes;
pub mod lds;
pub mod maker;
pub mod types;
pub mod values;
pub mod waits;

pub use builder::Builder;
pub use error::{Class, Error, Result};

/// The marker comment of an LDS fence.
pub const LDS_FENCE: &str = "[lds fence]";

/// magic(d) = (m, s, short) for dividing a u32 by the constant d.
pub fn magic(d: u32) -> std::result::Result<(u64, u32, bool), String> {
    match d {
        0 => Err("magic(0): a divisor of 1 or more".into()),
        d => Ok(expr::magic_triple(u64::from(d))),
    }
}
