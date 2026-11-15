//! The compilation pipeline from a built kernel to a code object.
//!
//! Owns phase ordering, deferred transforms, placement audits, wait and hazard
//! drivers, ABI/resource manifests and code-object construction. Immutable
//! snapshots separate analyses from the stream changes the pipeline applies.
//!
//! The target's assembly reader selects the assembler and records byte layout.
//! The independent wait checker deliberately shares no compiler parser or
//! scheduling implementation. Python receives plain artifacts and diagnostics;
//! it does not duplicate these algorithms.

pub mod abi;
pub mod checker;
pub mod compile;
pub mod json;
pub mod manifest;
pub mod meta;
pub mod object;
pub mod pipeline;
pub mod place;
pub mod spell;

pub use checker::counter_waits;
pub use object::code_object;
