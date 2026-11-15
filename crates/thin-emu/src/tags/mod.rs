//! Provenance tags: the tag format and
//! its append-only tables, the transfer rules, combining records, the
//! accumulator state and the checker interface.
//!
//! **The module rule.** This module uses no other module of the crate: the
//! machine (`machine`) and the GEMM checker (`gemm`) both build on it.

pub mod checker;
pub mod combine;
pub mod encoding;
pub mod matrix;
pub mod rules;

pub use checker::{AtomicAnswer, Checker, MatrixOutcome, Plain, StoreCall, TagError};
pub use encoding::*;
pub use matrix::{AccEntry, AccState, MatrixForm, MatrixOp, MatrixOperand, OperandFormat};
pub use rules::{Ctx, Lane};
