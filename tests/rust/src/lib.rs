//! Shared builders and exact golden-data assertions for integration tests.

pub mod adapters;
pub mod asm;
pub mod budgets;
pub mod golden;
pub mod golden_tests;
pub mod listing;
pub mod notation;
pub mod tagjson;

pub use golden_tests::{Case, Outcome, Suite};
