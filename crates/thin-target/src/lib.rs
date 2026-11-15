//! Generic target descriptions and shared assembly syntax.
//!
//! Owns instruction forms, operand and field vocabulary, canonical operations,
//! the shared assembly reader, assembler driver and generic code-object facts.
//! Concrete target definitions live in thin-targets. An inconsistent definition
//! fails during construction; a valid target is built once and shared.
//!
//! The assembler, compiler and emulator use one reader. Independent scheduling
//! checkers intentionally parse their own inputs to avoid sharing compiler bugs.
//! This crate has no dependencies and names no target instance.

mod asm;
mod code_object;
mod form;
mod matrix;
mod own;
mod read_s;
mod rules;
mod syntax;
mod target;
mod vocab;

use std::fmt;

pub use asm::{assemble, assemble_with_layout, Assembly};
pub use code_object::{descriptor, interface_descriptor, Block, Footer, Resources, COMMON};
pub use form::{base, mem_bytes, reads_dst, Field, Form, Pseudo, Slot, WaitText};
pub use matrix::{
    ClockRead, Elem, IcacheInvalidate, LibHooks, MatrixForm, Scope, ScopeRule, Summation,
};
pub use own::OWN_OPS;
pub use read_s::{read_s, CodeItem, SFile};
pub use rules::{slot_kind, unit_of, FormRules, Row, WaitInsn};
pub use syntax::{
    dpp16_control, float_text, int_text, int_value, label_text, parse_line, parse_operand,
    reg_text, Line, ModValue, Modifiers, Operand, OperandKind, Syntax, Value,
};
pub use target::{
    Chip, Counter, Definition, Encoded, File, Hazards, Hooks, Lands, Operands, Target, WaitCounts,
    GROUPS,
};
pub use vocab::{
    dst_half, dst_sel, is_line_field, mem_of_encoding, phys_named, DstHalf, DstSel, FieldKind,
    Flags, Kind, Mem, Phys, SlotKind, Space, FIELD_NAMES, FLAG_BITS, LINE_FIELDS, LINE_FIELD_NAMES,
    PHYS,
};

/// A failure of the target interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// An inconsistent target definition: a bug of the target.
    Definition(String),
    /// An unknown target, or an op the target cannot spell (`ValueError`).
    Value(String),
    /// A missing form or own op (`KeyError`).
    Key(String),
    /// A refusal of the reader or the assembler, with its line (`AsmError`).
    Asm(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (Error::Definition(msg) | Error::Value(msg) | Error::Key(msg) | Error::Asm(msg)) = self;
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}
