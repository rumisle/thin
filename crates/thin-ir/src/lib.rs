//! Immutable instructions, snapshots, dataflow and assembly printing.
//!
//! A transformed instruction is a new value. Snapshots retain captured rows and
//! never observe subsequent host stream edits. CFG and dataflow analyses read
//! snapshots and explicit inputs. Diagnostics retain structured message, row
//! and source parts for host-side rendering.
//!
//! Host objects are opaque identity handles with facts captured once. Core
//! analyses never call back into the host. The host retains source locations
//! and its stream of rows; this crate depends only on thin-target.

mod attr;
pub mod cfg;
pub mod control;
mod diag;
pub mod exec;
mod field;
pub mod flight;
mod instr;
pub mod live;
pub mod m0;
mod operand;
mod options;
pub mod print;
pub mod pyfmt;
pub mod reach;
pub mod sets;
mod snapshot;
pub mod solve;
pub mod verify;

pub use attr::{Attr, AttrClass, Attrs, LdsAccess};
pub use diag::{Diag, DiagClass, Part};
pub use field::{tag_repr, Atom, Datum, FieldId, FieldValue, HostObject, ObjectFacts, Tag};
pub use instr::{dpp_reads_old, Instr, Request, R, W};
pub use operand::{Imm, ImmValue, Operand, Reg};
pub use options::Options;
pub use snapshot::{
    Gather, GatherFile, GatherRegion, GatherValue, Region, Snapshot, ValueInfo, WaveClass,
    WaveGroup,
};

use std::fmt;

/// A refusal of an API call: the host raises `TypeError` or `ValueError`
/// with the message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A wrong type.
    Type(String),
    /// A wrong value.
    Value(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (Error::Type(msg) | Error::Value(msg)) = self;
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}
