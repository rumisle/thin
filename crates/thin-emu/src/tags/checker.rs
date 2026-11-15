//! The interface between the machine and a checker.
//!
//! The plain checker is [`Plain`]; the GEMM checker (module `gemm`) is the
//! only other implementation. Checkers are native: nothing here calls back
//! into a host language.

use super::encoding::{is_provenance_record, Tables, Tag};
use super::matrix::{AccState, MatrixOp};

/// A failure of a tag rule or a check: a kernel fault, or a limit of the
/// model (never the kernel's fault).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagError {
    /// A limit (`EmuLimit`) rather than a fault.
    pub limit: bool,
    /// The message.
    pub message: String,
}

impl TagError {
    /// A kernel fault.
    pub fn fault(message: impl Into<String>) -> TagError {
        TagError {
            limit: false,
            message: message.into(),
        }
    }

    /// A limit of the model.
    pub fn limit(message: impl Into<String>) -> TagError {
        TagError {
            limit: true,
            message: message.into(),
        }
    }
}

/// A global or buffer store into one buffer, for its active lanes.
pub struct StoreCall<'a> {
    /// The buffer's index in the memory.
    pub buffer: usize,
    /// The buffer's name.
    pub name: &'a str,
    /// Per active lane, the byte offsets written.
    pub offsets: &'a [Vec<u64>],
    /// Per active lane, the bytes written.
    pub data: &'a [Vec<u8>],
    /// Per active lane, their tags.
    pub tags: &'a [Vec<Tag>],
    /// The storing wave.
    pub wave: usize,
}

/// The packed-bf16 atomic's call: what the checker answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicAnswer {
    /// The buffer is no problem's: the machine applies R3c.
    Scratch,
    /// The checker handled it: the memory's tags are unchanged.
    Checked,
}

/// What the matrix-op dispatch did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixOutcome {
    /// Checked by a GEMM checker: the records changed, the values did not.
    Checked,
    /// No checker took it: the value mode if `products`, else nothing.
    Unchecked,
}

/// The checker the machine calls (the "hooks").
pub trait Checker {
    /// Whether the machine computes matrix-op values.
    fn products(&self) -> bool {
        true
    }

    /// After a global or buffer store into a buffer.
    fn store(&mut self, _tables: &mut Tables, _call: &StoreCall<'_>) -> Result<(), TagError> {
        Ok(())
    }

    /// Before the packed-bf16 atomic updates memory (R3c): a checker that
    /// checks no problem leaves it to the machine, tagged data or not.
    fn atomic(
        &mut self,
        _tables: &mut Tables,
        _call: &StoreCall<'_>,
    ) -> Result<AtomicAnswer, TagError> {
        Ok(AtomicAnswer::Scratch)
    }

    /// The matrix-op check; a checker that does not check returns
    /// `Unchecked`.
    fn matrix(
        &mut self,
        _tables: &mut Tables,
        _op: &MatrixOp<'_>,
        _acc: &mut AccState,
    ) -> Result<MatrixOutcome, TagError> {
        Ok(MatrixOutcome::Unchecked)
    }

    /// The problems' errors and logs, for a binding to read (a GEMM checker).
    fn as_gemm(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// The plain checker: every default.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plain;

impl Checker for Plain {}

/// Whether a store's tags hold any record (for a caller's fast path).
pub fn holds_records(tags: &[Vec<Tag>]) -> bool {
    tags.iter().flatten().any(|t| is_provenance_record(*t))
}
