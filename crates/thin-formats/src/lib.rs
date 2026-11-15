//! Element encodings, layouts and AMDHSA code-object containers.
//!
//! Bit-exact encode/decode routines define rounding once. Fast paths use tables
//! generated here rather than implementing their own numerical rules. Inputs are
//! borrowed slices and outputs use caller-owned storage. Host MX quantization,
//! MessagePack and byte-layout walking share these format definitions.
//!
//! This crate depends on no other thin crate. Machine-number data is supplied
//! by callers; no target instance is named here. Layout sets are fixed at build
//! time, with no runtime registration or global mutable state.

mod code_object;
mod decode;
mod dtype;
mod encode;
pub mod layout;
mod msgpack;
mod mx;
mod parallel;

use std::fmt;

pub use code_object::{code, read, target_of, write, Code, CodeObject, Machine, Section};
pub use decode::{decode, decode_table};
pub use dtype::{dtype, Dtype, Encoding, NanKind, DTYPES};
pub use encode::{check_encode, encode, encode_table, EncodeOptions, TABLE_INDEX_BITS};
pub use layout::{coords, from_bytes, nbytes, to_bytes, Layout};
pub use msgpack::{canonical_json, msgpack_read, msgpack_write, Value};
pub use mx::{quantize, scale_bytes, GROUP};

/// A refusal, by the error class gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The wrong kind or type of an argument (`TypeError`).
    Type(String),
    /// A value of the right type that is not allowed (`ValueError`).
    Value(String),
    /// A layout refusal (`LayoutError`).
    Layout(String),
    /// A code-object refusal (`CodeObjectError`).
    CodeObject(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (Error::Type(msg) | Error::Value(msg) | Error::Layout(msg) | Error::CodeObject(msg)) =
            self;
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// The shape of a stack of matrices: (stack, rows, cols).
pub type Shape = (usize, usize, usize);
