//! The refusals of the language.

use std::fmt;

/// The class of a refusal: the IR's diagnostics and the
/// Python errors of a misused API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// An expression or a statement.
    Expr,
    /// A control construct.
    Flow,
    /// The LDS layout, an address, an LDS access.
    Mem,
    /// Any other build-time refusal.
    Schedule,
    /// A register assignment refusal.
    Assign,
    /// A bad argument value.
    Value,
    /// A bad argument type.
    Type,
    /// A bad index.
    Index,
    /// A float constant past f32.
    Overflow,
    /// An attribute an object lacks.
    Attribute,
}

impl Class {
    /// The Python exception class's name.
    pub fn name(self) -> &'static str {
        match self {
            Class::Expr => "ExprError",
            Class::Flow => "FlowError",
            Class::Mem => "MemError",
            Class::Schedule => "ScheduleError",
            Class::Assign => "AssignError",
            Class::Value => "ValueError",
            Class::Type => "TypeError",
            Class::Index => "IndexError",
            Class::Overflow => "OverflowError",
            Class::Attribute => "AttributeError",
        }
    }
}

/// A refusal: its class and its message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// The class.
    pub class: Class,
    /// The message.
    pub msg: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class.name(), self.msg)
    }
}

impl std::error::Error for Error {}

/// A result of the language.
pub type Result<T> = std::result::Result<T, Error>;

/// A refusal of that class.
pub fn err<T>(class: Class, msg: impl Into<String>) -> Result<T> {
    Err(Error {
        class,
        msg: msg.into(),
    })
}

/// An ExprError.
pub fn expr<T>(msg: impl Into<String>) -> Result<T> {
    err(Class::Expr, msg)
}

/// A FlowError.
pub fn flow<T>(msg: impl Into<String>) -> Result<T> {
    err(Class::Flow, msg)
}

/// A MemError.
pub fn mem<T>(msg: impl Into<String>) -> Result<T> {
    err(Class::Mem, msg)
}

/// A ScheduleError.
pub fn schedule<T>(msg: impl Into<String>) -> Result<T> {
    err(Class::Schedule, msg)
}

/// A ValueError.
pub fn value<T>(msg: impl Into<String>) -> Result<T> {
    err(Class::Value, msg)
}

impl From<thin_ir::Error> for Error {
    fn from(e: thin_ir::Error) -> Error {
        match e {
            thin_ir::Error::Type(msg) => Error {
                class: Class::Type,
                msg,
            },
            thin_ir::Error::Value(msg) => Error {
                class: Class::Value,
                msg,
            },
        }
    }
}
