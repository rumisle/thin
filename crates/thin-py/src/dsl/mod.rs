//! The functions of `thin._native` over the kernel-writing language
//! the Builder and its statements, trees, typed values, the
//! maker, the argument block, the entry and the LDS layout. The
//! conversions are in [`convert`].

pub(crate) mod args;
pub(crate) mod builder;
mod convert;
mod kernel;
mod lds;
mod maker;
mod matrix;
mod stmts;
mod trees;
mod typed;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// magic(d) = (m, s, short) for dividing a u32 by the constant d.
#[pyfunction]
fn magic(d: u32) -> PyResult<(u64, u32, bool)> {
    thin_dsl::magic(d).map_err(PyValueError::new_err)
}

/// Adds the language's functions and classes.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(magic, m)?)?;
    m.add_class::<builder::DslBuilder>()?;
    trees::register(m)?;
    typed::register(m)?;
    maker::register(m)?;
    matrix::register(m)?;
    lds::register(m)?;
    stmts::register(m)?;
    args::register(m)?;
    kernel::register(m)
}
