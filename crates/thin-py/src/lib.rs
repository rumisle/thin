//! The Python extension module `thin._native`.
//!
//! This crate converts types, maps errors and manages object lifetime, and
//! nothing else: each function unpacks its arguments, calls one core entry
//! point and packs the result. Large arrays are borrowed from numpy without
//! a copy, results are written into arrays Python allocated, and the GIL is
//! released around each core call. The
//! exception classes belong to the Python package; this module looks them
//! up and raises them.
//!
//! There is one module per component, named after it (a directory when it
//! outgrows one file); this root holds only the shared helpers and the
//! module's init. `tools/check_binding.py` holds every module to the rules
//! in that script's docstring.

mod codegen;
mod dsl;
mod emu;
mod emu_tags;
mod formats;
mod ir;
mod target;

use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Raises the Python class `class` of module `module` with one message.
pub(crate) fn raise(py: Python<'_>, module: &str, class: &str, message: String) -> PyErr {
    let instance = py
        .import(module)
        .and_then(|m| m.getattr(class))
        .and_then(|c| c.call1((message,)));
    match instance {
        Ok(instance) => PyErr::from_value(instance),
        Err(lookup_failed) => lookup_failed,
    }
}

/// The elements of a borrowed, C-contiguous input array.
pub(crate) fn input<'a, T: numpy::Element>(
    array: &'a PyReadonlyArrayDyn<'_, T>,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyTypeError::new_err("an array that is not C-contiguous"))
}

/// The elements of a borrowed, C-contiguous output array.
pub(crate) fn output<'a, T: numpy::Element>(
    array: &'a mut PyReadwriteArrayDyn<'_, T>,
) -> PyResult<&'a mut [T]> {
    let contiguous = array.is_c_contiguous();
    match array.as_slice_mut() {
        Ok(slice) if contiguous => Ok(slice),
        _ => Err(PyTypeError::new_err(
            "an output array that is not C-contiguous",
        )),
    }
}

/// Builds the extension module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    formats::register(m)?;
    target::register(m)?;
    ir::register(m)?;
    dsl::register(m)?;
    codegen::register(m)?;
    emu::register(m)?;
    emu_tags::register(m)
}
