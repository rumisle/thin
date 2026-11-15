//! The functions of `thin._native` over thin-emu: reading,
//! numbers, memory, machines and clusters. The GEMM checker's binding is
//! `emu_tags`.

mod memory;
mod numbers;
mod program;
mod run;

use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::raise;

/// An emulator error as the class of its cause; a reading
/// error is a `ValueError`.
pub(crate) fn emu_error(py: Python<'_>, error: thin_emu::EmuError) -> PyErr {
    use thin_emu::Cause as C;
    let class = match error.cause {
        None => return PyValueError::new_err(error.message),
        Some(C::Fault) => "EmuError",
        Some(C::Emulator) => "EmuLimit",
        Some(C::Deadlock) => "Deadlock",
        Some(C::Ipc) => "IpcPolicy",
        Some(C::Scope) => "ScopeModel",
        Some(C::InFlight) => "RegisterInFlight",
    };
    raise(py, "thin.emu", class, error.message)
}

/// The process's provenance tables: append-only for
/// the life of the process, shared by every machine and run.
static TABLES: LazyLock<Mutex<thin_emu::Tables>> = LazyLock::new(Mutex::default);

/// The process's tables, locked.
pub(crate) fn tables() -> MutexGuard<'static, thin_emu::Tables> {
    TABLES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A test-only context manager whose block runs on fresh tables; the
/// process's come back at its end. Nothing else resets them.
#[pyclass(module = "thin._native")]
struct FreshTables {
    saved: Option<thin_emu::Tables>,
}

#[pymethods]
impl FreshTables {
    #[new]
    fn new() -> FreshTables {
        FreshTables { saved: None }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.saved = Some(std::mem::take(&mut *tables()));
        slf
    }

    #[pyo3(signature = (*_exc))]
    fn __exit__(&mut self, _exc: &Bound<'_, pyo3::types::PyTuple>) {
        if let Some(saved) = self.saved.take() {
            *tables() = saved;
        }
    }
}

/// Adds this module's functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FreshTables>()?;
    program::register(m)?;
    numbers::register(m)?;
    memory::register(m)?;
    run::register(m)
}
