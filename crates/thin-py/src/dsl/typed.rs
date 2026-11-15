//! Typed values.

use pyo3::prelude::*;
use thin_dsl::values::Typed;

use super::convert::{ok, reg_out};

/// The author's handle on some dwords of a group.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslTyped {
    pub(crate) t: Typed,
}

fn wrap(t: Typed) -> DslTyped {
    DslTyped { t }
}

#[pymethods]
impl DslTyped {
    /// `<file>:<group><part>`.
    fn __repr__(&self) -> String {
        self.t.repr()
    }

    /// Its file letter.
    #[getter]
    fn file(&self) -> char {
        self.t.file
    }

    /// Its group's name.
    #[getter]
    fn name(&self) -> &str {
        &self.t.group
    }

    /// Its type's name.
    #[getter]
    fn ty(&self) -> &'static str {
        self.t.ty.name()
    }

    /// Its elements.
    #[getter]
    fn n(&self) -> u32 {
        self.t.n
    }

    /// Its dword offset into the group's first value.
    #[getter]
    fn sub(&self) -> u32 {
        self.t.sub
    }

    /// Its label.
    #[getter]
    fn label(&self) -> Option<&str> {
        self.t.label.as_deref()
    }

    /// Its part.
    #[getter]
    fn part(&self) -> &str {
        &self.t.part
    }

    /// Its dwords.
    #[getter]
    fn width(&self) -> u32 {
        self.t.width()
    }

    /// The author's line.
    #[getter]
    fn where_(&self) -> &str {
        &self.t.where_
    }

    /// Its Builder's identity.
    #[getter]
    fn builder(&self) -> u64 {
        self.t.builder
    }

    /// Its group's first value's id.
    #[getter]
    fn value(&self) -> usize {
        self.t.value
    }

    /// Whether it is a divisor argument.
    #[getter]
    fn divisor(&self) -> bool {
        self.t.divisor
    }

    /// Its register as (value, file, lo, n).
    fn reg(&self) -> (i32, char, u32, u32) {
        reg_out(&self.t.reg())
    }

    /// Element j.
    fn element(&self, j: i64) -> PyResult<DslTyped> {
        ok(self.t.element(j)).map(wrap)
    }

    /// Elements a..b; `written` is the slice as written.
    fn slice(
        &self,
        a: Option<i64>,
        b: Option<i64>,
        step: Option<i64>,
        written: &str,
    ) -> PyResult<DslTyped> {
        ok(self.t.slice(a, b, step, written)).map(wrap)
    }

    /// `.lo` (0) or `.hi` (1).
    fn half(&self, k: u32) -> PyResult<DslTyped> {
        ok(self.t.half(k)).map(wrap)
    }

    /// A divisor's `mul` (1) or `shr` (2).
    fn divisor_part(&self, k: u32) -> DslTyped {
        wrap(self.t.divisor_part(k))
    }
}

/// Adds the typed values.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DslTyped>()
}
