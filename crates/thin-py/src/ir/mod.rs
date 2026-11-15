//! The functions of `thin._native` over the IR and the printer
//! the value conversions are in [`convert`].

mod convert;

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use thin_ir::print::{self, Named};
use thin_ir::verify::ValueFacts;
use thin_ir::{AttrClass, FieldId, Instr, Snapshot};
use thin_target::{Target, PHYS};

use crate::target::get;
use convert::{
    attribute, attribute_out, class, error, field_out, field_value, operand, operand_out,
};

/// An instruction's native data, shared and immutable.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct Ins {
    target: &'static Target,
    i: Arc<Instr>,
}

impl Ins {
    /// An instruction of a target, for the other modules' results.
    pub(crate) fn of(target: &'static Target, i: Arc<Instr>) -> Ins {
        Ins { target, i }
    }

    /// The native instruction.
    pub(crate) fn instr(&self) -> &Arc<Instr> {
        &self.i
    }

    fn with(&self, i: Instr) -> Ins {
        Ins {
            target: self.target,
            i: Arc::new(i),
        }
    }
}

/// An instruction's derived facts: roles, def and use operand indices,
/// named writes and reads, and (sgpr_valu_def, writes_exec, m0_write).
type Facts = (
    Vec<u32>,
    Vec<u16>,
    Vec<u16>,
    Vec<&'static str>,
    Vec<&'static str>,
    (bool, bool, bool),
);

#[pymethods]
impl Ins {
    #[new]
    #[pyo3(signature = (target, form, ops, fields, mods=None))]
    fn new(
        py: Python<'_>,
        target: &str,
        form: i32,
        ops: Vec<Bound<'_, PyAny>>,
        fields: Vec<(String, Bound<'_, PyAny>)>,
        mods: Option<Vec<u8>>,
    ) -> PyResult<Ins> {
        let target = get(py, target)?;
        let form = target
            .form_id(form)
            .ok_or_else(|| PyValueError::new_err("a form of the target"))?;
        let ops = ops.iter().map(operand).collect::<PyResult<_>>()?;
        let fields = (fields.iter())
            .map(|(k, v)| Ok((FieldId::named(k).map_err(error)?, field_value(v)?)))
            .collect::<PyResult<_>>()?;
        let i = Instr::new(form, ops, fields, mods.map(Into::into));
        Ok(Ins {
            target,
            i: Arc::new(i),
        })
    }

    fn with_ops(&self, ops: Vec<Bound<'_, PyAny>>) -> PyResult<Ins> {
        Ok(self.with(
            self.i
                .with_ops(ops.iter().map(operand).collect::<PyResult<_>>()?),
        ))
    }

    fn put(&self, class: &str, payload: &Bound<'_, PyAny>) -> PyResult<Ins> {
        Ok(self.with(
            self.i
                .with_attr(attribute(class, payload)?)
                .map_err(error)?,
        ))
    }

    fn drop(&self, name: &str) -> PyResult<Ins> {
        Ok(self.with(self.i.without_attr(class(name)?)))
    }

    fn attr(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        self.i
            .attrs()
            .get(class(name)?)
            .map(|a| attribute_out(py, &a))
            .transpose()
    }

    /// Its form's (id, name).
    fn form(&self) -> (i32, &'static str) {
        let f = self.i.form();
        (f.id, f.name.as_str())
    }

    fn attr_names(&self) -> Vec<&'static str> {
        AttrClass::ALL
            .into_iter()
            .filter(|&c| self.i.attrs().has(c))
            .map(AttrClass::name)
            .collect()
    }

    fn ops<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.i
            .ops()
            .iter()
            .enumerate()
            .map(|(k, op)| operand_out(py, k, op))
            .collect()
    }

    fn fields(&self, py: Python<'_>) -> PyResult<Vec<(&'static str, Py<PyAny>)>> {
        self.i
            .fields()
            .iter()
            .map(|(id, v)| Ok((id.name(), field_out(py, v)?)))
            .collect()
    }

    fn field(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        self.i
            .field_named(name)
            .map(|v| field_out(py, v))
            .transpose()
    }

    fn mods<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.i.mods().map(|m| PyBytes::new(py, m))
    }

    fn facts(&self) -> Facts {
        let i = &self.i;
        let names = |l: &[usize]| l.iter().map(|&p| PHYS[p].name).collect();
        let flags = (i.sgpr_valu_def(), i.writes_exec(), i.m0_write());
        (
            i.roles().iter().map(|&r| r.into()).collect(),
            i.def_ops().to_vec(),
            i.use_ops().to_vec(),
            names(i.writes()),
            names(i.reads()),
            flags,
        )
    }

    fn width_errors(&self, reprs: Vec<String>) -> Vec<String> {
        self.i.width_errors(|k| reprs[k].clone())
    }

    fn line(&self, regs: Vec<String>) -> PyResult<String> {
        print::line(&self.i, &regs).map_err(error)
    }
}

/// The instructions' shared data.
fn instrs(list: &[PyRef<'_, Ins>]) -> Arc<[Arc<Instr>]> {
    list.iter().map(|i| i.i.clone()).collect()
}

/// The verifier's verdict on a body: the first failing instruction's index
/// and message; `values` maps each value number to (repr,
/// dwords, alignment, pinned).
#[pyfunction]
fn verify(
    py: Python<'_>,
    target: &str,
    body: Vec<PyRef<'_, Ins>>,
    values: HashMap<i32, (String, u32, u32, bool)>,
    printable: bool,
) -> PyResult<Option<(usize, String)>> {
    let values = (values.into_iter())
        .map(|(k, (repr, n, align, pinned))| {
            (
                k,
                ValueFacts {
                    repr,
                    n,
                    align,
                    pinned,
                },
            )
        })
        .collect();
    Ok(thin_ir::verify::verify(
        get(py, target)?,
        &instrs(&body),
        &values,
        printable,
    ))
}

/// A snapshot, opaque to Python but for its wave classes.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct Snap {
    s: Arc<Snapshot>,
}

impl Snap {
    /// The handle of a snapshot made elsewhere.
    pub(crate) fn of(s: Snapshot) -> Snap {
        Snap { s: Arc::new(s) }
    }

    /// The handle of a shared snapshot.
    pub(crate) fn of_arc(s: Arc<Snapshot>) -> Snap {
        Snap { s }
    }
}

#[pymethods]
impl Snap {
    /// The snapshot of a free list of instructions.
    #[staticmethod]
    fn of_list(py: Python<'_>, target: &str, body: Vec<PyRef<'_, Ins>>) -> PyResult<Snap> {
        let s = Snapshot::of_list(get(py, target)?, instrs(&body)).map_err(error)?;
        Ok(Snap { s: Arc::new(s) })
    }

    /// The wave classes: per class its (index, taken) pairs, or none.
    fn wave_classes(&self) -> Vec<thin_ir::WaveClass> {
        self.s.wave_classes()
    }
}

/// A register spelled physically at its value's base.
#[pyfunction]
fn placed_text(
    repr: &str,
    file: char,
    base: u32,
    lo: u32,
    n: u32,
    value_n: u32,
) -> PyResult<String> {
    print::placed_text(repr, file, base, lo, n, value_n).map_err(error)
}

/// A register spelled by name: (file, name, index, group size, lo, n,
/// value dwords, short loc).
#[pyfunction]
fn name_text(r: (char, String, u32, usize, u32, u32, u32, Option<String>)) -> String {
    let (file, name, index, group, lo, n, value_n, short) = r;
    print::name_text(&Named {
        file,
        name: &name,
        index,
        group,
        lo,
        n,
        value_n,
        short: short.as_deref(),
    })
}

/// The .s line of a line with its comment.
#[pyfunction]
#[pyo3(signature = (line, comment=None))]
fn commented(line: &str, comment: Option<&str>) -> String {
    print::commented(line, comment)
}

/// The line with label serials masked.
#[pyfunction]
fn mask_serials(line: &str) -> String {
    print::mask_serials(line)
}

/// Adds this module's classes and functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Ins>()?;
    m.add_class::<Snap>()?;
    for function in [
        wrap_pyfunction!(verify, m)?,
        wrap_pyfunction!(placed_text, m)?,
        wrap_pyfunction!(name_text, m)?,
        wrap_pyfunction!(commented, m)?,
        wrap_pyfunction!(mask_serials, m)?,
    ] {
        m.add_function(function)?;
    }
    Ok(())
}
