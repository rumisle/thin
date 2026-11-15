//! The functions of `thin._native` over code objects:
//! the msgpack metadata conversion, reading, writing and the machine
//! number.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyInt, PyList, PyString};
use thin_formats::Value;

use super::error;
use crate::raise;

/// A metadata value refused by msgpack: always a
/// CodeObjectError.
fn refused(py: Python<'_>, why: String) -> PyErr {
    raise(
        py,
        "thin.formats",
        "CodeObjectError",
        format!("msgpack: {why}"),
    )
}

/// A Python metadata value as msgpack data. Types are
/// exact (a bool is not an int, a str or int subclass is refused), and any
/// refusal, at any depth, is a CodeObjectError.
fn to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = value.py();
    if value.is_none() {
        Ok(Value::Nil)
    } else if value.is_exact_instance_of::<PyBool>() {
        Ok(Value::Bool(value.extract()?))
    } else if value.is_exact_instance_of::<PyInt>() {
        let int = value.extract::<i128>();
        int.map(Value::Int)
            .map_err(|_| refused(py, format!("{value} does not fit 64 bits")))
    } else if value.is_exact_instance_of::<PyString>() {
        Ok(Value::Str(value.extract()?))
    } else if let Ok(list) = value.cast_exact::<PyList>() {
        list.iter()
            .map(|item| to_value(&item))
            .collect::<PyResult<_>>()
            .map(Value::Array)
    } else if let Ok(dict) = value.cast_exact::<PyDict>() {
        let entry = |(k, v): (Bound<'_, PyAny>, Bound<'_, PyAny>)| {
            let key = k
                .cast_exact::<PyString>()
                .map_err(|_| refused(py, format!("the map key {k} is not a str")))?;
            Ok((key.to_string(), to_value(&v)?))
        };
        dict.iter()
            .map(entry)
            .collect::<PyResult<_>>()
            .map(Value::Map)
    } else {
        let kind = value.get_type().name()?;
        Err(refused(
            py,
            format!("a {kind} is not a metadata value (None, bool, int, str, list, dict)"),
        ))
    }
}

/// msgpack data as a Python value.
fn from_value<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Nil => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let items = items.iter().map(|i| from_value(py, i));
            PyList::new(py, items.collect::<PyResult<Vec<_>>>()?)?.into_any()
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                dict.set_item(k, from_value(py, v)?)?;
            }
            dict.into_any()
        }
    })
}

/// A section as Python receives it: (name, type, flags, addr, offset, size, link).
type SectionRow = (String, u32, u64, u64, u64, u64, u32);

/// A code object's parts as Python receives them: (ident, e_flags,
/// sections, symbol, descriptor, metadata, text, (lds, private, kernarg)).
type CodeObjectRow<'py> = (
    Bound<'py, PyBytes>,
    u32,
    Vec<SectionRow>,
    String,
    Bound<'py, PyBytes>,
    Bound<'py, PyAny>,
    Bound<'py, PyBytes>,
    (u32, u32, u32),
);

/// A code object's parts.
#[pyfunction]
fn read_code_object<'py>(py: Python<'py>, img: &[u8]) -> PyResult<CodeObjectRow<'py>> {
    let o = thin_formats::read(img).map_err(|e| error(py, e))?;
    let section =
        |s: thin_formats::Section| (s.name, s.kind, s.flags, s.addr, s.offset, s.size, s.link);
    let sizes = (o.lds(), o.private(), o.kernarg());
    Ok((
        PyBytes::new(py, &o.ident),
        o.e_flags,
        o.sections.into_iter().map(section).collect(),
        o.symbol,
        PyBytes::new(py, &o.descriptor),
        from_value(py, &o.metadata)?,
        PyBytes::new(py, &o.text),
        sizes,
    ))
}

/// The normalized code as Python receives it: (ident, e_flags, text,
/// descriptor, metadata, [(name, digest)]).
type CodeRow<'py> = (
    Bound<'py, PyBytes>,
    u32,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    String,
    Vec<(&'static str, String)>,
);

/// The normalized "what runs" of a code object.
#[pyfunction]
fn code<'py>(py: Python<'py>, img: &[u8]) -> PyResult<CodeRow<'py>> {
    let c = thin_formats::code(img).map_err(|e| error(py, e))?;
    let digests = thin_formats::Code::DIGESTS
        .into_iter()
        .zip(c.digests())
        .collect();
    Ok((
        PyBytes::new(py, &c.ident),
        c.e_flags,
        PyBytes::new(py, &c.text),
        PyBytes::new(py, &c.descriptor),
        c.metadata,
        digests,
    ))
}

/// The code object of one kernel.
#[pyfunction]
fn write_code_object<'py>(
    py: Python<'py>,
    symbol: &str,
    text: &[u8],
    descriptor: &[u8],
    metadata: &Bound<'py, PyAny>,
    e_flags: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let metadata = to_value(metadata)?;
    let machines = &thin_targets::MACHINES;
    let image = thin_formats::write(symbol, text, descriptor, &metadata, e_flags, machines)
        .map_err(|e| error(py, e))?;
    Ok(PyBytes::new(py, &image))
}

/// The target an ELF's e_flags select.
#[pyfunction]
fn target_of(py: Python<'_>, e_flags: u32) -> PyResult<&'static str> {
    thin_formats::target_of(e_flags, &thin_targets::MACHINES).map_err(|e| error(py, e))
}

/// Adds this module's functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for function in [
        wrap_pyfunction!(read_code_object, m)?,
        wrap_pyfunction!(code, m)?,
        wrap_pyfunction!(write_code_object, m)?,
        wrap_pyfunction!(target_of, m)?,
    ] {
        m.add_function(function)?;
    }
    Ok(())
}
