//! The functions of `thin._native` over thin-codegen: the
//! text-in entry points here, the pipeline over a Builder in
//! [`pipeline`].

mod pipeline;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use thin_codegen::checker::Record;
use thin_codegen::json::{self, Json};
use thin_codegen::{manifest, object};
use thin_formats::Value;

use crate::raise;

fn asm(py: Python<'_>, msg: String) -> PyErr {
    raise(py, "thin.ir", "AsmError", msg)
}

/// The code object of a `.s` text.
#[pyfunction]
fn code_object<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let image = object::code_object(text).map_err(|m| asm(py, m))?;
    Ok(PyBytes::new(py, &image))
}

/// The `.text` bytes of a `.s`.
#[pyfunction]
fn encode<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = object::source(text)
        .and_then(|f| object::encode(&f))
        .map_err(|m| asm(py, m))?;
    Ok(PyBytes::new(py, &bytes))
}

/// The 64-byte descriptor of a `.s`.
#[pyfunction]
fn descriptor<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let d = object::source(text)
        .and_then(|f| object::descriptor(&f))
        .map_err(|m| asm(py, m))?;
    Ok(PyBytes::new(py, &d))
}

/// A metadata value as JSON.
fn json_of(v: &Value) -> Json {
    match v {
        Value::Nil => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::Int(*i),
        Value::Str(s) => Json::Str(s.clone()),
        Value::Array(a) => Json::List(a.iter().map(json_of).collect()),
        Value::Map(m) => Json::Map(m.iter().map(|(k, x)| (k.clone(), json_of(x))).collect()),
    }
}

/// The metadata of a `.s`, as JSON text.
#[pyfunction]
fn metadata(py: Python<'_>, text: &str) -> PyResult<String> {
    let doc = object::source(text)
        .and_then(|f| object::metadata(&f))
        .map_err(|m| asm(py, m))?;
    Ok(json_of(&doc).dumps())
}

/// A manifest's whole record of JSON text, as JSON text.
#[pyfunction]
fn manifest_record(text: &str) -> PyResult<String> {
    let r = json::parse(text).and_then(manifest::record);
    r.map(|j| j.dumps())
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// A record's manifest line.
#[pyfunction]
fn manifest_line(text: &str) -> PyResult<String> {
    let r = json::parse(text).and_then(manifest::record);
    r.map(|j| manifest::line(&j))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// `split_manifest(text)`: the body and the record's JSON, or none.
#[pyfunction]
fn split_manifest(text: &str) -> PyResult<(String, Option<String>)> {
    let (body, rec) = manifest::split(text).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((body, rec.map(|r| r.dumps())))
}

/// gfx1201's split-counter wait check: (record, message).
#[pyfunction]
fn counter_waits(text: &str) -> Vec<(Record, String)> {
    thin_codegen::counter_waits(text)
}

/// Test-only: turns the hold of in-flight loads on or off; returns the
/// previous setting.
#[pyfunction]
fn hold_loads(on: bool) -> bool {
    thin_codegen::place::hold_loads(on)
}

/// Adds this module's functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(code_object, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(metadata, m)?)?;
    m.add_function(wrap_pyfunction!(manifest_record, m)?)?;
    m.add_function(wrap_pyfunction!(manifest_line, m)?)?;
    m.add_function(wrap_pyfunction!(split_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(hold_loads, m)?)?;
    m.add_function(wrap_pyfunction!(counter_waits, m)?)?;
    pipeline::register(m)
}
