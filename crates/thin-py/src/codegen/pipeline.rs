//! The pipeline over a Builder.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_codegen::compile::{self, Call};
use thin_codegen::pipeline::{records, Options};
use thin_codegen::spell::{Piece, Refusal};
use thin_codegen::{json, place};
use thin_dsl::values::Placed;

use crate::dsl::args::DslArgs;
use crate::dsl::builder::DslBuilder;
use crate::ir::Snap;

/// The options as Python passes them: (audit, keep_dead, waits_unique,
/// safe_waits, regions, peak_list, peak_report, pressure, lds_wave_order).
type OptionsIn = (bool, bool, bool, bool, bool, usize, bool, bool, bool);

fn options(o: OptionsIn) -> Options {
    let (
        audit,
        keep_dead,
        waits_unique,
        safe_waits,
        regions,
        peak_list,
        peak_report,
        pressure,
        lds_wave_order,
    ) = o;
    Options {
        audit,
        keep_dead,
        waits_unique,
        safe_waits,
        regions,
        peak_list,
        peak_report,
        pressure,
        lds_wave_order,
    }
}

/// A refusal's text pieces as thin.ir's native parts.
fn parts(py: Python<'_>, pieces: &[Piece]) -> PyResult<Vec<Py<PyTuple>>> {
    let part = |p: &Piece| match p {
        Piece::Text(t) => ("text", 0u32, 0u32, t.clone())
            .into_pyobject(py)
            .map(Bound::unbind),
        Piece::Where(loc) => ("where", *loc, 0u32, String::new())
            .into_pyobject(py)
            .map(Bound::unbind),
    };
    pieces.iter().map(part).collect()
}

/// A refusal as its Python exception: thin.ir's classes through
/// `from_native` (locations spelled by `LOCS.where`), else the builtin.
fn err(py: Python<'_>, r: Refusal) -> PyErr {
    let msg = r.text();
    match r.class {
        "Value" => pyo3::exceptions::PyValueError::new_err(msg),
        "Type" => pyo3::exceptions::PyTypeError::new_err(msg),
        "Assertion" => pyo3::exceptions::PyAssertionError::new_err(msg),
        "Index" => pyo3::exceptions::PyIndexError::new_err(msg),
        "Overflow" => pyo3::exceptions::PyOverflowError::new_err(msg),
        "Attribute" => pyo3::exceptions::PyAttributeError::new_err(msg),
        class => native(py, class, r).unwrap_or_else(|e| e),
    }
}

fn native(py: Python<'_>, class: &str, r: Refusal) -> PyResult<PyErr> {
    let ir = py.import("thin.ir")?;
    let place = ir.getattr("LOCS")?.getattr("where")?;
    let notes: Vec<(u32, Vec<Py<PyTuple>>)> = r
        .notes
        .iter()
        .map(|(loc, p)| Ok((*loc, parts(py, p)?)))
        .collect::<PyResult<_>>()?;
    let e = ir.getattr("from_native")?.call1((
        class,
        r.loc,
        parts(py, &r.msg)?,
        notes,
        py.None(),
        place,
    ))?;
    Ok(PyErr::from_value(e))
}

/// `finish(kernel, nops)`.
#[pyfunction]
fn pipeline_finish(
    py: Python<'_>,
    mut b: PyRefMut<'_, DslBuilder>,
    o: OptionsIn,
    nops: bool,
) -> PyResult<()> {
    compile::finish(&mut b.b, &options(o), nops).map_err(|r| err(py, r))
}

/// `place(kernel)`: with `audit`, the placement for the
/// generator's second run.
#[pyfunction]
fn pipeline_place(
    py: Python<'_>,
    mut b: PyRefMut<'_, DslBuilder>,
    o: OptionsIn,
    audit: bool,
) -> PyResult<Option<[Vec<Placed>; 3]>> {
    place::place(&mut b.b, &options(o), audit).map_err(|r| err(py, r))
}

/// The audit of the first run against the second.
#[pyfunction]
fn pipeline_audit(
    py: Python<'_>,
    mut first: PyRefMut<'_, DslBuilder>,
    mut second: PyRefMut<'_, DslBuilder>,
    o: OptionsIn,
) -> PyResult<()> {
    place::audit(&mut first.b, &mut second.b, &options(o)).map_err(|r| err(py, r))
}

/// `rendered(kernel)`: the lines and the snapshot.
#[pyfunction]
fn pipeline_rendered(
    py: Python<'_>,
    mut b: PyRefMut<'_, DslBuilder>,
    o: OptionsIn,
) -> PyResult<(Vec<String>, Snap)> {
    let (lines, s) = compile::rendered(&mut b.b, &options(o)).map_err(|r| err(py, r))?;
    Ok((lines, Snap::of_arc(s)))
}

/// What compile_kernel takes: (symbol, wgsize, args, lds, build JSON,
/// grid, max_grid).
type CallIn<'py> = (
    String,
    u32,
    Option<PyRef<'py, DslArgs>>,
    u64,
    String,
    Vec<u64>,
    Vec<u64>,
);

/// `compile_kernel`: the text, the record's JSON and the summary.
#[pyfunction]
fn pipeline_compile(
    py: Python<'_>,
    mut b: PyRefMut<'_, DslBuilder>,
    c: CallIn<'_>,
    o: OptionsIn,
) -> PyResult<(String, String, String)> {
    let (symbol, wgsize, args, lds, build, grid, max_grid) = c;
    let build = json::parse(&build).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let call = Call {
        symbol: &symbol,
        wgsize,
        args: args.map(|a| a.a.clone()),
        lds,
        build,
        grid,
        max_grid,
    };
    let a = compile::compile(&mut b.b, call, &options(o)).map_err(|r| err(py, r))?;
    Ok((a.text, a.manifest.dumps(), a.summary))
}

/// The kernel's pass times, in the order first recorded.
#[pyfunction]
fn pass_times(mut b: PyRefMut<'_, DslBuilder>) -> Vec<(String, f64)> {
    records(&mut b.b).pass_times.clone()
}

/// The LDS proof's report and the ablation line (`final_report`).
#[pyfunction]
fn final_report(mut b: PyRefMut<'_, DslBuilder>) -> Vec<String> {
    records(&mut b.b).final_report.clone()
}

/// Adds the pipeline's functions.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pipeline_finish, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_place, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_audit, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_rendered, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_compile, m)?)?;
    m.add_function(wrap_pyfunction!(pass_times, m)?)?;
    m.add_function(wrap_pyfunction!(final_report, m)?)
}
