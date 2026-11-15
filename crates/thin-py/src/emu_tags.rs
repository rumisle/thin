//! The functions of `thin._native` over thin-emu's tags and GEMM checker
//!

use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::emu::tables;
use crate::input;
use thin_emu::gemm::{data, GemmHooks};
use thin_emu::tags::{self, combine::combine, Cols, KSet, ProblemFacts, Record};

/// The memory tag of a buffer byte.
#[pyfunction]
fn emu_tag_make(buf: usize, off: u64) -> i64 {
    tags::mem_tag(buf, off)
}

/// A memory tag's buffer.
#[pyfunction]
fn emu_tag_buf(tag: i64) -> i64 {
    tags::tag_buf(tag)
}

/// A memory tag's offset.
#[pyfunction]
fn emu_tag_off(tag: i64) -> u64 {
    tags::tag_off(tag)
}

/// A dword's one tag.
#[pyfunction]
fn emu_lane_source(t: Vec<i64>) -> i64 {
    tags::src(&t)
}

/// The canonical tag of a record: columns one or a tuple.
#[pyfunction]
#[pyo3(signature = (m, n, kset, stat, pid))]
fn emu_record_tag(m: i64, n: Vec<i64>, kset: Vec<i64>, stat: bool, pid: i64) -> i64 {
    let cols = match n.as_slice() {
        [one] => Cols::One(*one),
        many => Cols::Many(many.to_vec()),
    };
    let r = Record {
        m,
        n: cols,
        k: KSet::Set(kset.into_iter().collect()),
        stat,
        pid,
    };
    tables().record(&r)
}

/// (m, n, kset, stat, pid) of a record tag.
#[pyfunction]
fn emu_record_of(tag: i64) -> (i64, Vec<i64>, Vec<i64>, bool, i64) {
    let r = tables().record_of(tag);
    let k = r.k.set().into_iter().collect();
    (r.m, r.n.list(), k, r.stat, r.pid)
}

/// The poison record of a reason.
#[pyfunction]
fn emu_poison_tag(msg: &str) -> i64 {
    tables().poison(msg)
}

/// combine per element.
#[pyfunction]
fn emu_union_records(xt: Vec<i64>, yt: Vec<i64>, stat: bool) -> Vec<i64> {
    let mut t = tables();
    xt.iter()
        .zip(&yt)
        .map(|(x, y)| combine(&mut t, *x, *y, stat))
        .collect()
}

/// A sum record's leaves, left to right.
#[pyfunction]
fn emu_sum_sources(tag: i64) -> Vec<i64> {
    tables().sources(tag)
}

/// The sum rule of two operands' lanes.
#[pyfunction]
fn emu_sum_record_tags(xt: Vec<[i64; 4]>, yt: Vec<[i64; 4]>) -> Vec<[i64; 4]> {
    let mut t = tables();
    xt.iter()
        .zip(&yt)
        .map(|(x, y)| tags::rules::sum_lane(&mut t, x, y))
        .collect()
}

/// Registers a problem's facts; its pid.
#[pyfunction]
#[pyo3(signature = (k, n, source))]
fn emu_register_problem(k: i64, n: Option<i64>, source: Option<i64>) -> i64 {
    tables().register(ProblemFacts { k, n, source })
}

/// A layout's decode of offsets: row, k, sub arrays.
#[pyfunction]
fn emu_layout_decode<'py>(
    py: Python<'py>,
    layout: &str,
    offsets: PyReadonlyArrayDyn<'_, i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let l = data::layout_text(layout).map_err(PyValueError::new_err)?;
    let d: Vec<_> = input(&offsets)?
        .iter()
        .map(|o| l.decode(*o as u64))
        .collect();
    let out = PyDict::new(py);
    out.set_item("row", PyArray1::from_iter(py, d.iter().map(|x| x.row)))?;
    out.set_item("k", PyArray1::from_iter(py, d.iter().map(|x| x.k)))?;
    out.set_item("sub", PyArray1::from_iter(py, d.iter().map(|x| x.sub)))?;
    Ok(out)
}

/// A problem's results: (errors, store log, stores).
type Results = (
    Vec<String>,
    Vec<(i64, i64, Option<i64>)>,
    Vec<((i64, i64, Option<i64>), u64)>,
);

/// The GEMM checker over problems (JSON) and the memory's buffer names.
#[pyclass(module = "thin._native", unsendable)]
pub(crate) struct EmuGemm {
    /// The checker, while not lent to a run.
    pub(crate) inner: Option<GemmHooks>,
}

#[pymethods]
impl EmuGemm {
    #[new]
    fn new(problems: &str, names: Vec<String>) -> PyResult<EmuGemm> {
        let probs = data::problems_text(problems).map_err(PyValueError::new_err)?;
        Ok(EmuGemm {
            inner: Some(GemmHooks::new(&mut tables(), probs, names)),
        })
    }

    /// Each problem's results.
    fn results(&self) -> Vec<Results> {
        let probs = self.inner.as_ref().map_or(&[][..], |g| &g.probs[..]);
        probs
            .iter()
            .map(|p| (p.errors.clone(), p.store_log.clone(), p.stores().to_vec()))
            .collect()
    }
}

/// Adds this module's functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmuGemm>()?;
    m.add_function(wrap_pyfunction!(emu_tag_make, m)?)?;
    m.add_function(wrap_pyfunction!(emu_tag_buf, m)?)?;
    m.add_function(wrap_pyfunction!(emu_tag_off, m)?)?;
    m.add_function(wrap_pyfunction!(emu_lane_source, m)?)?;
    m.add_function(wrap_pyfunction!(emu_record_tag, m)?)?;
    m.add_function(wrap_pyfunction!(emu_record_of, m)?)?;
    m.add_function(wrap_pyfunction!(emu_poison_tag, m)?)?;
    m.add_function(wrap_pyfunction!(emu_union_records, m)?)?;
    m.add_function(wrap_pyfunction!(emu_sum_sources, m)?)?;
    m.add_function(wrap_pyfunction!(emu_sum_record_tags, m)?)?;
    m.add_function(wrap_pyfunction!(emu_register_problem, m)?)?;
    m.add_function(wrap_pyfunction!(emu_layout_decode, m)?)
}
