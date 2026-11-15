//! Convert kernel values and register files during host synchronization.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::builder::{Deferred, Make};
use thin_dsl::values::{File, Placed};

use super::builder::DslBuilder;
use super::convert::{ok, reg, reg_out, tree, ty, typed};
use super::maker::regs_of;
use super::typed::DslTyped;
use crate::ir::Ins;

fn file(b: &mut DslBuilder, f: char) -> (&mut File, &mut Vec<thin_dsl::values::ValueRec>) {
    let k = "vsa".find(f).unwrap_or(0);
    (&mut b.b.files[k], &mut b.b.values)
}

/// `F.<type>(init, n=, name=, at=, same_as=)`.
#[pyfunction]
#[pyo3(signature = (b, f, t, n, n_repr, name=None, at=None, same_as=None, init=None))]
#[allow(clippy::too_many_arguments)]
fn value_make(
    mut b: PyRefMut<'_, DslBuilder>,
    f: char,
    t: &str,
    n: Option<i64>,
    n_repr: String,
    name: Option<String>,
    at: Option<u32>,
    same_as: Option<Bound<'_, PyAny>>,
    init: Option<Bound<'_, PyTuple>>,
) -> PyResult<DslTyped> {
    let where_ = b.b.where_text.clone();
    let same_as = same_as.map(|y| typed(&y)).transpose()?;
    let m = Make {
        n,
        n_repr,
        name,
        at,
        same_as,
        where_,
    };
    let init = init.map(|x| tree(&x)).transpose()?;
    ok(b.b.make_init(f, ty(t)?, m, init)).map(|t| DslTyped { t })
}

/// `F.alloc(name, count, width, align)`; returns the group's value ids.
#[pyfunction]
fn file_alloc(
    mut b: PyRefMut<'_, DslBuilder>,
    f: char,
    name: &str,
    count: u32,
    width: u32,
    align: u32,
) -> PyResult<Vec<usize>> {
    let loc = b.b.loc;
    let (fl, values) = file(&mut b, f);
    let g = ok(fl.alloc(values, name, count, width, align, loc))?;
    Ok(fl.groups[g].values.clone())
}

/// `F.pin(name, at)`.
#[pyfunction]
fn file_pin(mut b: PyRefMut<'_, DslBuilder>, f: char, name: &str, at: u32) -> PyResult<()> {
    let (fl, values) = file(&mut b, f);
    ok(fl.pin(values, name, at))
}

/// `F.same_as(name, other)`.
#[pyfunction]
fn file_same_as(mut b: PyRefMut<'_, DslBuilder>, f: char, name: &str, other: &str) -> PyResult<()> {
    let (fl, values) = file(&mut b, f);
    ok(fl.same_as(values, name, other))
}

/// `F.reg(name, idx, width, sub)`.
#[pyfunction]
fn file_reg(
    mut b: PyRefMut<'_, DslBuilder>,
    f: char,
    name: &str,
    idx: i64,
    width: Option<u32>,
    sub: u32,
) -> PyResult<(i32, char, u32, u32)> {
    let (fl, values) = file(&mut b, f);
    ok(fl.reg(values, name, idx, width, sub)).map(|r| reg_out(&r))
}

/// `F.value_of(name)`.
#[pyfunction]
fn file_value_of(mut b: PyRefMut<'_, DslBuilder>, f: char, name: &str) -> PyResult<String> {
    let (fl, values) = file(&mut b, f);
    ok(fl.value_of(values, name)).map(|s| s.to_string())
}

/// `F.phys(r)` (`phys`) or the register's number.
#[pyfunction]
fn file_number(
    mut b: PyRefMut<'_, DslBuilder>,
    f: char,
    r: (i32, char, u32, u32),
    repr: &str,
    phys: bool,
) -> PyResult<u32> {
    let (fl, values) = file(&mut b, f);
    let r = reg(r);
    ok(match phys {
        true => fl.phys(values, &r, repr),
        false => fl.number(values, &r, repr),
    })
}

/// `F.used()`.
#[pyfunction]
fn file_used(mut b: PyRefMut<'_, DslBuilder>, f: char) -> u32 {
    let (fl, values) = file(&mut b, f);
    fl.used(values)
}

/// `F.apply(placement)`.
#[pyfunction]
fn file_apply(mut b: PyRefMut<'_, DslBuilder>, f: char, placement: Vec<Placed>) -> PyResult<()> {
    let (fl, values) = file(&mut b, f);
    ok(fl.apply(values, &placement))
}

/// `F.entry(*groups)`.
#[pyfunction]
fn file_entry(
    mut b: PyRefMut<'_, DslBuilder>,
    f: char,
    groups: Vec<(String, u32)>,
) -> PyResult<Vec<DslTyped>> {
    let v = ok(b.b.entry_values(f, &groups))?;
    Ok(v.into_iter().map(|t| DslTyped { t }).collect())
}

/// `F.lanes_covered(*names)`.
#[pyfunction]
fn file_lanes_covered(mut b: PyRefMut<'_, DslBuilder>, f: char, names: Vec<String>) {
    file(&mut b, f).0.lanes_covered(&names);
}

/// A file's facts: (limit, groups as (name, value ids), entered, pinned,
/// lanes covered, placed).
type FileRow = (
    u32,
    Vec<(String, Vec<usize>)>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    bool,
);

/// A file's facts.
#[pyfunction]
fn file_facts(mut b: PyRefMut<'_, DslBuilder>, f: char) -> FileRow {
    let fl = file(&mut b, f).0;
    let names = |xs: &[std::sync::Arc<str>]| xs.iter().map(|s| s.to_string()).collect();
    let groups = fl
        .groups
        .iter()
        .map(|g| (g.name.to_string(), g.values.clone()))
        .collect();
    (
        fl.limit,
        groups,
        names(&fl.entered),
        names(&fl.pinned),
        fl.lanes.clone(),
        fl.placed,
    )
}

/// A value's facts: (file, n, align, name, loc, index, pin, same_as).
type ValueRow = (char, u32, u32, String, u32, u32, Option<u32>, Option<usize>);

/// A value's (pin, same_as).
type Link = (Option<u32>, Option<usize>);

/// The values from id `k` on.
#[pyfunction]
fn values(b: PyRef<'_, DslBuilder>, k: usize) -> Vec<ValueRow> {
    let row = |v: &thin_dsl::values::ValueRec| {
        (
            v.file,
            v.n,
            v.align,
            v.name.to_string(),
            v.loc,
            v.index,
            v.pin,
            v.same_as,
        )
    };
    b.b.values.get(k..).unwrap_or(&[]).iter().map(row).collect()
}

/// Every value's (pin, same_as).
#[pyfunction]
fn value_links(b: PyRef<'_, DslBuilder>) -> Vec<Link> {
    b.b.values.iter().map(|v| (v.pin, v.same_as)).collect()
}

/// A stream row: (id, instruction, loc, registers per operand, whether
/// its loc is the hazard pass's).
type RowOut = (u64, Ins, u32, Vec<Option<(i32, u32, u32)>>, bool);

/// The rows changed since the last sync: the first changed index and the
/// rows from it on.
#[pyfunction]
fn sync(mut b: PyRefMut<'_, DslBuilder>) -> (usize, Vec<RowOut>) {
    let t = b.b.t;
    let (from, rows) = b.b.sync();
    let row = |r: &thin_dsl::builder::Row| {
        (
            r.id,
            Ins::of(t, r.ins.clone()),
            r.loc,
            regs_of(&r.ins),
            r.hazard,
        )
    };
    (from, rows.iter().map(row).collect())
}

/// A record for the pipeline: ("exit", rows, region) or ("peel", stem,
/// label, max, starts, last).
type DeferredOut = (
    &'static str,
    String,
    String,
    u32,
    Vec<Option<u64>>,
    Option<u64>,
    Option<u32>,
);

fn deferred_out(d: &Deferred) -> DeferredOut {
    match d {
        Deferred::Exit { exit, upto, region } => (
            "exit",
            String::new(),
            String::new(),
            0,
            vec![Some(*exit)],
            Some(*upto),
            *region,
        ),
        Deferred::Peel {
            stem,
            label,
            max,
            starts,
            last,
        } => (
            "peel",
            stem.clone(),
            label.clone(),
            *max,
            starts.clone(),
            *last,
            None,
        ),
    }
}

/// The records: loops (stem, first, last, conservative), regions (id,
/// entry, save, checked), and the deferred records.
type Records = (
    Vec<(String, u64, u64, bool)>,
    Vec<(u32, u64, Option<(i32, char, u32, u32)>, bool)>,
    Vec<DeferredOut>,
);

/// The program's records.
#[pyfunction]
fn records(b: PyRef<'_, DslBuilder>) -> Records {
    let loops =
        b.b.loops
            .iter()
            .map(|l| (l.stem.clone(), l.first, l.last, l.conservative))
            .collect();
    let regions =
        b.b.regions
            .iter()
            .map(|r| (r.id, r.entry, r.save.map(|s| reg_out(&s)), r.checked))
            .collect();
    (
        loops,
        regions,
        b.b.deferred.iter().map(deferred_out).collect(),
    )
}

/// Adds the values and the sync.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(value_make, m)?)?;
    m.add_function(wrap_pyfunction!(file_alloc, m)?)?;
    m.add_function(wrap_pyfunction!(file_pin, m)?)?;
    m.add_function(wrap_pyfunction!(file_same_as, m)?)?;
    m.add_function(wrap_pyfunction!(file_reg, m)?)?;
    m.add_function(wrap_pyfunction!(file_value_of, m)?)?;
    m.add_function(wrap_pyfunction!(file_number, m)?)?;
    m.add_function(wrap_pyfunction!(file_used, m)?)?;
    m.add_function(wrap_pyfunction!(file_apply, m)?)?;
    m.add_function(wrap_pyfunction!(file_entry, m)?)?;
    m.add_function(wrap_pyfunction!(file_lanes_covered, m)?)?;
    m.add_function(wrap_pyfunction!(file_facts, m)?)?;
    m.add_function(wrap_pyfunction!(values, m)?)?;
    m.add_function(wrap_pyfunction!(value_links, m)?)?;
    m.add_function(wrap_pyfunction!(sync, m)?)?;
    m.add_function(wrap_pyfunction!(records, m)?)
}
