//! Statements, control flow, lane regions, wave blocks and waits
//!

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::flow::{Cond, LoopArgs};
use thin_dsl::lanes::{LaneCond, Save};
use thin_dsl::waits::{Thing, Wait};

use super::builder::DslBuilder;
use super::convert::{arg, ok, reg, tag, tree, typed};
use super::trees::DslTree;
use super::typed::DslTyped;
use crate::ir::Ins;

/// A branch condition from `("t", tree)`, `("i", ins)` or `("o", repr)`.
fn cond(t: &Bound<'_, PyTuple>) -> PyResult<Cond> {
    let kind: String = t.get_item(0)?.extract()?;
    let item = t.get_item(1)?;
    Ok(match kind.as_str() {
        "t" => Cond::Tree(item.extract::<PyRef<'_, DslTree>>()?.t.clone()),
        "i" => Cond::Instr(Box::new(
            (**item.extract::<PyRef<'_, Ins>>()?.instr()).clone(),
        )),
        _ => Cond::Other(item.extract()?),
    })
}

/// `x[...] = tree`: a statement.
#[pyfunction]
#[pyo3(signature = (b, dst, x, comment=None))]
fn assign(
    mut b: PyRefMut<'_, DslBuilder>,
    dst: Bound<'_, PyAny>,
    x: Bound<'_, PyTuple>,
    comment: Option<String>,
) -> PyResult<()> {
    let d = typed(&dst)?;
    let t = tree(&x)?;
    ok(b.b.assign(&d, t, comment))
}

/// `b.when(c)` / `b.unless(c)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn when(
    mut b: PyRefMut<'_, DslBuilder>,
    c: Bound<'_, PyTuple>,
    display: &str,
    unless: bool,
    name: &str,
    comment: Option<String>,
    end: Option<String>,
) -> PyResult<()> {
    let c = cond(&c)?;
    ok(b.b.when(&c, display, unless, name, comment, end))
}

/// `b.elif_(c)`, or (no condition) `b.otherwise()`.
#[pyfunction]
fn elif_(
    mut b: PyRefMut<'_, DslBuilder>,
    c: Option<Bound<'_, PyTuple>>,
    display: &str,
    comment: Option<String>,
) -> PyResult<()> {
    let c = c.map(|c| cond(&c)).transpose()?;
    ok(b.b.elif(c.as_ref().map(|c| (c, display)), comment))
}

/// Leaves an arm (`error`: by an exception).
#[pyfunction]
fn leave_arm(mut b: PyRefMut<'_, DslBuilder>, error: bool) -> PyResult<()> {
    ok(b.b.leave_arm(error))
}

/// `b.exit_when(c, to=, comment=)`.
#[pyfunction]
fn exit_when(
    mut b: PyRefMut<'_, DslBuilder>,
    c: Bound<'_, PyTuple>,
    display: &str,
    to: &str,
    comment: Option<String>,
) -> PyResult<()> {
    let c = cond(&c)?;
    ok(b.b.exit_when(&c, display, to, comment))
}

/// A loop's arguments from Python: (n, until and its display, i, start,
/// start given, peel, end, name, conservative).
type LoopIn<'py> = (
    Option<Bound<'py, PyTuple>>,
    Option<(Bound<'py, PyTuple>, String)>,
    Option<Bound<'py, PyAny>>,
    Bound<'py, PyTuple>,
    bool,
    u32,
    Option<String>,
    Option<String>,
    bool,
);

/// Opens a loop; returns the copies of the body to run.
#[pyfunction]
fn loop_begin(mut b: PyRefMut<'_, DslBuilder>, a: LoopIn<'_>) -> PyResult<u32> {
    let (n, until, i, start, start_given, peel, end, name, conservative) = a;
    let until = until
        .map(|(t, d)| Ok::<_, PyErr>((tree(&t)?, d)))
        .transpose()?;
    let args = LoopArgs {
        n: n.map(|n| arg(&n)).transpose()?,
        until,
        i: i.map(|i| typed(&i)).transpose()?,
        start: arg(&start)?,
        start_given,
        peel,
        end,
        name,
        conservative,
    };
    ok(b.b.loop_begin(args))
}

/// Before copy k of a loop body.
#[pyfunction]
fn loop_copy_begin(mut b: PyRefMut<'_, DslBuilder>, k: u32) -> PyResult<()> {
    ok(b.b.loop_copy_begin(k))
}

/// After copy k of a loop body.
#[pyfunction]
fn loop_copy_end(mut b: PyRefMut<'_, DslBuilder>, k: u32) -> PyResult<()> {
    ok(b.b.loop_copy_end(k))
}

/// The loop body was left by a Python `break`.
#[pyfunction]
fn loop_broken(mut b: PyRefMut<'_, DslBuilder>) -> PyResult<()> {
    ok(b.b.loop_broken())
}

/// Closes a loop.
#[pyfunction]
fn loop_end(mut b: PyRefMut<'_, DslBuilder>) -> PyResult<()> {
    ok(b.b.loop_end())
}

/// `lp.i`.
#[pyfunction]
fn loop_index(b: PyRef<'_, DslBuilder>) -> Option<DslTyped> {
    b.b.loop_index().map(|t| DslTyped { t })
}

/// `lp.break_when(c, comment=)`.
#[pyfunction]
fn loop_break(
    mut b: PyRefMut<'_, DslBuilder>,
    c: Bound<'_, PyTuple>,
    display: &str,
    comment: Option<String>,
) -> PyResult<()> {
    let c = cond(&c)?;
    ok(b.b.loop_break(&c, display, comment))
}

/// Opens `b.steady_state_check`'s body; returns the passes to run.
#[pyfunction]
fn steady_begin(mut b: PyRefMut<'_, DslBuilder>, stem: &str, conservative: bool) -> PyResult<u32> {
    ok(b.b.steady_begin(stem, conservative))
}

/// Closes `b.steady_state_check`'s body.
#[pyfunction]
fn steady_end(mut b: PyRefMut<'_, DslBuilder>) {
    b.b.steady_end();
}

/// A lane condition from `("n",)`, `("t", tree)`, `("i", ins)`, `("int",
/// v)`, `("s", text)` or `("m", mask value)`.
fn lane_cond(t: &Bound<'_, PyTuple>) -> PyResult<LaneCond> {
    let kind: String = t.get_item(0)?.extract()?;
    let item = |k: usize| t.get_item(k);
    Ok(match kind.as_str() {
        "n" => LaneCond::None,
        "t" => LaneCond::Tree(item(1)?.extract::<PyRef<'_, DslTree>>()?.t.clone()),
        "i" => LaneCond::Instr(Box::new(
            (**item(1)?.extract::<PyRef<'_, Ins>>()?.instr()).clone(),
        )),
        "int" => LaneCond::Int(item(1)?.extract()?),
        "m" => LaneCond::Mask(typed(&item(1)?)?),
        _ => LaneCond::Str(item(1)?.extract()?),
    })
}

/// `save=` from `("n",)`, `("fresh",)` or `("r", reg, repr)`.
fn save(t: &Bound<'_, PyTuple>) -> PyResult<Save> {
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "n" => Save::None,
        "fresh" => Save::Fresh,
        _ => Save::Reg(reg(t.get_item(1)?.extract()?), t.get_item(2)?.extract()?),
    })
}

/// Opens `b.lanes(*conds, save=, skip=, comment=)`; returns whether a
/// region opened.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn lanes_begin(
    mut b: PyRefMut<'_, DslBuilder>,
    conds: Vec<Bound<'_, PyTuple>>,
    display: Vec<String>,
    sv: Bound<'_, PyTuple>,
    skip: bool,
    comment: &str,
) -> PyResult<bool> {
    let conds = conds.iter().map(lane_cond).collect::<PyResult<Vec<_>>>()?;
    ok(b.b.lanes_begin(conds, &display, save(&sv)?, skip, comment))
}

/// Closes the innermost lane region (`error`: by an exception).
#[pyfunction]
fn lanes_end(mut b: PyRefMut<'_, DslBuilder>, error: bool) -> PyResult<()> {
    ok(b.b.lanes_end(error))
}

/// `b.narrow(cond, comment=)`.
#[pyfunction]
fn narrow(
    mut b: PyRefMut<'_, DslBuilder>,
    c: Bound<'_, PyTuple>,
    display: &str,
    comment: &str,
) -> PyResult<()> {
    let c = lane_cond(&c)?;
    ok(b.b.narrow(c, display, comment))
}

/// Opens `b.raw_exec(why)`.
#[pyfunction]
fn raw_exec_begin(mut b: PyRefMut<'_, DslBuilder>, why: &str) -> PyResult<()> {
    ok(b.b.raw_exec_begin(why))
}

/// Closes `b.raw_exec`.
#[pyfunction]
fn raw_exec_end(mut b: PyRefMut<'_, DslBuilder>) {
    b.b.raw_exec_end();
}

/// Opens `b.waves(*waves)`.
#[pyfunction]
fn waves_begin(mut b: PyRefMut<'_, DslBuilder>, waves: Vec<i128>) -> PyResult<()> {
    ok(b.b.waves_begin(&waves))
}

/// Closes a wave block (`error`: by an exception).
#[pyfunction]
fn waves_end(mut b: PyRefMut<'_, DslBuilder>, error: bool) -> PyResult<()> {
    ok(b.b.waves_end(error))
}

/// A wait's argument from `("c", name)`, `("s", text)`, `("r", reg)`,
/// `("g", region name)`, `("l", [things])` or `("o", repr)`.
fn thing(t: &Bound<'_, PyTuple>) -> PyResult<Thing> {
    let kind: String = t.get_item(0)?.extract()?;
    let item = t.get_item(1)?;
    Ok(match kind.as_str() {
        "c" => Thing::Counter(item.extract()?),
        "s" => Thing::Str(item.extract()?),
        "r" => Thing::Reg(reg(item.extract()?)),
        "g" => Thing::Region(item.extract()?),
        "l" => Thing::List(
            item.extract::<Vec<Bound<'_, PyTuple>>>()?
                .iter()
                .map(thing)
                .collect::<PyResult<_>>()?,
        ),
        _ => Thing::Other(item.extract()?),
    })
}

/// `b.wait(*what, left=, tags=, written=, comment=, abl=)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wait(
    mut b: PyRefMut<'_, DslBuilder>,
    what: Vec<Bound<'_, PyTuple>>,
    left: Option<i128>,
    tags: Vec<Bound<'_, PyAny>>,
    written: bool,
    comment: String,
    abl: Vec<String>,
) -> PyResult<()> {
    let what = what.iter().map(thing).collect::<PyResult<Vec<_>>>()?;
    let tags = tags.iter().map(tag).collect::<PyResult<Vec<_>>>()?;
    ok(b.b.wait(
        what,
        Wait {
            left,
            tags,
            written,
            comment,
            abl,
        },
    ))
}

/// `b.land_arguments(why)`.
#[pyfunction]
fn land_arguments(mut b: PyRefMut<'_, DslBuilder>, why: &str) -> PyResult<()> {
    ok(b.b.land_arguments(why))
}

/// Adds the statements.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assign, m)?)?;
    m.add_function(wrap_pyfunction!(when, m)?)?;
    m.add_function(wrap_pyfunction!(elif_, m)?)?;
    m.add_function(wrap_pyfunction!(leave_arm, m)?)?;
    m.add_function(wrap_pyfunction!(exit_when, m)?)?;
    m.add_function(wrap_pyfunction!(loop_begin, m)?)?;
    m.add_function(wrap_pyfunction!(loop_copy_begin, m)?)?;
    m.add_function(wrap_pyfunction!(loop_copy_end, m)?)?;
    m.add_function(wrap_pyfunction!(loop_broken, m)?)?;
    m.add_function(wrap_pyfunction!(loop_end, m)?)?;
    m.add_function(wrap_pyfunction!(loop_index, m)?)?;
    m.add_function(wrap_pyfunction!(loop_break, m)?)?;
    m.add_function(wrap_pyfunction!(steady_begin, m)?)?;
    m.add_function(wrap_pyfunction!(steady_end, m)?)?;
    m.add_function(wrap_pyfunction!(lanes_begin, m)?)?;
    m.add_function(wrap_pyfunction!(lanes_end, m)?)?;
    m.add_function(wrap_pyfunction!(narrow, m)?)?;
    m.add_function(wrap_pyfunction!(raw_exec_begin, m)?)?;
    m.add_function(wrap_pyfunction!(raw_exec_end, m)?)?;
    m.add_function(wrap_pyfunction!(waves_begin, m)?)?;
    m.add_function(wrap_pyfunction!(waves_end, m)?)?;
    m.add_function(wrap_pyfunction!(wait, m)?)?;
    m.add_function(wrap_pyfunction!(land_arguments, m)?)
}
