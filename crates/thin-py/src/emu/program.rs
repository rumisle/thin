//! Decoded programs, reading and the claims.

use pyo3::prelude::*;

use super::emu_error;
use crate::target::target_error;
use thin_emu::api::{ins_row, InsRow};
use thin_emu::machine::claims;

/// A decoded program as host values: (instructions, labels, section lines,
/// user SGPR count, label lines, descriptor enables, kernarg SGPR, PC layout).
type ProgramRow = (
    Vec<InsRow>,
    Vec<(String, usize)>,
    Vec<(usize, String, usize)>,
    u32,
    Vec<(usize, usize)>,
    Option<[bool; 3]>,
    Option<u32>,
    Option<(Vec<(u64, u64)>, u64)>,
);

/// The flat rows of a program.
fn program_row(p: thin_emu::Program) -> ProgramRow {
    (
        p.ins.iter().map(ins_row).collect(),
        p.labels,
        p.sections,
        p.user_sgprs,
        p.label_lines,
        p.workgroup_ids,
        p.kernarg_sgpr,
        p.pc_layout.map(|l| (l.ranges, l.end)),
    )
}

/// The decoded program of a `.s` for a target.
#[pyfunction]
fn emu_read_program(py: Python<'_>, text: &str, target: &str) -> PyResult<ProgramRow> {
    let target = thin_targets::get(target).map_err(|e| target_error(py, e))?;
    let program = thin_emu::read_program(text, target).map_err(|e| emu_error(py, e))?;
    Ok(program_row(program))
}

/// A program from decoded JSON (JSON text) at W lanes.
#[pyfunction]
fn emu_program_of(py: Python<'_>, json: &str, w: usize) -> PyResult<ProgramRow> {
    let program = thin_emu::api::program_of(json, w).map_err(|e| emu_error(py, e))?;
    Ok(program_row(program))
}

/// A claim register as host values.
type Reg = (String, u32, u32);

/// emu_claims' result: uniform, assume, undefined, the refusal, edges.
type ClaimsOut = (
    Vec<(usize, Vec<Reg>, String)>,
    Vec<(usize, Vec<(String, u32, u32, u64, u64, String)>)>,
    Vec<(usize, Vec<Reg>)>,
    Option<String>,
    Vec<((usize, usize), Vec<Reg>)>,
);

/// The claims of a `.s`: uniform {index: (regs, text)},
/// assume {index: [(file, first, count, lo, hi, text)]}, undefined {index:
/// regs}, edge claims {(from, to): regs}; a kind's refusal as its message.
#[pyfunction]
fn emu_claims(py: Python<'_>, text: &str, target: &str, kind: &str) -> PyResult<ClaimsOut> {
    let target = thin_targets::get(target).map_err(|e| target_error(py, e))?;
    let program = thin_emu::read_program(text, target).map_err(|e| emu_error(py, e))?;
    let which = match kind {
        "uniform" => claims::Kind::Uniform,
        "assume" => claims::Kind::Assume,
        _ => claims::Kind::Undefined,
    };
    let mut c = claims::Claims::default();
    let facts = thin_emu::machine::facts::Facts::of(target);
    let refused = claims::read(&program, &facts, which, &mut c)
        .err()
        .map(|e| e.message);
    let uniform = c
        .uniform
        .into_iter()
        .map(|(i, u)| (i, u.regs, u.text))
        .collect();
    let assume = c
        .assume
        .into_iter()
        .map(|(i, list)| {
            let rows = list
                .into_iter()
                .map(|a| (a.reg.0, a.reg.1, a.reg.2, a.lo, a.hi, a.text));
            (i, rows.collect())
        })
        .collect();
    let edges = c.edges.into_iter().collect();
    Ok((
        uniform,
        assume,
        c.undefined.into_iter().collect(),
        refused,
        edges,
    ))
}

/// The descriptor's user SGPR count of a `.s`.
#[pyfunction]
fn emu_user_sgprs(py: Python<'_>, text: &str) -> PyResult<u32> {
    thin_emu::machine::program::user_sgprs(text).map_err(|e| emu_error(py, e))
}

/// The held-values spelling of a uniform claim's message.
#[pyfunction]
fn emu_held_text(values: Vec<Vec<u32>>) -> String {
    claims::held_text(&values)
}

/// The message of a false assume claim.
#[pyfunction]
#[pyo3(signature = (text, pc, wave, lane, value))]
fn emu_assume_false(text: &str, pc: usize, wave: usize, lane: Option<usize>, value: u64) -> String {
    claims::assume_false(text, pc, wave, lane, value)
}

/// Adds this module's functions.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emu_read_program, m)?)?;
    m.add_function(wrap_pyfunction!(emu_program_of, m)?)?;
    m.add_function(wrap_pyfunction!(emu_claims, m)?)?;
    m.add_function(wrap_pyfunction!(emu_user_sgprs, m)?)?;
    m.add_function(wrap_pyfunction!(emu_held_text, m)?)?;
    m.add_function(wrap_pyfunction!(emu_assume_false, m)?)
}
