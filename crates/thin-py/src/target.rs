//! The functions of `thin._native` over the targets.

use pyo3::prelude::*;
use thin_target::{Form, Slot, Target, FLAG_BITS, PHYS};

use pyo3::exceptions::{PyKeyError, PyValueError};

use crate::raise;

/// A target refusal as the class names.
pub(crate) fn target_error(py: Python<'_>, error: thin_target::Error) -> PyErr {
    use thin_target::Error as E;
    match error {
        E::Definition(msg) | E::Value(msg) => PyValueError::new_err(msg),
        E::Key(msg) => PyKeyError::new_err(msg),
        E::Asm(msg) => raise(py, "thin.ir", "AsmError", msg),
    }
}

/// Validate a target name, preserving its native refusal.
#[pyfunction]
fn check_target(py: Python<'_>, name: &str) -> PyResult<()> {
    thin_targets::get(name)
        .map(|_| ())
        .map_err(|e| target_error(py, e))
}

/// Look up a target by name, preserving its native refusal.
pub(crate) fn get(py: Python<'_>, name: &str) -> PyResult<&'static Target> {
    thin_targets::get(name).map_err(|e| target_error(py, e))
}

/// The known target names in registry order; the first is the default.
#[pyfunction]
fn target_names() -> Vec<&'static str> {
    thin_targets::names().to_vec()
}

/// Every form flag's (name, bit), in table order.
#[pyfunction]
fn flag_bits() -> Vec<(&'static str, u32)> {
    FLAG_BITS.to_vec()
}

/// The named registers: (name, dwords, file, whole).
#[pyfunction]
fn phys() -> Vec<(&'static str, u32, &'static str, &'static str)> {
    let row = |p: &thin_target::Phys| (p.name, p.dwords, p.space.name(), PHYS[p.whole].name);
    PHYS.iter().map(row).collect()
}

/// The chip facts in order.
type ChipRow = (
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Vec<&'static str>,
);

/// A target's record: (wave, lds_max, chip, counters as (name, field),
/// waits as (mnemonic, fields), barrier, hazard window).
type TargetRow = (
    u32,
    u32,
    ChipRow,
    Vec<(&'static str, &'static str)>,
    Vec<(&'static str, Vec<&'static str>)>,
    Vec<(&'static str, Vec<i64>)>,
    u32,
);

/// The instruction-cache invalidate: (instruction, nop, nops, per_cu), or None.
#[pyfunction]
fn target_icache_inv(
    py: Python<'_>,
    name: &str,
) -> PyResult<Option<(&'static str, &'static str, u32, u32)>> {
    let hook = get(py, name)?.hooks().icache_invalidate();
    Ok(hook.map(|i| (i.instruction, i.nop, i.nops, i.per_cu)))
}

/// The kernel library's hooks by name: each a value or None.
#[pyfunction]
fn target_lib_hooks(py: Python<'_>, name: &str) -> PyResult<Vec<(&'static str, Option<String>)>> {
    Ok(get(py, name)?.hooks().lib_hooks().named().to_vec())
}

/// A target's record.
#[pyfunction]
fn target_info(py: Python<'_>, name: &str) -> PyResult<TargetRow> {
    let d = get(py, name)?.definition();
    let c = &d.chip;
    let chip = (
        c.xcds,
        c.cus,
        c.simds,
        c.lds,
        c.vgprs,
        c.vgprs_per_lane,
        c.sgprs,
        c.user_sgprs,
        c.wave_slots,
        c.preload,
        c.wgids.to_vec(),
    );
    Ok((
        d.wave,
        d.lds_max,
        chip,
        d.counters.iter().map(|c| (c.name, c.field)).collect(),
        d.waits
            .iter()
            .map(|w| (w.mnemonic, w.fields.to_vec()))
            .collect(),
        d.barrier.clone(),
        d.hazards.window,
    ))
}

/// A slot as Python receives it.
type SlotRow = (
    &'static str,
    u32,
    u8,
    String,
    Vec<u32>,
    bool,
    bool,
    Vec<u32>,
);

fn slot_row(s: &Slot) -> SlotRow {
    let kind = s.kind.name();
    let flags = (s.modifiers, s.literal);
    (
        kind,
        s.dwords,
        s.role,
        s.xml_type.clone(),
        s.sgpr_widths.clone(),
        flags.0,
        flags.1,
        s.widths.clone(),
    )
}

/// A field as Python receives it.
type FieldRow = (&'static str, &'static str, u32, bool, bool, Vec<String>);

/// A form's identity, its class and its counter.
type FormHead = (
    i32,
    String,
    String,
    String,
    String,
    &'static str,
    Option<&'static str>,
    bool,
    u64,
    String,
);

/// A form's operands and the rest of its record.
type FormTail = (
    usize,
    bool,
    Vec<SlotRow>,
    bool,
    Vec<FieldRow>,
    Vec<&'static str>,
    Vec<&'static str>,
    u32,
    Option<&'static str>,
    (
        bool,
        &'static str,
        Option<&'static str>,
        Option<&'static str>,
        Option<&'static str>,
    ),
);

fn form_head(t: &Target, f: &Form) -> FormHead {
    let counter = f.counter.map(|c| t.definition().counters[c].name);
    (
        f.id,
        f.name.clone(),
        f.op.clone(),
        f.base.clone(),
        f.canon.clone(),
        f.kind.name(),
        counter,
        f.unordered,
        f.flags.0,
        f.encoding.clone(),
    )
}

fn form_tail(f: &Form) -> FormTail {
    let names = |regs: &[usize]| regs.iter().map(|&r| PHYS[r].name).collect();
    let field = |x: &thin_target::Field| {
        (
            x.name,
            x.kind.name(),
            x.bits,
            x.signed,
            x.required,
            x.choices.clone(),
        )
    };
    let rest = (
        f.dpp,
        f.dst_sel.name(),
        f.mem.map(|m| m.name()),
        f.wait_text.map(|w| w.name()),
        f.dst_half.map(|h| h.name()),
    );
    (
        f.arity,
        f.returning,
        f.slots.iter().map(slot_row).collect(),
        f.variadic,
        f.fields.iter().map(field).collect(),
        names(&f.implicit_defs),
        names(&f.implicit_uses),
        f.mem_bytes,
        f.pseudo.map(|p| p.name()),
        rest,
    )
}

/// Every form of a target in id order, then its own ops.
#[pyfunction]
fn target_forms(py: Python<'_>, name: &str) -> PyResult<Vec<(FormHead, FormTail)>> {
    let t = get(py, name)?;
    let all = t.forms().iter().chain(t.own_ops());
    Ok(all.map(|f| (form_head(t, f), form_tail(f))).collect())
}

/// The id of the form of a key, if any.
#[pyfunction]
fn form_id(
    py: Python<'_>,
    name: &str,
    op: &str,
    arity: usize,
    returning: bool,
) -> PyResult<Option<i32>> {
    Ok(get(py, name)?.form(op, arity, returning).map(|f| f.id))
}

/// The mnemonic of a dotted op, if the target has it.
#[pyfunction]
fn spell(py: Python<'_>, name: &str, op: &str) -> PyResult<Option<String>> {
    Ok(get(py, name)?.spell(op))
}

/// Whether an atomic with this temporal hint is its returning form.
#[pyfunction]
#[pyo3(signature = (name, mnemonic, th=None))]
fn returning(py: Python<'_>, name: &str, mnemonic: &str, th: Option<&str>) -> PyResult<bool> {
    Ok(get(py, name)?.hooks().returning(mnemonic, th))
}

/// The vector registers allocated for (vgpr, agpr).
#[pyfunction]
fn allocated_vgprs(py: Python<'_>, name: &str, vgpr: u32, agpr: u32) -> PyResult<u32> {
    Ok(get(py, name)?.allocated_vgprs(vgpr, agpr))
}

/// A matrix form as Python receives it: `(a, b, d, (m, n, k, blocks, wave, passes), (sparse, cbsz_abid, blgp, provisional), (da, db, dc, dd))`.
type MatrixRow = (
    &'static str,
    &'static str,
    &'static str,
    (u32, u32, u32, u32, u32, u32),
    (bool, bool, bool, bool),
    (u32, u32, u32, u32),
);

/// The matrix form of a mnemonic on a target, or None.
#[pyfunction]
fn target_matrix_form(py: Python<'_>, name: &str, mnemonic: &str) -> PyResult<Option<MatrixRow>> {
    let mf = get(py, name)?.matrix_form(mnemonic);
    Ok(mf.map(|m| {
        (
            m.a.name(),
            m.b.name(),
            m.d.name(),
            (m.m, m.n, m.k, m.blocks, m.wave, m.passes),
            (m.sparse, m.cbsz_abid, m.blgp, m.provisional),
            (m.dwords('a'), m.dwords('b'), m.dwords('c'), m.dwords('d')),
        )
    }))
}

/// Adds this module's functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for function in [
        wrap_pyfunction!(check_target, m)?,
        wrap_pyfunction!(target_names, m)?,
        wrap_pyfunction!(flag_bits, m)?,
        wrap_pyfunction!(phys, m)?,
        wrap_pyfunction!(target_info, m)?,
        wrap_pyfunction!(target_lib_hooks, m)?,
        wrap_pyfunction!(target_icache_inv, m)?,
        wrap_pyfunction!(target_forms, m)?,
        wrap_pyfunction!(form_id, m)?,
        wrap_pyfunction!(spell, m)?,
        wrap_pyfunction!(returning, m)?,
        wrap_pyfunction!(allocated_vgprs, m)?,
        wrap_pyfunction!(target_matrix_form, m)?,
    ] {
        m.add_function(function)?;
    }
    Ok(())
}
