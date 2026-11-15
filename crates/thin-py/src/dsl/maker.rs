//! The instruction maker and the memory families.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::maker::families::{self, Cache, Family, Opts, Val};
use thin_dsl::maker::{self as mk, Ctx, Made, Notes, Stage};
use thin_dsl::values::ValueRec;
use thin_ir::Operand;
use thin_target::Target;

use super::builder::DslBuilder;
use super::convert::{args, fields, ok, reg, tag, tree};
use super::lds::DslAddress;
use crate::ir::Ins;
use crate::target::get;

/// What a family made that is not yet an instruction: an LDS access, an
/// address split or a DMA (`b.emit` emits it).
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslMade {
    pub(crate) f: Family,
}

/// An instruction made, its registers per operand, or what else was made.
pub(crate) type Out = (Option<Ins>, Vec<Option<(i32, u32, u32)>>, Option<DslMade>);

/// The registers of an instruction per operand index.
pub(crate) fn regs_of(i: &thin_ir::Instr) -> Vec<Option<(i32, u32, u32)>> {
    let one = |op: &Operand| match op {
        Operand::Reg(r) => Some((r.value, r.lo, r.n)),
        _ => None,
    };
    i.ops().iter().map(one).collect()
}

fn out(t: &'static Target, f: Family) -> Out {
    match f {
        Family::Made(Made::Instr(i)) => (Some(Ins::of(t, Arc::new(*i.clone()))), regs_of(&i), None),
        f => (None, Vec::new(), Some(DslMade { f })),
    }
}

fn values<'a>(b: &'a Option<PyRef<'_, DslBuilder>>) -> &'a [ValueRec] {
    b.as_ref().map_or(&[], |b| &b.b.values)
}

/// A family's operand from its tagged tuple: `("n",)`, `("r", reg)`,
/// `("i", v)`, `("t", leaf)`, `("a", address)`, `("s", text)`.
fn val(t: &Bound<'_, PyTuple>) -> PyResult<Val> {
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "n" => Val::None,
        "r" => Val::Reg(reg(t.get_item(1)?.extract()?)),
        "i" => Val::Int(t.get_item(1)?.extract()?),
        "t" => Val::Tree(tree(&t.get_item(1)?.extract()?)?),
        "a" => Val::Addr(t.get_item(1)?.extract::<PyRef<'_, DslAddress>>()?.a.clone()),
        _ => Val::Str(t.get_item(1)?.extract()?),
    })
}

/// A stage from `("add", reg, offset)` or `("mov", offset)`.
fn stage(t: &Bound<'_, PyTuple>) -> PyResult<Stage> {
    let kind: String = t.get_item(0)?.extract()?;
    match kind.as_str() {
        "add" => Ok(Stage::Add(
            reg(t.get_item(1)?.extract()?),
            t.get_item(2)?.extract()?,
        )),
        _ => Ok(Stage::Mov(t.get_item(1)?.extract()?)),
    }
}

/// `O.op(name, *operands, comment=, tag=, abl=, stage=, **fields)`.
#[pyfunction]
#[pyo3(signature = (target, b, name, operands, f, comment, t, abl, st))]
#[allow(clippy::too_many_arguments)]
fn maker_op(
    py: Python<'_>,
    target: &str,
    b: Option<PyRef<'_, DslBuilder>>,
    name: &str,
    operands: Vec<Bound<'_, PyTuple>>,
    f: Vec<(String, Bound<'_, PyAny>)>,
    comment: String,
    t: Option<Bound<'_, PyAny>>,
    abl: Vec<String>,
    st: Option<Bound<'_, PyTuple>>,
) -> PyResult<Out> {
    let target = get(py, target)?;
    let notes = Notes {
        comment,
        tag: t.map(|t| tag(&t)).transpose()?,
        abl,
        stage: st.map(|s| stage(&s)).transpose()?,
    };
    let c = Ctx {
        t: target,
        values: values(&b),
    };
    let made = ok(mk::op(&c, name, args(operands)?, fields(f)?, notes))?;
    Ok(out(target, Family::Made(made)))
}

/// A family's options from (offset, th, scope, flags, comment, tag, bytes,
/// bytes_repr, signed, ordered).
type OptsIn<'py> = (
    Bound<'py, PyTuple>,
    String,
    String,
    Vec<String>,
    String,
    Option<Bound<'py, PyAny>>,
    Option<i128>,
    Option<String>,
    bool,
    String,
);

fn opts(o: OptsIn<'_>) -> PyResult<Opts> {
    let (offset, th, scope, flags, comment, t, bytes, bytes_repr, signed, ordered) = o;
    let bytes = match (bytes, bytes_repr) {
        (Some(b), _) => Some(Ok(b)),
        (None, Some(r)) => Some(Err(r)),
        (None, None) => None,
    };
    Ok(Opts {
        offset: val(&offset)?,
        cache: Cache { th, scope, flags },
        comment,
        tag: t.map(|t| tag(&t)).transpose()?,
        bytes,
        signed,
        ordered,
    })
}

/// A memory family: `s_load`, `s_store`, `global_load`, `global_store`,
/// `buffer_load`, `buffer_store`, `ds_read`, `ds_write`; `xs` its
/// operands after the data in the table's order.
#[pyfunction]
fn maker_family(
    py: Python<'_>,
    target: &str,
    b: Option<PyRef<'_, DslBuilder>>,
    family: &str,
    data: (i32, char, u32, u32),
    xs: Vec<Bound<'_, PyTuple>>,
    o: OptsIn<'_>,
) -> PyResult<Out> {
    let target = get(py, target)?;
    let c = Ctx {
        t: target,
        values: values(&b),
    };
    let v = xs.iter().map(val).collect::<PyResult<Vec<_>>>()?;
    let data = reg(data);
    let o = opts(o)?;
    Ok(out(target, ok(families::family(&c, family, &data, v, o))?))
}

/// Adds the maker.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DslMade>()?;
    m.add_function(wrap_pyfunction!(maker_op, m)?)?;
    m.add_function(wrap_pyfunction!(maker_family, m)?)
}
