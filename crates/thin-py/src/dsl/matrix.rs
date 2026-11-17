//! Matrix instruction makers and accumulator Builder bindings.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::builder::{AccIn, Make};
use thin_dsl::maker::families::Family;
use thin_dsl::maker::matrix::{self as mx, MfmaOpts};
use thin_dsl::maker::{Ctx, Notes};

use super::builder::DslBuilder;
use super::convert::{arg, ok, reg, tag, tree, typed};
use super::maker::{out, values, Out};
use super::typed::DslTyped;
use crate::target::get;

type MfmaOptsIn<'py> = (
    i128,
    i128,
    i128,
    Option<String>,
    Option<String>,
    i128,
    i128,
    Option<Vec<i128>>,
    String,
    Option<Bound<'py, PyAny>>,
    Vec<String>,
);

fn mfma_opts(o: MfmaOptsIn<'_>) -> PyResult<MfmaOpts> {
    let (cbsz, abid, blgp, fmt_a, fmt_b, sel_a, sel_b, neg, comment, t, abl) = o;
    Ok(MfmaOpts {
        cbsz,
        abid,
        blgp,
        fmt_a,
        fmt_b,
        sel_a,
        sel_b,
        neg,
        notes: Notes {
            comment,
            tag: t.map(|t| tag(&t)).transpose()?,
            abl,
            stage: None,
        },
    })
}

fn acc_in(t: &Bound<'_, PyTuple>) -> PyResult<AccIn> {
    let kind: String = t.get_item(0)?.extract()?;
    let item = t.get_item(1)?;
    Ok(match kind.as_str() {
        "t" => AccIn::Typed(typed(&item)?),
        _ => AccIn::Reg(reg(item.extract()?)),
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn maker_mfma(
    py: Python<'_>,
    target: &str,
    b: Option<PyRef<'_, DslBuilder>>,
    form: &str,
    dst: Bound<'_, PyTuple>,
    a: Bound<'_, PyTuple>,
    rhs: Bound<'_, PyTuple>,
    c_in: Option<Bound<'_, PyTuple>>,
    opts: MfmaOptsIn<'_>,
) -> PyResult<Out> {
    let target = get(py, target)?;
    let ctx = Ctx {
        t: target,
        values: values(&b),
    };
    let c_arg = c_in.map(|x| arg(&x)).transpose()?;
    let made = ok(mx::mfma(
        &ctx,
        form,
        arg(&dst)?,
        arg(&a)?,
        arg(&rhs)?,
        c_arg,
        mfma_opts(opts)?,
    ))?;
    Ok(out(target, Family::Made(made)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn maker_mfma_scale(
    py: Python<'_>,
    target: &str,
    b: Option<PyRef<'_, DslBuilder>>,
    form: &str,
    dst: Bound<'_, PyTuple>,
    a: Bound<'_, PyTuple>,
    rhs: Bound<'_, PyTuple>,
    scale_a: Bound<'_, PyTuple>,
    scale_b: Bound<'_, PyTuple>,
    c_in: Option<Bound<'_, PyTuple>>,
    opts: MfmaOptsIn<'_>,
) -> PyResult<Out> {
    let target = get(py, target)?;
    let ctx = Ctx {
        t: target,
        values: values(&b),
    };
    let c_arg = c_in.map(|x| arg(&x)).transpose()?;
    let made = ok(mx::mfma_scale(
        &ctx,
        form,
        arg(&dst)?,
        arg(&a)?,
        arg(&rhs)?,
        arg(&scale_a)?,
        arg(&scale_b)?,
        c_arg,
        mfma_opts(opts)?,
    ))?;
    Ok(out(target, Family::Made(made)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn maker_smfmac(
    py: Python<'_>,
    target: &str,
    b: Option<PyRef<'_, DslBuilder>>,
    form: &str,
    dst: Bound<'_, PyTuple>,
    a: Bound<'_, PyTuple>,
    rhs: Bound<'_, PyTuple>,
    idx: Bound<'_, PyTuple>,
    opts: MfmaOptsIn<'_>,
) -> PyResult<Out> {
    let target = get(py, target)?;
    let ctx = Ctx {
        t: target,
        values: values(&b),
    };
    let made = ok(mx::smfmac(
        &ctx,
        form,
        arg(&dst)?,
        arg(&a)?,
        arg(&rhs)?,
        arg(&idx)?,
        mfma_opts(opts)?,
    ))?;
    Ok(out(target, Family::Made(made)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn builder_mfma_acc(
    mut b: PyRefMut<'_, DslBuilder>,
    form: &str,
    acc: Option<String>,
    agpr: Option<bool>,
    name: Option<String>,
    at: Option<u32>,
    same_as: Option<Bound<'_, PyAny>>,
    init: Option<Bound<'_, PyTuple>>,
) -> PyResult<DslTyped> {
    let m = Make {
        name,
        at,
        same_as: same_as.map(|y| typed(&y)).transpose()?,
        ..Make::default()
    };
    let init = init.map(|x| tree(&x)).transpose()?;
    let t = ok(b.b.mfma_acc(form, acc.as_deref(), agpr, m, init))?;
    Ok(DslTyped { t })
}

type BuilderMfmaCfg<'py> = (
    Option<Bound<'py, PyTuple>>,
    Option<Bound<'py, PyTuple>>,
    Option<Bound<'py, PyTuple>>,
    Option<String>,
    Option<bool>,
    Option<String>,
    Vec<String>,
);

#[pyfunction]
fn builder_mfma(
    mut b: PyRefMut<'_, DslBuilder>,
    form: &str,
    a: Bound<'_, PyTuple>,
    rhs: Bound<'_, PyTuple>,
    cfg: BuilderMfmaCfg<'_>,
    opts: MfmaOptsIn<'_>,
) -> PyResult<DslTyped> {
    let (c_arg, c_acc, dst, acc, agpr, name, abl) = cfg;
    let c_arg = c_arg.map(|x| arg(&x)).transpose()?;
    let c_acc = c_acc.map(|x| acc_in(&x)).transpose()?;
    let dst = dst.map(|x| acc_in(&x)).transpose()?;
    let t = ok(b.b.mfma_emit(
        form,
        arg(&a)?,
        arg(&rhs)?,
        c_arg,
        c_acc,
        dst,
        acc.as_deref(),
        agpr,
        name,
        mfma_opts(opts)?,
        &abl,
    ))?;
    Ok(DslTyped { t })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn builder_mfma_scale(
    mut b: PyRefMut<'_, DslBuilder>,
    form: &str,
    a: Bound<'_, PyTuple>,
    rhs: Bound<'_, PyTuple>,
    scale_a: Bound<'_, PyTuple>,
    scale_b: Bound<'_, PyTuple>,
    cfg: BuilderMfmaCfg<'_>,
    opts: MfmaOptsIn<'_>,
) -> PyResult<DslTyped> {
    let (c_arg, c_acc, dst, acc, agpr, name, abl) = cfg;
    let c_arg = c_arg.map(|x| arg(&x)).transpose()?;
    let c_acc = c_acc.map(|x| acc_in(&x)).transpose()?;
    let dst = dst.map(|x| acc_in(&x)).transpose()?;
    let t = ok(b.b.mfma_scale_emit(
        form,
        arg(&a)?,
        arg(&rhs)?,
        arg(&scale_a)?,
        arg(&scale_b)?,
        c_arg,
        c_acc,
        dst,
        acc.as_deref(),
        agpr,
        name,
        mfma_opts(opts)?,
        &abl,
    ))?;
    Ok(DslTyped { t })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn builder_smfmac(
    mut b: PyRefMut<'_, DslBuilder>,
    form: &str,
    a: Bound<'_, PyTuple>,
    rhs: Bound<'_, PyTuple>,
    idx: Bound<'_, PyTuple>,
    c: Bound<'_, PyTuple>,
    acc: Option<String>,
    agpr: Option<bool>,
    abl: Vec<String>,
    opts: MfmaOptsIn<'_>,
) -> PyResult<DslTyped> {
    let t = ok(b.b.smfmac_emit(
        form,
        arg(&a)?,
        arg(&rhs)?,
        arg(&idx)?,
        acc_in(&c)?,
        acc.as_deref(),
        agpr,
        mfma_opts(opts)?,
        &abl,
    ))?;
    Ok(DslTyped { t })
}

#[pyfunction]
fn builder_read_acc(
    mut b: PyRefMut<'_, DslBuilder>,
    src: Bound<'_, PyTuple>,
    dst: Option<Bound<'_, PyTuple>>,
    copy: bool,
    name: Option<String>,
) -> PyResult<DslTyped> {
    let src = acc_in(&src)?;
    let dst = dst.map(|x| acc_in(&x)).transpose()?;
    let t = ok(b.b.read_acc(src, dst, copy, name))?;
    Ok(DslTyped { t })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn builder_write_acc(
    mut b: PyRefMut<'_, DslBuilder>,
    src: Bound<'_, PyTuple>,
    dst: Option<Bound<'_, PyTuple>>,
    acc: Option<String>,
    agpr: Option<bool>,
    copy: bool,
    name: Option<String>,
) -> PyResult<DslTyped> {
    let src = acc_in(&src)?;
    let dst = dst.map(|x| acc_in(&x)).transpose()?;
    let t = ok(b.b.write_acc(src, dst, acc.as_deref(), agpr, copy, name))?;
    Ok(DslTyped { t })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maker_mfma, m)?)?;
    m.add_function(wrap_pyfunction!(maker_mfma_scale, m)?)?;
    m.add_function(wrap_pyfunction!(maker_smfmac, m)?)?;
    m.add_function(wrap_pyfunction!(builder_mfma_acc, m)?)?;
    m.add_function(wrap_pyfunction!(builder_mfma, m)?)?;
    m.add_function(wrap_pyfunction!(builder_mfma_scale, m)?)?;
    m.add_function(wrap_pyfunction!(builder_smfmac, m)?)?;
    m.add_function(wrap_pyfunction!(builder_read_acc, m)?)?;
    m.add_function(wrap_pyfunction!(builder_write_acc, m)?)
}
