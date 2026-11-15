//! Conversions between Python values and the Builder's data.

use pyo3::exceptions::{
    PyAttributeError, PyIndexError, PyOverflowError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyString, PyTuple};
use thin_dsl::expr::node::{Leafish, Tree};
use thin_dsl::maker::{Arg, FArg};
use thin_dsl::types::Ty;
use thin_dsl::values::Typed;
use thin_dsl::{Class, Error};
use thin_ir::{Atom, Reg, Tag};
use thin_target::phys_named;

use super::lds::DslAddress;
use super::trees::DslTree;
use super::typed::DslTyped;
use crate::raise;

/// A refusal of the language as its Python class.
pub(crate) fn err(e: Error) -> PyErr {
    match e.class {
        Class::Value => PyValueError::new_err(e.msg),
        Class::Type => PyTypeError::new_err(e.msg),
        Class::Index => PyIndexError::new_err(e.msg),
        Class::Overflow => PyOverflowError::new_err(e.msg),
        Class::Attribute => PyAttributeError::new_err(e.msg),
        class => Python::attach(|py| raise(py, "thin.ir", class.name(), e.msg)),
    }
}

/// A result of the language as a Python result.
pub(crate) fn ok<T>(r: thin_dsl::Result<T>) -> PyResult<T> {
    r.map_err(err)
}

/// A type by name.
pub(crate) fn ty(name: &str) -> PyResult<Ty> {
    Ty::named(name).ok_or_else(|| PyValueError::new_err(format!("no type {name}")))
}

/// A register from its Python tuple (value, file, lo, n).
pub(crate) fn reg(r: (i32, char, u32, u32)) -> Reg {
    Reg::new(r.0, r.1, r.2, r.3)
}

/// A register as its Python tuple.
pub(crate) fn reg_out(r: &Reg) -> (i32, char, u32, u32) {
    (r.value, r.file, r.lo, r.n)
}

/// A named register by name.
pub(crate) fn phys(name: &str) -> PyResult<&'static thin_target::Phys> {
    phys_named(name).ok_or_else(|| PyValueError::new_err(format!("no named register {name}")))
}

/// A maker operand from its tagged tuple: `("n",)`, `("b", v)`, `("i",
/// v)`, `("f", v)`, `("s", v)`, `("r", value, file, lo, n, mods)`, `("p",
/// name, mods)`, `("a", address)`.
pub(crate) fn arg(t: &Bound<'_, PyTuple>) -> PyResult<Arg> {
    let tag: String = t.get_item(0)?.extract()?;
    let item = |k: usize| t.get_item(k);
    Ok(match tag.as_str() {
        "n" => Arg::None,
        "b" => Arg::Bool(item(1)?.extract()?),
        "i" => Arg::Int(item(1)?.extract()?),
        "f" => Arg::Float(item(1)?.extract()?),
        "s" => Arg::Str(item(1)?.extract()?),
        "r" => Arg::Reg(reg(item(1)?.extract()?), item(2)?.extract()?),
        "p" => Arg::Phys(phys(&item(1)?.extract::<String>()?)?, item(2)?.extract()?),
        _ => Arg::Addr(item(1)?.extract::<PyRef<'_, DslAddress>>()?.a.clone()),
    })
}

/// Maker operands.
pub(crate) fn args(xs: Vec<Bound<'_, PyTuple>>) -> PyResult<Vec<Arg>> {
    xs.iter().map(arg).collect()
}

/// A field value by its Python type.
pub(crate) fn farg(v: &Bound<'_, PyAny>) -> PyResult<FArg> {
    if v.is_none() {
        return Ok(FArg::None);
    }
    if v.is_instance_of::<PyBool>() {
        return Ok(FArg::Bool(v.extract()?));
    }
    if v.is_instance_of::<PyInt>() {
        return Ok(FArg::Int(v.extract()?));
    }
    if v.is_instance_of::<PyString>() {
        return Ok(FArg::Str(v.extract()?));
    }
    if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = v.extract()?;
        return Ok(FArg::List(items.iter().map(farg).collect::<PyResult<_>>()?));
    }
    Ok(FArg::Str(v.str()?.to_string()))
}

/// Fields from their Python pairs.
pub(crate) fn fields(xs: Vec<(String, Bound<'_, PyAny>)>) -> PyResult<Vec<(String, FArg)>> {
    xs.into_iter().map(|(k, v)| Ok((k, farg(&v)?))).collect()
}

/// A tag from a tuple of ints and strings.
pub(crate) fn tag(v: &Bound<'_, PyAny>) -> PyResult<Tag> {
    let items: Vec<Bound<'_, PyAny>> = v.extract()?;
    let atom = |x: &Bound<'_, PyAny>| -> PyResult<Atom> {
        match x.is_instance_of::<PyInt>() {
            true => Ok(Atom::Int(x.extract()?)),
            false => Ok(Atom::Str(x.extract::<String>()?.into())),
        }
    };
    Ok(items.iter().map(atom).collect::<PyResult<Vec<_>>>()?.into())
}

/// A tree's leaf from its tagged tuple: `("t", tree)`, `("v", value)`,
/// `("p", name)`, `("b", repr)`, `("i", int)`, `("big", repr)`, `("f",
/// float)`, `("o", repr)`.
pub(crate) fn leafish(t: &Bound<'_, PyTuple>) -> PyResult<Leafish> {
    let tag: String = t.get_item(0)?.extract()?;
    let item = t.get_item(1)?;
    Ok(match tag.as_str() {
        "t" => Leafish::Node(item.extract::<PyRef<'_, DslTree>>()?.t.clone()),
        "v" => Leafish::Value(item.extract::<PyRef<'_, DslTyped>>()?.t.clone()),
        "p" => Leafish::Phys(phys(&item.extract::<String>()?)?),
        "b" => Leafish::Bool(item.extract()?),
        "i" => Leafish::Int(item.extract()?),
        "big" => Leafish::Big(item.extract()?),
        "f" => Leafish::Float(item.extract()?),
        _ => Leafish::Other(item.extract()?),
    })
}

/// A tree from a leafish tuple.
pub(crate) fn tree(t: &Bound<'_, PyTuple>) -> PyResult<Tree> {
    ok(thin_dsl::expr::node::leaf(leafish(t)?))
}

/// Trees from leafish tuples.
pub(crate) fn trees(xs: &[Bound<'_, PyTuple>]) -> PyResult<Vec<Tree>> {
    xs.iter().map(tree).collect()
}

/// A typed value of its Python handle.
pub(crate) fn typed(v: &Bound<'_, PyAny>) -> PyResult<Typed> {
    Ok(v.extract::<PyRef<'_, DslTyped>>()?.t.clone())
}
