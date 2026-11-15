//! Trees: their nodes and the functions that make them.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::expr::funcs::{self, Cast, FmtArg};
use thin_dsl::expr::node::{self, Divisor, Kind, Num, Op, Tree, Val};
use thin_dsl::expr::text::top;

use super::convert::{fields, ok, tree, trees, ty, typed};
use super::typed::DslTyped;

/// A node of a tree, shared and immutable.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslTree {
    pub(crate) t: Tree,
}

fn wrap(t: Tree) -> DslTree {
    DslTree { t }
}

fn op(name: &str) -> PyResult<Op> {
    Op::named(name).ok_or_else(|| PyValueError::new_err(format!("no operator {name}")))
}

#[pymethods]
impl DslTree {
    /// Its operator's name.
    #[getter]
    fn op(&self) -> &'static str {
        self.t.op.name()
    }

    /// Its operands.
    #[getter]
    fn args(&self) -> Vec<DslTree> {
        self.t.args.iter().cloned().map(wrap).collect()
    }

    /// Its type's name, or None.
    #[getter]
    fn ty(&self) -> Option<&'static str> {
        self.t.ty.map(|t| t.name())
    }

    /// Its file, or None.
    #[getter]
    fn file(&self) -> Option<char> {
        self.t.file
    }

    /// Its Builder's identity, or None.
    #[getter]
    fn builder(&self) -> Option<u64> {
        self.t.builder
    }

    /// Its text at the top level.
    fn text(&self) -> String {
        top(&self.t)
    }

    /// Its value, as Python reads `val`.
    #[getter]
    fn val<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let any = |x: Bound<'py, PyAny>| Ok(x);
        match &self.t.val {
            Val::None => any(py.None().into_bound(py)),
            Val::Value(v) => any(Bound::new(py, DslTyped { t: v.clone() })?.into_any()),
            Val::Special(p) => any(p.name.into_pyobject(py)?.into_any()),
            Val::Const(Num::Int(v)) => any(v.into_pyobject(py)?.into_any()),
            Val::Const(Num::Float(x)) => any(x.into_pyobject(py)?.into_any()),
            Val::Cmp(k, t, n) => any((k.name(), t.name(), *n).into_pyobject(py)?.into_any()),
            Val::Divisor(Divisor::Const(c)) => any(c.into_pyobject(py)?.into_any()),
            Val::Divisor(Divisor::Arg(v)) => {
                any(Bound::new(py, DslTyped { t: v.clone() })?.into_any())
            }
            Val::Bfe(o, w, s) => any((*o, *w, *s).into_pyobject(py)?.into_any()),
            Val::F1(name) => any(name.into_pyobject(py)?.into_any()),
            Val::Fmt(f) => any((f.name.as_str(), f.at).into_pyobject(py)?.into_any()),
            Val::Lanes(o, l) => any((*o, *l).into_pyobject(py)?.into_any()),
            Val::Instr(name, _) => any(name.into_pyobject(py)?.into_any()),
        }
    }

    /// `(kind, [part, part])` for a junction, else None.
    fn junction(&self) -> Option<(&'static str, Vec<DslTree>)> {
        thin_dsl::expr::cond::junction(&self.t).map(|(k, p)| (k, p.into_iter().map(wrap).collect()))
    }

    /// Whether it is this same node.
    fn same(&self, other: PyRef<'_, DslTree>) -> bool {
        std::sync::Arc::ptr_eq(&self.t, &other.t)
    }
}

/// A tree's leaf.
#[pyfunction]
fn tree_leaf(x: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    tree(&x).map(wrap)
}

/// `a op b` for `+ - * & | ^ << >>`.
#[pyfunction]
fn tree_binop(name: &str, a: Bound<'_, PyTuple>, b: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    ok(node::binop(op(name)?, tree(&a)?, tree(&b)?)).map(wrap)
}

/// `a <kind> b`.
#[pyfunction]
fn tree_compare(kind: &str, a: Bound<'_, PyTuple>, b: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    let k = Kind::named(kind).ok_or_else(|| PyValueError::new_err(format!("no compare {kind}")))?;
    ok(node::compare(k, tree(&a)?, tree(&b)?)).map(wrap)
}

/// `x // d`, `x % d`: d an int, a divisor's value, or (None) anything else.
#[pyfunction]
#[pyo3(signature = (name, x, c=None, v=None, text=String::new()))]
fn tree_divmod(
    name: &str,
    x: Bound<'_, PyTuple>,
    c: Option<i128>,
    v: Option<Bound<'_, PyAny>>,
    text: String,
) -> PyResult<DslTree> {
    let d = match (c, v) {
        (Some(c), _) => Some(Divisor::Const(c)),
        (None, Some(v)) => typed(&v).ok().map(Divisor::Arg),
        (None, None) => None,
    };
    ok(node::divmod(op(name)?, tree(&x)?, d, &text)).map(wrap)
}

/// `x op= t`: the tree `x op t`.
#[pyfunction]
fn tree_update(name: &str, x: Bound<'_, PyTuple>, t: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    ok(funcs::update(op(name)?, tree(&x)?, tree(&t)?)).map(wrap)
}

/// `~x`.
#[pyfunction]
fn tree_invert(x: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    ok(node::invert(tree(&x)?)).map(wrap)
}

/// `-x` (`neg`) and `abs(x)` (`abs`).
#[pyfunction]
fn tree_sign(name: &str, x: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    ok(node::sign(op(name)?, tree(&x)?)).map(wrap)
}

/// `T(x)` of a tree.
#[pyfunction]
fn tree_cast(t: &str, x: Bound<'_, PyTuple>) -> PyResult<DslTree> {
    ok(funcs::cast(ty(t)?, tree(&x)?)).map(wrap)
}

/// `f32x2(x)` of a typed value of Builder `b`.
#[pyfunction]
fn cast_pair(b: PyRef<'_, super::builder::DslBuilder>, x: Bound<'_, PyAny>) -> PyResult<DslTyped> {
    let x = typed(&x)?;
    match ok(funcs::cast_pair(&x, b.b.pair_reg(&x)))? {
        Cast::Value(v) => Ok(DslTyped { t: v }),
        Cast::Tree(_) => Err(PyValueError::new_err("f32x2: a value")),
    }
}

/// `select(c, a, b)`.
#[pyfunction]
fn tree_select(
    c: Bound<'_, PyTuple>,
    a: Bound<'_, PyTuple>,
    b: Bound<'_, PyTuple>,
) -> PyResult<DslTree> {
    ok(funcs::select(tree(&c)?, tree(&a)?, tree(&b)?)).map(wrap)
}

/// The functions over operand lists: `mul24`, `mad24`, `mulhi`,
/// `mul_wide`, `min`…`med3`, `fma`, the conversions, `pair`, the lane
/// functions.
#[pyfunction]
fn tree_call(name: &str, xs: Vec<Bound<'_, PyTuple>>) -> PyResult<DslTree> {
    ok(funcs::call(name, trees(&xs)?)).map(wrap)
}

/// `across(x, op, lanes)` (`scan` false) or `scan(x, op, lanes)`.
#[pyfunction]
fn tree_lanes(scan: bool, x: Bound<'_, PyTuple>, name: &str, lanes: i128) -> PyResult<DslTree> {
    ok(funcs::lanes_op(scan, tree(&x)?, name, lanes)).map(wrap)
}

/// `bfe(a, offset, width)`.
#[pyfunction]
#[pyo3(signature = (a, offset, width, signed=None))]
fn tree_bfe(
    a: Bound<'_, PyTuple>,
    offset: u32,
    width: u32,
    signed: Option<bool>,
) -> PyResult<DslTree> {
    ok(funcs::bfe(tree(&a)?, offset, width, signed)).map(wrap)
}

fn fmt_arg(name: String, at: Option<Bound<'_, PyAny>>, exact: bool) -> PyResult<FmtArg> {
    let at = match (at, exact) {
        (None, _) => None,
        (Some(a), true) => Some(Ok(a.extract()?)),
        (Some(a), false) => Some(Err(a.repr()?.to_string())),
    };
    Ok(FmtArg { name, at })
}

/// `unpack(w, fmt, at)` (`pair` false) or `unpack2`.
#[pyfunction]
#[pyo3(signature = (pair, w, fmt, at=None, exact=true))]
fn tree_unpack(
    pair: bool,
    w: Bound<'_, PyTuple>,
    fmt: String,
    at: Option<Bound<'_, PyAny>>,
    exact: bool,
) -> PyResult<DslTree> {
    ok(funcs::unpack(pair, tree(&w)?, fmt_arg(fmt, at, exact)?)).map(wrap)
}

/// `pack2(a, b, fmt, into, at)`.
#[pyfunction]
#[pyo3(signature = (a, b, fmt, at=None, exact=true, into=None))]
fn tree_pack2(
    a: Bound<'_, PyTuple>,
    b: Bound<'_, PyTuple>,
    fmt: String,
    at: Option<Bound<'_, PyAny>>,
    exact: bool,
    into: Option<Bound<'_, PyTuple>>,
) -> PyResult<DslTree> {
    let into = into.map(|s| tree(&s)).transpose()?;
    ok(funcs::pack2(
        tree(&a)?,
        tree(&b)?,
        fmt_arg(fmt, at, exact)?,
        into,
    ))
    .map(wrap)
}

/// `op(name, *args, ty=, **fields)`.
#[pyfunction]
#[pyo3(signature = (name, xs, t=None, f=Vec::new()))]
fn tree_instr(
    name: &str,
    xs: Vec<Bound<'_, PyTuple>>,
    t: Option<String>,
    f: Vec<(String, Bound<'_, PyAny>)>,
) -> PyResult<DslTree> {
    let t = t.map(|t| ty(&t)).transpose()?;
    ok(funcs::instr(name, trees(&xs)?, t, fields(f)?)).map(wrap)
}

/// Adds the tree functions.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DslTree>()?;
    m.add_function(wrap_pyfunction!(tree_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(tree_binop, m)?)?;
    m.add_function(wrap_pyfunction!(tree_compare, m)?)?;
    m.add_function(wrap_pyfunction!(tree_divmod, m)?)?;
    m.add_function(wrap_pyfunction!(tree_update, m)?)?;
    m.add_function(wrap_pyfunction!(tree_invert, m)?)?;
    m.add_function(wrap_pyfunction!(tree_sign, m)?)?;
    m.add_function(wrap_pyfunction!(tree_cast, m)?)?;
    m.add_function(wrap_pyfunction!(cast_pair, m)?)?;
    m.add_function(wrap_pyfunction!(tree_select, m)?)?;
    m.add_function(wrap_pyfunction!(tree_call, m)?)?;
    m.add_function(wrap_pyfunction!(tree_lanes, m)?)?;
    m.add_function(wrap_pyfunction!(tree_bfe, m)?)?;
    m.add_function(wrap_pyfunction!(tree_unpack, m)?)?;
    m.add_function(wrap_pyfunction!(tree_pack2, m)?)?;
    m.add_function(wrap_pyfunction!(tree_instr, m)?)
}
