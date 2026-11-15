//! The argument block and the entry.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use thin_dsl::args::{Arg, Args, Given, Kind};
use thin_dsl::entry::{Derived, Loaded};
use thin_dsl::maker::families::Val;

use super::builder::DslBuilder;
use super::convert::{ok, tree, ty};
use super::typed::DslTyped;

/// An argument kind.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslKind {
    k: Kind,
}

#[pymethods]
impl DslKind {
    /// The kind as written.
    fn text(&self) -> String {
        self.k.text()
    }

    /// `.multiple_of(m)`: `m` None when not an int, `written` its repr.
    fn multiple_of(&self, m: Option<i128>, written: &str) -> PyResult<DslKind> {
        ok(self.k.multiple_of(m, written)).map(|k| DslKind { k })
    }

    /// `.divisor()`.
    fn divisor(&self) -> PyResult<DslKind> {
        ok(self.k.divisor()).map(|k| DslKind { k })
    }

    /// `.uncached`.
    fn uncached(&self) -> PyResult<DslKind> {
        ok(self.k.uncached()).map(|k| DslKind { k })
    }
}

/// A type as a kind.
#[pyfunction]
fn kind_type(t: &str) -> PyResult<DslKind> {
    Ok(DslKind {
        k: Kind::Type {
            ty: ty(t)?,
            range: None,
            divisor: false,
            multiple: 1,
        },
    })
}

/// `T.within(lo, hi)`.
#[pyfunction]
fn kind_within(t: &str, lo: i128, hi: i128) -> PyResult<DslKind> {
    ok(Kind::within(ty(t)?, lo, hi)).map(|k| DslKind { k })
}

/// `scratch`, `kept`, `carried`, `gens`.
#[pyfunction]
fn kind_own(owned: String) -> DslKind {
    DslKind {
        k: Kind::Own {
            owned,
            uncached: false,
            ranks: None,
        },
    }
}

/// `mapped(w)` and `peers(w)`.
#[pyfunction]
fn kind_ranked(peers: bool, w: Option<i128>, written: &str) -> PyResult<DslKind> {
    ok(Kind::ranked(peers, w, written)).map(|k| DslKind { k })
}

/// `lib_arg(kind, library)`.
#[pyfunction]
fn kind_lib(k: PyRef<'_, DslKind>, library: String) -> DslKind {
    DslKind {
        k: Kind::Lib(Box::new(k.k.clone()), library),
    }
}

/// The argument block.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslArgs {
    pub(crate) a: Args,
}

fn record<'py>(py: Python<'py>, a: &Arg) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", &a.name)?;
    d.set_item("offset", a.offset)?;
    d.set_item("bytes", a.bytes)?;
    d.set_item("kind", &a.kind)?;
    d.set_item("lo", a.lo)?;
    d.set_item("hi", a.hi)?;
    d.set_item("divisor", a.divisor)?;
    d.set_item("multiple", a.multiple)?;
    d.set_item("owned", &a.owned)?;
    d.set_item("library", &a.library)?;
    d.set_item("mem", &a.mem)?;
    d.set_item("ranks", a.ranks)?;
    Ok(d)
}

#[pymethods]
impl DslArgs {
    /// `Args(at=, **kinds)`: each kind a DslKind, else its repr.
    #[new]
    fn new(
        kinds: Vec<(String, Bound<'_, PyAny>)>,
        at: BTreeMap<String, i128>,
    ) -> PyResult<DslArgs> {
        let given = |(n, k): (String, Bound<'_, PyAny>)| -> PyResult<(String, Given)> {
            match k.extract::<PyRef<'_, DslKind>>() {
                Ok(k) => Ok((n, Ok(k.k.clone()))),
                Err(_) => Ok((n, Err(k.repr()?.to_string()))),
            }
        };
        let kinds = kinds.into_iter().map(given).collect::<PyResult<Vec<_>>>()?;
        ok(Args::new(kinds, &at)).map(|a| DslArgs { a })
    }

    /// `args.size`.
    #[getter]
    fn size(&self) -> u64 {
        self.a.size()
    }

    /// `args.names()`.
    fn names(&self) -> Vec<String> {
        self.a.args.iter().map(|a| a.name.clone()).collect()
    }

    /// `args.own()`.
    fn own(&self) -> Vec<String> {
        self.a.own()
    }

    /// `args[name]`, or None.
    fn record<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.a.get(name).map(|a| record(py, a)).transpose()
    }
}

/// `b.entry(args, wg=, preload=, wgsize=)`.
#[pyfunction]
fn entry(
    mut b: PyRefMut<'_, DslBuilder>,
    args: Option<PyRef<'_, DslArgs>>,
    wg: i128,
    preload: u32,
    wgsize: Option<u32>,
) -> PyResult<()> {
    let a = args.map(|a| a.a.clone()).unwrap_or_default();
    ok(b.b.entry(a, wg, preload, wgsize))
}

/// The entry: (karg, tid, wgid, wgid_y, preload, wg, wgsize).
type EntryOut = (
    DslTyped,
    DslTyped,
    DslTyped,
    Option<DslTyped>,
    u32,
    u32,
    Option<u32>,
);

/// The entry's values, or None before `b.entry`.
#[pyfunction]
fn entry_info(b: PyRef<'_, DslBuilder>) -> Option<EntryOut> {
    let w = |t: &thin_dsl::values::Typed| DslTyped { t: t.clone() };
    b.b.entered.as_ref().map(|k| {
        (
            w(&k.karg),
            w(&k.tid),
            w(&k.wgid),
            k.wgid_y.as_ref().map(w),
            k.preload,
            k.wg,
            k.wgsize,
        )
    })
}

/// `k.entry_moves`: the values the entry moves copy, in order.
#[pyfunction]
fn entry_moves(b: PyRef<'_, DslBuilder>) -> Vec<DslTyped> {
    (b.b.entered.iter())
        .flat_map(|k| k.entry_moves.iter())
        .map(|(_, t)| DslTyped { t: t.clone() })
        .collect()
}

/// `k.lane`.
#[pyfunction]
fn entry_lane(mut b: PyRefMut<'_, DslBuilder>) -> PyResult<DslTyped> {
    ok(b.b.entry_lane()).map(|t| DslTyped { t })
}

/// `k.wave`: a value or the int 0.
#[pyfunction]
fn entry_wave(py: Python<'_>, mut b: PyRefMut<'_, DslBuilder>) -> PyResult<Py<PyAny>> {
    match ok(b.b.entry_wave())? {
        Derived::Value(t) => Ok(Bound::new(py, DslTyped { t })?.into_any().unbind()),
        Derived::Int(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
    }
}

fn loaded(py: Python<'_>, l: Loaded) -> PyResult<Py<PyAny>> {
    let w = |t| DslTyped { t };
    match l {
        Loaded::Value(t) => Ok(Bound::new(py, w(t))?.into_any().unbind()),
        Loaded::Ranked(ranks, own) => {
            let ranks: Vec<DslTyped> = ranks.into_iter().map(w).collect();
            Ok((ranks, w(own)).into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// `k.load(*names, comment=)`.
#[pyfunction]
fn entry_load(
    py: Python<'_>,
    mut b: PyRefMut<'_, DslBuilder>,
    names: Vec<String>,
    comment: &str,
) -> PyResult<Vec<Py<PyAny>>> {
    let out = ok(b.b.load(&names, comment))?;
    out.into_iter().map(|l| loaded(py, l)).collect()
}

/// `k.load_rank(name, d, comment=)`: d an int, a value or a tree.
#[pyfunction]
fn entry_load_rank(
    mut b: PyRefMut<'_, DslBuilder>,
    name: &str,
    d: Bound<'_, PyTuple>,
    comment: &str,
) -> PyResult<DslTyped> {
    let d = Val::Tree(tree(&d)?);
    ok(b.b.load_rank(name, d, comment)).map(|t| DslTyped { t })
}

/// Adds the argument kinds, the block and the entry.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DslKind>()?;
    m.add_class::<DslArgs>()?;
    m.add_function(wrap_pyfunction!(kind_type, m)?)?;
    m.add_function(wrap_pyfunction!(kind_within, m)?)?;
    m.add_function(wrap_pyfunction!(kind_own, m)?)?;
    m.add_function(wrap_pyfunction!(kind_ranked, m)?)?;
    m.add_function(wrap_pyfunction!(kind_lib, m)?)?;
    m.add_function(wrap_pyfunction!(entry, m)?)?;
    m.add_function(wrap_pyfunction!(entry_info, m)?)?;
    m.add_function(wrap_pyfunction!(entry_moves, m)?)?;
    m.add_function(wrap_pyfunction!(entry_lane, m)?)?;
    m.add_function(wrap_pyfunction!(entry_wave, m)?)?;
    m.add_function(wrap_pyfunction!(entry_load, m)?)?;
    m.add_function(wrap_pyfunction!(entry_load_rank, m)?)
}
