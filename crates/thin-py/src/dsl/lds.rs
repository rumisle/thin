//! The LDS layout, its regions, addresses and pointers.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use thin_dsl::lds::{elem_of_name, elem_of_type, Address, Layout, Off, Pointer, RegionArgs, Xor};

use super::builder::DslBuilder;
use super::convert::{leafish, ok, ty};
use super::trees::DslTree;
use super::typed::DslTyped;

/// An LDS address: a region, a byte offset and a label.
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslAddress {
    pub(crate) a: Address,
}

/// A pointer into a region (`A.pointer`).
#[pyclass(module = "thin._native", frozen)]
pub(crate) struct DslPointer {
    p: Pointer,
}

/// An offset from a leafish tuple: an int stays one.
fn off(t: &Bound<'_, PyTuple>) -> PyResult<Off> {
    let tag: String = t.get_item(0)?.extract()?;
    match tag.as_str() {
        "i" => Ok(Off::Int(t.get_item(1)?.extract()?)),
        "b" => ok(thin_dsl::error::mem(format!(
            "{}: a byte offset",
            t.get_item(1)?.extract::<String>()?
        ))),
        _ => Ok(Off::Tree(ok(thin_dsl::expr::node::leaf(leafish(t)?))?)),
    }
}

fn offs(xs: &[Bound<'_, PyTuple>]) -> PyResult<Vec<Off>> {
    xs.iter().map(off).collect()
}

/// An offset as Python: an int or a tree.
fn off_out(py: Python<'_>, o: Off) -> PyResult<Py<PyAny>> {
    match o {
        Off::Int(c) => Ok(c.into_pyobject(py)?.into_any().unbind()),
        Off::Tree(t) => Ok(Bound::new(py, DslTree { t })?.into_any().unbind()),
    }
}

#[pymethods]
impl DslAddress {
    /// Its text.
    fn text(&self) -> String {
        self.a.text()
    }

    /// Its region's index.
    #[getter]
    fn region(&self) -> usize {
        self.a.region
    }

    /// Its byte offset: an int or a tree.
    #[getter]
    fn off(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        off_out(py, self.a.off.clone())
    }

    /// `addr + t`.
    fn plus(&self, t: Bound<'_, PyTuple>) -> PyResult<DslAddress> {
        Ok(DslAddress {
            a: self.a.plus(off(&t)?),
        })
    }

    /// `addr - t`.
    fn minus(&self, t: Bound<'_, PyTuple>) -> PyResult<DslAddress> {
        Ok(DslAddress {
            a: ok(self.a.minus(off(&t)?))?,
        })
    }
}

#[pymethods]
impl DslPointer {
    /// Its value.
    #[getter]
    fn value(&self) -> DslTyped {
        DslTyped {
            t: self.p.value.clone(),
        }
    }

    /// Its region's index.
    #[getter]
    fn region(&self) -> usize {
        self.p.region
    }
}

fn layout(b: &DslBuilder) -> &Layout {
    b.b.layout.as_ref().expect("a layout")
}

fn layout_mut(b: &mut DslBuilder) -> &mut Layout {
    b.b.layout.as_mut().expect("a layout")
}

/// A region's facts: (name, shape, element, element bytes, strides, size,
/// nbytes given, align, parent and index, swizzle, parts).
type RegionRow = (
    String,
    Vec<u64>,
    String,
    u64,
    Vec<u64>,
    u64,
    bool,
    u64,
    Option<(usize, u64)>,
    Option<(i64, i64, u64, u32)>,
    Vec<usize>,
);

/// The region's arguments from Python: (shape, element type or name,
/// pitch, pitch dict, nbytes, align, at, swizzle).
type RegionIn<'py> = (
    Vec<i128>,
    Bound<'py, PyAny>,
    Option<i128>,
    Option<Vec<(i64, i128)>>,
    Option<i128>,
    Option<u64>,
    Option<u64>,
    Option<(i64, i64, u64, u32)>,
);

fn region_args(a: RegionIn<'_>, typed: bool) -> PyResult<RegionArgs> {
    let (shape, elem, pitch, pitch_dims, nbytes, align, at, swizzle) = a;
    let name: String = elem.extract()?;
    let elem = match typed {
        true => elem_of_type(ty(&name)?),
        false => elem_of_name(&name),
    };
    let swizzle = swizzle.map(|(by, into, mask, shift)| Xor {
        by,
        into,
        mask,
        shift,
    });
    Ok(RegionArgs {
        shape,
        elem: Some(ok(elem)?),
        pitch,
        pitch_dims,
        nbytes,
        align,
        at,
        swizzle,
    })
}

/// `b.lds(budget)`.
#[pyfunction]
fn lds_make(mut b: PyRefMut<'_, DslBuilder>, budget: Option<i128>) -> PyResult<()> {
    ok(b.b.lds(budget))
}

/// Whether the kernel has a layout.
#[pyfunction]
fn lds_exists(b: PyRef<'_, DslBuilder>) -> bool {
    b.b.layout.is_some()
}

/// Opens a struct or union group.
#[pyfunction]
fn lds_open(
    mut b: PyRefMut<'_, DslBuilder>,
    union: bool,
    name: Option<String>,
    align: Option<u64>,
    at: Option<u64>,
) -> PyResult<()> {
    ok(layout_mut(&mut b).open_group(union, name, align, at))
}

/// Closes the innermost group.
#[pyfunction]
fn lds_close(mut b: PyRefMut<'_, DslBuilder>) {
    layout_mut(&mut b).close_group();
}

/// `L.region(name, …)`; `typed` when the element is a type.
#[pyfunction]
fn lds_region(
    mut b: PyRefMut<'_, DslBuilder>,
    name: &str,
    a: RegionIn<'_>,
    typed: bool,
) -> PyResult<usize> {
    let args = region_args(a, typed)?;
    ok(layout_mut(&mut b).region(name, args))
}

/// `A.parts(*names)`.
#[pyfunction]
fn lds_parts(
    mut b: PyRefMut<'_, DslBuilder>,
    a: usize,
    names: Vec<String>,
) -> PyResult<Vec<usize>> {
    ok(layout_mut(&mut b).parts(a, &names))
}

/// Freezes the layout and returns its end (`L.size`).
#[pyfunction]
fn lds_size(mut b: PyRefMut<'_, DslBuilder>) -> PyResult<u64> {
    ok(b.b.frozen_layout()).map(|l| l.end)
}

/// `L.leaves()`.
#[pyfunction]
fn lds_leaves(b: PyRef<'_, DslBuilder>) -> Vec<usize> {
    layout(&b).leaves()
}

/// `L.table()`.
#[pyfunction]
fn lds_table(mut b: PyRefMut<'_, DslBuilder>) -> PyResult<Vec<String>> {
    ok(b.b.frozen_layout()).map(|l| l.table())
}

/// The regions' names in index order.
#[pyfunction]
fn lds_names(b: PyRef<'_, DslBuilder>) -> Vec<String> {
    layout(&b).regions.iter().map(|r| r.name.clone()).collect()
}

/// A region's index by name.
#[pyfunction]
fn lds_named(b: PyRef<'_, DslBuilder>, name: &str) -> Option<usize> {
    layout(&b).by_name.get(name).copied()
}

/// A region's facts.
#[pyfunction]
fn lds_info(b: PyRef<'_, DslBuilder>, k: usize) -> RegionRow {
    let r = &layout(&b).regions[k];
    let sw = r.swizzle.map(|x| (x.by, x.into, x.mask, x.shift));
    (
        r.name.clone(),
        r.shape.clone(),
        r.elem.name.clone(),
        r.elem.bytes,
        r.strides.clone(),
        r.size,
        r.nbytes,
        r.align,
        r.parent,
        sw,
        r.parts.clone(),
    )
}

/// A region's base (freezes the layout).
#[pyfunction]
fn lds_base(mut b: PyRefMut<'_, DslBuilder>, k: usize) -> PyResult<u64> {
    ok(b.b.frozen_layout()).map(|l| l.regions[k].base)
}

/// `A.tags()`.
#[pyfunction]
fn lds_tags(b: PyRef<'_, DslBuilder>, k: usize) -> Vec<String> {
    layout(&b).tags(k)
}

/// `A.offset(*idx, swizzled=)`.
#[pyfunction]
fn lds_offset(
    b: PyRef<'_, DslBuilder>,
    py: Python<'_>,
    k: usize,
    idx: Vec<Bound<'_, PyTuple>>,
    swizzled: bool,
) -> PyResult<Py<PyAny>> {
    let o = ok(layout(&b).regions[k].offset(&offs(&idx)?, swizzled))?;
    off_out(py, o)
}

/// `A.stored(*idx)`.
#[pyfunction]
fn lds_stored(
    b: PyRef<'_, DslBuilder>,
    py: Python<'_>,
    k: usize,
    idx: Vec<Bound<'_, PyTuple>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let s = ok(layout(&b).regions[k].stored(&offs(&idx)?))?;
    s.into_iter().map(|o| off_out(py, o)).collect()
}

/// `A + t` (t a leafish tuple; `("i", 0)` for the region itself).
#[pyfunction]
fn lds_address(b: PyRef<'_, DslBuilder>, k: usize, t: Bound<'_, PyTuple>) -> PyResult<DslAddress> {
    let r = &layout(&b).regions[k];
    let a = Address {
        builder: b.b.id,
        region: k,
        name: r.name.clone(),
        off: Off::Int(0),
        label: None,
    };
    Ok(DslAddress {
        a: a.plus(off(&t)?),
    })
}

/// `A.pointer(*idx, name=)`.
#[pyfunction]
fn lds_pointer(
    mut b: PyRefMut<'_, DslBuilder>,
    k: usize,
    idx: Vec<Bound<'_, PyTuple>>,
    name: Option<String>,
) -> PyResult<DslPointer> {
    let w = b.b.where_text.clone();
    Ok(DslPointer {
        p: ok(b.b.pointer(k, &offs(&idx)?, name, w))?,
    })
}

/// `p.at(*d)`.
#[pyfunction]
fn lds_pointer_at(
    mut b: PyRefMut<'_, DslBuilder>,
    p: PyRef<'_, DslPointer>,
    d: Vec<Bound<'_, PyTuple>>,
) -> PyResult<DslAddress> {
    let id = b.b.id;
    let d = offs(&d)?;
    let l = ok(b.b.frozen_layout())?;
    Ok(DslAddress {
        a: ok(p.p.at(l, id, &d))?,
    })
}

/// Adds the LDS classes.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DslAddress>()?;
    m.add_function(wrap_pyfunction!(lds_make, m)?)?;
    m.add_function(wrap_pyfunction!(lds_exists, m)?)?;
    m.add_function(wrap_pyfunction!(lds_open, m)?)?;
    m.add_function(wrap_pyfunction!(lds_close, m)?)?;
    m.add_function(wrap_pyfunction!(lds_region, m)?)?;
    m.add_function(wrap_pyfunction!(lds_parts, m)?)?;
    m.add_function(wrap_pyfunction!(lds_size, m)?)?;
    m.add_function(wrap_pyfunction!(lds_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(lds_table, m)?)?;
    m.add_function(wrap_pyfunction!(lds_named, m)?)?;
    m.add_function(wrap_pyfunction!(lds_names, m)?)?;
    m.add_function(wrap_pyfunction!(lds_info, m)?)?;
    m.add_function(wrap_pyfunction!(lds_base, m)?)?;
    m.add_function(wrap_pyfunction!(lds_tags, m)?)?;
    m.add_function(wrap_pyfunction!(lds_offset, m)?)?;
    m.add_function(wrap_pyfunction!(lds_stored, m)?)?;
    m.add_function(wrap_pyfunction!(lds_address, m)?)?;
    m.add_function(wrap_pyfunction!(lds_pointer, m)?)?;
    m.add_function(wrap_pyfunction!(lds_pointer_at, m)?)?;
    m.add_class::<DslPointer>()
}
