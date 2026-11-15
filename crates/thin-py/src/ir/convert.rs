//! Conversions between Python values and the IR's native data: host
//! objects, field values, operands and attribute payloads
//!

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyString, PyTuple};
use thin_ir::{
    Atom, Attr, AttrClass, Datum, FieldValue, HostObject, Imm, ImmValue, LdsAccess, ObjectFacts,
    Operand, Reg,
};
use thin_target::phys_named;

/// An IR refusal as `TypeError` or `ValueError`.
pub(super) fn error(e: thin_ir::Error) -> PyErr {
    match e {
        thin_ir::Error::Type(msg) => PyTypeError::new_err(msg),
        thin_ir::Error::Value(msg) => PyValueError::new_err(msg),
    }
}

/// A Python object behind the IR's handle, with the facts it reads.
pub(super) fn object(v: &Bound<'_, PyAny>) -> PyResult<HostObject> {
    let is_int = v.is_instance_of::<PyInt>();
    let facts = ObjectFacts {
        truthy: v.is_truthy()?,
        str: v.str()?.to_string(),
        repr: v.repr()?.to_string(),
        is_int,
        int: v.extract::<i128>().ok().filter(|_| is_int),
        is_str: v.is_instance_of::<PyString>(),
        is_tuple: v.is_instance_of::<PyTuple>(),
        is_none: v.is_none(),
    };
    let handle: Arc<dyn std::any::Any + Send + Sync> = Arc::new(v.clone().unbind());
    Ok(HostObject {
        handle,
        facts: Arc::new(facts),
    })
}

/// The Python object of a handle; native data (a Builder's) as nested
/// tuples, a register as `("reg", value, file, lo, n)`.
pub(super) fn host(py: Python<'_>, o: &HostObject) -> Py<PyAny> {
    match (o.handle.downcast_ref::<Py<PyAny>>(), o.datum()) {
        (Some(obj), _) => obj.clone_ref(py),
        (None, Some(d)) => datum(py, d).unwrap_or_else(|_| py.None()),
        (None, None) => py.None(),
    }
}

/// Native data as a Python value.
fn datum(py: Python<'_>, d: &Datum) -> PyResult<Py<PyAny>> {
    Ok(match d {
        Datum::None => py.None(),
        Datum::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        Datum::Str(s) => PyString::new(py, s).into_any().unbind(),
        Datum::Reg(r) => ("reg", r.value, r.file, r.lo, r.n)
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        Datum::Tuple(t) => {
            let items = t
                .iter()
                .map(|x| datum(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, items)?.into_any().unbind()
        }
    })
}

/// A tag atom: an exact int that fits, or an exact str.
pub(super) fn atom(v: &Bound<'_, PyAny>) -> Option<Atom> {
    if v.is_exact_instance_of::<PyInt>() {
        v.extract().ok().map(Atom::Int)
    } else if v.is_exact_instance_of::<PyString>() {
        Some(Atom::Str(v.extract::<String>().ok()?.into()))
    } else {
        None
    }
}

/// A tuple's field value using exact host-type checks, or none for an Object.
pub(super) fn tuple_value(t: &Bound<'_, PyTuple>) -> Option<FieldValue> {
    let items: Vec<Bound<'_, PyAny>> = t.iter().collect();
    let exact_ints = items.iter().all(|x| x.is_exact_instance_of::<PyInt>());
    if let (true, Ok(ints)) = (
        exact_ints,
        items
            .iter()
            .map(|x| x.extract())
            .collect::<PyResult<Vec<i128>>>(),
    ) {
        return Some(FieldValue::Ints(ints.into()));
    }
    if items.iter().all(|x| x.is_exact_instance_of::<PyString>()) {
        let names = items.iter().map(|x| x.extract::<String>().map(Into::into));
        return names
            .collect::<PyResult<Vec<Arc<str>>>>()
            .ok()
            .map(|n| FieldValue::Names(n.into()));
    }
    let tag = |x: &Bound<'_, PyAny>| -> Option<thin_ir::Tag> {
        let t = x.cast_exact::<PyTuple>().ok()?;
        t.iter()
            .map(|a| atom(&a))
            .collect::<Option<Vec<Atom>>>()
            .map(Into::into)
    };
    items
        .iter()
        .map(tag)
        .collect::<Option<Vec<_>>>()
        .map(|t| FieldValue::Tags(t.into()))
}

/// Convert a field value, accepting only its exact host type.
pub(super) fn field_value(v: &Bound<'_, PyAny>) -> PyResult<FieldValue> {
    let exact = if v.is_exact_instance_of::<PyBool>() {
        Some(FieldValue::Bool(v.extract()?))
    } else if v.is_exact_instance_of::<PyInt>() {
        v.extract().ok().map(FieldValue::Int)
    } else if v.is_exact_instance_of::<PyString>() {
        Some(FieldValue::Text(v.extract::<String>()?.into()))
    } else {
        v.cast_exact::<PyTuple>().ok().and_then(tuple_value)
    };
    exact.map_or_else(|| object(v).map(FieldValue::Object), Ok)
}

/// A tag as a Python tuple.
pub(super) fn tag_out<'py>(py: Python<'py>, tag: &[Atom]) -> PyResult<Bound<'py, PyTuple>> {
    let atom = |a: &Atom| -> PyResult<Bound<'py, PyAny>> {
        Ok(match a {
            Atom::Int(v) => v.into_pyobject(py)?.into_any(),
            Atom::Str(s) => PyString::new(py, s).into_any(),
        })
    };
    PyTuple::new(py, tag.iter().map(atom).collect::<PyResult<Vec<_>>>()?)
}

/// A field value as the Python value it was made from.
pub(super) fn field_out(py: Python<'_>, v: &FieldValue) -> PyResult<Py<PyAny>> {
    Ok(match v {
        FieldValue::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        FieldValue::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        FieldValue::Text(t) => PyString::new(py, t).into_any().unbind(),
        FieldValue::Ints(v) => PyTuple::new(py, v.iter())?.into_any().unbind(),
        FieldValue::Names(v) => PyTuple::new(py, v.iter().map(|s| &**s))?
            .into_any()
            .unbind(),
        FieldValue::Tags(v) => {
            let tags = v
                .iter()
                .map(|t| tag_out(py, t))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, tags)?.into_any().unbind()
        }
        FieldValue::Object(o) => host(py, o),
    })
}

/// A register as Python gives it: (value, file, lo, n, part).
pub(super) type RegIn = (i32, char, u32, u32, u32);

pub(super) fn reg((value, file, lo, n, part): RegIn) -> Reg {
    Reg {
        value,
        lo,
        n,
        part,
        file,
    }
}

/// An operand as Python gives it: `("r", RegIn)`, `("p", name)`,
/// `("i", int, text)`, `("f", float, text)` or `("b", name)`.
pub(super) fn operand(o: &Bound<'_, PyAny>) -> PyResult<Operand> {
    let tag: String = o.get_item(0)?.extract()?;
    let arg = o.get_item(1)?;
    let text = || -> PyResult<Arc<str>> { Ok(o.get_item(2)?.extract::<String>()?.into()) };
    Ok(match tag.as_str() {
        "r" => Operand::Reg(reg(arg.extract()?)),
        "p" => Operand::Phys(
            phys_named(&arg.extract::<String>()?)
                .ok_or_else(|| PyValueError::new_err("a named register"))?,
        ),
        "i" => Operand::Imm(Imm {
            value: ImmValue::Int(arg.extract()?),
            text: text()?,
        }),
        "f" => Operand::Imm(Imm {
            value: ImmValue::Float(arg.extract()?),
            text: text()?,
        }),
        "b" => Operand::Block(arg.extract::<String>()?.into()),
        _ => return Err(PyTypeError::new_err(format!("an operand tag {tag:?}"))),
    })
}

/// An operand as Python reads it back: `("r", k)` for register operand
/// k, else as given.
pub(super) fn operand_out<'py>(
    py: Python<'py>,
    k: usize,
    op: &Operand,
) -> PyResult<Bound<'py, PyTuple>> {
    let row: (&str, Bound<'py, PyAny>, &str) = match op {
        Operand::Reg(_) => ("r", k.into_pyobject(py)?.into_any(), ""),
        Operand::Phys(p) => ("p", PyString::new(py, p.name).into_any(), ""),
        Operand::Imm(Imm {
            value: ImmValue::Int(v),
            text,
        }) => ("i", v.into_pyobject(py)?.into_any(), text),
        Operand::Imm(Imm {
            value: ImmValue::Float(x),
            text,
        }) => ("f", x.into_pyobject(py)?.into_any(), text),
        Operand::Block(name) => ("b", PyString::new(py, name).into_any(), ""),
    };
    row.into_pyobject(py)
}

/// An attribute that holds plain data, from its class name and payload.
pub(super) fn data_attribute(class: &str, p: &Bound<'_, PyAny>) -> PyResult<Option<Attr>> {
    let strs = || -> PyResult<Vec<Arc<str>>> {
        Ok(p.extract::<Vec<String>>()?
            .into_iter()
            .map(Into::into)
            .collect())
    };
    let pair = |(r, w): (String, Option<String>)| (r.into(), w.map(Into::into));
    Ok(Some(match class {
        "Comment" => Attr::Comment(p.extract::<String>()?.into()),
        "Published" => Attr::Published(p.extract::<String>()?.into()),
        "Ablation" => Attr::Ablation(strs()?),
        "Places" => Attr::Places(strs()?),
        "AliasOk" => Attr::AliasOk(p.extract::<Vec<_>>()?.into_iter().map(pair).collect()),
        "InRegion" => Attr::InRegion(p.extract()?),
        "M0Set" => Attr::M0Set(p.extract()?),
        "WavesTaken" => Attr::WavesTaken(p.extract()?),
        "Hazard" => Attr::Hazard,
        "Fence" => Attr::Fence,
        _ => return Ok(None),
    }))
}

/// An attribute from its class name and the payload Python prepares.
pub(super) fn attribute(class: &str, p: &Bound<'_, PyAny>) -> PyResult<Attr> {
    if let Some(a) = data_attribute(class, p)? {
        return Ok(a);
    }
    let atom =
        |a: PyResult<Bound<'_, PyAny>>| atom(&a?).ok_or_else(|| PyTypeError::new_err("a tag atom"));
    Ok(match class {
        "Tag" => Attr::Tag(
            p.try_iter()?
                .map(atom)
                .collect::<PyResult<Vec<_>>>()?
                .into(),
        ),
        "Staged" => Attr::Staged(object(p)?),
        "WaitUses" => {
            let regs: Vec<RegIn> = p.get_item(1)?.extract()?;
            Attr::WaitUses(
                object(&p.get_item(0)?)?,
                regs.into_iter().map(reg).collect(),
            )
        }
        "LdsAccess" => {
            let (region, field, what, ordered): (u64, i128, String, String) =
                p.get_item(1)?.extract()?;
            let object = object(&p.get_item(0)?)?;
            Attr::LdsAccess(LdsAccess {
                object,
                region,
                field,
                what: what.into(),
                ordered: ordered.into(),
            })
        }
        _ => {
            return Err(PyTypeError::new_err(format!(
                "{class}: not an attribute of the IR"
            )))
        }
    })
}

/// An attribute's payload as Python rebuilds it: the instance for the
/// object-kept classes, else its stored data.
pub(super) fn attribute_out(py: Python<'_>, a: &Attr) -> PyResult<Py<PyAny>> {
    let any = |b: Bound<'_, PyAny>| b.unbind();
    Ok(match a {
        Attr::Comment(t) | Attr::Published(t) => any(PyString::new(py, t).into_any()),
        Attr::Ablation(v) | Attr::Places(v) => {
            any(PyTuple::new(py, v.iter().map(|s| &**s))?.into_any())
        }
        Attr::AliasOk(v) => {
            any(PyTuple::new(py, v.iter().map(|(r, w)| (&**r, w.as_deref())))?.into_any())
        }
        Attr::Tag(t) => any(tag_out(py, t)?.into_any()),
        Attr::Staged(o) | Attr::WaitUses(o, _) => host(py, o),
        Attr::LdsAccess(l) => host(py, &l.object),
        Attr::InRegion(v) | Attr::M0Set(v) => any(v.into_pyobject(py)?.into_any()),
        Attr::WavesTaken(w) => any(PyTuple::new(py, w.iter())?.into_any()),
        Attr::Hazard | Attr::Fence => any(PyTuple::empty(py).into_any()),
    })
}

/// An attribute class by name.
pub(super) fn class(name: &str) -> PyResult<AttrClass> {
    AttrClass::named(name)
        .ok_or_else(|| PyTypeError::new_err(format!("{name}: not an attribute of the IR")))
}
