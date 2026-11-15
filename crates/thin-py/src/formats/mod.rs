//! The functions of `thin._native` over thin-formats:
//! dtypes, encode/decode, MX and layouts here, code objects and their
//! metadata in [`code_object`].

mod code_object;

use numpy::{PyArray1, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use thin_formats::{dtype, EncodeOptions, Layout, Shape};

use crate::{input, output, raise};

/// A formats refusal as the class names.
pub(crate) fn error(py: Python<'_>, error: thin_formats::Error) -> PyErr {
    use thin_formats::Error as E;
    match error {
        E::Type(msg) => PyTypeError::new_err(msg),
        E::Value(msg) => PyValueError::new_err(msg),
        E::Layout(msg) => raise(py, "thin.formats", "LayoutError", msg),
        E::CodeObject(msg) => raise(py, "thin.formats", "CodeObjectError", msg),
    }
}

/// A dtype's encoding facts: (exp, man, inf, NaN kind, signed,
/// subnormals, bias, max_bits, max).
type EncodingRow = (u32, u32, bool, &'static str, bool, bool, i32, u32, f64);

/// A dtype as Python receives it: (name, width, numpy, torch, dlpack, encoding).
type DtypeRow = (
    &'static str,
    u32,
    &'static str,
    &'static str,
    (u8, u8),
    Option<EncodingRow>,
);

/// The dtype table.
#[pyfunction]
fn dtypes() -> Vec<DtypeRow> {
    use thin_formats::NanKind as N;
    let encoding = |e: &thin_formats::Encoding| {
        let nan = match e.nan {
            N::Ieee => "IEEE",
            N::AllOnes => "ALL_ONES",
            N::NegZero => "NEG_ZERO",
            N::None => "none",
        };
        let facts = (e.bias(), e.max_bits(), e.max());
        (
            e.exp,
            e.man,
            e.inf,
            nan,
            e.signed,
            e.subnormals,
            facts.0,
            facts.1,
            facts.2,
        )
    };
    let row = |d: &thin_formats::Dtype| {
        let enc = d.encoding.as_ref().map(encoding);
        (d.name, d.width, d.numpy, d.torch, d.dlpack, enc)
    };
    thin_formats::DTYPES.iter().map(row).collect()
}

/// Encodes f32 bit patterns into a dtype's storage bytes.
#[pyfunction]
#[pyo3(signature = (x, name, saturate, out, nan=None, inf=None))]
fn encode_into(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<'_, u32>,
    name: &str,
    saturate: bool,
    mut out: PyReadwriteArrayDyn<'_, u8>,
    nan: Option<i64>,
    inf: Option<i64>,
) -> PyResult<()> {
    let options = EncodeOptions { saturate, nan, inf };
    let (x, out) = (input(&x)?, output(&mut out)?);
    py.detach(|| dtype(name).and_then(|d| thin_formats::encode(x, d, options, out)))
        .map_err(|e| error(py, e))
}

/// Checks encode options for a dtype.
#[pyfunction]
#[pyo3(signature = (name, saturate, nan=None, inf=None))]
fn check_encode(
    py: Python<'_>,
    name: &str,
    saturate: bool,
    nan: Option<i64>,
    inf: Option<i64>,
) -> PyResult<()> {
    let options = EncodeOptions { saturate, nan, inf };
    dtype(name)
        .and_then(|d| thin_formats::check_encode(d, options))
        .map_err(|e| error(py, e))
}

/// The encode table of a dtype of at most 6 mantissa bits.
#[pyfunction]
#[pyo3(signature = (name, saturate, out, nan=None, inf=None))]
fn encode_table_into(
    py: Python<'_>,
    name: &str,
    saturate: bool,
    mut out: PyReadwriteArrayDyn<'_, u8>,
    nan: Option<i64>,
    inf: Option<i64>,
) -> PyResult<()> {
    let options = EncodeOptions { saturate, nan, inf };
    let out = output(&mut out)?;
    dtype(name)
        .and_then(|d| thin_formats::encode_table(d, options, out))
        .map_err(|e| error(py, e))
}

/// Decodes a dtype's storage bytes into f32 bit patterns.
#[pyfunction]
fn decode_into(
    py: Python<'_>,
    bits: PyReadonlyArrayDyn<'_, u8>,
    name: &str,
    mut out: PyReadwriteArrayDyn<'_, u32>,
) -> PyResult<()> {
    let (bits, out) = (input(&bits)?, output(&mut out)?);
    py.detach(|| dtype(name).and_then(|d| thin_formats::decode(bits, d, out)))
        .map_err(|e| error(py, e))
}

/// The f32 bits of every pattern of a dtype of at most 16 bits.
#[pyfunction]
fn decode_table_into(
    py: Python<'_>,
    name: &str,
    mut out: PyReadwriteArrayDyn<'_, u32>,
) -> PyResult<()> {
    let out = output(&mut out)?;
    dtype(name)
        .and_then(|d| thin_formats::decode_table(d, out))
        .map_err(|e| error(py, e))
}

/// One scale byte per f32 group maximum.
#[pyfunction]
fn scale_bytes_into(
    py: Python<'_>,
    amax: PyReadonlyArrayDyn<'_, u32>,
    name: &str,
    mut out: PyReadwriteArrayDyn<'_, u8>,
) -> PyResult<()> {
    let (amax, out) = (input(&amax)?, output(&mut out)?);
    py.detach(|| dtype(name).and_then(|d| thin_formats::scale_bytes(amax, d, out)))
        .map_err(|e| error(py, e))
}

/// MX codes and scales of f32 values of a shape.
#[pyfunction]
fn quantize_into(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<'_, u32>,
    shape: Vec<usize>,
    name: &str,
    mut codes: PyReadwriteArrayDyn<'_, u8>,
    mut scales: PyReadwriteArrayDyn<'_, u8>,
) -> PyResult<()> {
    let (x, codes, scales) = (input(&x)?, output(&mut codes)?, output(&mut scales)?);
    py.detach(|| dtype(name).and_then(|d| thin_formats::quantize(x, &shape, d, codes, scales)))
        .map_err(|e| error(py, e))
}

/// A layout as Python receives it: (name, non-default args, key).
type LayoutRow = (&'static str, Vec<(&'static str, i64)>, String);

fn layout_row(layout: Layout) -> LayoutRow {
    (layout.name(), layout.args().to_vec(), layout.key())
}

/// Every layout definition: (name, [(param, default)]).
#[pyfunction]
fn layout_definitions() -> Vec<(&'static str, Vec<(&'static str, i64)>)> {
    let row = |d: &&thin_formats::layout::Definition| (d.name, d.params.to_vec());
    thin_formats::layout::definitions()
        .iter()
        .map(row)
        .collect()
}

/// A layout of a definition and arguments by name.
#[pyfunction]
fn layout_new(py: Python<'_>, name: &str, args: Vec<(String, i64)>) -> PyResult<LayoutRow> {
    let args: Vec<(&str, i64)> = args.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    Layout::new(name, &args)
        .map(layout_row)
        .map_err(|e| error(py, e))
}

/// The layout of a key, refused unless canonical.
#[pyfunction]
fn layout_parse(py: Python<'_>, key: &str) -> PyResult<LayoutRow> {
    Layout::parse(key).map(layout_row).map_err(|e| error(py, e))
}

/// The bytes a layout occupies.
#[pyfunction]
fn layout_nbytes(py: Python<'_>, key: &str, name: &str, shape: Shape) -> PyResult<usize> {
    Layout::parse(key)
        .and_then(|l| dtype(name).and_then(|d| thin_formats::nbytes(&l, d, shape)))
        .map_err(|e| error(py, e))
}

/// The bytes of storage elements in a layout, as a new uint8 array
///
#[pyfunction]
fn to_bytes<'py>(
    py: Python<'py>,
    elements: PyReadonlyArrayDyn<'py, u8>,
    key: &str,
    name: &str,
    shape: Shape,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let elements = input(&elements)?;
    let run = |l: Layout| dtype(name).and_then(|d| thin_formats::to_bytes(elements, &l, d, shape));
    let bytes = py
        .detach(|| Layout::parse(key).and_then(run))
        .map_err(|e| error(py, e))?;
    Ok(PyArray1::from_vec(py, bytes))
}

/// Storage elements back from a layout's bytes.
#[pyfunction]
fn from_bytes_into(
    py: Python<'_>,
    raw: PyReadonlyArrayDyn<'_, u8>,
    key: &str,
    name: &str,
    shape: Shape,
    mut out: PyReadwriteArrayDyn<'_, u8>,
) -> PyResult<()> {
    let (raw, out) = (input(&raw)?, output(&mut out)?);
    let run =
        |l: Layout| dtype(name).and_then(|d| thin_formats::from_bytes(raw, &l, d, shape, out));
    py.detach(|| Layout::parse(key).and_then(run))
        .map_err(|e| error(py, e))
}

/// The coordinates of layout offsets into an int64 array of shape [n, 3]
///
#[pyfunction]
fn coords_into(
    py: Python<'_>,
    key: &str,
    name: &str,
    shape: Shape,
    offsets: PyReadonlyArrayDyn<'_, u64>,
    mut out: PyReadwriteArrayDyn<'_, i64>,
) -> PyResult<()> {
    let (offsets, out) = (input(&offsets)?, output(&mut out)?);
    let run =
        |l: Layout| dtype(name).and_then(|d| thin_formats::coords(&l, d, shape, offsets, out));
    py.detach(|| Layout::parse(key).and_then(run))
        .map_err(|e| error(py, e))
}

/// Adds this module's functions to `thin._native`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for function in [
        wrap_pyfunction!(dtypes, m)?,
        wrap_pyfunction!(encode_into, m)?,
        wrap_pyfunction!(check_encode, m)?,
        wrap_pyfunction!(encode_table_into, m)?,
        wrap_pyfunction!(decode_into, m)?,
        wrap_pyfunction!(decode_table_into, m)?,
        wrap_pyfunction!(scale_bytes_into, m)?,
        wrap_pyfunction!(quantize_into, m)?,
        wrap_pyfunction!(layout_definitions, m)?,
        wrap_pyfunction!(layout_new, m)?,
        wrap_pyfunction!(layout_parse, m)?,
        wrap_pyfunction!(layout_nbytes, m)?,
        wrap_pyfunction!(to_bytes, m)?,
        wrap_pyfunction!(from_bytes_into, m)?,
        wrap_pyfunction!(coords_into, m)?,
    ] {
        m.add_function(function)?;
    }
    code_object::register(m)
}
