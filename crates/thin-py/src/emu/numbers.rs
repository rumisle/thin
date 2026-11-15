//! The bit-exact numbers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use thin_emu::machine::numbers as n;

/// A small format by name.
fn small(fmt: &str) -> PyResult<n::Small> {
    n::Small::named(fmt).ok_or_else(|| {
        PyValueError::new_err(format!("format {fmt:?}: e4m3, e5m2, e2m1, e2m3 or e3m2"))
    })
}

/// canon32 of f32 bits.
#[pyfunction]
fn emu_fcanon(bits: u32) -> u32 {
    n::canon32(bits)
}

/// bf16_rne of f32 bits.
#[pyfunction]
fn emu_bf16_round(bits: u32) -> u32 {
    n::bf16_rne(bits)
}

/// bf16_conv of f32 bits.
#[pyfunction]
fn emu_bf16_canon(bits: u32) -> u32 {
    n::bf16_conv(bits)
}

/// f32 bits → f16 bits (RNE).
#[pyfunction]
fn emu_f16_of_f32(bits: u32) -> u32 {
    n::f16_of_f32(bits)
}

/// f32 bits → f16 bits toward zero.
#[pyfunction]
fn emu_f16_rtz(bits: u32) -> u32 {
    n::f16_rtz(bits)
}

/// An exact value → f16 bits, one RNE rounding.
#[pyfunction]
fn emu_f16_one(x: f64) -> u32 {
    n::f16_one(x)
}

/// f16 bits → f32 bits.
#[pyfunction]
fn emu_f32_of_f16(h: u32) -> u32 {
    n::f32_of_f16(h)
}

/// A binary16 pattern's value.
#[pyfunction]
fn emu_f16_val(h: u32) -> f64 {
    n::f16_value(h)
}

/// A small format's code values, in code order.
#[pyfunction]
fn emu_code_values(fmt: &str) -> PyResult<Vec<f64>> {
    let f = small(fmt)?;
    let (mb, _, _, _, eb) = f.params();
    Ok((0..1u32 << (mb + eb + 1)).map(|c| f.value(c)).collect())
}

/// The scale exponent E of a scale operand.
#[pyfunction]
fn emu_cvt_scale_e(scale: u32) -> i32 {
    n::scale_e(scale)
}

/// down(x, scale, fmt): an exact value to a code.
#[pyfunction]
fn emu_cvt_down(x: f64, scale: u32, fmt: &str) -> PyResult<u32> {
    Ok(n::down(x, scale, small(fmt)?))
}

/// up(code, scale, fmt) and its (f32, bf16, f16) destination bits.
#[pyfunction]
fn emu_cvt_up(code: u32, scale: u32, fmt: &str) -> PyResult<(f64, u32, u32, u32)> {
    let v = n::up(code, scale, small(fmt)?);
    let (a, b, c) = n::up_bits(v);
    Ok((v, a, b, c))
}

/// f32 → i32 and f32 → u32.
#[pyfunction]
fn emu_int_of_f32(bits: u32) -> (u32, u32) {
    (n::i32_of_f32(bits), n::u32_of_f32(bits))
}

/// Adds this module's functions.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emu_fcanon, m)?)?;
    m.add_function(wrap_pyfunction!(emu_bf16_round, m)?)?;
    m.add_function(wrap_pyfunction!(emu_bf16_canon, m)?)?;
    m.add_function(wrap_pyfunction!(emu_f16_of_f32, m)?)?;
    m.add_function(wrap_pyfunction!(emu_f16_rtz, m)?)?;
    m.add_function(wrap_pyfunction!(emu_f16_one, m)?)?;
    m.add_function(wrap_pyfunction!(emu_f32_of_f16, m)?)?;
    m.add_function(wrap_pyfunction!(emu_f16_val, m)?)?;
    m.add_function(wrap_pyfunction!(emu_code_values, m)?)?;
    m.add_function(wrap_pyfunction!(emu_cvt_scale_e, m)?)?;
    m.add_function(wrap_pyfunction!(emu_cvt_down, m)?)?;
    m.add_function(wrap_pyfunction!(emu_cvt_up, m)?)?;
    m.add_function(wrap_pyfunction!(emu_int_of_f32, m)?)
}
