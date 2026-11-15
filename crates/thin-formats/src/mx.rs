//! MX quantization: 32 elements share one e8m0 scale.

use rayon::prelude::*;

use crate::dtype::Dtype;
use crate::encode::{EncodeOptions, Plan};
use crate::parallel::GRAIN;
use crate::{Error, Result};

/// The number of elements that share a scale.
pub const GROUP: usize = 32;

/// The exponent of the largest finite element (emax) and the encode of an
/// MX element dtype: saturating, and for e2m1 a NaN becomes ±6.
fn element(dtype: &Dtype, what: &str) -> Result<(i32, Plan)> {
    let nan = match dtype.name {
        "e2m1" => Some(7),
        "e4m3fn" => None,
        other => {
            return Err(Error::Value(format!(
                "{what}: {other} is not an MX element dtype (e2m1 or e4m3fn)"
            )))
        }
    };
    let enc = dtype.float_encoding(what)?;
    let options = EncodeOptions {
        saturate: true,
        nan,
        inf: None,
    };
    Ok((enc.emax(), Plan::new(dtype, options)?))
}

/// The scale byte of one group maximum: the exponent of `amax` rounded up
/// from 1.75·2^e, less emax, clamped to 0..=254.
#[inline]
fn scale_byte(amax: u32, emax: i32) -> u8 {
    let exponent = (amax.wrapping_add(0x20_0000) & 0x7F80_0000) >> 23;
    (exponent as i32 - emax).clamp(0, 254) as u8
}

/// One scale byte per f32 group maximum.
pub fn scale_bytes(amax: &[u32], dtype: &Dtype, out: &mut [u8]) -> Result<()> {
    let (emax, _) = element(dtype, "scale_bytes")?;
    if out.len() != amax.len() {
        return Err(Error::Value(format!(
            "scale_bytes: {} outputs for {} maxima",
            out.len(),
            amax.len()
        )));
    }
    amax.iter()
        .zip(out.iter_mut())
        .for_each(|(&a, o)| *o = scale_byte(a, emax));
    Ok(())
}

/// Quantizes one group of 32 values.
#[inline]
fn quantize_group(group: &[u32], emax: i32, plan: &Plan, codes: &mut [u8], scale: &mut u8) {
    let amax = group.iter().map(|b| b & 0x7FFF_FFFF).max().unwrap_or(0);
    *scale = scale_byte(amax, emax);
    let factor = 2f64.powi(127 - i32::from(*scale));
    for (&b, code) in group.iter().zip(codes.iter_mut()) {
        let value = f32::from_bits(b);
        let q = if value.is_nan() {
            b
        } else {
            ((f64::from(value) * factor) as f32).to_bits()
        };
        *code = plan.encode(q) as u8;
    }
}

/// MX codes (one per byte) and scale bytes of f32 values of `shape`
/// `[..][K]`.
///
/// Each run of 32 values of a row is a group whose scale byte comes from
/// its largest magnitude (a NaN propagates), and each value is scaled by
/// 2^(127 − byte) in f64, rounded to f32 and encoded saturating. K must be
/// a multiple of 32, 0 included; any dimension may be empty. The codes
/// have the shape of x and the scales `[..][K/32]`.
pub fn quantize(
    x: &[u32],
    shape: &[usize],
    dtype: &Dtype,
    codes: &mut [u8],
    scales: &mut [u8],
) -> Result<()> {
    let (emax, plan) = element(dtype, "quantize")?;
    let Some(&k) = shape.last() else {
        return Err(Error::Value(
            "quantize: x has no dimension; MX groups run along the last".into(),
        ));
    };
    if k % GROUP != 0 {
        return Err(Error::Value(format!(
            "quantize: rows of {k} values; MX groups need a multiple of {GROUP}"
        )));
    }
    let n: Option<usize> = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if n != Some(x.len()) || codes.len() != x.len() || scales.len() != x.len() / GROUP {
        return Err(Error::Value(format!(
            "quantize: {} values, {} codes and {} scales for shape {shape:?}",
            x.len(),
            codes.len(),
            scales.len()
        )));
    }
    let run = |x: &[u32], codes: &mut [u8], scales: &mut [u8]| {
        let groups = x.chunks_exact(GROUP).zip(codes.chunks_exact_mut(GROUP));
        for ((group, codes), scale) in groups.zip(scales.iter_mut()) {
            quantize_group(group, emax, &plan, codes, scale);
        }
    };
    if x.len() <= GRAIN {
        run(x, codes, scales);
    } else {
        x.par_chunks(GRAIN)
            .zip(codes.par_chunks_mut(GRAIN))
            .zip(scales.par_chunks_mut(GRAIN / GROUP))
            .for_each(|((x, codes), scales)| run(x, codes, scales));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::dtype;

    #[test]
    fn scales_round_up_from_one_and_three_quarters() {
        let d = dtype("e4m3fn").unwrap();
        let amax = [1.0f32, 1.74, 1.75, 0.0].map(f32::to_bits);
        let mut out = [0; 4];
        scale_bytes(&amax, d, &mut out).unwrap();
        assert_eq!(out, [119, 119, 120, 0]);
    }

    #[test]
    fn a_nan_saturates_to_six_with_its_sign_in_e2m1() {
        let d = dtype("e2m1").unwrap();
        let mut x = [1f32.to_bits(); 32];
        x[3] = 0xFFC0_0000;
        let (mut codes, mut scales) = ([0; 32], [0; 1]);
        quantize(&x, &[32], d, &mut codes, &mut scales).unwrap();
        assert_eq!(codes[3], 0xF);
    }

    #[test]
    fn refusals() {
        let (mut codes, mut scales) = ([0; 48], [0; 1]);
        let x = [0; 48];
        assert!(quantize(&x, &[48], dtype("e2m1").unwrap(), &mut codes, &mut scales).is_err());
        let (mut codes, mut scales) = ([0; 32], [0; 1]);
        let x = [0; 32];
        assert!(quantize(&x, &[32], dtype("e5m2").unwrap(), &mut codes, &mut scales).is_err());
        let e2m1 = dtype("e2m1").unwrap();
        assert!(quantize(&[], &[3, 0], e2m1, &mut [], &mut []).is_ok());
        assert!(quantize(&[], &[], e2m1, &mut [], &mut []).is_err());
    }
}
