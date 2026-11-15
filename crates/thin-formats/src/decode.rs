//! Decode: the exact f32 value of every pattern, and the tables a fast
//! path gathers from.

use crate::dtype::{Dtype, Encoding, NanKind};
use crate::{Error, Result};

/// The quiet NaN every narrow NaN decodes to, before its sign.
const QUIET_NAN: u32 = 0x7FC0_0000;

/// The f32 bits of one pattern of a float dtype.
fn decode_one(p: u32, dtype: &Dtype, enc: &Encoding) -> u32 {
    match dtype.width {
        32 => return p,
        16 if enc.exp == 8 => return p << 16,
        _ => {}
    }
    if !enc.signed {
        return if p == enc.ones() {
            QUIET_NAN
        } else {
            f32_power(p as i32 - enc.bias())
        };
    }
    let sign = (p >> (enc.exp + enc.man) & 1) << 31;
    let magnitude = p & enc.ones();
    let exponent = magnitude >> enc.man;
    let mantissa = magnitude & ((1 << enc.man) - 1);
    let all_ones_exponent = exponent == (1 << enc.exp) - 1;
    match enc.nan {
        NanKind::Ieee if all_ones_exponent && mantissa == 0 => return sign | 0x7F80_0000,
        NanKind::Ieee if all_ones_exponent && dtype.width == 16 => {
            return sign | QUIET_NAN | mantissa << (23 - enc.man)
        }
        NanKind::Ieee if all_ones_exponent => return sign | QUIET_NAN,
        NanKind::AllOnes if magnitude == enc.ones() => return sign | QUIET_NAN,
        _ => {}
    }
    if exponent != 0 {
        let biased = (exponent as i32 - enc.bias() + 127) as u32;
        return sign | biased << 23 | mantissa << (23 - enc.man);
    }
    if mantissa == 0 {
        return sign;
    }
    // A subnormal m · 2^(1 − bias − man) is a normal f32: normalize m.
    let top = 31 - mantissa.leading_zeros();
    let biased = (top as i32 + 1 - enc.bias() - enc.man as i32 + 127) as u32;
    sign | biased << 23 | (mantissa << (23 - top)) & 0x7F_FFFF
}

/// The f32 bits of 2^e, for e ≥ −149.
fn f32_power(e: i32) -> u32 {
    if e >= -126 {
        ((e + 127) as u32) << 23
    } else {
        1 << (e + 149)
    }
}

/// Decodes a dtype's storage bytes into the f32 bits of their exact values.
pub fn decode(bits: &[u8], dtype: &Dtype, out: &mut [u32]) -> Result<()> {
    let enc = dtype.float_encoding("decode")?;
    let width = dtype.storage_bytes();
    if bits.len() != out.len() * width {
        return Err(Error::Value(format!(
            "decode: {} input bytes for {} {} elements",
            bits.len(),
            out.len(),
            dtype.name
        )));
    }
    if width == 1 {
        let limit = (1u32 << dtype.width) - 1;
        if let Some(i) = bits.iter().position(|&p| u32::from(p) > limit) {
            return Err(Error::Value(format!(
                "decode: element {i} is {}, above {limit} for the {}-bit {}",
                bits[i], dtype.width, dtype.name
            )));
        }
        let mut table = vec![0; 1 << dtype.width];
        decode_table(dtype, &mut table)?;
        table.resize(256, 0);
        crate::parallel::zip(bits, 1, out, 1, |bits, out| {
            for (&p, o) in bits.iter().zip(out.iter_mut()) {
                *o = table[usize::from(p)];
            }
        });
        return Ok(());
    }
    crate::parallel::zip(bits, width, out, 1, |bits, out| {
        for (p, o) in bits.chunks_exact(width).zip(out.iter_mut()) {
            let mut le = [0; 4];
            le[..width].copy_from_slice(p);
            *o = decode_one(u32::from_le_bytes(le), dtype, enc);
        }
    });
    Ok(())
}

/// The f32 bits of every pattern of a float dtype of at most 16 bits, in
/// pattern order: the table a fast path gathers from.
pub fn decode_table(dtype: &Dtype, out: &mut [u32]) -> Result<()> {
    let enc = dtype.float_encoding("decode_table")?;
    if dtype.width > 16 || out.len() != 1 << dtype.width {
        return Err(Error::Value(format!(
            "decode_table: {} takes 2^{} entries, not {}",
            dtype.name,
            dtype.width,
            out.len()
        )));
    }
    for (p, o) in out.iter_mut().enumerate() {
        *o = decode_one(p as u32, dtype, enc);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::dtype;

    fn one(p: u32, name: &str) -> Result<u32> {
        let d = dtype(name).unwrap();
        let bytes = p.to_le_bytes();
        let mut out = [0];
        decode(&bytes[..d.storage_bytes()], d, &mut out)?;
        Ok(out[0])
    }

    #[test]
    fn narrow_nans_drop_their_payload_and_keep_their_sign() {
        assert_eq!(one(0xFF, "e4m3fn").unwrap(), 0xFFC0_0000);
        assert_eq!(one(0x7D, "e5m2").unwrap(), 0x7FC0_0000);
        assert_eq!(one(0xFC, "e5m2").unwrap(), 0xFF80_0000);
        assert_eq!(one(0xFF, "e8m0").unwrap(), 0x7FC0_0000);
    }

    #[test]
    fn f16_nans_are_forced_quiet_and_bf16_nans_are_kept() {
        assert_eq!(one(0x7C01, "f16").unwrap(), 0x7FC0_2000);
        assert_eq!(one(0x7F81, "bf16").unwrap(), 0x7F81_0000);
    }

    #[test]
    fn subnormals_and_scales() {
        assert_eq!(one(0x01, "e4m3fn").unwrap(), 2f32.powi(-9).to_bits());
        assert_eq!(one(0x0001, "f16").unwrap(), 2f32.powi(-24).to_bits());
        assert_eq!(one(0x00, "e8m0").unwrap(), 0x0040_0000);
        assert_eq!(one(0xF, "e2m1").unwrap(), (-6f32).to_bits());
    }

    #[test]
    fn refusals() {
        assert!(matches!(one(0x10, "e2m1"), Err(Error::Value(_))));
        assert!(matches!(one(0x40, "e3m2"), Err(Error::Value(_))));
        assert!(matches!(one(1, "u8"), Err(Error::Value(_))));
    }
}
