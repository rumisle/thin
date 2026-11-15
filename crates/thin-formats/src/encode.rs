//! Encode: the normative definition, the options' refusals and the table a
//! fast path applies.

use crate::dtype::{Dtype, Encoding, NanKind};
use crate::{Error, Result};

/// The options of an encode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Clamp overflow to the largest finite value.
    pub saturate: bool,
    /// The pattern a NaN becomes, for an encoding with neither NaN nor inf.
    pub nan: Option<i64>,
    /// The pattern an overflow becomes when not saturating, for such an encoding.
    pub inf: Option<i64>,
}

/// An encode with its options checked: either a copy (f32) or the
/// definition's rounding into a narrower encoding.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Plan {
    /// f32 to f32.
    Copy,
    /// Rounding into a narrower encoding.
    Round(Rounding),
}

/// The constants of the definition for one encoding and option set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rounding {
    bias: i32,
    man: u32,
    max_bits: u32,
    /// The magnitude an overflow becomes.
    overflow: u32,
    /// The pattern a NaN becomes, before its sign.
    nan: u32,
    /// Whether a NaN keeps the input's sign.
    nan_signed: bool,
    /// Whether zero keeps the input's sign (`false` for FNUZ unsigned zero).
    zero_signed: bool,
    /// The position of the sign bit, exp + man.
    sign_shift: u32,
}

impl Plan {
    /// Validate rounding and overflow options for a dtype.
    pub(crate) fn new(dtype: &Dtype, options: EncodeOptions) -> Result<Plan> {
        let enc = dtype.float_encoding("encode")?;
        let name = dtype.name;
        if !enc.signed || !enc.subnormals {
            return Err(Error::Value(format!("encode: {name} is not encoded ")));
        }
        if !enc.has_no_specials() {
            if let Some(given) = options.nan.map(|_| "nan").or(options.inf.map(|_| "inf")) {
                return Err(Error::Value(format!(
                    "encode: {given}= is only for an encoding with neither NaN nor inf, not {name}"
                )));
            }
        }
        if enc.man == 23 {
            if options.saturate {
                return Err(Error::Value(
                    "encode: f32 has no overflow to saturate (saturate=True)".into(),
                ));
            }
            return Ok(Plan::Copy);
        }
        let sign_shift = enc.exp + enc.man;
        let (overflow, nan) = if enc.has_no_specials() {
            special_patterns(name, enc, options)?
        } else {
            let sign_bit = 1 << sign_shift;
            let all_exponent = ((1 << enc.exp) - 1) << enc.man;
            let overflow = match (options.saturate, enc.nan) {
                (true, _) => enc.max_bits(),
                (false, NanKind::Ieee) => all_exponent,
                (false, NanKind::NegZero) => sign_bit,
                (false, _) => enc.ones(),
            };
            let nan = match enc.nan {
                NanKind::Ieee => all_exponent | 1 << (enc.man - 1),
                NanKind::NegZero => sign_bit,
                _ => enc.ones(),
            };
            (overflow, nan)
        };
        Ok(Plan::Round(Rounding {
            bias: enc.bias(),
            man: enc.man,
            max_bits: enc.max_bits(),
            overflow,
            nan,
            nan_signed: !matches!(enc.nan, NanKind::Ieee | NanKind::NegZero),
            zero_signed: enc.nan != NanKind::NegZero,
            sign_shift,
        }))
    }

    /// The bits of one f32 pattern.
    #[inline]
    pub(crate) fn encode(&self, b: u32) -> u32 {
        match self {
            Plan::Copy => b,
            Plan::Round(rounding) => rounding.encode(b),
        }
    }
}

/// The overflow and NaN patterns the caller gives for an encoding with
/// neither NaN nor inf, with their refusals.
fn special_patterns(name: &str, enc: &Encoding, options: EncodeOptions) -> Result<(u32, u32)> {
    let pattern = |what: &str, value: i64| {
        u32::try_from(value)
            .ok()
            .filter(|&v| v <= enc.max_bits())
            .ok_or_else(|| {
                Error::Value(format!(
                    "encode: {what}={value} is not a pattern of {name} (0..={})",
                    enc.max_bits()
                ))
            })
    };
    let nan = options
        .nan
        .ok_or_else(|| Error::Type(format!("encode: {name} has no NaN, so nan= is required")))?;
    let nan = pattern("nan", nan)?;
    let overflow = match (options.saturate, options.inf) {
        (true, None) => enc.max_bits(),
        (true, Some(_)) => {
            return Err(Error::Value(format!(
                "encode: inf= is for an overflow that does not saturate (saturate=True, {name})"
            )))
        }
        (false, Some(inf)) => pattern("inf", inf)?,
        (false, None) => {
            return Err(Error::Type(format!(
                "encode: {name} has no inf, so inf= is required when not saturating"
            )))
        }
    };
    Ok((overflow, nan))
}

impl Rounding {
    /// Round one f32 bit pattern into this element encoding.
    ///
    /// It is written without branches so that it compiles to selects: the
    /// bounds preserve the same rounding result. The shift is
    /// capped at 25 instead of 40: `full` < 2^24, so any shift of 25 or more
    /// leaves a zero base and a remainder below half, as 40 does. And every
    /// quantity fits 32 bits, since `man` ≤ 10 and `te` ≤ 254 here.
    #[inline]
    fn encode(&self, b: u32) -> u32 {
        let sign = (b >> 31) << self.sign_shift;
        let a = b & 0x7FFF_FFFF;
        let e32 = a >> 23;
        let full = (a & 0x7F_FFFF) | u32::from(e32 != 0) << 23;
        let te = e32.max(1) as i32 - 127 + self.bias;
        let sh = (23 - self.man as i32 + (1 - te).max(0)).min(25) as u32;
        let base = (((te - 1).max(0) as u32) << self.man) + (full >> sh);
        let rem = full & ((1 << sh) - 1);
        let half = 1 << (sh - 1);
        let bits = base + u32::from(rem > half || (rem == half && base & 1 == 1));
        let overflow = bits > self.max_bits || a == 0x7F80_0000;
        let mag = if overflow { self.overflow } else { bits };
        let finite = mag
            | if mag != 0 || self.zero_signed {
                sign
            } else {
                0
            };
        let nan = self.nan | if self.nan_signed { sign } else { 0 };
        if a > 0x7F80_0000 {
            nan
        } else {
            finite
        }
    }
}

/// Check encoding options before constructing a
/// path that applies a table or a native cast instead of [`encode`].
pub fn check_encode(dtype: &Dtype, options: EncodeOptions) -> Result<()> {
    Plan::new(dtype, options).map(drop)
}

/// Encodes f32 bit patterns into a dtype's storage, written little-endian
/// into `out` (one byte per element of 8 bits or fewer).
pub fn encode(x: &[u32], dtype: &Dtype, options: EncodeOptions, out: &mut [u8]) -> Result<()> {
    let plan = Plan::new(dtype, options)?;
    let width = dtype.storage_bytes();
    if out.len() != x.len() * width {
        return Err(Error::Value(format!(
            "encode: {} output bytes for {} {} elements",
            out.len(),
            x.len(),
            dtype.name
        )));
    }
    crate::parallel::zip(x, 1, out, width, |x, out| match width {
        1 => x
            .iter()
            .zip(out.iter_mut())
            .for_each(|(&b, o)| *o = plan.encode(b) as u8),
        2 => x
            .iter()
            .zip(out.chunks_exact_mut(2))
            .for_each(|(&b, o)| o.copy_from_slice(&(plan.encode(b) as u16).to_le_bytes())),
        _ => x
            .iter()
            .zip(out.chunks_exact_mut(4))
            .for_each(|(&b, o)| o.copy_from_slice(&plan.encode(b).to_le_bytes())),
    });
    Ok(())
}

/// The number of index bits of an [`encode_table`]: the top 16 bits of an
/// f32 and a sticky bit for the low 16.
pub const TABLE_INDEX_BITS: u32 = 17;

/// The table a fast path applies to encode into a dtype of at most 6
/// mantissa bits. Entry `(b >> 16) << 1 | (b & 0xFFFF != 0)` is
/// the encode of `b`, in the dtype's storage bytes.
///
/// Why the top 16 bits and a sticky bit decide the result: with `man` ≤ 6
/// the shift is at least 23 − 6 = 17, so the half the remainder is
/// compared with is 2^(sh−1) ≥ 2^16 and has no low 16 bits. The remainder
/// is above, at or below half according to its bits above 16 and whether
/// any low bit is set; the base, the NaN test (`a > 0x7F800000`) and the
/// overflow test read nothing else. Each entry is the definition applied
/// to one input of its class, so a path that gathers from the table is
/// the definition once its index is right.
pub fn encode_table(dtype: &Dtype, options: EncodeOptions, out: &mut [u8]) -> Result<()> {
    let enc = dtype.float_encoding("encode_table")?;
    if enc.man > 6 {
        return Err(Error::Value(format!(
            "encode_table: {} has {} mantissa bits; a table serves 6 or fewer",
            dtype.name, enc.man
        )));
    }
    let inputs: Vec<u32> = (0..1u32 << TABLE_INDEX_BITS)
        .map(|i| (i >> 1) << 16 | (i & 1))
        .collect();
    encode(&inputs, dtype, options, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::dtype;

    fn one(x: f32, name: &str, options: EncodeOptions) -> Result<u32> {
        let d = dtype(name).unwrap();
        let mut out = vec![0; d.storage_bytes()];
        encode(&[x.to_bits()], d, options, &mut out)?;
        let mut le = [0; 4];
        le[..out.len()].copy_from_slice(&out);
        Ok(u32::from_le_bytes(le))
    }

    const SAT: EncodeOptions = EncodeOptions {
        saturate: true,
        nan: None,
        inf: None,
    };
    const NO_SAT: EncodeOptions = EncodeOptions {
        saturate: false,
        nan: None,
        inf: None,
    };

    #[test]
    fn negatives_that_round_to_zero_keep_their_sign() {
        assert_eq!(one(-0.0, "e4m3fn", SAT).unwrap(), 0x80);
        assert_eq!(one(-1e-30, "e4m3fn", SAT).unwrap(), 0x80);
        assert_eq!(one(-1e-45, "bf16", NO_SAT).unwrap(), 0x8000);
        assert_eq!(one(-0.0, "e4m3fnuz", SAT).unwrap(), 0x00);
        assert_eq!(one(-0.0, "e4m3fnuz", NO_SAT).unwrap(), 0x00);
        assert_eq!(one(-1e-30, "e4m3fnuz", SAT).unwrap(), 0x00);
        assert_eq!(one(-0.0, "e5m2fnuz", NO_SAT).unwrap(), 0x00);
        assert_eq!(one(-1e-30, "e5m2fnuz", NO_SAT).unwrap(), 0x00);
    }

    #[test]
    fn fnuz_overflow_and_nan() {
        let nan = f32::from_bits(0xFFC0_0001);
        assert_eq!(one(240.0, "e4m3fnuz", NO_SAT).unwrap(), 0x7F);
        assert_eq!(one(-240.0, "e4m3fnuz", NO_SAT).unwrap(), 0xFF);
        assert_eq!(one(247.0, "e4m3fnuz", NO_SAT).unwrap(), 0x7F);
        assert_eq!(one(248.0, "e4m3fnuz", NO_SAT).unwrap(), 0x80);
        assert_eq!(one(-248.0, "e4m3fnuz", NO_SAT).unwrap(), 0x80);
        assert_eq!(one(248.0, "e4m3fnuz", SAT).unwrap(), 0x7F);
        assert_eq!(one(-248.0, "e4m3fnuz", SAT).unwrap(), 0xFF);
        assert_eq!(one(f32::INFINITY, "e4m3fnuz", SAT).unwrap(), 0x7F);
        assert_eq!(one(f32::NEG_INFINITY, "e4m3fnuz", SAT).unwrap(), 0xFF);
        assert_eq!(one(f32::INFINITY, "e4m3fnuz", NO_SAT).unwrap(), 0x80);
        assert_eq!(one(f32::NEG_INFINITY, "e4m3fnuz", NO_SAT).unwrap(), 0x80);
        assert_eq!(one(nan, "e4m3fnuz", SAT).unwrap(), 0x80);
        assert_eq!(one(nan, "e4m3fnuz", NO_SAT).unwrap(), 0x80);
        assert_eq!(one(57344.0, "e5m2fnuz", NO_SAT).unwrap(), 0x7F);
        assert_eq!(one(-57344.0, "e5m2fnuz", NO_SAT).unwrap(), 0xFF);
        assert_eq!(one(61440.0, "e5m2fnuz", NO_SAT).unwrap(), 0x80);
        assert_eq!(one(-61440.0, "e5m2fnuz", SAT).unwrap(), 0xFF);
        assert_eq!(one(nan, "e5m2fnuz", NO_SAT).unwrap(), 0x80);
    }

    #[test]
    fn overflow_follows_the_nan_kind() {
        assert_eq!(one(1e6, "e4m3fn", SAT).unwrap(), 0x7E);
        assert_eq!(one(1e6, "e4m3fn", NO_SAT).unwrap(), 0x7F);
        assert_eq!(one(-1e6, "e5m2", NO_SAT).unwrap(), 0xFC);
        assert_eq!(one(f32::INFINITY, "f16", SAT).unwrap(), 0x7BFF);
        let caller = EncodeOptions {
            saturate: false,
            nan: Some(0),
            inf: Some(5),
        };
        assert_eq!(one(-7.0, "e2m1", caller).unwrap(), 0xD);
    }

    #[test]
    fn nan_is_quiet_and_unsigned_only_for_ieee() {
        let nan = f32::from_bits(0xFFC0_0001);
        assert_eq!(one(nan, "f16", NO_SAT).unwrap(), 0x7E00);
        assert_eq!(one(nan, "e4m3fn", SAT).unwrap(), 0xFF);
        let caller = EncodeOptions {
            saturate: true,
            nan: Some(7),
            inf: None,
        };
        assert_eq!(one(nan, "e2m1", caller).unwrap(), 0xF);
    }

    #[test]
    fn ties_round_to_even_and_carry_into_the_exponent() {
        assert_eq!(one(1.0625, "e4m3fn", SAT).unwrap(), 0x38);
        assert_eq!(one(1.1875, "e4m3fn", SAT).unwrap(), 0x3A);
        assert_eq!(one(1.9375, "e4m3fn", SAT).unwrap(), 0x40);
    }

    #[test]
    fn f32_copies_and_refuses_saturate() {
        let nan = f32::from_bits(0x7F80_0001);
        assert_eq!(one(nan, "f32", NO_SAT).unwrap(), 0x7F80_0001);
        assert!(matches!(one(1.0, "f32", SAT), Err(Error::Value(_))));
    }

    #[test]
    fn option_refusals() {
        let with = |saturate, nan, inf| EncodeOptions { saturate, nan, inf };
        let class = |r: Result<u32>| match r {
            Err(Error::Type(_)) => "TypeError",
            Err(Error::Value(_)) => "ValueError",
            Err(_) => "other",
            Ok(_) => "ok",
        };
        assert_eq!(class(one(1.0, "u8", SAT)), "ValueError");
        assert_eq!(class(one(1.0, "e8m0", SAT)), "ValueError");
        assert_eq!(
            class(one(1.0, "e4m3fn", with(true, Some(0), None))),
            "ValueError"
        );
        assert_eq!(
            class(one(1.0, "bf16", with(false, None, Some(0)))),
            "ValueError"
        );
        assert_eq!(class(one(1.0, "e2m1", SAT)), "TypeError");
        assert_eq!(
            class(one(1.0, "e2m1", with(false, Some(0), None))),
            "TypeError"
        );
        assert_eq!(
            class(one(1.0, "e2m1", with(true, Some(0), Some(7)))),
            "ValueError"
        );
        assert_eq!(
            class(one(1.0, "e2m1", with(true, Some(8), None))),
            "ValueError"
        );
        assert_eq!(
            class(one(1.0, "e2m1", with(true, Some(-1), None))),
            "ValueError"
        );
        assert_eq!(
            class(one(1.0, "e2m1", with(false, Some(0), Some(8)))),
            "ValueError"
        );
        assert_eq!(class(one(1.0, "e2m1", with(false, Some(7), Some(7)))), "ok");
    }

    #[test]
    fn the_table_matches_the_definition_on_its_own_inputs() {
        let d = dtype("e5m2").unwrap();
        let mut table = vec![0; 1 << TABLE_INDEX_BITS];
        encode_table(d, NO_SAT, &mut table).unwrap();
        for b in [
            0x3F80_0000u32,
            0x3F90_0001,
            0xC77F_F000,
            0x7F80_0001,
            0x0000_0001,
        ] {
            let index = ((b >> 16) << 1 | u32::from(b & 0xFFFF != 0)) as usize;
            assert_eq!(
                u32::from(table[index]),
                one(f32::from_bits(b), "e5m2", NO_SAT).unwrap()
            );
        }
        assert!(encode_table(dtype("bf16").unwrap(), NO_SAT, &mut table).is_err());
    }
}
