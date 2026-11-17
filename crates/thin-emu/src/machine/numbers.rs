//! The bit-exact numbers: canonical NaNs, conversions, the small
//! formats and the transcendentals.

/// The canonical f32 NaN.
pub const QNAN: u32 = 0x7FC0_0000;

/// Any NaN becomes the canonical NaN; other bits are kept.
#[inline]
pub fn canon32(u: u32) -> u32 {
    if f32::from_bits(u).is_nan() {
        QNAN
    } else {
        u
    }
}

/// The canonical f64 NaN.
pub const QNAN64: u64 = 0x7FF8_0000_0000_0000;

/// Any binary64 NaN becomes the canonical NaN; other bits are unchanged.
#[inline]
pub fn canon64(u: u64) -> u64 {
    if f64::from_bits(u).is_nan() {
        QNAN64
    } else {
        u
    }
}

/// An f32 result, canonicalised.
#[inline]
pub fn fbits(x: f32) -> u32 {
    canon32(x.to_bits())
}

/// The f32 view of bits.
#[inline]
pub fn f32_of(u: u32) -> f32 {
    f32::from_bits(u)
}

/// gfx1201's default NaN of an invalid f32 operation.
pub const DNAN32: u32 = 0xFFC0_0000;
/// gfx1201's default NaN of an invalid f16 operation.
pub const DNAN16: u32 = 0xFE00;
/// gfx1201's default NaN of an invalid f64 operation.
pub const DNAN64: u64 = 0xFFF8_0000_0000_0000;

/// Whether f32 bits are a NaN.
#[inline]
pub fn is_nan32(u: u32) -> bool {
    u & 0x7FFF_FFFF > 0x7F80_0000
}

/// Whether f16 bits (the low half) are a NaN.
#[inline]
pub fn is_nan16(h: u32) -> bool {
    h & 0x7FFF > 0x7C00
}

/// Whether f64 bits are a NaN.
#[inline]
pub fn is_nan64(u: u64) -> bool {
    u & !(1 << 63) > 0x7FF0_0000_0000_0000
}

/// An f32 result whose IEEE value is `r`, of operands `ops` in the
/// operation's order (negated where it negates one, as v_sub's
/// subtrahend). Not a NaN: its bits. A NaN: the first NaN operand quieted
/// (bit 22 set), payload and sign kept, or, with none (an invalid operation
/// such as ∞ − ∞, 0 · ∞, √−1 or the log of a negative), the default NaN
/// 0xFFC00000. RDNA4 ISA guide ("All operands that can produce
/// exceptions quiet SNaNs"); the payload, order and default NaN measured on
/// gfx1201 (tests/hw/test_emu_ops.py, `test_nans_and_specials_are_the_gpus`).
pub fn nan32(r: f32, ops: &[u32]) -> u32 {
    if !r.is_nan() {
        return r.to_bits();
    }
    ops.iter()
        .find(|u| is_nan32(**u))
        .map_or(DNAN32, |u| u | 0x0040_0000)
}

/// An f16 result's bits `r` (the low half), of operands `ops`: as
/// [`nan32`] at binary16 (quiet bit 9, default NaN 0xFE00).
pub fn nan16(r: u32, ops: &[u32]) -> u32 {
    if !is_nan16(r) {
        return r;
    }
    ops.iter()
        .map(|u| u & 0xFFFF)
        .find(|u| is_nan16(*u))
        .map_or(DNAN16, |u| u | 0x200)
}

/// An f64 result: as [`nan32`] at binary64 (quiet bit 51, default NaN
/// 0xFFF8000000000000).
pub fn nan64(r: f64, ops: &[u64]) -> u64 {
    if !r.is_nan() {
        return r.to_bits();
    }
    ops.iter()
        .find(|u| is_nan64(**u))
        .map_or(DNAN64, |u| u | 1 << 51)
}

/// The bits of fma(x, y, z) whose IEEE value is `r`. A NaN is the
/// product's first (x, then y; an invalid 0 · ∞ the default NaN), then
/// z's (measured on gfx1201: the product is formed before the addend is
/// looked at).
pub fn fma_nan32(r: f32, x: u32, y: u32, z: u32) -> u32 {
    if !r.is_nan() {
        return r.to_bits();
    }
    let p = f32_of(x) * f32_of(y);
    if p.is_nan() {
        return nan32(p, &[x, y]);
    }
    nan32(r, &[z])
}

/// [`fma_nan32`] at binary64.
pub fn fma_nan64(r: f64, x: u64, y: u64, z: u64) -> u64 {
    if !r.is_nan() {
        return r.to_bits();
    }
    let p = f64::from_bits(x) * f64::from_bits(y);
    if p.is_nan() {
        return nan64(p, &[x, y]);
    }
    nan64(r, &[z])
}

/// bf16_rne of f32 bits: a NaN may become an infinity.
pub fn bf16_rne(u: u32) -> u32 {
    (u.wrapping_add(0x7FFF).wrapping_add((u >> 16) & 1) >> 16) & 0xFFFF
}

/// bf16_conv: bf16_rne, but a NaN gives 0x7FC0.
pub fn bf16_conv(u: u32) -> u32 {
    if f32_of(u).is_nan() {
        0x7FC0
    } else {
        bf16_rne(u)
    }
}

/// One RNE rounding of an exact value to binary16, subnormals and infinity
/// past the range; NaN → 0x7E00 (`f16_one`).
pub fn f16_one(x: f64) -> u32 {
    if x.is_nan() {
        return 0x7E00;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_infinite() {
        return sign | 0x7C00;
    }
    if a == 0.0 {
        return sign;
    }
    let e = floor_log2(a).max(-14);
    let step = 2f64.powi(e - 10);
    let r = (a / step).round_ties_even() * step;
    if r > 65504.0 {
        return sign | 0x7C00;
    }
    if r < 2f64.powi(-14) {
        return sign | (r / 2f64.powi(-24)) as u32;
    }
    let ex = floor_log2(r);
    let man = ((r / 2f64.powi(ex) - 1.0) * 1024.0) as u32;
    sign | (((ex + 15) as u32) << 10) | man
}

/// f32 → f16.
pub fn f16_of_f32(u: u32) -> u32 {
    let x = f32_of(u);
    if x.is_nan() {
        return ((u >> 16) & 0x8000) | 0x7E00 | ((u >> 13) & 0x3FF);
    }
    f16_one(f64::from(x))
}

/// f32 → f16 toward zero.
pub fn f16_rtz(u: u32) -> u32 {
    let x = f32_of(u);
    let h = f16_of_f32(u);
    if x.is_finite() && f16_value(h).abs() > f64::from(x).abs() {
        return h - 1;
    }
    h
}

/// f16 → f32, exact; a NaN keeps sign and payload, quieted.
pub fn f32_of_f16(h: u32) -> u32 {
    let h = h & 0xFFFF;
    let (exp, man) = ((h >> 10) & 0x1F, h & 0x3FF);
    if exp == 0x1F && man != 0 {
        return ((h & 0x8000) << 16) | QNAN | (man << 13);
    }
    let v = f16_value(h);
    (v as f32).to_bits()
}

/// A binary16 pattern's exact value.
pub fn f16_value(h: u32) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (exp, man) = ((h >> 10) & 0x1F, f64::from(h & 0x3FF));
    match exp {
        0 => sign * man * 2f64.powi(-24),
        0x1F if man == 0.0 => sign * f64::INFINITY,
        0x1F => f64::NAN,
        _ => sign * (1.0 + man / 1024.0) * 2f64.powi(exp as i32 - 15),
    }
}

/// f32 → i32: NaN → 0, truncate, clamp.
pub fn i32_of_f32(u: u32) -> u32 {
    let x = f32_of(u);
    if x.is_nan() {
        return 0;
    }
    (x.trunc() as f64).clamp(-2147483648.0, 2147483647.0) as i32 as u32
}

/// f32 → u32: NaN → 0, truncate, clamp.
pub fn u32_of_f32(u: u32) -> u32 {
    let x = f32_of(u);
    if x.is_nan() {
        return 0;
    }
    (x.trunc() as f64).clamp(0.0, 4294967295.0) as u32
}

/// A small element format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Small {
    /// e4m3: no infinity, NaN 0x7F/0xFF.
    E4M3,
    /// e5m2: IEEE-like.
    E5M2,
    /// e4m3fnuz: CDNA3 FNUZ (bias 8, max 240, 0x00 zero, 0x80 NaN, no infinity).
    E4M3Fnuz,
    /// e5m2fnuz: CDNA3 FNUZ (bias 16, max 57344, 0x00 zero, 0x80 NaN, no infinity).
    E5M2Fnuz,
    /// e2m1: no NaN, no infinity.
    E2M1,
    /// e2m3.
    E2M3,
    /// e3m2.
    E3M2,
}

impl Small {
    /// The format of a name (`e4m3`, `e4m3fn`, `e5m2`, `e4m3fnuz`, `e5m2fnuz`, `e2m1`, `e2m3`, `e3m2`).
    pub fn named(s: &str) -> Option<Small> {
        Some(match s {
            "e4m3" | "e4m3fn" => Small::E4M3,
            "e5m2" => Small::E5M2,
            "e4m3fnuz" => Small::E4M3Fnuz,
            "e5m2fnuz" => Small::E5M2Fnuz,
            "e2m1" => Small::E2M1,
            "e2m3" => Small::E2M3,
            "e3m2" => Small::E3M2,
            _ => return None,
        })
    }

    /// (mantissa bits, least normal exponent, bias, largest finite, exponent bits).
    pub fn params(self) -> (i32, i32, i32, f64, i32) {
        match self {
            Small::E4M3 => (3, -6, 7, 448.0, 4),
            Small::E5M2 => (2, -14, 15, 57344.0, 5),
            Small::E4M3Fnuz => (3, -7, 8, 240.0, 4),
            Small::E5M2Fnuz => (2, -15, 16, 57344.0, 5),
            Small::E2M1 => (1, 0, 1, 6.0, 2),
            Small::E2M3 => (3, 0, 1, 7.5, 2),
            Small::E3M2 => (2, -2, 3, 28.0, 3),
        }
    }

    /// The code's value (−0 kept; NaN with its sign as NaN).
    pub fn value(self, code: u32) -> f64 {
        let (mb, emin, bias, _, eb) = self.params();
        let sign_bit = 1 << (mb + eb);
        let code = code & ((sign_bit << 1) - 1);
        let sign = if code & sign_bit != 0 { -1.0 } else { 1.0 };
        let mag = code & (sign_bit - 1);
        let (exp, man) = ((mag >> mb) as i32, f64::from(mag & ((1 << mb) - 1)));
        match self {
            Small::E4M3 if mag == 0x7F => return f64::NAN,
            Small::E4M3Fnuz | Small::E5M2Fnuz if code == 0x80 => return f64::NAN,
            Small::E5M2 if exp == 31 => {
                return if man == 0.0 {
                    sign * f64::INFINITY
                } else {
                    f64::NAN
                };
            }
            _ => {}
        }
        let scale = 2f64.powi(mb);
        if exp == 0 {
            sign * man / scale * 2f64.powi(emin)
        } else {
            sign * (1.0 + man / scale) * 2f64.powi(exp - bias)
        }
    }

    fn largest(self) -> u32 {
        let (mb, _, _, _, eb) = self.params();
        (1 << (mb + eb)) - 1
    }
}

/// The scale exponent E of a scale operand.
pub fn scale_e(scale: u32) -> i32 {
    ((scale >> 23) & 0xFF) as i32
}

/// ⌊log2 a⌋ of a finite positive value.
pub fn floor_log2(a: f64) -> i32 {
    let bits = a.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    if exp == 0 {
        // subnormal f64
        let m = bits & ((1 << 52) - 1);
        return -1074 + (63 - m.leading_zeros() as i32);
    }
    exp - 1023
}

/// `down(x, scale, fmt)`: an exact value to a small format's code.
pub fn down(x: f64, scale: u32, fmt: Small) -> u32 {
    let (mb, emin, bias, max, eb) = fmt.params();
    let e = scale_e(scale);
    let fnuz = matches!(fmt, Small::E4M3Fnuz | Small::E5M2Fnuz);
    let sign = if x.is_sign_negative() {
        1 << (mb + eb)
    } else {
        0
    };
    if x.is_nan() || e == 255 {
        return match fmt {
            Small::E4M3 => 0xFF,
            Small::E5M2 => 0xFE,
            Small::E4M3Fnuz | Small::E5M2Fnuz => 0x80,
            _ => sign | fmt.largest(),
        };
    }
    let v = x / 2f64.powi(e - 127);
    let a = v.abs();
    let r = if a.is_infinite() {
        f64::INFINITY
    } else {
        let lead = floor_log2(a.max(2f64.powi(emin - mb - 1)));
        let step = 2f64.powi(lead.max(emin) - mb);
        (a / step).round_ties_even() * step
    };
    if r > max {
        return match fmt {
            Small::E4M3 => 0x7F | sign,
            Small::E5M2 => 0x7C | sign,
            Small::E4M3Fnuz | Small::E5M2Fnuz => 0x80,
            _ => fmt.largest() | sign,
        };
    }
    let code = if r < 2f64.powi(emin) {
        (r / 2f64.powi(emin - mb)) as u32
    } else {
        let ex = floor_log2(r);
        let man = ((r / 2f64.powi(ex) - 1.0) * 2f64.powi(mb)) as u32;
        (((ex + bias) as u32) << mb) | man
    };
    if code == 0 && fnuz {
        0
    } else {
        code | sign
    }
}

/// The exact value of a code times its scale, NaN as NaN.
pub fn up(code: u32, scale: u32, fmt: Small) -> f64 {
    let e = scale_e(scale);
    let v = fmt.value(code);
    if e == 255 || v.is_nan() {
        return f64::NAN;
    }
    v * 2f64.powi(e - 127)
}

/// Destination bits of a converted value: (f32, bf16, f16).
pub fn up_bits(v: f64) -> (u32, u32, u32) {
    if v.is_nan() {
        return (0xFFC0_0000, 0xFFC0, 0xFE00);
    }
    let f = (v as f32).to_bits();
    (f, bf16_conv(f), f16_one(v))
}
