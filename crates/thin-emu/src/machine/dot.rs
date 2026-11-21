//! gfx1201's float dot unit, bit for bit: one step of two (v_dot2_f32_bf16),
//! four (a WMMA with f16 or bf16 inputs) or eight (fp8 or bf8 inputs)
//! products added to an accumulator in D's format.
//!
//! The algorithm is AMD rocjitsu's (ROCm/rocm-systems, emulation/rocjitsu,
//! MIT; see NOTICE), extended to every gfx1201 WMMA by measurement. Products
//! are exact. The products come in two groups (pairs for 16-bit inputs,
//! quads for 8-bit ones); each group is aligned to its grid, 2^(e_a + e_b −
//! 24) for its largest alignment-exponent sum (a denormal input's exponent
//! is its format's smallest), by floor; the group sums and the accumulator
//! are aligned to the step's grid, the larger of the products' grid and
//! 2^(e_c − 26) (e_c the accumulator's leading exponent, at least D's
//! smallest normal one), again by floor; a negative accumulator with no bit
//! left contributes nothing while the grid is above D's smallest normal
//! exponent, and the floor's -1 unit at or below it. The total is rounded once to D
//! (RNE; a bf16 D is the binary32 rounding's high half), a zero +0.
//! Measured on gfx1201: 32 768 random WMMA elements of f32_16x16x16_f16 and
//! _bf16, v_dot2_f32_bf16's random, cancelling and accumulator-dominated sets
//! (tests/hw/test_emu_ops.py, tests/hw/test_wmma.py), and over 10^5 random
//! and edge-case elements of every dense gfx1201 WMMA form.

/// No exponent: a zero significand.
const EMPTY: i32 = -1024;

/// A float format the dot unit reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    /// Total width in bits.
    bits: u32,
    /// Exponent bits.
    eb: u32,
    /// Fraction bits.
    fb: u32,
    /// An all-ones exponent is ∞ / NaN (else only the all-ones code is NaN:
    /// OCP e4m3fn).
    ieee: bool,
    /// The NaN for a NaN factor (D formats only).
    factor_nan: u32,
}

/// binary32.
pub const F32: Format = Format {
    bits: 32,
    eb: 8,
    fb: 23,
    ieee: true,
    factor_nan: 0xFFC0_0A3D,
};
/// binary16.
pub const F16: Format = Format {
    bits: 16,
    eb: 5,
    fb: 10,
    ieee: true,
    factor_nan: 0xFE3D,
};
/// bfloat16.
pub const BF16: Format = Format {
    bits: 16,
    eb: 8,
    fb: 7,
    ieee: true,
    factor_nan: 0xFFFD,
};
/// OCP fp8 e4m3fn: no ∞, ±0x7F the NaNs.
pub const FP8: Format = Format {
    bits: 8,
    eb: 4,
    fb: 3,
    ieee: false,
    factor_nan: 0,
};
/// OCP bf8 e5m2.
pub const BF8: Format = Format {
    bits: 8,
    eb: 5,
    fb: 2,
    ieee: true,
    factor_nan: 0,
};

/// A decoded input: NaN, ±∞, or a finite (-1)^neg · sig · 2^(align − fb).
enum Value {
    Nan,
    Inf(bool),
    Finite { sig: u64, align: i32, neg: bool },
}

impl Format {
    fn sign(self) -> u32 {
        1 << (self.bits - 1)
    }
    fn exp_max(self) -> u32 {
        (1 << self.eb) - 1
    }
    fn bias(self) -> i32 {
        (1 << (self.eb - 1)) - 1
    }
    /// The smallest normal exponent.
    pub fn emin(self) -> i32 {
        1 - self.bias()
    }
    /// The quiet bit, the top fraction bit.
    fn quiet(self) -> u32 {
        1 << (self.fb - 1)
    }
    /// The default NaN.
    fn default_nan(self) -> u32 {
        self.sign() | self.exp_max() << self.fb | self.quiet()
    }
    fn decode(self, bits: u32) -> Value {
        let bits = bits & (u32::MAX >> (32 - self.bits));
        let neg = bits & self.sign() != 0;
        let field = (bits >> self.fb) & self.exp_max();
        let frac = bits & ((1 << self.fb) - 1);
        if field == self.exp_max() {
            if self.ieee {
                return if frac == 0 {
                    Value::Inf(neg)
                } else {
                    Value::Nan
                };
            }
            if frac == (1 << self.fb) - 1 {
                return Value::Nan;
            }
        }
        let sig = u64::from(if field != 0 {
            1 << self.fb | frac
        } else {
            frac
        });
        Value::Finite {
            sig,
            align: field.max(1) as i32 - self.bias(),
            neg,
        }
    }
}

/// A signed significand times 2^exponent.
#[derive(Clone, Copy)]
struct Term {
    sig: u64,
    exp: i32,
    neg: bool,
}

impl Term {
    fn leading(self) -> i32 {
        if self.sig == 0 {
            EMPTY
        } else {
            self.exp + 63 - self.sig.leading_zeros() as i32
        }
    }
}

/// `value >> shift` rounded half to even (a left shift for shift ≤ 0).
fn shift_rne(value: u64, shift: i32) -> u64 {
    if shift <= 0 {
        return value << -shift;
    }
    if shift >= 64 {
        return 0;
    }
    let tail = value & ((1 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let whole = value >> shift;
    whole + u64::from(tail > half || (tail == half && whole & 1 == 1))
}

/// `units` · 2^grid rounded once to `d` (RNE, subnormals kept, overflow ∞),
/// a zero +0; a bf16 D the binary32 rounding's high half.
fn pack(units: i64, grid: i32, d: Format) -> u32 {
    if d == BF16 {
        return pack(units, grid, F32) >> 16;
    }
    if units == 0 {
        return 0;
    }
    let sign = if units < 0 { d.sign() } else { 0 };
    let (fb, emin) = (d.fb as i32, d.emin());
    let mag = units.unsigned_abs();
    let top = 63 - mag.leading_zeros() as i32;
    let mut e = top + grid;
    if e < emin {
        // Denormal; a carry into the exponent field is the smallest normal, and
        // a result that rounds to zero is +0.
        let f = shift_rne(mag, emin - fb - grid) as u32;
        return if f == 0 { 0 } else { sign | f };
    }
    let mut s = shift_rne(mag, top - fb);
    if s == 1 << (fb + 1) {
        s >>= 1;
        e += 1;
    }
    if e > d.exp_max() as i32 - 1 - d.bias() {
        return sign | d.exp_max() << d.fb;
    }
    sign | ((e + d.bias()) as u32) << d.fb | (s as u32 & ((1 << fb) - 1))
}

/// `t` on the grid 2^grid by floor; with `empty_is_zero`, a negative `t` with
/// no bit left is 0 instead of -1.
fn align(t: Term, grid: i32, empty_is_zero: bool) -> i64 {
    let shift = t.exp - grid;
    let mag = if t.sig == 0 || shift <= -64 {
        0
    } else if shift < 0 {
        t.sig >> -shift
    } else {
        t.sig << shift
    };
    let tail = t.sig != 0 && shift < 0 && (shift <= -64 || t.sig & ((1 << -shift) - 1) != 0);
    if t.neg {
        -(mag as i64) - i64::from(tail && (!empty_is_zero || mag != 0))
    } else {
        mag as i64
    }
}

/// One step's formats: A's, B's and D's (the accumulator's), and the
/// products per group (2 for 16-bit inputs, 4 for 8-bit ones).
#[derive(Clone, Copy, Debug)]
pub struct Step {
    /// A's format.
    pub a: Format,
    /// B's format.
    pub b: Format,
    /// D's and the accumulator's format.
    pub d: Format,
    /// Products per group.
    pub group: usize,
}

impl Step {
    /// The form of a gfx1201 dot step for these inputs and D.
    pub fn new(a: Format, b: Format, d: Format) -> Step {
        let group = if a.bits == 8 { 4 } else { 2 };
        Step { a, b, d, group }
    }
}

/// acc + Σ a[i]·b[i] (two groups of `form.group` products, or one), all
/// raw bits, as D's bits. NaNs: a NaN factor gives D's factor NaN; an
/// invalid product or ∞ − ∞ the default NaN (the factor NaN for 8-bit
/// inputs); else the accumulator's NaN quieted.
pub fn step(a: &[u32], b: &[u32], acc: u32, form: Step) -> u32 {
    debug_assert!(a.len() == b.len() && a.len() % form.group == 0 && a.len() <= 8);
    let d = form.d;
    let invalid_nan = if form.a.bits == 8 {
        d.factor_nan
    } else {
        d.default_nan()
    };
    let (mut pinf, mut ninf, mut invalid) = (false, false, false);
    let mut prods = [Term {
        sig: 0,
        exp: 0,
        neg: false,
    }; 8];
    let mut grids = [EMPTY; 8];
    for i in 0..a.len() {
        let (l, r) = (form.a.decode(a[i]), form.b.decode(b[i]));
        match (l, r) {
            (Value::Nan, _) | (_, Value::Nan) => return d.factor_nan,
            (Value::Inf(x), Value::Inf(y)) => {
                (pinf, ninf) = (pinf || x == y, ninf || x != y);
            }
            (Value::Inf(x), Value::Finite { sig, neg, .. })
            | (Value::Finite { sig, neg, .. }, Value::Inf(x)) => {
                invalid |= sig == 0;
                (pinf, ninf) = (pinf || x == neg, ninf || x != neg);
            }
            (
                Value::Finite {
                    sig: ls,
                    align: la,
                    neg: ln,
                },
                Value::Finite {
                    sig: rs,
                    align: ra,
                    neg: rn,
                },
            ) => {
                prods[i] = Term {
                    sig: ls * rs,
                    exp: la + ra - (form.a.fb + form.b.fb) as i32,
                    neg: ln != rn,
                };
                if prods[i].sig != 0 {
                    grids[i] = la + ra - 24;
                }
            }
        }
    }
    if invalid || (pinf && ninf) {
        return invalid_nan;
    }
    let c = match d.decode(acc) {
        Value::Nan => return acc & (u32::MAX >> (32 - d.bits)) | d.quiet(),
        Value::Inf(neg) => {
            (pinf, ninf) = (pinf || !neg, ninf || neg);
            Term {
                sig: 0,
                exp: 0,
                neg: false,
            }
        }
        Value::Finite { sig, align, neg } => Term {
            sig,
            exp: align - d.fb as i32,
            neg,
        },
    };
    if pinf || ninf {
        return match (pinf, ninf) {
            (true, true) => invalid_nan,
            (false, true) => d.sign() | d.exp_max() << d.fb,
            _ => d.exp_max() << d.fb,
        };
    }
    let products = grids.iter().copied().max().unwrap_or(EMPTY);
    let grid = products.max(c.leading().max(d.emin()) - 26);
    // A negative accumulator below one grid unit adds nothing above D's emin
    // and the floor's -1 at or below it (measured: f16 D grids -14 / -13;
    // bf16 products at -126 / -125).
    let mut total = align(c, grid, grid > d.emin());
    for g in (0..a.len()).step_by(form.group) {
        let gg = form.group;
        let pg = grids[g..g + gg].iter().copied().max().unwrap_or(EMPTY);
        let sum: i64 = (g..g + gg).map(|i| align(prods[i], pg, false)).sum();
        let sum = Term {
            sig: sum.unsigned_abs(),
            exp: pg,
            neg: sum < 0,
        };
        total += align(sum, grid, false);
    }
    pack(total, grid, d)
}

/// The steps of one gfx1201 WMMA D element: the k each step starts at,
/// and its products (four or eight).
pub fn schedule(form: Step) -> (&'static [usize], usize) {
    if form.a.bits == 8 {
        (&[0, 8], 8)
    } else {
        (&[0, 8, 4, 12], 4)
    }
}

/// One D element of a gfx1201 float WMMA: C's bits (after its modifiers)
/// plus Σ a[k]·b[k] over 16 k, every input its raw bits, as D's bits.
pub fn wmma_element(a: &[u32], b: &[u32], c: u32, form: Step) -> u32 {
    let (starts, n) = schedule(form);
    starts
        .iter()
        .fold(c, |acc, &s| step(&a[s..s + n], &b[s..s + n], acc, form))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16 inputs, zero but at the given (k, a, b).
    fn sparse(at: &[(usize, u32, u32)]) -> ([u32; 16], [u32; 16]) {
        let (mut a, mut b) = ([0; 16], [0; 16]);
        for &(k, x, y) in at {
            (a[k], b[k]) = (x, y);
        }
        (a, b)
    }

    // Each case below is a gfx1201 output, measured on an RX 9070 XT.
    const F16D: Step = Step {
        a: F16,
        b: F16,
        d: F16,
        group: 2,
    };

    #[test]
    fn an_f16_d_overflows_per_step_and_keeps_its_sign() {
        let a = [
            0x2963, 0x820e, 0x3ffb, 0xb94d, 0xa460, 0x72e2, 0x1f63, 0x0cdb, 0xf192, 0x5905, 0xba58,
            0x4918, 0x1834, 0x66da, 0x52e0, 0x7936,
        ];
        let b = [
            0xddf1, 0xe628, 0x64e4, 0xc11b, 0x3e31, 0x1cf8, 0x17e7, 0xea57, 0xc278, 0x4c10, 0x0000,
            0xfbc7, 0x0b2a, 0xb975, 0x18d4, 0x7ae8,
        ];
        assert_eq!(wmma_element(&a, &b, 0xa36a, F16D), 0xfc00);
        // An overflowed step's +∞ meets a later step's −∞: the default NaN.
        let a = [
            0x5c01, 0xe927, 0x2fa9, 0x7ae4, 0x47ab, 0x2403, 0xedf6, 0x8e14, 0x3c4e, 0x2b03, 0x88d4,
            0x073c, 0xbfc7, 0xfa53, 0x9887, 0x3bb4,
        ];
        let b = [
            0x209b, 0x50cc, 0xed9a, 0xbb10, 0x1786, 0xa35a, 0x0502, 0xfc00, 0x03b7, 0x1612, 0x56ca,
            0x6b4d, 0x5f7b, 0x2790, 0x826e, 0x7859,
        ];
        assert_eq!(wmma_element(&a, &b, 0xd9b7, F16D), 0xfe00);
    }

    #[test]
    fn an_f16_d_aligns_no_finer_than_its_smallest_normal_less_26() {
        // 2^-25 + 2^-41: the grid is 2^-40, the 2^-41 is lost.
        let (a, b) = sparse(&[(0, 0x0c00, 0x0800), (1, 0x0010, 0x0008)]);
        assert_eq!(wmma_element(&a, &b, 0, F16D), 0x0000);
    }

    #[test]
    fn nans_and_zeros_are_decided_step_by_step() {
        // A NaN factor in step 0, 0 · ∞ in step 2 (k = 4): the later wins.
        let (a, b) = sparse(&[(0, 0x7c01, 0), (4, 0, 0x7c00)]);
        assert_eq!(wmma_element(&a, &b, 0, F16D), 0xfe00);
        // −0 products and C: +0.
        assert_eq!(wmma_element(&[0x8000; 16], &[0; 16], 0x8000, F16D), 0);
    }

    #[test]
    fn fp8_products_align_in_quads() {
        let fp8 = Step::new(FP8, FP8, F32);
        // 2^16 and, in the second quad, -2^-10 twice: the quad's sum is
        // floored once onto the step's grid 2^-8.
        let (a, b) = sparse(&[(0, 0x78, 0x78), (4, 0x90, 0x10), (5, 0x90, 0x10)]);
        assert_eq!(wmma_element(&a, &b, 0, fp8), 0x477f_ffff);
        // The same products in one quad, C = -2^-11 below the grid: C adds
        // nothing.
        let (a, b) = sparse(&[(0, 0x78, 0x78), (1, 0x90, 0x10), (2, 0x90, 0x10)]);
        assert_eq!(wmma_element(&a, &b, 0xba00_0000, fp8), 0x477f_fffe);
    }

    #[test]
    fn a_denormal_8_bit_input_aligns_at_its_formats_smallest_exponent() {
        // 0x81, 0x86, 0x87 are fp8 denormals; aligned by their magnitudes
        // instead, D's last bit is 0x46cd977e.
        let a = [
            0xb3, 0xf7, 0x75, 0x5d, 0x73, 0x2b, 0xc9, 0xb5, 0xaf, 0xed, 0x00, 0x33, 0xec, 0xcb,
            0x81, 0x7b,
        ];
        let b = [
            0xf5, 0x75, 0x29, 0x00, 0x00, 0x86, 0x68, 0x87, 0xfc, 0x78, 0x55, 0xf8, 0xf4, 0xcb,
            0x47, 0x77,
        ];
        let fp8 = Step::new(FP8, FP8, F32);
        assert_eq!(wmma_element(&a, &b, 0x8886_804e, fp8), 0x46cd_9780);
    }

    #[test]
    fn infinities_of_different_steps_meet_cs_nan_quieted() {
        let a = [
            0xeb, 0x1e, 0x8d, 0xde, 0x32, 0x2d, 0xbf, 0x29, 0x6c, 0xae, 0xbd, 0xfc, 0x88, 0x2b,
            0xc3, 0x11,
        ];
        let b = [
            0x9d, 0x02, 0x50, 0x3c, 0x1b, 0x13, 0x0a, 0x7c, 0x28, 0x53, 0xaf, 0x3d, 0x6c, 0xe0,
            0xa0, 0x74,
        ];
        let bf8 = Step::new(BF8, BF8, F32);
        assert_eq!(wmma_element(&a, &b, 0x7ff2_8f8b, bf8), 0x7ff2_8f8b);
    }
}
