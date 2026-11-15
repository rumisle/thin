//! gfx1201's float dot unit, bit for bit: one step of two (v_dot2_f32_bf16)
//! or four (a WMMA with f16 or bf16 inputs and f32 D) products added to an
//! f32 accumulator.
//!
//! The algorithm is AMD rocjitsu's (ROCm/rocm-systems, emulation/rocjitsu,
//! MIT; see NOTICE). Products are exact. Each pair of products is aligned to
//! the pair's grid, 2^(e_a + e_b − 24) for the larger exponent sum (a
//! denormal input's exponent its format's smallest), by floor; the pair sums
//! and the accumulator are aligned to the step's grid, the larger of the
//! products' grid and 2^(e_c − 26) (e_c the accumulator's leading exponent,
//! at least −126), again by floor, an accumulator with no bit left
//! contributing nothing even when negative; the total is rounded once to
//! binary32 (RNE), a zero +0. Measured on gfx1201: 32 768 random WMMA
//! elements of f32_16x16x16_f16 and _bf16, and v_dot2_f32_bf16's random,
//! cancelling and accumulator-dominated sets (tests/hw/test_emu_ops.py,
//! tests/hw/test_wmma.py).

/// No exponent: a zero significand.
const EMPTY: i32 = -1024;

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

/// One f16 or bf16 input: its significand, its alignment exponent (a
/// denormal's the smallest normal one) and fraction width.
struct Factor {
    bits: u32,
    fb: i32,
    bias: i32,
    inf: u32,
}

impl Factor {
    fn new(bits: u32, bf16: bool) -> Factor {
        let (fb, bias, inf) = if bf16 {
            (7, 127, 0x7F80)
        } else {
            (10, 15, 0x7C00)
        };
        Factor {
            bits: bits & 0xFFFF,
            fb,
            bias,
            inf,
        }
    }
    fn nan(&self) -> bool {
        self.bits & 0x7FFF > self.inf
    }
    fn is_inf(&self) -> bool {
        self.bits & 0x7FFF == self.inf
    }
    fn negative(&self) -> bool {
        self.bits >> 15 == 1
    }
    fn field(&self) -> u32 {
        (self.bits & 0x7FFF) >> self.fb
    }
    fn sig(&self) -> u64 {
        let frac = u64::from(self.bits & ((1 << self.fb) - 1));
        if self.field() != 0 {
            1 << self.fb | frac
        } else {
            frac
        }
    }
    fn align_exp(&self) -> i32 {
        self.field().max(1) as i32 - self.bias
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

/// `units` · 2^grid rounded once to binary32 bits, a zero +0.
fn pack(units: i64, grid: i32) -> u32 {
    if units == 0 {
        return 0;
    }
    let sign = u32::from(units < 0) << 31;
    let mag = units.unsigned_abs();
    let top = 63 - mag.leading_zeros() as i32;
    let mut e = top + grid;
    if e < -126 {
        let f = shift_rne(mag, -149 - grid);
        return if f == 0 { 0 } else { sign | f as u32 };
    }
    let mut s = shift_rne(mag, top - 23);
    if s == 1 << 24 {
        s >>= 1;
        e += 1;
    }
    if e > 127 {
        return sign | 0x7F80_0000;
    }
    sign | ((e + 127) as u32) << 23 | (s as u32 & 0x7F_FFFF)
}

/// `t` on the grid 2^grid by floor; an accumulator with no bit left is 0.
fn align(t: Term, grid: i32, accumulator: bool) -> i64 {
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
        -(mag as i64) - i64::from(tail && (!accumulator || mag != 0))
    } else {
        mag as i64
    }
}

/// gfx1201's NaN when a factor is a NaN.
pub const FACTOR_NAN: u32 = 0xFFC0_0A3D;

/// acc + Σ a[i]·b[i] for 2 or 4 products of f16 (`bf16`: bfloat16) inputs
/// (the low 16 bits of each), as binary32 bits. NaNs: a NaN factor gives
/// [`FACTOR_NAN`]; an invalid product or ∞ − ∞ the default NaN; else the
/// accumulator's NaN quieted.
pub fn step(a: &[u32], b: &[u32], acc: u32, bf16: bool) -> u32 {
    debug_assert!(a.len() == b.len() && (a.len() == 2 || a.len() == 4));
    let (mut pinf, mut ninf, mut invalid) = (false, false, false);
    let mut prods = [Term {
        sig: 0,
        exp: 0,
        neg: false,
    }; 4];
    let mut grids = [EMPTY; 4];
    for i in 0..a.len() {
        let (l, r) = (Factor::new(a[i], bf16), Factor::new(b[i], bf16));
        if l.nan() || r.nan() {
            return FACTOR_NAN;
        }
        invalid |= (l.is_inf() && r.sig() == 0) || (r.is_inf() && l.sig() == 0);
        if l.is_inf() || r.is_inf() {
            if l.negative() != r.negative() {
                ninf = true;
            } else {
                pinf = true;
            }
        }
        prods[i] = Term {
            sig: l.sig() * r.sig(),
            exp: l.align_exp() + r.align_exp() - 2 * l.fb,
            neg: l.negative() != r.negative(),
        };
        if prods[i].sig != 0 {
            grids[i] = l.align_exp() + r.align_exp() - 24;
        }
    }
    if invalid || (pinf && ninf) {
        return super::numbers::DNAN32;
    }
    if acc & 0x7FFF_FFFF > 0x7F80_0000 {
        return acc | 0x0040_0000;
    }
    pinf |= acc == 0x7F80_0000;
    ninf |= acc == 0xFF80_0000;
    if pinf || ninf {
        return match (pinf, ninf) {
            (true, true) => super::numbers::DNAN32,
            (false, true) => 0xFF80_0000,
            _ => 0x7F80_0000,
        };
    }
    let field = (acc >> 23) & 0xFF;
    let c = Term {
        sig: u64::from(if field != 0 {
            acc & 0x7F_FFFF | 0x80_0000
        } else {
            acc & 0x7F_FFFF
        }),
        exp: field.max(1) as i32 - 150,
        neg: acc >> 31 == 1,
    };
    let products = grids.iter().copied().max().unwrap_or(EMPTY);
    let grid = products.max(c.leading().max(-126) - 26);
    let mut total = align(c, grid, true);
    for i in (0..a.len()).step_by(2) {
        let pg = grids[i].max(grids[i + 1]);
        let pair = align(prods[i], pg, false) + align(prods[i + 1], pg, false);
        let sum = Term {
            sig: pair.unsigned_abs(),
            exp: pg,
            neg: pair < 0,
        };
        total += align(sum, grid, false);
    }
    pack(total, grid)
}
