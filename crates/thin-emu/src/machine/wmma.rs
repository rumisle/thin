//! Evaluate D = A·B + C using the target's matrix lane layout.
//! Every lane participates regardless of EXEC; results carry no provenance tags.

use thin_target::{Elem, MatrixForm, Summation};

use super::alu::list_at;
use super::numbers::{f16_one, f16_value, f32_of, Small};
use super::wave::{Wave, MAX_W};
use crate::{Base, Cause, EmuError, Ins};

fn limit(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Emulator),
        message,
    }
}

/// Element `e` of a lane's operand (its dwords), as a value: floats
/// exact in binary64, integers exact (signed when `signed`).
fn element(dwords: &[u32], ty: Elem, e: usize, signed: bool) -> f64 {
    let bits = ty.bits() as usize;
    let (dw, at) = (e * bits / 32, e * bits % 32);
    let raw = (dwords[dw] >> at) & (u32::MAX >> (32 - bits));
    match ty {
        Elem::F16 => f16_value(raw),
        Elem::Bf16 => f64::from(f32_of(raw << 16)),
        Elem::F32 => f64::from(f32_of(raw)),
        Elem::Xf32 => f64::from(f32_of(raw & 0xFFFF_E000)),
        Elem::F64 => 0.0,
        Elem::I32 => f64::from(raw as i32),
        Elem::I8 => f64::from(raw as u8 as i8),
        Elem::Iu8 | Elem::Iu4 => {
            if signed && raw >> (bits - 1) & 1 != 0 {
                f64::from(raw) - f64::from(1u32 << bits)
            } else {
                f64::from(raw)
            }
        }
        Elem::Fp8 => Small::E4M3.value(raw),
        Elem::Bf8 => Small::E5M2.value(raw),
        Elem::Fp8Fnuz => Small::E4M3Fnuz.value(raw),
        Elem::Bf8Fnuz => Small::E5M2Fnuz.value(raw),
    }
}

/// One RNE rounding of an exact value to bfloat16, subnormals kept;
/// NaN → 0x7FC0.
fn bf16_one(x: f64) -> u32 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let a = x.abs();
    let r = if a == 0.0 || a.is_infinite() {
        a
    } else {
        let e = super::numbers::floor_log2(a).max(-126);
        let step = 2f64.powi(e - 7);
        (a / step).round_ties_even() * step
    };
    let bits = (r as f32).to_bits() >> 16;
    bits | if x.is_sign_negative() { 0x8000 } else { 0 }
}

/// The value of a D element rounded to its type, as its bits.
fn round_to(x: f64, ty: Elem) -> u32 {
    match ty {
        Elem::F16 => f16_one(x),
        Elem::Bf16 => bf16_one(x),
        _ => (x as f32).to_bits(),
    }
}

/// The exponent of the lowest set bit of a finite nonzero f64.
fn lsb_exponent(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    let (field, frac) = ((bits >> 52) as i32, bits & ((1 << 52) - 1));
    let (m, e) = if field == 0 {
        (frac, -1074)
    } else {
        (frac | 1 << 52, field - 1075)
    };
    e + m.trailing_zeros() as i32
}

/// An exact nonzero f64 as m · 2^e, |m| below 2^53.
pub(crate) fn parts(x: f64) -> (i128, i32) {
    let e = lsb_exponent(x);
    let m = (x.abs() / 2f64.powi(e)) as i128;
    (if x < 0.0 { -m } else { m }, e)
}

/// D's format: (significand bits, smallest normal exponent, largest
/// exponent, toward zero).
fn format_of(d: Elem) -> (u32, i32, i32, bool) {
    match d {
        Elem::F16 => (11, -14, 15, false),
        Elem::Bf16 => (8, -126, 127, true),
        _ => (24, -126, 127, false),
    }
}

/// m · 2^e rounded to D's format (subnormals kept, overflow to ±inf), as
/// an f64 that holds it exactly.
pub(crate) fn round_exact(m: i128, e: i32, d: Elem) -> f64 {
    if m == 0 {
        return 0.0;
    }
    let (p, emin, emax, toward_zero) = format_of(d);
    let a = m.unsigned_abs();
    let top = e + (127 - a.leading_zeros() as i32);
    let t = (top - (p as i32 - 1)).max(emin - (p as i32 - 1));
    let shift = t - e;
    // shift <= 0: m already fits D's format (shift >= -(p - 1)).
    let (q, t) = if shift <= 0 {
        (a, e)
    } else if shift >= 128 {
        (0, t)
    } else {
        let (q, rem) = (a >> shift, a & ((1u128 << shift) - 1));
        let half = 1u128 << (shift - 1);
        let up = !toward_zero && (rem > half || (rem == half && q & 1 == 1));
        (q + u128::from(up), t)
    };
    let v = q as f64 * 2f64.powi(t);
    let v = if v >= 2f64.powi(emax + 1) {
        f64::INFINITY
    } else {
        v
    };
    if m < 0 {
        -v
    } else {
        v
    }
}

/// One D element of the grouped model, as D's bits: a[k],
/// b[k] the inputs' values (after their modifiers), c C's value.
///
/// `byte_inputs`: fp8 / bf8, groups of `group_byte`; else of `group_half`.
/// A product's reference is its inputs' exponent sum; a group aligns to its
/// largest, the accumulator floored with the products, and the sum is
/// rounded to D's format after every group.
pub fn grouped_element(
    a: &[f64],
    b: &[f64],
    c: f64,
    d: Elem,
    byte_inputs: bool,
    (group_half, group_byte, fraction): (usize, usize, i32),
) -> u32 {
    use super::numbers::floor_log2;
    let k = a.len();
    let products: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    // NaN, or both infinities: the canonical NaN here; `wmma` gives a
    // target's GPU NaN instead (`gpu_nan`, measured).
    let terms = || products.iter().copied().chain([c]);
    let nan = terms().any(f64::is_nan)
        || (terms().any(|t| t == f64::INFINITY) && terms().any(|t| t == f64::NEG_INFINITY));
    let acc: f64 = if nan {
        f64::NAN
    } else if let Some(inf) = terms().find(|t| t.is_infinite()) {
        inf
    } else {
        let g = if byte_inputs { group_byte } else { group_half };
        let half = k / 2;
        let starts: Vec<usize> = if g >= half {
            (0..k).step_by(g).collect()
        } else {
            (0..half / g).flat_map(|i| [i * g, half + i * g]).collect()
        };
        let mut acc = c;
        for s0 in starts {
            let group: Vec<usize> = (s0..(s0 + g).min(k))
                .filter(|&q| products[q] != 0.0)
                .collect();
            let Some(top) = group
                .iter()
                .map(|&q| floor_log2(a[q].abs()) + floor_log2(b[q].abs()))
                .max()
            else {
                continue;
            };
            let qe = top - fraction;
            let q = 2f64.powi(qe);
            // The products in units of q, floored: each below 2^(fraction + 2).
            let s: i128 = group
                .iter()
                .map(|&i| (products[i] / q).floor() as i128)
                .sum();
            acc = if acc == 0.0 {
                round_exact(s, qe, d)
            } else {
                let (am, ae) = parts(acc);
                if ae >= qe {
                    if ae - qe <= 90 {
                        round_exact((am << (ae - qe)) + s, qe, d)
                    } else {
                        // The products lie far below acc's last bit: only their
                        // sign can move its rounding (a sticky bit).
                        round_exact((am << 64) + s.signum(), ae - 64, d)
                    }
                } else {
                    let floored = (acc / q).floor() as i128;
                    round_exact(floored + s, qe, d)
                }
            };
        }
        if acc == 0.0 {
            let negative = |x: f64| x == 0.0 && x.is_sign_negative();
            let all_negative = negative(c) && products.iter().all(|p| negative(*p));
            if all_negative {
                -0.0
            } else {
                0.0
            }
        } else {
            acc
        }
    };
    match d {
        Elem::F16 => f16_one(acc),
        Elem::Bf16 => {
            if acc.is_nan() {
                0x7FC0
            } else {
                (acc as f32).to_bits() >> 16
            }
        }
        _ => (acc as f32).to_bits(),
    }
}

/// Reads `n` dwords of operand `k` for every lane: `out[l][d]`.
fn lanes_of(w: &Wave, ins: &Ins, k: usize, n: u32) -> Result<Vec<[u32; 8]>, EmuError> {
    let op = ins
        .operands
        .get(k)
        .ok_or_else(|| limit(format!("{}: operand {k} missing", ins.text)))?;
    let mut out = vec![[0u32; 8]; w.w];
    for d in 0..n {
        let s = w.vread(op, d)?;
        for (l, lane) in out.iter_mut().enumerate() {
            lane[d as usize] = s.v[l];
        }
    }
    Ok(out)
}

/// The modifier bits of one list field (`neg_lo`, `neg_hi`): A, B, C.
fn bits3(ins: &Ins, name: &str) -> [bool; 3] {
    [0, 1, 2].map(|i| list_at(ins, name, i, 0) != 0)
}

/// The NaN gfx1201 gives a D element of a float WMMA whose sum is not a
/// number (measured, tests/hw/test_emu_ops.py): a NaN A or B input of its
/// dot product gives the signature NaN (f32 0xFFC00A3D, f16 0xFE3D, bf16
/// 0xFFFD); else an invalid product or sum (0 · ∞, ∞ − ∞, C's ∞ included)
/// gives it too on fp8 / bf8 inputs and the default NaN (0xFFC00000, 0xFE00,
/// 0xFFC0) on f16 / bf16 ones, products before C; else C's NaN, quieted
/// (`c_bits`, after its modifiers).
fn gpu_nan(av: &[f64], bv: &[f64], c_bits: u32, d: Elem, byte: bool) -> u32 {
    let (sig, dnan, quiet, nan) = match d {
        Elem::F16 => (0xFE3D, 0xFE00, 0x200, super::numbers::is_nan16(c_bits)),
        Elem::Bf16 => (0xFFFD, 0xFFC0, 0x40, c_bits & 0x7FFF > 0x7F80),
        _ => (
            0xFFC0_0A3D,
            0xFFC0_0000,
            0x40_0000,
            super::numbers::is_nan32(c_bits),
        ),
    };
    let invalid = if byte { sig } else { dnan };
    if av.iter().chain(bv).any(|v| v.is_nan()) {
        return sig;
    }
    let p: Vec<f64> = av.iter().zip(bv).map(|(x, y)| x * y).collect();
    let products_nan = p.iter().any(|v| v.is_nan())
        || (p.contains(&f64::INFINITY) && p.contains(&f64::NEG_INFINITY));
    if products_nan {
        return invalid;
    }
    if nan {
        return c_bits | quiet;
    }
    invalid
}

/// `v_wmma_*` in value mode: the target's summation model; a NaN result by
/// [`gpu_nan`].
pub fn wmma(w: &mut Wave, ins: &Ins, form: &MatrixForm) -> Result<(), EmuError> {
    if form.wave == 64 {
        return super::mfma::mfma(w, ins, form);
    }
    if w.w != 32 {
        return Err(limit(format!(
            "{}: a matrix form of 32 lanes on a {}-lane wave",
            ins.text, w.w
        )));
    }
    if ins.field("clamp").is_some() {
        return Err(limit(format!("{}: clamp is not emulated", ins.text)));
    }
    let (lo, hi) = (bits3(ins, "neg_lo"), bits3(ins, "neg_hi"));
    let int = form.d.integer();
    let halves = matches!(form.a, Elem::F16 | Elem::Bf16);
    let refused = if int {
        lo[2] || hi.iter().any(|b| *b)
    } else {
        !halves && (lo[0] || lo[1] || hi[0] || hi[1])
    };
    if refused {
        return Err(limit(format!(
            "{}: neg_lo / neg_hi not allowed on this form ",
            ins.text
        )));
    }
    let k = form.k as usize;
    let a = lanes_of(w, ins, 1, form.dwords('a'))?;
    let b = lanes_of(w, ins, 2, form.dwords('b'))?;
    let c_zero = matches!(
        ins.operands.get(3).map(|o| &o.base),
        Some(Base::Literal(0, _))
    );
    let c = if c_zero {
        vec![[0u32; 8]; 32]
    } else {
        lanes_of(w, ins, 3, form.dwords('c'))?
    };
    // A[i][kk] from lane i + 16h, element e = kk - (K/2)h; B[kk][j] likewise.
    let half = k / 2;
    let fetch = |ops: &[[u32; 8]], ty: Elem, idx: usize, kk: usize, side: usize| -> f64 {
        let (h, e) = (kk / half, kk % half);
        let v = element(&ops[idx + 16 * h], ty, e, int && lo[side]);
        let neg = if e % 2 == 0 { lo[side] } else { hi[side] };
        if halves && neg {
            -v
        } else {
            v
        }
    };
    // The raw 16-bit input A[idx][kk] (B[kk][idx]), its neg modifier a sign
    // flip.
    let raw16 = |ops: &[[u32; 8]], idx: usize, kk: usize, side: usize| -> u32 {
        let (h, e) = (kk / half, kk % half);
        let v = (ops[idx + 16 * h][e / 2] >> (16 * (e % 2))) & 0xFFFF;
        let neg = if e % 2 == 0 { lo[side] } else { hi[side] };
        if neg {
            v ^ 0x8000
        } else {
            v
        }
    };
    let dot_unit = halves && k == 16 && (form.d == Elem::F32 || form.d == Elem::Bf16);
    let mut out = [[0u32; MAX_W]; 8];
    for i in 0..16 {
        for j in 0..16 {
            // D[i][j]: lane j + 16 (i / 8), element i % 8.
            let (lane, e) = (j + 16 * (i / 8), i % 8);
            let cv = element(&c[lane], form.d, e, true);
            let bits = if dot_unit {
                // gfx1201's dot unit, four products a step, the steps over
                // k in 0..4, 8..12, 4..8, 12..16 ([`super::dot::step`]).
                // A bf16 D is the f32 sum's high half after every step
                // (truncated; a factor NaN 0xFFFD).
                let narrow = form.d == Elem::Bf16;
                let mut acc = match (c_zero, narrow) {
                    (true, _) => 0,
                    (false, false) => c[lane][e],
                    (false, true) => (c[lane][e / 2] >> (16 * (e % 2))) << 16,
                };
                if hi[2] {
                    acc &= 0x7FFF_FFFF;
                }
                if lo[2] {
                    acc ^= 0x8000_0000;
                }
                let bf16 = form.a == Elem::Bf16;
                for s0 in [0, 8, 4, 12] {
                    let av: Vec<u32> = (s0..s0 + 4).map(|kk| raw16(&a, i, kk, 0)).collect();
                    let bv: Vec<u32> = (s0..s0 + 4).map(|kk| raw16(&b, j, kk, 1)).collect();
                    acc = super::dot::step(&av, &bv, acc, bf16);
                    if narrow {
                        acc = if acc == super::dot::FACTOR_NAN {
                            0xFFFD_0000
                        } else {
                            acc & 0xFFFF_0000
                        };
                    }
                }
                if narrow {
                    acc >> 16
                } else {
                    acc
                }
            } else if int {
                let mut acc = cv as i64;
                for kk in 0..k {
                    let p = fetch(&a, form.a, i, kk, 0) * fetch(&b, form.b, j, kk, 1);
                    acc = acc.wrapping_add(p as i64);
                }
                acc as u32
            } else {
                let mut cv = cv;
                if hi[2] {
                    cv = cv.abs();
                }
                if lo[2] {
                    cv = -cv;
                }
                let av: Vec<f64> = (0..k).map(|kk| fetch(&a, form.a, i, kk, 0)).collect();
                let bv: Vec<f64> = (0..k).map(|kk| fetch(&b, form.b, j, kk, 1)).collect();
                let byte = matches!(form.a, Elem::Fp8 | Elem::Bf8);
                let r = match form.sum {
                    Summation::Ordered => {
                        let acc = av.iter().zip(&bv).fold(0.0f64, |s, (x, y)| s + x * y);
                        round_to(acc + cv, form.d)
                    }
                    Summation::Grouped {
                        group_half,
                        group_byte,
                        fraction,
                    } => {
                        let params = (group_half as usize, group_byte as usize, fraction);
                        grouped_element(&av, &bv, cv, form.d, byte, params)
                    }
                };
                let dbits = form.d.bits();
                let is_nan = match form.d {
                    Elem::F16 => super::numbers::is_nan16(r),
                    Elem::Bf16 => r & 0x7FFF > 0x7F80,
                    _ => super::numbers::is_nan32(r),
                };
                if is_nan {
                    let raw = (c[lane][e * dbits as usize / 32] >> (e * dbits as usize % 32))
                        & (u32::MAX >> (32 - dbits));
                    let sign = 1u32 << (dbits - 1);
                    let raw = if c_zero { 0 } else { raw };
                    let raw = if hi[2] { raw & !sign } else { raw };
                    let raw = if lo[2] { raw ^ sign } else { raw };
                    gpu_nan(&av, &bv, raw, form.d, byte)
                } else {
                    r
                }
            };
            let dbits = form.d.bits() as usize;
            let (dw, at) = (e * dbits / 32, e * dbits % 32);
            let mask = u32::MAX >> (32 - dbits);
            out[dw][lane] = (out[dw][lane] & !(mask << at)) | (bits & mask) << at;
        }
    }
    let dst = &ins.operands[0];
    for d in 0..form.dwords('c') {
        w.vwrite(dst, d, &out[d as usize], None, u64::from(u32::MAX))?;
    }
    Ok(())
}
