//! Evaluate D = A·B + C using the target's matrix lane layout.
//! Every lane participates regardless of EXEC; results carry no provenance tags.

use thin_target::{Elem, MatrixForm, Summation};

use super::alu::list_at;
use super::dot;
use super::numbers::{f16_value, f32_of, Small};
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
        Elem::Fp6 => Small::E2M3.value(raw),
        Elem::Bf6 => Small::E3M2.value(raw),
        Elem::Fp4 => Small::E2M1.value(raw),
        Elem::F8F6F4 => Small::E4M3.value(raw),
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

/// The dot unit's format of a float element type.
fn format(ty: Elem) -> Option<dot::Format> {
    Some(match ty {
        Elem::F32 => dot::F32,
        Elem::F16 => dot::F16,
        Elem::Bf16 => dot::BF16,
        Elem::Fp8 => dot::FP8,
        Elem::Bf8 => dot::BF8,
        _ => return None,
    })
}

/// `v_wmma_*` in value mode. Float forms: gfx1201's dot unit
/// ([`dot::wmma_element`]); integer forms exact, in two halves of k, each
/// wrapped to 32 bits, or with `clamp` saturated (measured on gfx1201).
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
    let int = form.d.integer();
    let clamp = ins.field("clamp").is_some();
    if clamp && !int {
        return Err(limit(format!("{}: clamp is not emulated", ins.text)));
    }
    let (lo, hi) = (bits3(ins, "neg_lo"), bits3(ins, "neg_hi"));
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
    let step = match (
        int,
        form.sum,
        format(form.a),
        format(form.b),
        format(form.d),
    ) {
        (true, _, _, _, _) => None,
        (false, Summation::DotUnit, Some(a), Some(b), Some(d)) => Some(dot::Step::new(a, b, d)),
        _ => {
            return Err(limit(format!(
                "{}: this matrix form's summation is not emulated",
                ins.text
            )))
        }
    };
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
    // The raw input A[idx][kk] (B[kk][idx]), a 16-bit one's neg modifier a
    // sign flip (even k neg_lo, odd neg_hi).
    let raw = |ops: &[[u32; 8]], ty: Elem, idx: usize, kk: usize, side: usize| -> u32 {
        let (h, e) = (kk / half, kk % half);
        let bits = ty.bits() as usize;
        let v = (ops[idx + 16 * h][e * bits / 32] >> (e * bits % 32)) & (u32::MAX >> (32 - bits));
        let neg = if e % 2 == 0 { lo[side] } else { hi[side] };
        if halves && neg {
            v ^ 1 << (bits - 1)
        } else {
            v
        }
    };
    let dbits = form.d.bits() as usize;
    let mut out = [[0u32; MAX_W]; 8];
    for i in 0..16 {
        for j in 0..16 {
            // D[i][j]: lane j + 16 (i / 8), element i % 8.
            let (lane, e) = (j + 16 * (i / 8), i % 8);
            let (dw, at) = (e * dbits / 32, e * dbits % 32);
            let mask = u32::MAX >> (32 - dbits);
            let cbits = (c[lane][dw] >> at) & mask;
            let bits = if let Some(step) = step {
                // C's modifiers: neg_hi abs, then neg_lo negation; 8-bit
                // inputs ignore both (measured).
                let sign = 1u32 << (dbits - 1);
                let mut cv = cbits;
                if halves && hi[2] {
                    cv &= !sign;
                }
                if halves && lo[2] {
                    cv ^= sign;
                }
                let av: Vec<u32> = (0..k).map(|kk| raw(&a, form.a, i, kk, 0)).collect();
                let bv: Vec<u32> = (0..k).map(|kk| raw(&b, form.b, j, kk, 1)).collect();
                dot::wmma_element(&av, &bv, cv, step)
            } else {
                let mut acc = i64::from(cbits as i32);
                for part in [0..half, half..k] {
                    acc += part
                        .map(|kk| {
                            let x = element(&a[i + 16 * (kk / half)], form.a, kk % half, lo[0]);
                            let y = element(&b[j + 16 * (kk / half)], form.b, kk % half, lo[1]);
                            (x * y) as i64
                        })
                        .sum::<i64>();
                    acc = if clamp {
                        acc.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                    } else {
                        i64::from(acc as i32)
                    };
                }
                acc as u32
            };
            out[dw][lane] = (out[dw][lane] & !(mask << at)) | (bits & mask) << at;
        }
    }
    let dst = &ins.operands[0];
    for d in 0..form.dwords('c') {
        w.vwrite(dst, d, &out[d as usize], None, u64::from(u32::MAX))?;
    }
    Ok(())
}
