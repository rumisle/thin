//! Wave64 MFMA and 4:2 SMFMAC execution (`D = A·B + C`) using the target's
//! matrix lane/register/block layout. Every lane participates regardless of
//! `EXEC`; value-mode outputs carry no provenance tags.

use thin_target::{Elem, MatrixForm};

use super::alu::{fma as fma_f32, int_field, list_at, tag_error};
use super::numbers::{self as num, f16_value, f32_of, floor_log2, Small};
use super::wave::{Wave, MAX_W};
use super::Exec;
use crate::tags::{self, is_record, MatrixOp, MatrixOperand, OperandFormat, Tag, ACC_MARK};
use crate::{Base, Cause, EmuError, Ins};

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
fn parts(x: f64) -> (i128, i32) {
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
fn round_exact(m: i128, e: i32, d: Elem) -> f64 {
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

/// Largest dword count of any MFMA operand (`v_mfma_*_32x32x*_2b_*` uses 32).
const MAX_DWORDS: usize = 32;

fn limit(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Emulator),
        message,
    }
}

/// Reads `n` dwords of a VGPR or AGPR operand `k` across all 64 lanes: `out[lane][dword]`.
fn reg_lanes(
    w: &Wave,
    ins: &Ins,
    k: usize,
    n: u32,
    only_v: bool,
) -> Result<(String, Vec<[u32; MAX_DWORDS]>), EmuError> {
    let op = ins
        .operands
        .get(k)
        .ok_or_else(|| limit(format!("{}: operand {k} missing", ins.text)))?;
    let file = match &op.base {
        Base::Reg(f, _, _) if f == "v" || (!only_v && f == "a") => f.clone(),
        _ => {
            return Err(limit(format!(
                "{}: operand {k} (`{}`) is not a legal matrix register operand",
                ins.text,
                op.spelling()
            )))
        }
    };
    let mut out = vec![[0u32; MAX_DWORDS]; w.w];
    for d in 0..n {
        let s = w.vread(op, d)?;
        for (l, lane) in out.iter_mut().enumerate() {
            lane[d as usize] = s.v[l];
        }
    }
    Ok((file, out))
}

/// Reads `n` dwords of dense MFMA `src2` (`C`): a matching `v`/`a` register range or an inline constant.
fn c_lanes(
    w: &Wave,
    ins: &Ins,
    n: u32,
    d_ty: Elem,
    dst_file: &str,
) -> Result<Vec<[u32; MAX_DWORDS]>, EmuError> {
    let op = ins
        .operands
        .get(3)
        .ok_or_else(|| limit(format!("{}: operand 3 missing", ins.text)))?;
    match &op.base {
        Base::Reg(f, _, _) if f == "v" || f == "a" => {
            if f != dst_file {
                return Err(limit(format!(
                    "{}: accumulator src2 (`{}`) must match destination register file `{dst_file}`",
                    ins.text,
                    op.spelling()
                )));
            }
            Ok(reg_lanes(w, ins, 3, n, false)?.1)
        }
        Base::Literal(_, text) => {
            let v0 = w.vread(op, 0)?.v[0];
            let mut lane = [0u32; MAX_DWORDS];
            if d_ty == Elem::F64 {
                let bits64 = if text.contains('.') {
                    f64::from(f32_of(v0)).to_bits()
                } else {
                    (v0 as i32 as i64) as u64
                };
                for d in (0..n as usize).step_by(2) {
                    lane[d] = bits64 as u32;
                    lane[d + 1] = (bits64 >> 32) as u32;
                }
            } else {
                for slot in &mut lane[..n as usize] {
                    *slot = v0;
                }
            }
            Ok(vec![lane; w.w])
        }
        _ => Err(limit(format!(
            "{}: invalid accumulator operand `{}`",
            ins.text,
            op.spelling()
        ))),
    }
}

/// Applies the `BLGP` lane group pattern to a B-matrix lane (`0..64`).
#[inline]
fn blgp_lane(lane: usize, blgp: usize) -> usize {
    match blgp {
        0 => lane,
        1 => lane % 32,
        2 => (lane % 32) + 32,
        3 => (lane + 16) % 64,
        4..=7 => (lane % 16) + 16 * (blgp - 4),
        _ => lane,
    }
}

/// Maps input matrix coordinates `(outer_dim, k_dim, blocks, i, k, b)` to `(dword, bit_lo, lane)`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn input_loc(
    outer: usize,
    k_dim: usize,
    blocks: usize,
    i: usize,
    k: usize,
    b: usize,
    bits: usize,
    sparse_or_k: bool,
    is_k: bool,
    cbsz: usize,
    abid: usize,
    blgp: usize,
) -> (usize, usize, usize) {
    let (k_eff_dim, k_eff, b_eff) = if sparse_or_k {
        (k_dim / 2, k / 2, b)
    } else if cbsz > 0 {
        let grp = 1usize << cbsz;
        (k_dim, k, (b & !(grp - 1)) + abid)
    } else {
        (k_dim, k, b)
    };
    let lane_groups = 64 / (outer * blocks);
    let elems_contig = k_eff_dim / lane_groups;
    if !is_k && bits <= 16 && elems_contig * bits == 256 {
        // 8-VGPR narrow operands (CDNA 4 SMFMAC dense B and F8F6F4 FP8/BF8)
        // stack two 4-VGPR (128-bit) K-halves: v0..v3 holds 0..k_eff_dim/2
        // and v4..v7 holds k_eff_dim/2..k_eff_dim.
        let half_k = k_eff_dim / 2;
        let half_idx = k_eff / half_k;
        let k_in_half = k_eff % half_k;
        let elems_half = half_k / lane_groups;
        let local = k_in_half % elems_half;
        let lane = blgp_lane(
            b_eff * outer + (k_in_half / elems_half) * (outer * blocks) + i,
            blgp,
        );
        let tot_bits = local * bits;
        return (4 * half_idx + tot_bits / 32, tot_bits % 32, lane);
    }
    let local = k_eff % elems_contig;
    let lane = blgp_lane(
        b_eff * outer + (k_eff / elems_contig) * (outer * blocks) + i,
        blgp,
    );
    if is_k {
        let mut slot = local;
        if cbsz == 0 {
            slot += elems_contig * abid;
        }
        (0, (slot / 2) * 4, lane)
    } else {
        let tot_bits = local * bits;
        (tot_bits / 32, tot_bits % 32, lane)
    }
}

/// Maps output/accumulator coordinates `(m, n, blocks, i, j, b)` to `(dword, lane)`.
#[inline]
fn output_loc(m: usize, n: usize, i: usize, j: usize, b: usize, bits: usize) -> (usize, usize) {
    let multirows_per_reg = 64 / n;
    let multirow_h = if bits == 64 { 1 } else { 4 };
    let chunk = (n * m) / multirow_h;
    let blocks_per_reg = 64usize.div_ceil(chunk);
    let local =
        b * ((m * n) / 64) + (i / (multirow_h * multirows_per_reg)) * multirow_h + (i % multirow_h);
    let lane =
        (b % blocks_per_reg) * n + ((i / multirow_h) % multirows_per_reg) * blocks_per_reg * n + j;
    if bits == 64 {
        (2 * local, lane)
    } else {
        (local, lane)
    }
}

/// Decodes one narrow/32-bit/64-bit input element at `(dword, bit_lo)` into `f64`.
#[inline]
fn decode_elem(dwords: &[u32; MAX_DWORDS], ty: Elem, dw: usize, bit_lo: usize) -> f64 {
    match ty {
        Elem::F64 => {
            let bits = u64::from(dwords[dw]) | (u64::from(dwords[dw + 1]) << 32);
            f64::from_bits(bits)
        }
        Elem::Xf32 => {
            let raw = dwords[dw];
            if num::is_nan32(raw) {
                f64::NAN
            } else {
                f64::from(f32_of(raw & 0xFFFF_E000))
            }
        }
        Elem::F32 => f64::from(f32_of(dwords[dw])),
        Elem::I32 => f64::from(dwords[dw] as i32),
        Elem::F16 => f16_value((dwords[dw] >> bit_lo) & 0xFFFF),
        Elem::Bf16 => f64::from(f32_of(((dwords[dw] >> bit_lo) & 0xFFFF) << 16)),
        Elem::I8 => f64::from(((dwords[dw] >> bit_lo) & 0xFF) as u8 as i8),
        Elem::Fp8 => Small::E4M3.value((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Bf8 => Small::E5M2.value((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Fp8Fnuz => Small::E4M3Fnuz.value((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Bf8Fnuz => Small::E5M2Fnuz.value((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Fp6 | Elem::Bf6 => {
            // 6-bit packed elements straddle dword boundaries at bit_lo == 28 and 30.
            let word64 =
                u64::from(dwords[dw]) | (u64::from(*dwords.get(dw + 1).unwrap_or(&0)) << 32);
            let raw = ((word64 >> bit_lo) & 0x3F) as u32;
            if ty == Elem::Fp6 {
                Small::E2M3.value(raw)
            } else {
                Small::E3M2.value(raw)
            }
        }
        Elem::Fp4 => Small::E2M1.value((dwords[dw] >> bit_lo) & 0xF),
        Elem::F8F6F4 => Small::E4M3.value((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Iu8 => f64::from((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Iu4 => f64::from((dwords[dw] >> bit_lo) & 0xF),
    }
}

/// Evaluates an XDLOP `f32`-accumulating dot product (`f16`, `bf16`, `xf32`,
/// `fp8`, `bf8`, `f8f6f4`, dense and sparse `v_smfmac_*`) using the grouped dot model:
/// - Any `NaN` in `A`, `B`, or `C`, or `0 * Inf` / `+Inf + (-Inf)`, returns `0xFFC0_0000`.
/// - Active reduction terms are processed sequentially in contiguous groups of `group_size`.
/// - Within each group, `top = max_q (floor_log2(|a[q]|) + floor_log2(|b[q]|))` over
///   non-zero products sets `qe = top - 24`; products are truncated toward zero at `2^qe`,
///   combined with `acc` (exact if `lsb_exp(acc) >= qe`, floored at `2^qe` below), and
///   rounded RNE to `f32` after every group.
fn eval_xdlop_dot(av: &[f64], bv: &[f64], c_bits: u32, group_size: usize) -> u32 {
    if num::is_nan32(c_bits) || av.iter().chain(bv).any(|v| v.is_nan()) {
        return num::DNAN32;
    }
    let cv = f64::from(f32_of(c_bits));
    let n = av.len();
    let product = |q: usize| av[q] * bv[q];
    let mut has_pos_inf = cv == f64::INFINITY;
    let mut has_neg_inf = cv == f64::NEG_INFINITY;
    for (&x, &y) in av.iter().zip(bv) {
        let p = x * y;
        if p.is_nan() {
            return num::DNAN32;
        }
        if p == f64::INFINITY {
            has_pos_inf = true;
        } else if p == f64::NEG_INFINITY {
            has_neg_inf = true;
        }
    }
    if has_pos_inf && has_neg_inf {
        return num::DNAN32;
    }
    if has_pos_inf {
        return 0x7F80_0000;
    }
    if has_neg_inf {
        return 0xFF80_0000;
    }

    let g = group_size.min(n).max(1);
    let mut acc = cv;
    for s0 in (0..n).step_by(g) {
        let end = (s0 + g).min(n);
        let Some(top) = (s0..end)
            .filter(|&q| product(q) != 0.0)
            .map(|q| floor_log2(av[q].abs()) + floor_log2(bv[q].abs()))
            .max()
        else {
            continue;
        };
        let qe = top - 24;
        let step = 2f64.powi(qe);
        let s: i128 = (s0..end)
            .map(product)
            .filter(|&p| p != 0.0)
            .map(|p| (p / step).trunc() as i128)
            .sum();
        acc = if acc == 0.0 {
            round_exact(s, qe, Elem::F32)
        } else {
            let (am, ae) = parts(acc);
            if ae >= qe {
                if ae - qe <= 90 {
                    round_exact((am << (ae - qe)) + s, qe, Elem::F32)
                } else {
                    round_exact((am << 64) + s.signum(), ae - 64, Elem::F32)
                }
            } else {
                let floored = (acc / step).floor() as i128;
                round_exact(floored + s, qe, Elem::F32)
            }
        };
    }

    if acc == 0.0 {
        let neg = |x: f64| x == 0.0 && x.is_sign_negative();
        let all_neg = neg(cv) && (0..n).all(|q| neg(product(q)));
        return if all_neg { 0x8000_0000 } else { 0 };
    }
    (acc as f32).to_bits()
}

/// Evaluates an `f32`-input SGEMM MFMA (`v_mfma_f32_*_f32`) via sequential
/// single-rounded `v_fma_f32` steps (`D = fma(A[k], B[k], acc)`), preserving
/// NaN sign and payload per step as measured on MI300X.
fn eval_sgemm_fma(a_raw: &[u32], b_raw: &[u32], c_bits: u32) -> u32 {
    let mut acc = c_bits;
    for (&x, &y) in a_raw.iter().zip(b_raw) {
        acc = fma_f32(x, y, acc);
    }
    acc
}

/// Evaluates an `f64`-accumulating MFMA dot product via fused `mul_add` steps
/// with MI300X-measured `fma_nan64` payload propagation.
fn eval_f64_dot(a_raw: &[u64], b_raw: &[u64], c_bits: u64) -> u64 {
    let mut acc = c_bits;
    for (&x, &y) in a_raw.iter().zip(b_raw) {
        let r = f64::from_bits(x).mul_add(f64::from_bits(y), f64::from_bits(acc));
        acc = num::fma_nan64(r, x, y, acc);
    }
    acc
}

/// Executes a wave64 `v_mfma_*` or `v_smfmac_*` instruction in value mode.
pub fn mfma(w: &mut Wave, ins: &Ins, form: &MatrixForm) -> Result<(), EmuError> {
    if w.w != 64 {
        return Err(limit(format!(
            "{}: a matrix form of 64 lanes on a {}-lane wave",
            ins.text, w.w
        )));
    }
    if ins.field("clamp").is_some() {
        return Err(limit(format!(
            "{}: clamp is not supported on MFMA",
            ins.text
        )));
    }

    let cbsz_i = int_field(ins, "cbsz", 0);
    let abid_i = int_field(ins, "abid", 0);
    let blgp_i = int_field(ins, "blgp", 0);

    let is_f64 = form.d == Elem::F64;
    let is_f8f6f4 = form.a == Elem::F8F6F4 || form.b == Elem::F8F6F4;
    let mut neg_f64 = [false; 3];
    let mut a_ty = form.a;
    let mut b_ty = form.b;

    if form.sparse {
        if blgp_i != 0 || ins.field("neg").is_some() {
            return Err(limit(format!(
                "{}: blgp/neg is not supported on sparse MFMA",
                ins.text
            )));
        }
        if !(0..=3).contains(&cbsz_i) {
            return Err(limit(format!(
                "{}: cbsz {cbsz_i} out of range 0..3",
                ins.text
            )));
        }
        let max_sets = i128::from(form.sparse_sets());
        if !(0..max_sets).contains(&abid_i) {
            return Err(limit(format!(
                "{}: abid {abid_i} out of range 0..{}",
                ins.text,
                max_sets - 1
            )));
        }
    } else if is_f8f6f4 {
        if abid_i != 0 || ins.field("neg").is_some() {
            return Err(limit(format!(
                "{}: abid/neg is not supported on f8f6f4 MFMA",
                ins.text
            )));
        }
        a_ty = u32::try_from(cbsz_i)
            .ok()
            .and_then(Elem::from_f8f6f4)
            .ok_or_else(|| {
                limit(format!(
                    "{}: cbsz {cbsz_i} out of range 0..4 for f8f6f4 A format",
                    ins.text
                ))
            })?;
        b_ty = u32::try_from(blgp_i)
            .ok()
            .and_then(Elem::from_f8f6f4)
            .ok_or_else(|| {
                limit(format!(
                    "{}: blgp {blgp_i} out of range 0..4 for f8f6f4 B format",
                    ins.text
                ))
            })?;
    } else if is_f64 {
        if cbsz_i != 0 || abid_i != 0 {
            return Err(limit(format!(
                "{}: cbsz/abid is not supported on f64 MFMA",
                ins.text
            )));
        }
        if !(0..=7).contains(&blgp_i) {
            return Err(limit(format!(
                "{}: blgp {blgp_i} out of range 0..7",
                ins.text
            )));
        }
        for i in 0..3 {
            neg_f64[i] = list_at(ins, "neg", i, 0) != 0 || ((blgp_i >> i) & 1) != 0;
        }
    } else {
        if ins.field("neg").is_some() {
            return Err(limit(format!(
                "{}: neg is only supported on f64 MFMA",
                ins.text
            )));
        }
        if cbsz_i != 0 || abid_i != 0 {
            if !form.cbsz_abid {
                return Err(limit(format!(
                    "{}: cbsz/abid is not supported on this form",
                    ins.text
                )));
            }
            let max_cbsz = i128::from(form.blocks.ilog2()).min(4);
            let max_abid = (1i128 << cbsz_i.clamp(0, 4)) - 1;
            if !(0..=max_cbsz).contains(&cbsz_i) {
                return Err(limit(format!(
                    "{}: cbsz {cbsz_i} out of range 0..{max_cbsz}",
                    ins.text
                )));
            }
            if !(0..=max_abid).contains(&abid_i) {
                return Err(limit(format!(
                    "{}: abid {abid_i} out of range 0..{max_abid}",
                    ins.text
                )));
            }
        }
        if blgp_i != 0 && (!form.blgp || !(0..=7).contains(&blgp_i)) {
            return Err(limit(format!(
                "{}: blgp {blgp_i} is not supported on this form",
                ins.text
            )));
        }
    }

    let cbsz = if is_f8f6f4 { 0 } else { cbsz_i as usize };
    let abid = abid_i as usize;
    let blgp = if is_f64 || is_f8f6f4 {
        0
    } else {
        blgp_i as usize
    };

    let d_dwords = form.dwords('d');
    let a_dwords = form
        .operand_dwords('a', cbsz_i as u32, blgp_i as u32)
        .unwrap_or_else(|| form.dwords('a'));
    let b_dwords = form
        .operand_dwords('b', cbsz_i as u32, blgp_i as u32)
        .unwrap_or_else(|| form.dwords('b'));
    let dst = ins
        .operands
        .first()
        .ok_or_else(|| limit(format!("{}: missing destination", ins.text)))?;
    let dst_file = match &dst.base {
        Base::Reg(f, _, _) if f == "v" || f == "a" => f.clone(),
        _ => {
            return Err(limit(format!(
                "{}: invalid destination `{}`",
                ins.text,
                dst.spelling()
            )))
        }
    };

    let (_, a_regs) = reg_lanes(w, ins, 1, a_dwords, false)?;
    let (_, b_regs) = reg_lanes(w, ins, 2, b_dwords, false)?;
    let (c_regs, k_regs) = if form.sparse {
        let (_, c) = reg_lanes(w, ins, 0, d_dwords, false)?;
        let (_, k) = reg_lanes(w, ins, 3, 1, true)?;
        (c, Some(k))
    } else {
        let c = c_lanes(w, ins, d_dwords, form.d, &dst_file)?;
        (c, None)
    };
    let scale_regs = if form.scaled {
        let (_, sa) = reg_lanes(w, ins, 4, 1, true)?;
        let (_, sb) = reg_lanes(w, ins, 5, 1, true)?;
        let sel_a = (((list_at(ins, "op_sel_hi", 0, 0) as usize) & 1) << 1)
            | ((list_at(ins, "op_sel", 0, 0) as usize) & 1);
        let sel_b = (((list_at(ins, "op_sel_hi", 1, 0) as usize) & 1) << 1)
            | ((list_at(ins, "op_sel", 1, 0) as usize) & 1);
        Some((sa, sb, sel_a, sel_b))
    } else {
        None
    };

    let m = form.m as usize;
    let n = form.n as usize;
    let k = form.k as usize;
    let blocks = form.blocks as usize;
    let a_bits = a_ty.bits() as usize;
    let b_bits = b_ty.bits() as usize;
    let d_bits = form.d.bits() as usize;
    let is_int = form.d.integer();
    let dot_group = if is_f8f6f4 { 32 } else { 128 / a_bits };

    let mut out = [[0u32; MAX_W]; MAX_DWORDS];

    let active_terms = if form.sparse { k / 2 } else { k };
    let mut av = vec![0.0f64; active_terms];
    let mut bv = vec![0.0f64; active_terms];
    // Sparse forms only take f16/bf16/fp8/bf8/i8 inputs: no raw f32/f64 terms.
    let (a_raw32, b_raw32, a_raw64, b_raw64) = (
        Vec::<u32>::new(),
        Vec::<u32>::new(),
        Vec::<u64>::new(),
        Vec::<u64>::new(),
    );

    // Dense operands depend on (row, k) or (column, k) alone: decode them once
    // per block instead of once per output element.
    let dense = !form.sparse;
    let (rows_a, rows_b) = if dense { (m, n) } else { (0, 0) };
    let mut a_val = vec![0.0f64; rows_a * k];
    let mut b_val = vec![0.0f64; rows_b * k];
    let mut a_r32 = vec![0u32; rows_a * k];
    let mut b_r32 = vec![0u32; rows_b * k];
    let mut a_r64 = vec![0u64; rows_a * k];
    let mut b_r64 = vec![0u64; rows_b * k];
    let raw32 = a_ty == Elem::F32 && b_ty == Elem::F32;
    let scale = |regs: &[[u32; MAX_DWORDS]], sel: usize, row: usize, rows: usize, kk: usize| {
        let byte = (regs[(kk / 32) * rows + row][0] >> (8 * sel)) & 0xFF;
        if byte == 0xFF {
            f64::NAN
        } else {
            2f64.powi(byte as i32 - 127)
        }
    };

    for b in 0..blocks {
        if dense {
            for i in 0..m {
                for kk in 0..k {
                    let (dw, lo, lane) =
                        input_loc(m, k, blocks, i, kk, b, a_bits, false, false, cbsz, abid, 0);
                    let at = i * k + kk;
                    if is_f64 {
                        let mut x =
                            u64::from(a_regs[lane][dw]) | (u64::from(a_regs[lane][dw + 1]) << 32);
                        if neg_f64[0] {
                            x ^= 1u64 << 63;
                        }
                        a_r64[at] = x;
                    } else if raw32 {
                        a_r32[at] = a_regs[lane][dw];
                    } else {
                        let mut x = decode_elem(&a_regs[lane], a_ty, dw, lo);
                        if let Some((sa, _, sel_a, _)) = scale_regs.as_ref() {
                            x *= scale(sa, *sel_a, i, m, kk);
                        }
                        a_val[at] = x;
                    }
                }
            }
            for j in 0..n {
                for kk in 0..k {
                    let (dw, lo, lane) =
                        input_loc(n, k, blocks, j, kk, b, b_bits, false, false, 0, 0, blgp);
                    let at = j * k + kk;
                    if is_f64 {
                        let mut y =
                            u64::from(b_regs[lane][dw]) | (u64::from(b_regs[lane][dw + 1]) << 32);
                        if neg_f64[1] {
                            y ^= 1u64 << 63;
                        }
                        b_r64[at] = y;
                    } else if raw32 {
                        b_r32[at] = b_regs[lane][dw];
                    } else {
                        let mut y = decode_elem(&b_regs[lane], b_ty, dw, lo);
                        if let Some((_, sb, _, sel_b)) = scale_regs.as_ref() {
                            y *= scale(sb, *sel_b, j, n, kk);
                        }
                        b_val[at] = y;
                    }
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                let (d_dw, d_lane) = output_loc(m, n, i, j, b, d_bits);
                if let Some(k_mat) = k_regs.as_ref() {
                    for g in 0..(k / 4) {
                        let (k_dw, k_lo, k_lane) =
                            input_loc(m, k, blocks, i, 4 * g, b, a_bits, true, true, cbsz, abid, 0);
                        let nib = ((k_mat[k_lane][k_dw] >> k_lo) & 0xF) as usize;
                        let idx0 = nib & 0b11;
                        let idx1 = (nib >> 2) & 0b11;

                        let (a0_dw, a0_lo, a0_lane) =
                            input_loc(m, k, blocks, i, 4 * g, b, a_bits, true, false, 0, 0, 0);
                        let (a1_dw, a1_lo, a1_lane) =
                            input_loc(m, k, blocks, i, 4 * g + 2, b, a_bits, true, false, 0, 0, 0);
                        let (b0_dw, b0_lo, b0_lane) = input_loc(
                            n,
                            k,
                            blocks,
                            j,
                            4 * g + idx0,
                            b,
                            b_bits,
                            false,
                            false,
                            0,
                            0,
                            0,
                        );
                        let (b1_dw, b1_lo, b1_lane) = input_loc(
                            n,
                            k,
                            blocks,
                            j,
                            4 * g + idx1,
                            b,
                            b_bits,
                            false,
                            false,
                            0,
                            0,
                            0,
                        );
                        av[2 * g] = decode_elem(&a_regs[a0_lane], form.a, a0_dw, a0_lo);
                        av[2 * g + 1] = decode_elem(&a_regs[a1_lane], form.a, a1_dw, a1_lo);
                        bv[2 * g] = decode_elem(&b_regs[b0_lane], form.b, b0_dw, b0_lo);
                        bv[2 * g + 1] = decode_elem(&b_regs[b1_lane], form.b, b1_dw, b1_lo);
                    }
                }
                let (av, bv, a_raw32, b_raw32, a_raw64, b_raw64) = if dense {
                    let (ra, rb) = (i * k..(i + 1) * k, j * k..(j + 1) * k);
                    (
                        &a_val[ra.clone()],
                        &b_val[rb.clone()],
                        &a_r32[ra.clone()],
                        &b_r32[rb.clone()],
                        &a_r64[ra],
                        &b_r64[rb],
                    )
                } else {
                    (
                        &av[..],
                        &bv[..],
                        &a_raw32[..],
                        &b_raw32[..],
                        &a_raw64[..],
                        &b_raw64[..],
                    )
                };

                if is_f64 {
                    let mut c_bits = u64::from(c_regs[d_lane][d_dw])
                        | (u64::from(c_regs[d_lane][d_dw + 1]) << 32);
                    if neg_f64[2] {
                        c_bits ^= 1u64 << 63;
                    }
                    let res = eval_f64_dot(a_raw64, b_raw64, c_bits);
                    out[d_dw][d_lane] = res as u32;
                    out[d_dw + 1][d_lane] = (res >> 32) as u32;
                } else if is_int {
                    let mut acc = c_regs[d_lane][d_dw] as i32 as i64;
                    for (&x, &y) in av.iter().zip(bv) {
                        acc = acc.wrapping_add((x as i64) * (y as i64));
                    }
                    out[d_dw][d_lane] = acc as i32 as u32;
                } else if a_ty == Elem::F32 && b_ty == Elem::F32 {
                    let c_bits = c_regs[d_lane][d_dw];
                    out[d_dw][d_lane] = eval_sgemm_fma(a_raw32, b_raw32, c_bits);
                } else {
                    let c_bits = c_regs[d_lane][d_dw];
                    out[d_dw][d_lane] = eval_xdlop_dot(av, bv, c_bits, dot_group);
                }
            }
        }
    }

    let full_mask = w.all();
    for d in 0..d_dwords {
        w.vwrite(dst, d, &out[d as usize], None, full_mask)?;
    }
    Ok(())
}

/// Checks one dense single-block 32-bit-accumulating matrix op (`wave64` MFMA or
/// `wave32` WMMA) against the active GEMM provenance checker (`!e.checker.products()`).
pub fn check_matrix(
    e: &mut Exec<'_>,
    w: &mut Wave,
    ins: &Ins,
    form: &MatrixForm,
) -> Result<(), EmuError> {
    if w.w != form.wave as usize {
        return Err(limit(format!(
            "{}: a matrix form of {} lanes on a {}-lane wave",
            ins.text, form.wave, w.w
        )));
    }
    if form.sparse
        || form.scaled
        || form.a == Elem::F8F6F4
        || form.b == Elem::F8F6F4
        || form.blocks != 1
        || form.m != form.n
        || form.d.bits() != 32
        || form.a.bits() < 8
        || form.b.bits() < 8
    {
        return Err(limit(format!(
            "{}: the GEMM checker does not support this matrix form",
            ins.text
        )));
    }
    if int_field(ins, "cbsz", 0) != 0
        || int_field(ins, "abid", 0) != 0
        || int_field(ins, "blgp", 0) != 0
        || ins.field("clamp").is_some()
        || ins.field("neg").is_some()
        || (0..3).any(|i| list_at(ins, "neg_lo", i, 0) != 0 || list_at(ins, "neg_hi", i, 0) != 0)
    {
        return Err(limit(format!(
            "{}: the GEMM checker does not support broadcast/neg/clamp modifiers",
            ins.text
        )));
    }

    let wave = w.w;
    let r = form.m as usize;
    let k = form.k as usize;
    let g = wave / r;
    let elems = k / g;
    let d_dwords = form.dwords('d') as usize;

    let hwk: Vec<Vec<i64>> = (0..g)
        .map(|q| (0..elems).map(|p| (q * elems + p) as i64).collect())
        .collect();
    let mut d_map = vec![vec![(0usize, 0usize); wave]; d_dwords];
    for i in 0..r {
        for j in 0..r {
            let (dw, lane) = if wave == 64 {
                output_loc(r, r, i, j, 0, 32)
            } else {
                (i % 8, j + 16 * (i / 8))
            };
            d_map[dw][lane] = (i, j);
        }
    }
    let tag_form = tags::MatrixForm {
        w: wave,
        r,
        k_hw: form.k as i64,
        d_map,
    };
    let fmt_a = OperandFormat {
        name: form.a.name().to_owned(),
        bytes: (form.dwords('a') * 4) as usize,
        elem_bytes: (form.a.bits() / 8) as usize,
        hwk: hwk.clone(),
    };
    let fmt_b = OperandFormat {
        name: form.b.name().to_owned(),
        bytes: (form.dwords('b') * 4) as usize,
        elem_bytes: (form.b.bits() / 8) as usize,
        hwk,
    };

    // Materialize `ACC_MARK` on `src0`/`src1` if a prior GEMM accumulator is fed directly.
    if let Some(op0) = ins.operands.get(1) {
        w.materialize_operand(e.tables, op0);
    }
    if let Some(op1) = ins.operands.get(2) {
        w.materialize_operand(e.tables, op1);
    }

    let read_byte_tags = |w: &Wave, k_op: usize, n_dw: u32| -> Result<Vec<Vec<Tag>>, EmuError> {
        let op = ins
            .operands
            .get(k_op)
            .ok_or_else(|| limit(format!("{}: operand {k_op} missing", ins.text)))?;
        match &op.base {
            Base::Reg(f, _, _) if f == "v" || f == "a" => {}
            _ => {
                return Err(limit(format!(
                    "{}: operand {k_op} (`{}`) is not a matrix register operand",
                    ins.text,
                    op.spelling()
                )))
            }
        }
        let mut out = vec![Vec::with_capacity((n_dw * 4) as usize); wave];
        for d in 0..n_dw {
            let s = w.vread(op, d)?;
            for (l, dst_lane) in out.iter_mut().enumerate() {
                dst_lane.extend_from_slice(&s.tag(l));
            }
        }
        Ok(out)
    };

    let s0_tags = read_byte_tags(w, 1, form.dwords('a'))?;
    let s1_tags = read_byte_tags(w, 2, form.dwords('b'))?;

    let dst = ins
        .operands
        .first()
        .ok_or_else(|| limit(format!("{}: missing destination", ins.text)))?;
    let (dst_file, dst_first) = match &dst.base {
        Base::Reg(f, first, _) if f == "v" || f == "a" => (f.clone(), *first),
        _ => {
            return Err(limit(format!(
                "{}: invalid destination `{}`",
                ins.text,
                dst.spelling()
            )))
        }
    };
    // Ensure dst range is within bounds, then get its base AccState entry.
    if (dst_first as usize) + d_dwords
        > if dst_file == "a" {
            w.at.len()
        } else {
            w.vt.len()
        }
    {
        return Err(limit(format!(
            "{}: destination `{}` exceeds register file",
            ins.text,
            dst.spelling()
        )));
    }
    let dst_entry = w.acc_index(&dst_file, dst_first as usize).ok_or_else(|| {
        limit(format!(
            "{}: invalid destination `{}`",
            ins.text,
            dst.spelling()
        ))
    })?;

    let src2 = ins
        .operands
        .get(3)
        .ok_or_else(|| limit(format!("{}: operand 3 missing", ins.text)))?;
    let mut c_zero = false;
    let mut marked = vec![false; d_dwords];
    match &src2.base {
        Base::Literal(0, _) => {
            c_zero = true;
        }
        Base::Literal(_, _) => {
            return Err(limit(format!(
                "{}: non-zero literal C (`{}`) is not supported by the GEMM checker",
                ins.text,
                src2.spelling()
            )));
        }
        Base::Reg(c_file, c_first, _) if c_file == "v" || c_file == "a" => {
            if wave == 64 && c_file != &dst_file {
                return Err(limit(format!(
                    "{}: accumulator src2 (`{}`) must match destination register file `{dst_file}`",
                    ins.text,
                    src2.spelling()
                )));
            }
            for (idx, m_flag) in marked.iter_mut().enumerate() {
                let c_reg = (*c_first as usize) + idx;
                let Some(c_acc_idx) = w.acc_index(c_file, c_reg) else {
                    return Err(limit(format!(
                        "{}: accumulator `{}` exceeds register file",
                        ins.text,
                        src2.spelling()
                    )));
                };
                if w.is_acc_marked(c_file, c_reg) {
                    if c_acc_idx != dst_entry + idx {
                        let cloned = w.acc.entry(c_acc_idx).clone();
                        *w.acc.entry(dst_entry + idx) = cloned;
                    }
                    *m_flag = true;
                } else {
                    let s = w.vread(src2, idx as u32)?;
                    let lanes = s.tags(wave);
                    if lanes.iter().flatten().any(|&t| t > 0) {
                        return Err(EmuError {
                            cause: Some(Cause::Fault),
                            message: format!(
                                "{}: accumulator C (`{}`) holds raw memory bytes",
                                ins.text,
                                src2.spelling()
                            ),
                        });
                    }
                    if lanes.iter().flatten().any(|&t| is_record(t)) {
                        *m_flag = w
                            .acc
                            .resume(e.tables, dst_entry + idx, lanes, &ins.text)
                            .map_err(tag_error)?;
                    }
                }
            }
        }
        _ => {
            return Err(limit(format!(
                "{}: invalid accumulator operand `{}`",
                ins.text,
                src2.spelling()
            )));
        }
    }

    let dst_name = dst.spelling();
    let mop = MatrixOp {
        form: &tag_form,
        text: &ins.text,
        dst: &dst_name,
        entry: dst_entry,
        marked,
        c_zero,
        src: [
            MatrixOperand {
                tags: &s0_tags,
                format: &fmt_a,
            },
            MatrixOperand {
                tags: &s1_tags,
                format: &fmt_b,
            },
        ],
    };
    e.checker
        .matrix(e.tables, &mop, &mut w.acc)
        .map_err(tag_error)?;

    let zeros = [0u32; MAX_W];
    let mark_lanes = [[ACC_MARK; 4]; MAX_W];
    let full_mask = w.all();
    for d in 0..d_dwords as u32 {
        w.vwrite(dst, d, &zeros, Some(&mark_lanes[..wave]), full_mask)?;
    }
    Ok(())
}
