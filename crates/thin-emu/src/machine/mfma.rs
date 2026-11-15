//! Wave64 MFMA and 4:2 SMFMAC execution (`D = A·B + C`) using the target's
//! matrix lane/register/block layout. Every lane participates regardless of
//! `EXEC`; value-mode outputs carry no provenance tags.

use thin_target::{Elem, MatrixForm};

use super::alu::{int_field, list_at};
use super::numbers::{self as num, f16_value, f32_of, Small};
use super::wave::{Wave, MAX_W};
use crate::{Base, Cause, EmuError, Ins};

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
    let elems_contig = k_eff_dim / (64 / (outer * blocks));
    let local = k_eff % elems_contig;
    let lane = blgp_lane(
        b_eff * outer + (k_eff / elems_contig) * (outer * blocks) + i,
        blgp,
    );
    if is_k {
        let mut slot = local;
        if cbsz == 0 {
            slot += (64 * abid) / bits;
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
        Elem::Iu8 => f64::from((dwords[dw] >> bit_lo) & 0xFF),
        Elem::Iu4 => f64::from((dwords[dw] >> bit_lo) & 0xF),
    }
}

/// Evaluates an `f32`-accumulating dot product `sum_k(a_k * b_k) + c` in IEEE RNE,
/// producing quieted `c_bits` when only `c` is NaN and default quiet NaN `0xFFC0_0000`
/// on factor NaN or invalid arithmetic (`0 * Inf`, `Inf - Inf`).
fn eval_f32_dot(av: &[f64], bv: &[f64], c_bits: u32) -> u32 {
    let cv = f64::from(f32_of(c_bits));
    if av.iter().chain(bv).any(|v| v.is_nan()) {
        return num::DNAN32;
    }
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    let mut sum = 0.0f64;
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
        sum += p;
    }
    if has_pos_inf && has_neg_inf {
        return num::DNAN32;
    }
    if num::is_nan32(c_bits) {
        return c_bits | 0x0040_0000;
    }
    let total = sum + cv;
    if total.is_nan() {
        return num::DNAN32;
    }
    if total == 0.0 {
        let neg = |x: f64| x == 0.0 && x.is_sign_negative();
        let all_neg = neg(cv) && av.iter().zip(bv).all(|(&x, &y)| neg(x * y));
        return if all_neg { 0x8000_0000 } else { 0 };
    }
    (total as f32).to_bits()
}

/// Evaluates an `f32`-input SGEMM MFMA (`v_mfma_f32_*_f32`) via sequential
/// single-rounded IEEE binary32 `mul_add` steps (`D = fma(A[k], B[k], acc)`).
fn eval_sgemm_fma(av: &[f64], bv: &[f64], c_bits: u32) -> u32 {
    for (&x, &y) in av.iter().zip(bv) {
        if x.is_nan()
            || y.is_nan()
            || (x == 0.0 && y.is_infinite())
            || (x.is_infinite() && y == 0.0)
        {
            return num::DNAN32;
        }
    }
    if num::is_nan32(c_bits) {
        return c_bits | 0x0040_0000;
    }
    let mut acc = f32_of(c_bits);
    for (&x, &y) in av.iter().zip(bv) {
        acc = (x as f32).mul_add(y as f32, acc);
        if acc.is_nan() {
            return num::DNAN32;
        }
    }
    acc.to_bits()
}

/// Evaluates an `f64`-accumulating MFMA dot product via fused `mul_add` steps.
fn eval_f64_dot(av: &[f64], bv: &[f64], c_bits: u64) -> u64 {
    const DNAN64: u64 = 0xFFF8_0000_0000_0000;
    const QUIET64: u64 = 0x0008_0000_0000_0000;
    for (&x, &y) in av.iter().zip(bv) {
        if x.is_nan()
            || y.is_nan()
            || (x == 0.0 && y.is_infinite())
            || (x.is_infinite() && y == 0.0)
        {
            return DNAN64;
        }
    }
    let mut acc = f64::from_bits(c_bits);
    if acc.is_nan() {
        return c_bits | QUIET64;
    }
    for (&x, &y) in av.iter().zip(bv) {
        acc = x.mul_add(y, acc);
        if acc.is_nan() {
            return DNAN64;
        }
    }
    acc.to_bits()
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
    let mut neg_f64 = [false; 3];

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
        let max_sets = i128::from(form.a.bits() / 4);
        if !(0..max_sets).contains(&abid_i) {
            return Err(limit(format!(
                "{}: abid {abid_i} out of range 0..{}",
                ins.text,
                max_sets - 1
            )));
        }
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

    let cbsz = cbsz_i as usize;
    let abid = abid_i as usize;
    let blgp = if is_f64 { 0 } else { blgp_i as usize };

    let d_dwords = form.dwords('d');
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

    let (_, a_regs) = reg_lanes(w, ins, 1, form.dwords('a'), false)?;
    let (_, b_regs) = reg_lanes(w, ins, 2, form.dwords('b'), false)?;
    let (c_regs, k_regs) = if form.sparse {
        let (_, c) = reg_lanes(w, ins, 0, d_dwords, false)?;
        let (_, k) = reg_lanes(w, ins, 3, 1, true)?;
        (c, Some(k))
    } else {
        let c = c_lanes(w, ins, d_dwords, form.d, &dst_file)?;
        (c, None)
    };

    let m = form.m as usize;
    let n = form.n as usize;
    let k = form.k as usize;
    let blocks = form.blocks as usize;
    let a_bits = form.a.bits() as usize;
    let b_bits = form.b.bits() as usize;
    let d_bits = form.d.bits() as usize;
    let is_int = form.d.integer();

    let mut out = [[0u32; MAX_W]; MAX_DWORDS];

    let active_terms = if form.sparse { k / 2 } else { k };
    let mut av = vec![0.0f64; active_terms];
    let mut bv = vec![0.0f64; active_terms];

    for b in 0..blocks {
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
                } else {
                    for kk in 0..k {
                        let (a_dw, a_lo, a_lane) =
                            input_loc(m, k, blocks, i, kk, b, a_bits, false, false, cbsz, abid, 0);
                        let (b_dw, b_lo, b_lane) =
                            input_loc(n, k, blocks, j, kk, b, b_bits, false, false, 0, 0, blgp);
                        let mut x = decode_elem(&a_regs[a_lane], form.a, a_dw, a_lo);
                        let mut y = decode_elem(&b_regs[b_lane], form.b, b_dw, b_lo);
                        if neg_f64[0] {
                            x = -x;
                        }
                        if neg_f64[1] {
                            y = -y;
                        }
                        av[kk] = x;
                        bv[kk] = y;
                    }
                }

                if is_f64 {
                    let mut c_bits = u64::from(c_regs[d_lane][d_dw])
                        | (u64::from(c_regs[d_lane][d_dw + 1]) << 32);
                    if neg_f64[2] {
                        c_bits ^= 1u64 << 63;
                    }
                    let res = eval_f64_dot(&av, &bv, c_bits);
                    out[d_dw][d_lane] = res as u32;
                    out[d_dw + 1][d_lane] = (res >> 32) as u32;
                } else if is_int {
                    let mut acc = c_regs[d_lane][d_dw] as i32 as i64;
                    for (&x, &y) in av.iter().zip(&bv) {
                        acc = acc.wrapping_add((x as i64) * (y as i64));
                    }
                    out[d_dw][d_lane] = acc as i32 as u32;
                } else if form.a == Elem::F32 && form.b == Elem::F32 {
                    let c_bits = c_regs[d_lane][d_dw];
                    out[d_dw][d_lane] = eval_sgemm_fma(&av, &bv, c_bits);
                } else {
                    let c_bits = c_regs[d_lane][d_dw];
                    out[d_dw][d_lane] = eval_f32_dot(&av, &bv, c_bits);
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
