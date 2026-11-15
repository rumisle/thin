//! The ALU semantics with their tag rules
//!

use super::numbers::{self as num, f32_of};
use super::wave::{Src, Wave, MAX_W};
use super::Exec;
use crate::tags::rules::{self, Ctx as TagCtx};
use crate::tags::{src as dword_src, Lane, TagError};
use crate::{Base, Cause, EmuError, Field, Ins, Operand};

fn fault(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Fault),
        message,
    }
}

fn limit(message: String) -> EmuError {
    EmuError {
        cause: Some(Cause::Emulator),
        message,
    }
}

/// A tag rule's failure as an emulator error.
pub fn tag_error(e: TagError) -> EmuError {
    EmuError {
        cause: Some(if e.limit {
            Cause::Emulator
        } else {
            Cause::Fault
        }),
        message: e.message,
    }
}

/// A list field's entry `i`, or `default`.
pub fn list_at(ins: &Ins, name: &str, i: usize, default: i128) -> i128 {
    match ins.field(name) {
        Some(Field::List(l)) => l.get(i).copied().unwrap_or(default),
        Some(Field::Int(v)) => *v,
        Some(Field::True) => 1,
        None => default,
    }
}

/// An integer field, or `default`.
pub fn int_field(ins: &Ins, name: &str, default: i128) -> i128 {
    match ins.field(name) {
        Some(Field::Int(v)) => *v,
        Some(Field::True) => 1,
        Some(Field::List(l)) => l.first().copied().unwrap_or(default),
        None => default,
    }
}

// ---------------------------------------------------------------- floats

fn f(u: u32) -> f32 {
    f32_of(u)
}

/// maximumNumber / minimumNumber's NaN cases: both NaN gives x quieted
/// (RDNA4 ISA guide V_MAX_NUM_F32: `cvtToQuietNAN(S0)`); one NaN,
/// the other operand.
fn num_nans(x: u32, y: u32) -> Option<u32> {
    match (num::is_nan32(x), num::is_nan32(y)) {
        (true, true) => Some(num::nan32(f32::NAN, &[x, y])),
        (true, false) => Some(y),
        (false, true) => Some(x),
        _ => None,
    }
}

/// `max(x, y)`: IEEE maximumNumber, +0 above −0 (RDNA4 ISA guide
/// V_MAX_NUM_F32); every f32 max is this one, three-operand and DPP forms
/// included.
pub fn fmax(x: u32, y: u32) -> u32 {
    if let Some(r) = num_nans(x, y) {
        return r;
    }
    let (a, b) = (f(x), f(y));
    if a > b || (a == b && a == 0.0 && !a.is_sign_negative()) {
        x
    } else {
        y
    }
}

/// `min(x, y)`: IEEE minimumNumber, −0 below +0.
pub fn fmin(x: u32, y: u32) -> u32 {
    if let Some(r) = num_nans(x, y) {
        return r;
    }
    let (a, b) = (f(x), f(y));
    if a < b || (a == b && a == 0.0 && a.is_sign_negative()) {
        x
    } else {
        y
    }
}

/// A float compare by its mnemonic's kind: the
/// ordered ones are false for a NaN operand (lg too: less or greater),
/// their `n` negations and `neq` true; `o` / `u` ordered / unordered.
/// None for an unknown kind.
pub fn float_compare(kind: &str, a: f64, b: f64) -> Option<bool> {
    let ordered = |k: &str| match k {
        "eq" => Some(a == b),
        "lg" => Some(a < b || b < a),
        "lt" => Some(a < b),
        "le" => Some(a <= b),
        "gt" => Some(a > b),
        "ge" => Some(a >= b),
        _ => None,
    };
    Some(match kind {
        "o" => !a.is_nan() && !b.is_nan(),
        "u" => a.is_nan() || b.is_nan(),
        "neq" => !ordered("eq")?,
        k => match k.strip_prefix('n') {
            Some(rest) if !rest.is_empty() => !ordered(rest)?,
            _ => ordered(k)?,
        },
    })
}

/// `v_med3_num_f32`, in the guide's order: any
/// NaN → min3_num; else with m = max3_num: m == x → max_num(y, z), m == y
/// → max_num(x, z), else max_num(x, y). The equalities compare bits, so −0
/// is not +0 (measured on gfx1201: med3(−4, −0, +0) = −0).
fn med3_num(x: u32, y: u32, z: u32) -> u32 {
    if [x, y, z].iter().any(|v| num::is_nan32(*v)) {
        return fmin(fmin(x, y), z);
    }
    let m = fmax(fmax(x, y), z);
    if m == x {
        fmax(y, z)
    } else if m == y {
        fmax(x, z)
    } else {
        fmax(x, y)
    }
}

/// IEEE minimum (`max`: maximum) (RDNA4 ISA guide
/// V_MINIMUM3_F32): NaN-propagating, the first NaN of x, y quieted whether
/// signaling or not (measured on gfx1201); −0 < +0; a tie gives y.
fn minimum(x: u32, y: u32, max: bool) -> u32 {
    let (a, b) = (f(x), f(y));
    if a.is_nan() || b.is_nan() {
        return num::nan32(f32::NAN, &[x, y]);
    }
    let less = a < b || (a == 0.0 && b == 0.0 && a.is_sign_negative() && !b.is_sign_negative());
    let more = a > b || (a == 0.0 && b == 0.0 && !a.is_sign_negative() && b.is_sign_negative());
    if (!max && less) || (max && more) {
        x
    } else {
        y
    }
}

/// fma(x, y, z): one RNE rounding of the exact x·y + z (RDNA4 ISA guide
/// V_FMA_F32 "fused"; measured on gfx1201 over binary32 midpoints,
/// tests/hw/test_emu_ops.py `test_a_fused_op_rounds_as_the_gpu_does`), its
/// NaN by [`num::fma_nan32`].
pub fn fma(x: u32, y: u32, z: u32) -> u32 {
    num::fma_nan32(f(x).mul_add(f(y), f(z)), x, y, z)
}

/// `v_dot2_f32_bf16`: D = S0.lo · S1.lo + S0.hi · S1.hi + S2; one step of
/// gfx1201's dot unit ([`super::dot::step`]).
fn dot2_bf16(x: u32, y: u32, z: u32) -> u32 {
    super::dot::step(&[x, x >> 16], &[y, y >> 16], z, true)
}

fn ldexp(x: u32, e: u32) -> u32 {
    let e = (e as i32).clamp(-400, 400);
    let v = f64::from(f(x)) * 2f64.powi(e);
    num::nan32(v as f32, &[x])
}

fn frexp(x: u32) -> (u32, u32) {
    let a = f(x);
    if a.is_nan() {
        return (num::nan32(a, &[x]), 0);
    }
    if a == 0.0 || a.is_infinite() {
        return (x, 0);
    }
    let d = f64::from(a);
    let bits = d.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32 - 1022;
    let m = d / 2f64.powi(exp);
    ((m as f32).to_bits(), exp as u32)
}

fn rint_even(a: f32) -> f32 {
    a.round_ties_even()
}

/// A unary f32 op: a NaN source propagates quieted, an invalid one the
/// default NaN ([`num::nan32`]). exp, log, rcp, sqrt and rsq are gfx1201's
/// transcendental unit, bit for bit ([`super::trans32`]).
fn unary_f32(name: &str, x: u32) -> Option<u32> {
    use super::trans32 as t;
    let a = f(x);
    let r = match name {
        "exp" => return Some(t::exp(x)),
        "log" => return Some(t::log(x)),
        "rcp" => return Some(t::rcp(x)),
        "sqrt" => return Some(t::sqrt(x)),
        "rsq" => return Some(t::rsq(x)),
        "ceil" => a.ceil(),
        "floor" => a.floor(),
        "rndne" => rint_even(a),
        "trunc" => a.trunc(),
        "frexp.mant" => return Some(frexp(x).0),
        "frexp.exp" => return Some(frexp(x).1),
        _ => return None,
    };
    Some(num::nan32(r, &[x]))
}

// ---------------------------------------------------------------- helpers

/// The tag context of a rule.
fn tctx<'a>(e: &'a mut Exec<'_>) -> TagCtx<'a> {
    TagCtx {
        tables: e.tables,
        sum_mode: e.sum_mode,
    }
}

fn zeros(w: usize) -> Vec<Lane> {
    vec![[0; 4]; w]
}

/// The integer rule of two sources, with a byte rule's
/// result when there is one.
fn int_rule(
    e: &mut Exec<'_>,
    w: usize,
    x: &Src,
    y: &Src,
    bytes: Option<Vec<Lane>>,
    shift: bool,
) -> Result<Option<Vec<Lane>>, EmuError> {
    if !x.tagged && !y.tagged {
        return Ok(None);
    }
    if !rules::has_records(x.tags(w)) && !rules::has_records(y.tags(w)) {
        return Ok(bytes);
    }
    let mut c = tctx(e);
    rules::integer(&mut c, x.tags(w), y.tags(w), bytes, shift)
        .map(Some)
        .map_err(tag_error)
}

/// The float rule of two sources.
fn float_rule(
    e: &mut Exec<'_>,
    w: usize,
    x: &[Lane],
    y: &[Lane],
    stat: bool,
    add: bool,
) -> Result<Option<Vec<Lane>>, EmuError> {
    let none = |t: &[Lane]| t.iter().all(|l| *l == [0; 4]);
    if none(x) && none(y) {
        return Ok(None);
    }
    if !e.sum_mode {
        let below = |t: &[Lane]| t.iter().any(|l| l.iter().any(|b| *b < -1));
        if !below(x) && !below(y) {
            return Ok(None);
        }
    }
    let _ = w;
    let mut c = tctx(e);
    rules::float(&mut c, x, y, stat, add)
        .map(Some)
        .map_err(tag_error)
}

/// A register operand moved by an M0 index (the movrel ops); an index
/// past the wave's file is the program's fault.
fn relative(wv: &Wave, o: &Operand, by: u32) -> Result<Operand, EmuError> {
    match &o.base {
        Base::Reg(f, r, n) if f == "v" || f == "s" => {
            let size = if f == "v" { wv.vt.len() } else { wv.s.len() };
            let at = r.checked_add(by).filter(|&at| (at + n) as usize <= size);
            let at = at.ok_or_else(|| {
                fault(format!(
                    "{}: relative index {by} is past the {size}-register file",
                    o.spelling()
                ))
            })?;
            Ok(Operand {
                base: Base::Reg(f.clone(), at, *n),
                neg: o.neg,
                abs: o.abs,
            })
        }
        _ => Err(fault(format!(
            "{}: relative indexing needs a numbered register",
            o.spelling()
        ))),
    }
}

fn copy_tags(x: &Src, w: usize) -> Option<Vec<Lane>> {
    x.tagged.then(|| x.tags(w).to_vec())
}

fn widen16(t: &Lane, half: usize) -> Lane {
    let (a, b) = (t[2 * half], t[2 * half + 1]);
    [a, b, a, b]
}

fn mask_bit(m: u64, l: usize) -> bool {
    m >> l & 1 != 0
}

// ------------------------------------------------------------ the ALU ops

/// Runs a VALU op whose canon is `canon` (no memory); `None` when the
/// canon is not an ALU op of this module.
pub fn valu(
    e: &mut Exec<'_>,
    wv: &mut Wave,
    ins: &Ins,
    canon: &str,
) -> Option<Result<(), EmuError>> {
    let r = valu_inner(e, wv, ins, canon);
    match r {
        Ok(true) => Some(Ok(())),
        Ok(false) => None,
        // A mask read or write of vcc_hi or exec_hi: "TEXT:...".
        Err(mut err)
            if err.message == super::wave::NOT_MASK
                || err.message == super::wave::NOT_MASK_EXEC =>
        {
            err.message = format!("{}: {}", ins.text, err.message);
            Some(Err(err))
        }
        Err(err) => Some(Err(err)),
    }
}

/// Vector operations used by shipped kernels (see docs/dev/lift.md): untagged
/// results; None for any other canon.
fn shipped_valu(
    wv: &mut Wave,
    ins: &Ins,
    canon: &str,
    exec: u64,
) -> Result<Option<bool>, EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    let arg = |k: usize| {
        ops.get(k)
            .ok_or_else(|| fault(format!("{}: operand {k} missing", ins.text)))
    };
    let mut out = vec![0u32; w];
    match canon {
        // opcode 583: D = (x + y) << z[4:0].
        "v.add.lshl.u32" => {
            let (x, y, z) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            for l in 0..w {
                out[l] = x.v[l].wrapping_add(y.v[l]) << (z.v[l] & 31);
            }
        }
        // opcode 806: D = x + y as i32, no carry.
        "v.add.nc.i32" => {
            if ins.field("clamp").is_some() {
                return Err(limit(format!("{}: clamp not modeled", ins.text)));
            }
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            for l in 0..w {
                out[l] = x.v[l].wrapping_add(y.v[l]);
            }
        }
        // VOP1 opcode 43, VOP3 427: the value of v.rcp.f32.
        "v.rcp.iflag.f32" => {
            let x = wv.vread(arg(1)?, 0)?;
            for l in 0..w {
                out[l] = unary_f32("rcp", x.v[l]).expect("rcp is a unary f32 op");
            }
        }
        _ => return Ok(None),
    }
    wv.vwrite(&ops[0], 0, &out, None, exec)?;
    Ok(Some(true))
}

/// Scalar operations used by shipped kernels; false for any other canon.
fn shipped_salu(wv: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let ops = &ins.operands;
    let r = |wv: &Wave, k: usize| -> Result<u32, EmuError> {
        let o = ops
            .get(k)
            .ok_or_else(|| fault(format!("{}: operand {k} missing", ins.text)))?;
        Ok(wv.sread(o)? as u32)
    };
    let d = match canon {
        // SOP1 opcode 2: if SCC, D = x; SCC unchanged.
        "s.cmov.b32" => {
            if wv.scc {
                let x = r(wv, 1)?;
                wv.swrite(&ops[0], u64::from(x))?;
            }
            return Ok(true);
        }
        // SOP2 opcode 46: the high half of the signed 64-bit product.
        "s.mul.hi.i32" => {
            let p = i64::from(r(wv, 1)? as i32) * i64::from(r(wv, 2)? as i32);
            (p >> 32) as u32
        }
        // SOP1 opcode 21: |x| as i32; SCC = D != 0.
        "s.abs.i32" => {
            let v = (r(wv, 1)? as i32).wrapping_abs() as u32;
            wv.scc = v != 0;
            v
        }
        // SOP2 opcode 3: D = x - y; SCC = signed overflow.
        "s.sub.co.i32" => {
            let (x, y) = (r(wv, 1)?, r(wv, 2)?);
            let v = x.wrapping_sub(y);
            wv.scc = (x >> 31 != y >> 31) && (x >> 31 != v >> 31);
            v
        }
        // SOP2 opcode 42: ((1 << x[4:0]) - 1) << y[4:0].
        "s.bfm.b32" => {
            let (x, y) = (r(wv, 1)?, r(wv, 2)?);
            ((1u32 << (x & 31)).wrapping_sub(1)) << (y & 31)
        }
        // SOP1 opcode 15: x's low 16 bits sign-extended.
        "s.sext.i32.i16" => i32::from(r(wv, 1)? as u16 as i16) as u32,
        _ => return Ok(false),
    };
    wv.swrite(&ops[0], u64::from(d))?;
    Ok(true)
}

fn binop_int(name: &str) -> Option<fn(u32, u32) -> u32> {
    Some(match name {
        "v.add.u32" => |x, y| x.wrapping_add(y),
        "v.sub.u32" => |x, y| x.wrapping_sub(y),
        "v.subrev.u32" => |x, y| y.wrapping_sub(x),
        "v.xor.b32" => |x, y| x ^ y,
        "v.mul.lo.u32" => |x, y| x.wrapping_mul(y),
        "v.min.i32" => |x, y| (x as i32).min(y as i32) as u32,
        "v.max.i32" => |x, y| (x as i32).max(y as i32) as u32,
        "v.min.u32" => |x, y| x.min(y),
        "v.max.u32" => |x, y| x.max(y),
        "v.ashrrev.i32" => |x, y| ((y as i32) >> (x & 31)) as u32,
        _ => return None,
    })
}

fn untagged_int2(name: &str) -> Option<fn(u32, u32, usize) -> u32> {
    Some(match name {
        "v.mul.hi.u32" => |x, y, _| ((u64::from(x) * u64::from(y)) >> 32) as u32,
        "v.bfm.b32" => |x, y, _| ((1u64 << (x & 31)) - 1).wrapping_shl(y & 31) as u32,
        "v.bcnt.u32.b32" => |x, y, _| x.count_ones().wrapping_add(y),
        "v.mbcnt.lo.u32.b32" => |x, y, l| {
            let n = l.min(32) as u32;
            let m = if n == 0 {
                0
            } else {
                x & (u32::MAX >> (32 - n))
            };
            m.count_ones().wrapping_add(y)
        },
        "v.mbcnt.hi.u32.b32" => |x, y, l| {
            let n = l.saturating_sub(32).min(32) as u32;
            let m = if n == 0 {
                0
            } else {
                x & (u32::MAX >> (32 - n))
            };
            m.count_ones().wrapping_add(y)
        },
        _ => return None,
    })
}

/// A float op with its rule's (stat, add).
type FloatOp = (fn(u32, u32) -> u32, bool, bool);

/// The two-source f32 arithmetic. A NaN by [`num::nan32`] with the operands
/// in source order, a subtraction's negated one negated (measured: v_sub's
/// NaN subtrahend and v_subrev's NaN src0 come out sign-flipped).
fn float2(name: &str) -> Option<FloatOp> {
    const S: u32 = 0x8000_0000;
    Some(match name {
        "v.add.f32" => (|x, y| num::nan32(f(x) + f(y), &[x, y]), false, true),
        "v.sub.f32" => (|x, y| num::nan32(f(x) - f(y), &[x, y ^ S]), false, true),
        "v.subrev.f32" => (|x, y| num::nan32(f(y) - f(x), &[x ^ S, y]), false, true),
        "v.mul.f32" => (|x, y| num::nan32(f(x) * f(y), &[x, y]), false, false),
        _ => return None,
    })
}

#[allow(clippy::too_many_lines)]
fn valu_inner(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    let exec = wv.exec;
    let mut out = [0u32; MAX_W];
    let arg = |k: usize| {
        ops.get(k)
            .ok_or_else(|| fault(format!("{}: operand {k} missing", ins.text)))
    };

    if let Some(op) = binop_int(canon) {
        let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
        for l in 0..w {
            out[l] = op(x.v[l], y.v[l]);
        }
        let t = int_rule(e, w, &x, &y, None, false)?;
        wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        return Ok(true);
    }
    if let Some(done) = shipped_valu(wv, ins, canon, exec)? {
        return Ok(done);
    }
    if let Some(op) = untagged_int2(canon) {
        let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
        for l in 0..w {
            out[l] = op(x.v[l], y.v[l], l);
        }
        wv.vwrite(&ops[0], 0, &out, None, exec)?;
        return Ok(true);
    }
    if let Some((op, stat, add)) = float2(canon) {
        let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
        for l in 0..w {
            out[l] = op(x.v[l], y.v[l]);
        }
        let t = float_rule(e, w, x.tags(w), y.tags(w), stat, add)?;
        wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        return Ok(true);
    }
    match canon {
        "v.max.f32" | "v.min.f32" | "v.max.num.f32" | "v.min.num.f32" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let max = canon.starts_with("v.max");
            for l in 0..w {
                out[l] = if max {
                    fmax(x.v[l], y.v[l])
                } else {
                    fmin(x.v[l], y.v[l])
                };
            }
            let t = float_rule(e, w, x.tags(w), y.tags(w), true, false)?;
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.and.b32" | "v.or.b32" | "v.lshlrev.b32" | "v.lshrrev.b32" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let mut bytes = zeros(w);
            for l in 0..w {
                let (a, b) = (x.v[l], y.v[l]);
                match canon {
                    "v.and.b32" => {
                        out[l] = a & b;
                        bytes[l] = rules::and_lane(a, &x.tag(l), b, &y.tag(l));
                    }
                    "v.or.b32" => {
                        out[l] = a | b;
                        bytes[l] = rules::or_lane(a, &x.tag(l), b, &y.tag(l));
                    }
                    "v.lshlrev.b32" => {
                        out[l] = b << (a & 31);
                        bytes[l] = rules::shift_lane(&y.tag(l), a, true);
                    }
                    _ => {
                        out[l] = b >> (a & 31);
                        bytes[l] = rules::shift_lane(&y.tag(l), a, false);
                    }
                }
            }
            let shift = canon.contains("sh");
            let t = if !x.tagged && !y.tagged {
                None
            } else if shift {
                // For shifts only the shifted value counts.
                let amount = Src::splat(0);
                int_rule(e, w, &y, &amount, Some(bytes), true)?
            } else {
                int_rule(e, w, &x, &y, Some(bytes), false)?
            };
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        // RDNA4 ISA guide V_MUL_U32_U24 / V_MAD_U32_U24: the
        // sources' low 24 bits, the high byte ignored (measured on gfx1201:
        // 2048 of 2048 lanes with high bytes set, tests/hw/test_emu_ops.py).
        "v.mul.u32.u24" | "v.mad.u32.u24" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let mad = canon.starts_with("v.mad");
            let z = if mad {
                Some(wv.vread(arg(3)?, 0)?)
            } else {
                None
            };
            for l in 0..w {
                out[l] = (x.v[l] & 0xFF_FFFF).wrapping_mul(y.v[l] & 0xFF_FFFF);
                if let Some(z) = &z {
                    out[l] = out[l].wrapping_add(z.v[l]);
                }
            }
            let t = match &z {
                Some(_) => None,
                None => int_rule(e, w, &x, &y, None, false)?,
            };
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        // RDNA4 ISA guide V_AND_OR_B32: D = (S0 & S1) | S2. The
        // tags are and of x and y, then or of that with z.
        "v.and.or.b32" => {
            let (x, y, z) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            let mut m = Src::splat(0);
            let mut bytes = zeros(w);
            for l in 0..w {
                m.v[l] = x.v[l] & y.v[l];
                bytes[l] = rules::and_lane(x.v[l], &x.tag(l), y.v[l], &y.tag(l));
            }
            if let Some(t) = int_rule(e, w, &x, &y, Some(bytes), false)? {
                m.t = t;
                m.tagged = true;
            }
            let mut bytes = zeros(w);
            for l in 0..w {
                out[l] = m.v[l] | z.v[l];
                bytes[l] = rules::or_lane(m.v[l], &m.tag(l), z.v[l], &z.tag(l));
            }
            let t = int_rule(e, w, &m, &z, Some(bytes), false)?;
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        // RDNA4 ISA guide V_MUL_I32_I24: D = 32'I(S0.i24) *
        // 32'I(S1.i24), the sources' low 24 bits sign-extended; the product's
        // low 32 bits. Tags as v_mul_u32_u24's.
        "v.mul.i32.i24" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let i24 = |v: u32| (v << 8) as i32 >> 8;
            for l in 0..w {
                out[l] = i24(x.v[l]).wrapping_mul(i24(y.v[l])) as u32;
            }
            let t = int_rule(e, w, &x, &y, None, false)?;
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.lshl.add.u32" | "v.lshl.or.b32" | "v.add3.u32" => {
            let (x, y, z) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            for l in 0..w {
                out[l] = match canon {
                    "v.lshl.add.u32" => (x.v[l] << (y.v[l] & 31)).wrapping_add(z.v[l]),
                    "v.lshl.or.b32" => (x.v[l] << (y.v[l] & 31)) | z.v[l],
                    _ => x.v[l].wrapping_add(y.v[l]).wrapping_add(z.v[l]),
                };
            }
            wv.vwrite(&ops[0], 0, &out, None, exec)?;
        }
        "v.med3.i32" | "v.max3.i32" | "v.min3.i32" => {
            let (x, y, z) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            for l in 0..w {
                let mut v = [x.v[l] as i32, y.v[l] as i32, z.v[l] as i32];
                out[l] = match canon {
                    "v.max3.i32" => v[0].max(v[1]).max(v[2]) as u32,
                    "v.min3.i32" => v[0].min(v[1]).min(v[2]) as u32,
                    _ => {
                        v.sort_unstable();
                        v[1] as u32
                    }
                };
            }
            wv.vwrite(&ops[0], 0, &out, None, exec)?;
        }
        // v.bfe.i32: RDNA4 ISA guide V_BFE_I32, tmp = (S0.i32 >>
        // S1[4:0]) & ((1 << S2[4:0]) - 1), D = signext_from_bit(tmp,
        // S2[4:0]) (width 0: 0); the field's bytes take the unsigned
        // extract's tags, the sign bytes above it none.
        "v.bfe.u32" | "v.bfe.i32" => {
            let (x, y, z) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            let signed = canon == "v.bfe.i32";
            let mut t = zeros(w);
            for l in 0..w {
                let (off, width) = (y.v[l] & 31, z.v[l] & 31);
                let shifted = if signed {
                    ((x.v[l] as i32) >> off) as u32
                } else {
                    x.v[l] >> off
                };
                out[l] = if width == 0 {
                    0
                } else if signed {
                    ((shifted << (32 - width)) as i32 >> (32 - width)) as u32
                } else {
                    shifted & ((1u64 << width) - 1) as u32
                };
                t[l] = rules::bfe_lane(&x.tag(l), y.v[l], z.v[l]);
            }
            let t = x.tagged.then_some(t);
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        // RDNA4 ISA guide Table 40 and V_MOVREL*: the raw
        // register number plus M0 (unsigned; MOVRELSD_2 takes M0[9:0] for
        // the source and M0[25:16] for the destination). Past the file is
        // a fault, as any register past it is.
        "v.movrels.b32" | "v.movreld.b32" | "v.movrelsd.b32" | "v.movrelsd.2.b32" => {
            let m0 = wv.m0;
            let (by_src, by_dst) = match canon {
                "v.movrels.b32" => (Some(m0), 0),
                "v.movreld.b32" => (None, m0),
                "v.movrelsd.b32" => (Some(m0), m0),
                _ => (Some(m0 & 0x3ff), m0 >> 16 & 0x3ff),
            };
            let src = match by_src {
                Some(by) => relative(wv, arg(1)?, by)?,
                None => arg(1)?.clone(),
            };
            let x = wv.vread(&src, 0)?;
            out[..w].copy_from_slice(&x.v[..w]);
            let dst = relative(wv, &ops[0], by_dst)?;
            wv.vwrite(&dst, 0, &out, copy_tags(&x, w).as_deref(), exec)?;
        }
        "v.mov.b32"
        | "v.accvgpr.read.b32"
        | "v.accvgpr.write.b32"
        | "v.accvgpr.mov.b32"
        | "v.not.b32" => {
            let x = wv.vread(arg(1)?, 0)?;
            for l in 0..w {
                out[l] = if canon == "v.not.b32" {
                    !x.v[l]
                } else {
                    x.v[l]
                };
            }
            wv.vwrite(&ops[0], 0, &out, copy_tags(&x, w).as_deref(), exec)?;
        }
        "v.cndmask.b32" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let m = wv.sread64(arg(3)?)?;
            let mut t = zeros(w);
            for l in 0..w {
                let pick = mask_bit(m, l);
                out[l] = if pick { y.v[l] } else { x.v[l] };
                t[l] = if pick { y.tag(l) } else { x.tag(l) };
            }
            let t = (x.tagged || y.tagged).then_some(t);
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.perm.b32" => {
            let (x, y, s) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            let mut t = zeros(w);
            for l in 0..w {
                let bytes: [u8; 8] = {
                    let (yb, xb) = (y.v[l].to_le_bytes(), x.v[l].to_le_bytes());
                    [yb[0], yb[1], yb[2], yb[3], xb[0], xb[1], xb[2], xb[3]]
                };
                let mut r = 0u32;
                for k in 0..4 {
                    let sel = (s.v[l] >> (8 * k)) & 0xFF;
                    let b = match sel {
                        0..=7 => u32::from(bytes[sel as usize]),
                        8..=11 => {
                            let src = bytes[(2 * (sel - 8) + 1) as usize];
                            if src & 0x80 != 0 {
                                0xFF
                            } else {
                                0
                            }
                        }
                        12 => 0,
                        _ => 0xFF,
                    };
                    r |= b << (8 * k);
                }
                out[l] = r;
                t[l] = rules::perm_lane(&x.tag(l), &y.tag(l), s.v[l]);
            }
            let t = (x.tagged || y.tagged).then_some(t);
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.pk.add.u16" | "v.pk.add.i16" | "v.pk.max.i16" => {
            let sel_ok = (0..2).all(|i| list_at(ins, "op_sel", i, 0) == 0)
                && (0..2).all(|i| list_at(ins, "op_sel_hi", i, 1) == 1)
                && ins.field("clamp").is_none();
            if !sel_ok {
                return Err(limit(format!("{}: op_sel / clamp not modelled", ins.text)));
            }
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            for l in 0..w {
                let mut r = 0u32;
                for h in 0..2 {
                    let (a, b) = ((x.v[l] >> (16 * h)) as u16, (y.v[l] >> (16 * h)) as u16);
                    let v = if canon == "v.pk.max.i16" {
                        (a as i16).max(b as i16) as u16
                    } else {
                        a.wrapping_add(b)
                    };
                    r |= u32::from(v) << (16 * h);
                }
                out[l] = r;
            }
            let t = if rules::has_records(x.tags(w)) || rules::has_records(y.tags(w)) {
                float_rule(e, w, x.tags(w), y.tags(w), false, false)?
            } else {
                None
            };
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.add.co.u32" | "v.addc.co.u32" | "v.subbrev.co.u32" => {
            let (x, y) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(3)?, 0)?);
            let cin = if canon == "v.add.co.u32" {
                0
            } else {
                wv.sread64(arg(4)?)?
            };
            let mut carry = 0u64;
            for l in 0..w {
                let c = u64::from(mask_bit(cin, l));
                let (r, wrapped) = if canon == "v.subbrev.co.u32" {
                    let r = i64::from(y.v[l]) - i64::from(x.v[l]) - c as i64;
                    (r as u32, r < 0)
                } else {
                    let r = u64::from(x.v[l]) + u64::from(y.v[l]) + c;
                    (r as u32, r >= 1 << 32)
                };
                out[l] = r;
                // The carry / borrow-out is a lane mask: 0 in inactive lanes.
                if wrapped && mask_bit(exec, l) {
                    carry |= 1 << l;
                }
            }
            let t = if canon == "v.subbrev.co.u32" {
                copy_tags(&y, w)
            } else {
                None
            };
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
            wv.swrite64(&ops[1], carry)?;
        }
        "v.sub.co.u32" | "v.subrev.co.u32" | "v.subb.co.u32" => {
            // gfx1201's vector borrow chain: the
            // borrow-out a full mask, 0 in inactive lanes, untagged.
            let (x, y) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(3)?, 0)?);
            let bin = if canon == "v.subb.co.u32" {
                wv.sread64(arg(4)?)?
            } else {
                0
            };
            let mut borrow = 0u64;
            for l in 0..w {
                let b = u64::from(mask_bit(bin, l));
                let (a, c) = if canon == "v.subrev.co.u32" {
                    (y.v[l], x.v[l])
                } else {
                    (x.v[l], y.v[l])
                };
                out[l] = a.wrapping_sub(c).wrapping_sub(b as u32);
                if u64::from(c) + b > u64::from(a) && mask_bit(exec, l) {
                    borrow |= 1 << l;
                }
            }
            wv.vwrite(&ops[0], 0, &out, None, exec)?;
            wv.swrite64(&ops[1], borrow)?;
        }
        "v.lshlrev.b64" | "v.lshrrev.b64" | "v.ashrrev.i64" => {
            let x = wv.vread(arg(1)?, 0)?;
            let (lo, hi) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(2)?, 1)?);
            let mut out2 = [0u32; MAX_W];
            for l in 0..w {
                let v = u64::from(lo.v[l]) | u64::from(hi.v[l]) << 32;
                let s = x.v[l] & 63;
                let r = match canon {
                    "v.lshlrev.b64" => v << s,
                    "v.lshrrev.b64" => v >> s,
                    _ => ((v as i64) >> s) as u64,
                };
                out[l] = r as u32;
                out2[l] = (r >> 32) as u32;
            }
            wv.vwrite(&ops[0], 0, &out, None, exec)?;
            wv.vwrite(&ops[0], 1, &out2, None, exec)?;
        }
        "v.readfirstlane.b32" | "v.readlane.b32" => {
            let x = wv.vread(arg(1)?, 0)?;
            let lane = if canon == "v.readfirstlane.b32" {
                if exec == 0 {
                    0
                } else {
                    exec.trailing_zeros() as usize
                }
            } else {
                (wv.sread(arg(2)?)? as usize) & (w - 1)
            };
            wv.swrite(&ops[0], u64::from(x.v[lane]))?;
        }
        "v.writelane.b32" => {
            let s = wv.vread(arg(1)?, 0)?;
            let lane = (wv.sread(arg(2)?)? as usize) & (w - 1);
            wv.vwrite(&ops[0], 0, &s.v, None, 1 << lane)?;
        }
        "v.permlane16.b32" => permlane16(wv, ins, false)?,
        "v.permlanex16.b32" => permlane16(wv, ins, true)?,
        "v.lshlrev.b16" | "v.max.num.f16" | "v.pack.b32.f16" => half16(e, wv, ins, canon)?,
        "v.exp.f16" | "v.log.f16" | "v.rcp.f16" | "v.sqrt.f16" | "v.rsq.f16" => {
            trans16(e, wv, ins, canon)?;
        }
        "v.fma.mix.f32" => fma_mix(e, wv, ins)?,
        "v.mul.f64" | "v.fma.f64" => f64_op(e, wv, ins, canon)?,
        "v.pk.fma.f32" | "v.pk.mul.f32" | "v.pk.add.f32" | "v.pk.mov.b32" => {
            pk_f32_op(e, wv, ins, canon)?;
        }
        c if c.ends_with(".dpp") => {
            if !dpp(e, wv, ins, c)? {
                return Ok(false);
            }
        }
        "v.fma.f32" | "v.fmac.f32" | "v.dot2.f32.bf16" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let z = if canon == "v.fma.f32" || canon == "v.dot2.f32.bf16" {
                wv.vread(arg(3)?, 0)?
            } else {
                wv.vread(&ops[0], 0)?
            };
            for l in 0..w {
                out[l] = if canon.contains("dot2") {
                    dot2_bf16(x.v[l], y.v[l], z.v[l])
                } else {
                    fma(x.v[l], y.v[l], z.v[l])
                };
            }
            let t = match float_rule(e, w, x.tags(w), y.tags(w), false, false)? {
                Some(p) => float_rule(e, w, &p, z.tags(w), false, true)?,
                None => float_rule(e, w, &zeros(w), z.tags(w), false, true)?,
            };
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.max3.f32" | "v.min3.f32" | "v.med3.f32" | "v.max3.num.f32" | "v.min3.num.f32"
        | "v.med3.num.f32" | "v.minimum3.f32" | "v.maximum3.f32" => {
            let (x, y, z) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            for l in 0..w {
                let (a, b, c) = (x.v[l], y.v[l], z.v[l]);
                out[l] = match canon {
                    "v.minimum3.f32" => minimum(minimum(a, b, false), c, false),
                    "v.maximum3.f32" => minimum(minimum(a, b, true), c, true),
                    c3 if c3.starts_with("v.max3") => fmax(fmax(a, b), c),
                    c3 if c3.starts_with("v.min3") => fmin(fmin(a, b), c),
                    _ => med3_num(a, b, c),
                };
            }
            let t = match float_rule(e, w, x.tags(w), y.tags(w), true, false)? {
                Some(p) => float_rule(e, w, &p, z.tags(w), true, false)?,
                None => float_rule(e, w, &zeros(w), z.tags(w), true, false)?,
            };
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        "v.pk.mul.f16" => {
            if ins.field("neg_lo").is_some() || ins.field("neg_hi").is_some() {
                return Err(limit(format!("{}: neg_lo / neg_hi not modelled", ins.text)));
            }
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let mut t = zeros(w);
            let mut any = false;
            for half in 0..2 {
                let (sx, sy) = if half == 0 {
                    (
                        list_at(ins, "op_sel", 0, 0) as usize & 1,
                        list_at(ins, "op_sel", 1, 0) as usize & 1,
                    )
                } else {
                    (
                        list_at(ins, "op_sel_hi", 0, 1) as usize & 1,
                        list_at(ins, "op_sel_hi", 1, 1) as usize & 1,
                    )
                };
                for l in 0..w {
                    let (xh, yh) = (x.v[l] >> (16 * sx) & 0xFFFF, y.v[l] >> (16 * sy) & 0xFFFF);
                    let (a, b) = (num::f16_value(xh), num::f16_value(yh));
                    let h = num::nan16(num::f16_one(a * b), &[xh, yh]);
                    out[l] = (out[l] & !(0xFFFF << (16 * half))) | (h << (16 * half));
                }
                let xt: Vec<Lane> = (0..w).map(|l| widen16(&x.tag(l), sx)).collect();
                let yt: Vec<Lane> = (0..w).map(|l| widen16(&y.tag(l), sy)).collect();
                if let Some(r) = float_rule(e, w, &xt, &yt, false, false)? {
                    any = true;
                    for l in 0..w {
                        t[l][2 * half] = r[l][0];
                        t[l][2 * half + 1] = r[l][1];
                    }
                }
            }
            wv.vwrite(&ops[0], 0, &out, any.then_some(t.as_slice()), exec)?;
        }
        "v.ldexp.f32" => {
            let (x, y) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            for l in 0..w {
                out[l] = ldexp(x.v[l], y.v[l]);
            }
            let t = copy_tags(&x, w);
            wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)?;
        }
        c if c.starts_with("v.cmp.") => {
            cmp(wv, ins, c)?;
        }
        c if c.starts_with("v.cmpx.") => {
            cmpx(wv, ins, c)?;
        }
        c if c.starts_with("v.cvt.") => {
            return super::cvt::cvt(wv, ins, c, e.facts.fnuz_fp8);
        }
        c => {
            let Some(name) = c.strip_prefix("v.").and_then(|r| r.strip_suffix(".f32")) else {
                return Ok(false);
            };
            let name = match name {
                "frexp.exp.i32" => "frexp.exp",
                n => n,
            };
            let x = wv.vread(arg(1)?, 0)?;
            for l in 0..w {
                match unary_f32(name, x.v[l]) {
                    Some(v) => out[l] = v,
                    None => return Ok(false),
                }
            }
            wv.vwrite(&ops[0], 0, &out, copy_tags(&x, w).as_deref(), exec)?;
        }
    }
    Ok(true)
}

fn is_nan16(h: u32) -> bool {
    num::is_nan16(h)
}

/// The low half of a 16-bit op's source (RDNA4 ISA guide: VOP3
/// OPSEL 0 reads bits 15:0), with abs and neg on its sign, bit 15
/// its tags with the half's two bytes in bytes 0–1. A float
/// constant has no f16 value here: gfx1201's encoder takes integer
/// constants only for 16-bit sources, their 32-bit pattern's low half.
fn low16(wv: &Wave, o: &Operand, text: &str) -> Result<Src, EmuError> {
    if let Base::Literal(_, t) = &o.base {
        let int = t.parse::<i64>().is_ok() || t.trim_start_matches('-').starts_with("0x");
        if !int && !t.is_empty() {
            return Err(limit(format!(
                "{text}: the float constant {t} of a 16-bit source is not modelled"
            )));
        }
    }
    let plain = Operand {
        base: o.base.clone(),
        neg: false,
        abs: false,
    };
    let mut x = wv.vread(&plain, 0)?;
    for v in &mut x.v[..wv.w] {
        *v &= 0xFFFF;
        if o.abs {
            *v &= 0x7FFF;
        }
        if o.neg {
            *v ^= 0x8000;
        }
    }
    if x.tagged {
        for t in x.tags_mut().iter_mut() {
            *t = [t[0], t[1], 0, 0];
        }
    }
    Ok(x)
}

/// `max_num` at binary16: IEEE
/// maximumNumber, +0 above −0; both NaN: `cvtToQuietNAN(S0)`.
fn fmax16(x: u32, y: u32) -> u32 {
    match (is_nan16(x), is_nan16(y)) {
        (true, true) => num::nan16(x, &[x, y]),
        (true, false) => y,
        (false, true) => x,
        _ => {
            let (a, b) = (num::f16_value(x), num::f16_value(y));
            let pos_zero = a == 0.0 && b == 0.0 && x & 0x8000 == 0 && y & 0x8000 != 0;
            if a > b || pos_zero {
                x
            } else {
                y
            }
        }
    }
}

/// The 16-bit ops of shipped kernels: RDNA4 ISA guide V_LSHLREV_B16
/// (`D0.u16 = S1.u16 << S0[3:0]`), V_MAX_NUM_F16 ([`fmax16`]) and
/// V_PACK_B32_F16 (`D0[31:16] = S1.f16; D0[15:0] = S0.f16`). The
/// sources are their low halves (OPSEL 0; gfx1201's forms take no op_sel,
/// one given is a limit). A 16-bit result goes to the low half and the high
/// half keeps the register's value and tags (VOP3 OPSEL, dest 0:
/// `{prev_dst_vgpr[31:16], result[15:0]}`). pack's halves are f16 values:
/// a NaN is quieted (measured on gfx1201: each half its own), anything else its
/// bits. Tags: the shift's
/// byte rule within the half, max's float rule on the halves, and
/// pack's byte move.
fn half16(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins, canon: &str) -> Result<(), EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    if ins.field("op_sel").is_some() {
        return Err(limit(format!(
            "{}: op_sel is not modelled (no {} form of {canon} takes it)",
            ins.text, e.facts.name
        )));
    }
    if ops.len() != 3 {
        return Err(fault(format!("{}: three operands", ins.text)));
    }
    let (x, y) = (
        low16(wv, &ops[1], &ins.text)?,
        low16(wv, &ops[2], &ins.text)?,
    );
    let exec = wv.exec;
    let mut out = [0u32; MAX_W];
    if canon == "v.pack.b32.f16" {
        let one = |h: u32| num::nan16(h, &[h]);
        let mut t = zeros(w);
        for l in 0..w {
            out[l] = one(x.v[l]) | one(y.v[l]) << 16;
            let (a, b) = (x.tag(l), y.tag(l));
            t[l] = [a[0], a[1], b[0], b[1]];
        }
        let tagged = x.tagged || y.tagged;
        return wv.vwrite(&ops[0], 0, &out, tagged.then_some(t.as_slice()), exec);
    }
    let old = wv.vread(&ops[0], 0)?;
    let low = if canon == "v.lshlrev.b16" {
        let mut bytes = zeros(w);
        for l in 0..w {
            let a = x.v[l] & 15;
            out[l] = (y.v[l] << a) & 0xFFFF;
            bytes[l] = rules::shift_lane(&y.tag(l), a, true);
            bytes[l][2..].fill(0);
        }
        let amount = Src::splat(0);
        int_rule(e, w, &y, &amount, Some(bytes), true)?
    } else {
        for l in 0..w {
            out[l] = fmax16(x.v[l], y.v[l]);
        }
        let wide = |s: &Src| -> Vec<Lane> { (0..w).map(|l| widen16(&s.tag(l), 0)).collect() };
        float_rule(e, w, &wide(&x), &wide(&y), true, false)?
    };
    let mut t = zeros(w);
    for l in 0..w {
        out[l] |= old.v[l] & 0xFFFF_0000;
        let (r, o) = (low.as_ref().map_or([0; 4], |r| r[l]), old.tag(l));
        t[l] = [r[0], r[1], o[2], o[3]];
    }
    let tagged = low.is_some() || old.tagged;
    wv.vwrite(&ops[0], 0, &out, tagged.then_some(t.as_slice()), exec)
}

/// An f16 transcendental of a binary16 value (`name`: exp, log, rcp, sqrt,
/// rsq; 2^x, log2 x, 1/x, √x, 1/√x): one RNE rounding of the exact result
/// to binary16, denormals kept in and out; a NaN source quieted, an invalid
/// one (log, √, 1/√ of a negative) the default NaN 0xFE00 (measured on
/// gfx1201 over all 65 536 inputs of each, tests/hw/test_emu_ops.py
/// `test_an_f16_transcendental_is_the_gpus`). The binary64 value rounded is
/// within 2^-52 relatively of the exact one, far closer than binary16's
/// midpoints come to an exact result of these functions (the same 65 536
/// inputs agree).
pub fn trans16_value(name: &str, h: u32) -> u32 {
    let h = h & 0xFFFF;
    if num::is_nan16(h) {
        return h | 0x200;
    }
    let x = num::f16_value(h);
    let r = match name {
        "exp" => x.exp2(),
        "log" => x.log2(),
        "rcp" => 1.0 / x,
        "sqrt" => x.sqrt(),
        _ => 1.0 / x.sqrt(),
    };
    if r.is_nan() {
        return num::DNAN16;
    }
    num::f16_one(r)
}

/// `v_exp_f16` … `v_rsq_f16 d, s`: the low half of s by [`trans16_value`]
/// into d's low half, the high half kept (OPSEL destination 0). Tags: the
/// source's low bytes, the old high bytes.
fn trans16(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins, canon: &str) -> Result<(), EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    if ins.field("op_sel").is_some() || ins.field("clamp").is_some() {
        return Err(limit(format!(
            "{}: op_sel and clamp are not modelled on {}",
            ins.text, e.facts.name
        )));
    }
    if ops.len() != 2 {
        return Err(fault(format!("{}: two operands", ins.text)));
    }
    let name = canon.trim_start_matches("v.").trim_end_matches(".f16");
    let x = low16(wv, &ops[1], &ins.text)?;
    let old = wv.vread(&ops[0], 0)?;
    let mut out = [0u32; MAX_W];
    let mut t = zeros(w);
    for l in 0..w {
        out[l] = trans16_value(name, x.v[l]) | old.v[l] & 0xFFFF_0000;
        let (a, o) = (x.tag(l), old.tag(l));
        t[l] = [a[0], a[1], o[2], o[3]];
    }
    let tagged = x.tagged || old.tagged;
    wv.vwrite(&ops[0], 0, &out, tagged.then_some(t.as_slice()), wv.exec)
}

/// `v_fma_mix_f32 d, s0, s1, s2` (RDNA4 ISA guide V_FMA_MIX_F32,
/// "OPSEL for MIX instructions"): source i is S[i].f32 when op_sel_hi[i] is
/// 0, else f16_to_f32 of S[i]'s half op_sel[i] (exact, [`num::f32_of_f16`]);
/// the operand's abs (the NEG_HI field, an absolute value for MIX ops) and
/// neg act on that f32 value (an f16 NaN keeps its payload, quieted);
/// D = fma(in0, in1, in2) rounded once ([`fma`]; measured on gfx1201 over
/// binary32 midpoints), its NaN by [`num::fma_nan32`]. Tags: the fused
/// rule, a source's tags those of the bytes it reads (an f16 half's two
/// bytes widened).
fn fma_mix(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins) -> Result<(), EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    if ops.len() != 4 {
        return Err(fault(format!("{}: four operands", ins.text)));
    }
    let mut ins_f32 = Vec::with_capacity(3);
    let mut tags = Vec::with_capacity(3);
    for i in 0..3 {
        let o = &ops[1 + i];
        let (half, sel) = (
            list_at(ins, "op_sel_hi", i, 0) != 0,
            list_at(ins, "op_sel", i, 0) as usize & 1,
        );
        if let (true, Base::Literal(_, t)) = (half, &o.base) {
            if t.contains('.') {
                return Err(limit(format!(
                    "{}: the float constant {t} as an f16 mix source is not modelled",
                    ins.text
                )));
            }
        }
        let plain = Operand {
            base: o.base.clone(),
            neg: false,
            abs: false,
        };
        let x = wv.vread(&plain, 0)?;
        let mut v = [0u32; MAX_W];
        for l in 0..w {
            v[l] = if half {
                num::f32_of_f16(x.v[l] >> (16 * sel) & 0xFFFF)
            } else {
                x.v[l]
            };
            if o.abs {
                v[l] &= 0x7FFF_FFFF;
            }
            if o.neg {
                v[l] ^= 0x8000_0000;
            }
        }
        ins_f32.push(v);
        tags.push(if half {
            (0..w).map(|l| widen16(&x.tag(l), sel)).collect()
        } else {
            x.tags(w).to_vec()
        });
    }
    let mut out = [0u32; MAX_W];
    for l in 0..w {
        let [a, b, c] = [ins_f32[0][l], ins_f32[1][l], ins_f32[2][l]];
        out[l] = fma(a, b, c);
    }
    let t = match float_rule(e, w, &tags[0], &tags[1], false, false)? {
        Some(p) => float_rule(e, w, &p, &tags[2], false, true)?,
        None => float_rule(e, w, &zeros(w), &tags[2], false, true)?,
    };
    let exec = wv.exec;
    wv.vwrite(&ops[0], 0, &out, t.as_deref(), exec)
}

/// A 64-bit float source (RDNA4 ISA guide "Integer constants are
/// sign-extended for 64-bit sources"; gfx1201's encoder takes no other
/// constant for one): per lane its value, with abs and neg on bit 63
/// and its tags, the source of its eight bytes on each byte.
fn src64(wv: &Wave, o: &Operand, text: &str) -> Result<([u64; MAX_W], Vec<Lane>), EmuError> {
    let w = wv.w;
    let mut v = [0u64; MAX_W];
    let mut t = zeros(w);
    match &o.base {
        Base::Literal(c, s) => {
            let hex = s.trim_start_matches('-').starts_with("0x");
            if !(s.is_empty() || hex || s.parse::<i64>().is_ok()) {
                return Err(limit(format!(
                    "{text}: the constant {s} of a 64-bit float source is not modelled"
                )));
            }
            v[..w].fill(*c as i32 as i64 as u64);
        }
        base => {
            let plain = Operand {
                base: base.clone(),
                neg: false,
                abs: false,
            };
            let (lo, hi) = (wv.vread(&plain, 0)?, wv.vread(&plain, 1)?);
            for l in 0..w {
                v[l] = u64::from(lo.v[l]) | u64::from(hi.v[l]) << 32;
                let (a, b) = (lo.tag(l), hi.tag(l));
                t[l] = [dword_src(&[a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]); 4];
            }
        }
    }
    for x in &mut v[..w] {
        if o.abs {
            *x &= !(1 << 63);
        }
        if o.neg {
            *x ^= 1 << 63;
        }
    }
    Ok((v, t))
}

/// `v_mul_f64 d, x, y` and `v_fma_f64 d, x, y, z` (RDNA4 ISA guide
/// V_MUL_F64: `D0.f64 = S0.f64 * S1.f64`; V_FMA_F64: `fma(S0.f64, S1.f64,
/// S2.f64)`, both 0.5 ULP with denormals): binary64 with one RNE rounding
/// (the fma fused), a NaN by [`num::nan64`] and [`num::fma_nan64`]. Tags: the float rule (the
/// fused rule for fma) on each source's eight bytes, on both dwords.
fn f64_op(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins, canon: &str) -> Result<(), EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    let fma = canon == "v.fma.f64";
    if ops.len() != if fma { 4 } else { 3 } {
        return Err(fault(format!("{}: operand count", ins.text)));
    }
    let (x, xt) = src64(wv, &ops[1], &ins.text)?;
    let (y, yt) = src64(wv, &ops[2], &ins.text)?;
    let z = if fma {
        Some(src64(wv, &ops[3], &ins.text)?)
    } else {
        None
    };
    let (mut lo, mut hi) = ([0u32; MAX_W], [0u32; MAX_W]);
    for l in 0..w {
        let (a, b) = (f64::from_bits(x[l]), f64::from_bits(y[l]));
        let r = match &z {
            Some((z, _)) => num::fma_nan64(a.mul_add(b, f64::from_bits(z[l])), x[l], y[l], z[l]),
            None => num::nan64(a * b, &[x[l], y[l]]),
        };
        (lo[l], hi[l]) = (r as u32, (r >> 32) as u32);
    }
    let p = float_rule(e, w, &xt, &yt, false, false)?;
    let t = match &z {
        Some((_, zt)) => float_rule(e, w, &p.unwrap_or_else(|| zeros(w)), zt, false, true)?,
        None => p,
    };
    let exec = wv.exec;
    wv.vwrite(&ops[0], 0, &lo, t.as_deref(), exec)?;
    wv.vwrite(&ops[0], 1, &hi, t.as_deref(), exec)
}

/// `v.permlane16.b32 d, x, s1, s2`: lane 16·row + i of an active
/// lane takes x of lane 16·row + nibble i of s2:s1 (including its tags);
/// `v.permlanex16.b32` (`cross`) takes it from the other row of its pair,
/// 16·(row ^ 1) + nibble i. op_sel[0] is FI, op_sel[1] BOUND_CTRL. An
/// inactive source lane leaves d unwritten (FI 0, BC 0), gives 0 (FI 0,
/// BC 1), or is read anyway (FI 1).
fn permlane16(wv: &mut Wave, ins: &Ins, cross: bool) -> Result<(), EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    if ops.len() != 4 {
        return Err(fault(format!("{}: four operands", ins.text)));
    }
    let modified = ops.iter().any(|o| o.neg || o.abs)
        || ["abs", "neg", "neg_lo", "neg_hi", "omod", "clamp"]
            .iter()
            .any(|f| ins.field(f).is_some());
    if modified {
        return Err(limit(format!(
            "{}: abs, neg and omod are not emulated",
            ins.text
        )));
    }
    let x = wv.vread(&ops[1], 0)?;
    let lanesel = wv.sread(&ops[2])? & 0xFFFF_FFFF | (wv.sread(&ops[3])? & 0xFFFF_FFFF) << 32;
    let (fi, bc) = (
        list_at(ins, "op_sel", 0, 0) != 0,
        list_at(ins, "op_sel", 1, 0) != 0,
    );
    let exec = wv.exec;
    let mut out = [0u32; MAX_W];
    let mut tags = vec![[0; 4]; w];
    let mut write = 0u64;
    for l in (0..w).filter(|l| exec >> l & 1 != 0) {
        let i = l % 16;
        let row = if cross { (l - i) ^ 16 } else { l - i };
        let src = row + ((lanesel >> (4 * i)) & 15) as usize;
        let live = exec >> src & 1 != 0;
        if live || fi {
            out[l] = x.v[src];
            tags[l] = x.tag(src);
        } else if !bc {
            continue;
        }
        write |= 1 << l;
    }
    let t = x.tagged.then_some(&tags[..]);
    wv.vwrite(&ops[0], 0, &out, t, write)
}

/// `v.cmp.K.T mask, x, y`.
fn cmp(wv: &mut Wave, ins: &Ins, canon: &str) -> Result<(), EmuError> {
    let m = compare(wv, ins, canon, 1)?;
    wv.swrite64(&ins.operands[0], m)
}

/// `v.cmpx.K.T [exec_lo,] x, y`: the compare per active lane, 0
/// in inactive lanes, replaces EXEC; VCC and the SGPRs are untouched.
fn cmpx(wv: &mut Wave, ins: &Ins, canon: &str) -> Result<(), EmuError> {
    let first = match ins.operands.len() {
        2 => 0,
        3 if matches!(&ins.operands[0].base, Base::Reg(f, _, _) if f == "exec") => 1,
        _ => return Err(limit(format!("{}: a v_cmpx writes only EXEC ", ins.text))),
    };
    let m = compare(wv, ins, &canon.replacen("v.cmpx.", "v.cmp.", 1), first)?;
    wv.exec = m & wv.all();
    Ok(())
}

/// The lane mask of a compare whose sources are operands `first` and
/// `first + 1`: active lanes only.
fn compare(wv: &Wave, ins: &Ins, canon: &str, first: usize) -> Result<u64, EmuError> {
    let w = wv.w;
    let parts: Vec<&str> = canon.trim_end_matches(".e64").split('.').collect();
    let (kind, ty) = (
        parts.get(2).copied().unwrap_or(""),
        parts.get(3).copied().unwrap_or(""),
    );
    let ops = &ins.operands;
    let get = |k: usize| {
        ops.get(k)
            .ok_or_else(|| fault(format!("{}: operand {k} missing", ins.text)))
    };
    let wide = ty == "u64" || ty == "i64";
    let (x, y) = (wv.vread(get(first)?, 0)?, wv.vread(get(first + 1)?, 0)?);
    let (x2, y2) = if wide {
        (
            Some(wv.vread(get(first)?, 1)?),
            Some(wv.vread(get(first + 1)?, 1)?),
        )
    } else {
        (None, None)
    };
    let mut m = 0u64;
    for l in 0..w {
        if !mask_bit(wv.exec, l) {
            continue;
        }
        let r = match ty {
            "f32" => {
                let (a, b) = (f(x.v[l]), f(y.v[l]));
                if kind == "class" {
                    y.v[l] >> class_f32(x.v[l]) & 1 != 0
                } else {
                    float_compare(kind, f64::from(a), f64::from(b))
                        .ok_or_else(|| limit(format!("instruction not emulated: {}", ins.text)))?
                }
            }
            _ => {
                let (a, b): (i128, i128) = if wide {
                    let a = u64::from(x.v[l]) | u64::from(x2.as_ref().map_or(0, |s| s.v[l])) << 32;
                    let b = u64::from(y.v[l]) | u64::from(y2.as_ref().map_or(0, |s| s.v[l])) << 32;
                    if ty == "i64" {
                        (i128::from(a as i64), i128::from(b as i64))
                    } else {
                        (i128::from(a), i128::from(b))
                    }
                } else if ty == "i32" {
                    (i128::from(x.v[l] as i32), i128::from(y.v[l] as i32))
                } else {
                    (i128::from(x.v[l]), i128::from(y.v[l]))
                };
                match kind {
                    "eq" => a == b,
                    "ne" | "lg" | "neq" => a != b,
                    "lt" => a < b,
                    "le" => a <= b,
                    "gt" => a > b,
                    "ge" => a >= b,
                    _ => return Err(limit(format!("instruction not emulated: {}", ins.text))),
                }
            }
        };
        if r {
            m |= 1 << l;
        }
    }
    Ok(m)
}

/// The bit of `v_cmp_class_f32`'s mask that names the class of x (RDNA4 ISA
/// guide V_CMP_CLASS_F32): 0 signaling NaN, 1 quiet NaN, then −∞,
/// −normal, −denormal, −0, +0, +denormal, +normal, +∞ (2–9).
fn class_f32(x: u32) -> u32 {
    let (sign, exp, man) = (x >> 31 != 0, (x >> 23) & 0xFF, x & 0x7F_FFFF);
    let (neg, pos) = match (exp, man) {
        (0xFF, m) if m != 0 => return if m & 0x40_0000 != 0 { 1 } else { 0 },
        (0xFF, _) => (2, 9),
        (0, 0) => (5, 6),
        (0, _) => (4, 7),
        _ => (3, 8),
    };
    if sign {
        neg
    } else {
        pos
    }
}

/// One 32-bit dword (`sel` in `0..=1`) of a 64-bit VOP3P packed source.
/// Inline integers sign-extend into dword 1; 32-bit inline floats occupy
/// `[31:0]` with `0` in `[63:32]`.
fn pk_src_dword(wv: &Wave, o: &Operand, sel: usize) -> Result<Src, EmuError> {
    match &o.base {
        Base::Literal(v, _) if sel == 0 => Ok(Src::splat(*v)),
        Base::Literal(v, text) => {
            let int = text.parse::<i64>().is_ok() || text.trim_start_matches('-').starts_with("0x");
            let hi = if int && (*v as i32) < 0 { u32::MAX } else { 0 };
            Ok(Src::splat(hi))
        }
        _ => wv.vread(o, sel as u32),
    }
}

/// CDNA 3 packed 32-bit VOP3P ops: `v_pk_fma_f32`, `v_pk_mul_f32`,
/// `v_pk_add_f32`, and `v_pk_mov_b32`.
///
/// For `v_pk_mov_b32`, `op_sel[0]` selects the source dword of `S0` written to
/// `D0[31:0]` and `op_sel[1]` selects the source dword of `S1` written to
/// `D0[63:32]` (`op_sel_hi`, `neg_lo`, `neg_hi`, and `clamp` are ignored by
/// hardware). For the arithmetic ops, `op_sel` / `neg_lo` control `D0[31:0]`
/// and `op_sel_hi` (defaulting to `[1, 1(, 1)]`) / `neg_hi` control
/// `D0[63:32]`, with optional DX10 `[+0.0, 1.0]` clamping.
fn pk_f32_op(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins, canon: &str) -> Result<(), EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    if ["mul", "div", "omod"]
        .iter()
        .any(|k| ins.field(k).is_some())
    {
        return Err(limit(format!(
            "{}: output modifier (mul, div, omod) is not modelled",
            ins.text
        )));
    }
    if ops.iter().any(|o| o.neg || o.abs) {
        return Err(limit(format!(
            "{}: source abs / neg is not modelled on packed 32-bit ops",
            ins.text
        )));
    }
    let fma3 = canon == "v.pk.fma.f32";
    let nsrc = if fma3 { 3 } else { 2 };
    if ops.len() != nsrc + 1 {
        return Err(fault(format!("{}: operand count", ins.text)));
    }
    let exec = wv.exec;
    if canon == "v.pk.mov.b32" {
        let s0 = pk_src_dword(wv, &ops[1], list_at(ins, "op_sel", 0, 0) as usize & 1)?;
        let s1 = pk_src_dword(wv, &ops[2], list_at(ins, "op_sel", 1, 0) as usize & 1)?;
        wv.vwrite(&ops[0], 0, &s0.v, copy_tags(&s0, w).as_deref(), exec)?;
        return wv.vwrite(&ops[0], 1, &s1.v, copy_tags(&s1, w).as_deref(), exec);
    }
    let clamp = ins.field("clamp").is_some_and(|f| field_int(f) != 0);
    let mut outs = [[0u32; MAX_W]; 2];
    let mut tags: [Option<Vec<Lane>>; 2] = [None, None];
    for half in 0..2 {
        let mut srcs = Vec::with_capacity(nsrc);
        for i in 0..nsrc {
            let (sel, neg) = if half == 0 {
                (
                    list_at(ins, "op_sel", i, 0) as usize & 1,
                    list_at(ins, "neg_lo", i, 0) & 1 != 0,
                )
            } else {
                (
                    list_at(ins, "op_sel_hi", i, 1) as usize & 1,
                    list_at(ins, "neg_hi", i, 0) & 1 != 0,
                )
            };
            let mut s = pk_src_dword(wv, &ops[1 + i], sel)?;
            if neg {
                for v in &mut s.v[..w] {
                    *v ^= 0x8000_0000;
                }
            }
            srcs.push(s);
        }
        for l in 0..w {
            let (a, b) = (srcs[0].v[l], srcs[1].v[l]);
            let mut r = match canon {
                "v.pk.fma.f32" => fma(a, b, srcs[2].v[l]),
                "v.pk.mul.f32" => num::nan32(f(a) * f(b), &[a, b]),
                _ => num::nan32(f(a) + f(b), &[a, b]),
            };
            if clamp {
                let fv = f(r);
                r = if fv.is_nan() || fv <= 0.0 {
                    0
                } else if fv >= 1.0 {
                    1.0f32.to_bits()
                } else {
                    r
                };
            }
            outs[half][l] = r;
        }
        tags[half] = match canon {
            "v.pk.fma.f32" => {
                match float_rule(e, w, srcs[0].tags(w), srcs[1].tags(w), false, false)? {
                    Some(p) => float_rule(e, w, &p, srcs[2].tags(w), false, true)?,
                    None => float_rule(e, w, &zeros(w), srcs[2].tags(w), false, true)?,
                }
            }
            "v.pk.mul.f32" => float_rule(e, w, srcs[0].tags(w), srcs[1].tags(w), false, false)?,
            _ => float_rule(e, w, srcs[0].tags(w), srcs[1].tags(w), false, true)?,
        };
    }
    wv.vwrite(&ops[0], 0, &outs[0], tags[0].as_deref(), exec)?;
    wv.vwrite(&ops[0], 1, &outs[1], tags[1].as_deref(), exec)
}

/// The DPP source lane of lane `l` in a `w`-lane wave, or `None` when out of
/// bounds so `bound_ctrl` applies (AMD CDNA 3 / RDNA 4 DPP_CTRL tables:
/// `row_bcast:15` reads `(l & 0x30) - 1` for `l > 15` else OOB; `row_bcast:31`
/// reads lane `31` for `l > 31` else OOB; `wave_shl:1` / `wave_shr:1` shift
/// across the wave with OOB at the boundary; `wave_rol:1` / `wave_ror:1`
/// rotate across `w`).
fn dpp_source(ins: &Ins, l: usize, w: usize) -> Result<Option<usize>, EmuError> {
    let (row, col) = (l / 16, l % 16);
    if let Some(Field::List(q)) = ins.field("quad_perm") {
        let sel = q.get(l & 3).copied().unwrap_or(0) as usize & 3;
        return Ok(Some((l & !3) + sel));
    }
    if let Some(n) = ins.field("row_shl") {
        let n = field_int(n) as usize;
        return Ok((col + n < 16).then_some(l + n));
    }
    if let Some(n) = ins.field("row_shr") {
        let n = field_int(n) as usize;
        return Ok((col >= n).then(|| l - n));
    }
    if let Some(n) = ins.field("row_ror") {
        let n = field_int(n) as usize;
        return Ok(Some(row * 16 + (col + 16 - n % 16) % 16));
    }
    if ins.field("row_mirror").is_some() {
        return Ok(Some(row * 16 + 15 - col));
    }
    if ins.field("row_half_mirror").is_some() {
        return Ok(Some((l & !7) + 7 - (l & 7)));
    }
    if let Some(n) = ins.field("row_share").or_else(|| ins.field("row_newbcast")) {
        return Ok(Some(row * 16 + field_int(n) as usize % 16));
    }
    if let Some(n) = ins.field("row_xmask") {
        return Ok(Some(row * 16 + (col ^ (field_int(n) as usize % 16))));
    }
    if let Some(n) = ins.field("row_bcast") {
        return match field_int(n) {
            15 => Ok((l > 15).then(|| (l & 0x30) - 1)),
            31 => Ok((l > 31).then_some(31)),
            _ => Err(limit("dpp control not emulated: row_bcast".into())),
        };
    }
    if let Some(n) = ins.field("wave_shr") {
        if field_int(n) != 1 {
            return Err(limit("dpp control not emulated: wave_shr".into()));
        }
        return Ok((l >= 1).then(|| l - 1));
    }
    if let Some(n) = ins.field("wave_shl") {
        if field_int(n) != 1 {
            return Err(limit("dpp control not emulated: wave_shl".into()));
        }
        return Ok((l + 1 < w).then_some(l + 1));
    }
    if let Some(n) = ins.field("wave_ror") {
        if field_int(n) != 1 {
            return Err(limit("dpp control not emulated: wave_ror".into()));
        }
        return Ok(Some((l + w - 1) % w));
    }
    if let Some(n) = ins.field("wave_rol") {
        if field_int(n) != 1 {
            return Err(limit("dpp control not emulated: wave_rol".into()));
        }
        return Ok(Some((l + 1) % w));
    }
    let control: Vec<&str> = ins
        .fields
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| !matches!(*k, "row_mask" | "bank_mask" | "bound_ctrl" | "fi"))
        .collect();
    Err(limit(format!(
        "dpp control not emulated: {}",
        control.join(" ")
    )))
}

fn field_int(f: &Field) -> i128 {
    match f {
        Field::Int(v) => *v,
        Field::True => 1,
        Field::List(l) => l.first().copied().unwrap_or(0),
    }
}

/// The lanes a DPP op writes: the active lanes
/// whose row is in `row_mask` and whose bank (four lanes of a row) is in
/// `bank_mask`; the others keep their value.
fn dpp_writes(ins: &Ins, exec: u64, w: usize) -> u64 {
    let mask = |k: &str| ins.field(k).map_or(0xF, field_int) as u64;
    let (rows, banks) = (mask("row_mask"), mask("bank_mask"));
    (0..w)
        .filter(|&l| rows >> (l / 16) & 1 != 0 && banks >> (l % 16 / 4) & 1 != 0)
        .fold(0, |m, l| m | 1 << l)
        & exec
}

fn dpp(e: &mut Exec<'_>, wv: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    if ops.len() < 2 {
        return Err(fault(format!("{}: operand missing", ins.text)));
    }
    let plain0 = Operand {
        base: ops[1].base.clone(),
        neg: false,
        abs: false,
    };
    let raw = wv.vread(&plain0, 0)?;
    let old = wv.vread(&ops[0], 0)?;
    let set = |k: &str| ins.field(k).is_some_and(|f| field_int(f) != 0);
    let bound = set("bound_ctrl");
    // A source lane out of the row, or inactive without fetch-inactive, is
    // invalid: the lane reads 0 with bound_ctrl, else keeps its value.
    let fetch_inactive = set("fi");
    let mut srcs = [None; MAX_W];
    for (l, s) in srcs.iter_mut().enumerate().take(w) {
        *s = dpp_source(ins, l, w)?.filter(|s| *s < w && (fetch_inactive || mask_bit(wv.exec, *s)));
    }
    let mut xs = Src::splat(0);
    for l in 0..w {
        if let Some(s) = srcs[l] {
            xs.v[l] = raw.v[s];
            if raw.tagged {
                xs.tags_mut()[l] = raw.tag(s);
            }
        }
    }
    if ops[1].abs {
        for v in &mut xs.v[..w] {
            *v &= 0x7FFF_FFFF;
        }
    }
    if ops[1].neg {
        for v in &mut xs.v[..w] {
            *v ^= 0x8000_0000;
        }
    }
    let writes = dpp_writes(ins, wv.exec, w);
    let mut out = [0u32; MAX_W];
    let mut t = old.tags(w).to_vec();
    let base = canon.strip_suffix(".dpp").unwrap_or(canon);
    if base == "v.mov.b32" || base == "v.not.b32" {
        for l in 0..w {
            if srcs[l].is_some() || bound {
                out[l] = if base == "v.not.b32" {
                    !xs.v[l]
                } else {
                    xs.v[l]
                };
                t[l] = xs.tag(l);
            } else {
                out[l] = old.v[l];
            }
        }
        let tagged = xs.tagged || old.tagged;
        wv.vwrite(&ops[0], 0, &out, tagged.then_some(t.as_slice()), writes)?;
        return Ok(true);
    }
    let y = ops
        .get(2)
        .ok_or_else(|| fault(format!("{}: operand 2 missing", ins.text)))?;
    let y = wv.vread(y, 0)?;
    let mut vals = [0u32; MAX_W];
    let rt = if let Some((op, stat, add)) = float2(base) {
        for l in 0..w {
            vals[l] = op(xs.v[l], y.v[l]);
        }
        float_rule(e, w, xs.tags(w), y.tags(w), stat, add)?
    } else if matches!(
        base,
        "v.max.f32" | "v.min.f32" | "v.max.num.f32" | "v.min.num.f32"
    ) {
        let max = base.starts_with("v.max");
        for l in 0..w {
            vals[l] = if max {
                fmax(xs.v[l], y.v[l])
            } else {
                fmin(xs.v[l], y.v[l])
            };
        }
        float_rule(e, w, xs.tags(w), y.tags(w), true, false)?
    } else if base == "v.fmac.f32" || base == "v.mac.f32" {
        for l in 0..w {
            vals[l] = fma(xs.v[l], y.v[l], old.v[l]);
        }
        match float_rule(e, w, xs.tags(w), y.tags(w), false, false)? {
            Some(p) => float_rule(e, w, &p, old.tags(w), false, true)?,
            None => float_rule(e, w, &zeros(w), old.tags(w), false, true)?,
        }
    } else if base == "v.cndmask.b32" {
        let m = match ops.get(3) {
            Some(m) => wv.sread64(m)?,
            None => wv.vcc,
        };
        let mut ct = zeros(w);
        for l in 0..w {
            let pick = mask_bit(m, l);
            vals[l] = if pick { y.v[l] } else { xs.v[l] };
            ct[l] = if pick { y.tag(l) } else { xs.tag(l) };
        }
        (xs.tagged || y.tagged).then_some(ct)
    } else if let Some(op) = binop_int(base) {
        for l in 0..w {
            vals[l] = op(xs.v[l], y.v[l]);
        }
        int_rule(e, w, &xs, &y, None, false)?
    } else if matches!(base, "v.and.b32" | "v.or.b32") {
        let mut bytes = zeros(w);
        for l in 0..w {
            let (a, b) = (xs.v[l], y.v[l]);
            if base == "v.and.b32" {
                vals[l] = a & b;
                bytes[l] = rules::and_lane(a, &xs.tag(l), b, &y.tag(l));
            } else {
                vals[l] = a | b;
                bytes[l] = rules::or_lane(a, &xs.tag(l), b, &y.tag(l));
            }
        }
        int_rule(e, w, &xs, &y, Some(bytes), false)?
    } else {
        return Ok(false);
    };
    let tagged = rt.is_some() || old.tagged;
    for l in 0..w {
        if srcs[l].is_some() || bound {
            out[l] = vals[l];
            t[l] = rt.as_ref().map_or([0; 4], |rt| rt[l]);
        } else {
            out[l] = old.v[l];
        }
    }
    wv.vwrite(&ops[0], 0, &out, tagged.then_some(t.as_slice()), writes)?;
    Ok(true)
}

// --------------------------------------------------------------- scalar

/// Runs a scalar ALU op; `None` when `canon` is not one.
pub fn salu(wv: &mut Wave, ins: &Ins, canon: &str) -> Option<Result<(), EmuError>> {
    match salu_inner(wv, ins, canon) {
        Ok(true) => Some(Ok(())),
        Ok(false) => None,
        Err(e) => Some(Err(e)),
    }
}

#[allow(clippy::too_many_lines)]
fn salu_inner(wv: &mut Wave, ins: &Ins, canon: &str) -> Result<bool, EmuError> {
    let ops = &ins.operands;
    let get = |k: usize| {
        ops.get(k)
            .ok_or_else(|| fault(format!("{}: operand {k} missing", ins.text)))
    };
    let r32 = |wv: &Wave, k: usize| -> Result<u32, EmuError> { Ok(wv.sread(get(k)?)? as u32) };
    let r64 = |wv: &Wave, k: usize| -> Result<u64, EmuError> { wv.sread64(get(k)?) };
    if shipped_salu(wv, ins, canon)? {
        return Ok(true);
    }
    let scc = u32::from(wv.scc);
    match canon {
        "s.mov.b32" | "s.movk.i32" => {
            let v = wv.sread(get(1)?)?;
            wv.swrite(&ops[0], v)?;
        }
        "s.mov.b64" => {
            let v = r64(wv, 1)?;
            wv.swrite64(&ops[0], v)?;
        }
        // RDNA4 ISA guide S_MOVRELS_B32/B64, S_MOVRELD_B32/B64: the
        // source's (S) or destination's (D) raw SGPR number plus M0; a
        // 64-bit move needs an even index and register.
        "s.movrels.b32" | "s.movrels.b64" | "s.movreld.b32" | "s.movreld.b64" => {
            let m0 = wv.m0;
            let wide = canon.ends_with("b64");
            if wide && m0 % 2 != 0 {
                return Err(fault(format!(
                    "{}: an odd M0 index {m0} for a 64-bit move",
                    ins.text
                )));
            }
            let reads = canon.starts_with("s.movrels");
            let src = if reads {
                relative(wv, get(1)?, m0)?
            } else {
                get(1)?.clone()
            };
            let dst = if reads {
                ops[0].clone()
            } else {
                relative(wv, &ops[0], m0)?
            };
            if wide {
                let v = wv.sread64(&src)?;
                wv.swrite64(&dst, v)?;
            } else {
                let v = wv.sread(&src)?;
                wv.swrite(&dst, v)?;
            }
        }
        "s.add.u32" | "s.addc.u32" => {
            let c = if canon == "s.addc.u32" {
                u64::from(scc)
            } else {
                0
            };
            let r = u64::from(r32(wv, 1)?) + u64::from(r32(wv, 2)?) + c;
            wv.swrite(&ops[0], r & 0xFFFF_FFFF)?;
            wv.scc = r >= 1 << 32;
        }
        "s.sub.u32" | "s.subb.u32" => {
            let (x, y) = (u64::from(r32(wv, 1)?), u64::from(r32(wv, 2)?));
            let y2 = if canon == "s.subb.u32" {
                y + u64::from(scc)
            } else {
                y
            };
            wv.swrite(&ops[0], x.wrapping_sub(y2) & 0xFFFF_FFFF)?;
            wv.scc = x < y2;
        }
        "s.mul.i32" => {
            let r = r32(wv, 1)?.wrapping_mul(r32(wv, 2)?);
            wv.swrite(&ops[0], u64::from(r))?;
        }
        "s.mul.hi.u32" => {
            let r = (u64::from(r32(wv, 1)?) * u64::from(r32(wv, 2)?)) >> 32;
            wv.swrite(&ops[0], r)?;
        }
        "s.and.b32" | "s.or.b32" | "s.xor.b32" | "s.andn2.b32" | "s.orn2.b32" | "s.nand.b32"
        | "s.nor.b32" | "s.xnor.b32" => {
            let (x, y) = (r32(wv, 1)?, r32(wv, 2)?);
            let r = match canon {
                "s.and.b32" => x & y,
                "s.or.b32" => x | y,
                "s.xor.b32" => x ^ y,
                "s.andn2.b32" => x & !y,
                "s.orn2.b32" => x | !y,
                "s.nand.b32" => !(x & y),
                "s.nor.b32" => !(x | y),
                _ => !(x ^ y),
            };
            wv.swrite(&ops[0], u64::from(r))?;
            wv.scc = r != 0;
        }
        "s.and.b64" | "s.or.b64" | "s.xor.b64" | "s.andn2.b64" | "s.orn2.b64" => {
            let (x, y) = (r64(wv, 1)?, r64(wv, 2)?);
            let r = match canon {
                "s.and.b64" => x & y,
                "s.or.b64" => x | y,
                "s.xor.b64" => x ^ y,
                "s.andn2.b64" => x & !y,
                _ => x | !y,
            };
            wv.swrite64(&ops[0], r)?;
            wv.scc = r != 0;
        }
        "s.lshl.b32" | "s.lshr.b32" | "s.ashr.i32" => {
            let (x, y) = (r32(wv, 1)?, r32(wv, 2)? & 31);
            let r = match canon {
                "s.lshl.b32" => x << y,
                "s.lshr.b32" => x >> y,
                _ => ((x as i32) >> y) as u32,
            };
            wv.swrite(&ops[0], u64::from(r))?;
            wv.scc = r != 0;
        }
        "s.lshl.b64" | "s.lshr.b64" => {
            let (x, y) = (r64(wv, 1)?, r32(wv, 2)? & 63);
            let r = if canon == "s.lshl.b64" {
                x << y
            } else {
                x >> y
            };
            wv.swrite64(&ops[0], r)?;
            wv.scc = r != 0;
        }
        "s.lshl1.add.u32" | "s.lshl2.add.u32" | "s.lshl3.add.u32" | "s.lshl4.add.u32" => {
            let n = u32::from(canon.as_bytes()[6] - b'0');
            let r = (u64::from(r32(wv, 1)?) << n) + u64::from(r32(wv, 2)?);
            wv.swrite(&ops[0], r & 0xFFFF_FFFF)?;
            wv.scc = r >= 1 << 32;
        }
        "s.bfe.u32" => {
            let (x, y) = (r32(wv, 1)?, r32(wv, 2)?);
            let (off, width) = (y & 31, (y >> 16) & 0x7F);
            let r = if width == 0 {
                0
            } else {
                ((u64::from(x) >> off) & ((1u64 << width.min(63)) - 1)) as u32
            };
            wv.swrite(&ops[0], u64::from(r))?;
            wv.scc = r != 0;
        }
        "s.min.u32" | "s.max.u32" | "s.min.i32" | "s.max.i32" => {
            let (x, y) = (r32(wv, 1)?, r32(wv, 2)?);
            let r = match canon {
                "s.min.u32" => x.min(y),
                "s.max.u32" => x.max(y),
                "s.min.i32" => (x as i32).min(y as i32) as u32,
                _ => (x as i32).max(y as i32) as u32,
            };
            wv.swrite(&ops[0], u64::from(r))?;
            wv.scc = r == x;
        }
        c if c.starts_with("s.cmp.") => {
            let parts: Vec<&str> = c.split('.').collect();
            let (kind, ty) = (
                parts.get(2).copied().unwrap_or(""),
                parts.get(3).copied().unwrap_or(""),
            );
            let (a, b): (i128, i128) = match ty {
                "u64" => (i128::from(r64(wv, 0)?), i128::from(r64(wv, 1)?)),
                "i32" => (
                    i128::from(r32(wv, 0)? as i32),
                    i128::from(r32(wv, 1)? as i32),
                ),
                _ => (i128::from(r32(wv, 0)?), i128::from(r32(wv, 1)?)),
            };
            wv.scc = match kind {
                "eq" => a == b,
                "lg" | "ne" => a != b,
                "lt" => a < b,
                "le" => a <= b,
                "gt" => a > b,
                "ge" => a >= b,
                _ => return Err(limit(format!("instruction not emulated: {}", ins.text))),
            };
        }
        "s.cselect.b32" => {
            let v = if wv.scc { r32(wv, 1)? } else { r32(wv, 2)? };
            wv.swrite(&ops[0], u64::from(v))?;
        }
        "s.cselect.b64" => {
            let v = if wv.scc { r64(wv, 1)? } else { r64(wv, 2)? };
            wv.swrite64(&ops[0], v)?;
        }
        "s.not.b32" => {
            let r = !r32(wv, 1)?;
            wv.swrite(&ops[0], u64::from(r))?;
            wv.scc = r != 0;
        }
        "s.not.b64" => {
            let r = !r64(wv, 1)?;
            wv.swrite64(&ops[0], r)?;
            wv.scc = r != 0;
        }
        "s.bcnt1.i32.b32" | "s.bcnt1.i32.b64" => {
            let r = if canon.ends_with("b64") {
                r64(wv, 1)?.count_ones()
            } else {
                r32(wv, 1)?.count_ones()
            };
            wv.swrite(&ops[0], u64::from(r))?;
            wv.scc = r != 0;
        }
        "s.bitcmp1.b32" | "s.bitcmp0.b32" => {
            let bit = r32(wv, 0)? >> (r32(wv, 1)? & 31) & 1 != 0;
            wv.scc = if canon == "s.bitcmp1.b32" { bit } else { !bit };
        }
        "s.bitset1.b32" | "s.bitset0.b32" => {
            let b = r32(wv, 1)? & 31;
            let d = r32(wv, 0)?;
            let r = if canon == "s.bitset1.b32" {
                d | 1 << b
            } else {
                d & !(1 << b)
            };
            wv.swrite(&ops[0], u64::from(r))?;
        }
        "s.bitset1.b64" | "s.bitset0.b64" => {
            let b = r32(wv, 1)? & 63;
            let d = r64(wv, 0)?;
            let r = if canon == "s.bitset1.b64" {
                d | 1 << b
            } else {
                d & !(1 << b)
            };
            wv.swrite64(&ops[0], r)?;
        }
        "s.ff1.i32.b64" | "s.ff1.i32.b32" => {
            let v = if canon.ends_with("b64") {
                r64(wv, 1)?
            } else {
                u64::from(r32(wv, 1)?)
            };
            let r = if v == 0 {
                0xFFFF_FFFF
            } else {
                u64::from(v.trailing_zeros())
            };
            wv.swrite(&ops[0], r)?;
        }
        // RDNA4 ISA guide, S_AND_SAVEEXEC_B32 / _B64 and
        // S_OR_SAVEEXEC_B32 / _B64: saveexec = EXEC; EXEC = S0 op EXEC;
        // D = saveexec; SCC = EXEC != 0. The b32 forms read and write
        // EXEC[31:0] alone, the b64 forms all of EXEC (EXEC_HI on wave32).
        "s.and.saveexec.b32" | "s.or.saveexec.b32" => {
            let x = u64::from(r32(wv, 1)?);
            let old = wv.exec64();
            let lo = old & 0xFFFF_FFFF;
            let new = if canon.starts_with("s.and") {
                x & lo
            } else {
                x | lo
            };
            wv.set_exec64(old & !0xFFFF_FFFF | new);
            wv.swrite(&ops[0], lo)?;
            wv.scc = new != 0;
        }
        "s.and.saveexec.b64" | "s.or.saveexec.b64" => {
            let x = r64(wv, 1)?;
            let old = wv.exec64();
            let new = if canon.starts_with("s.and") {
                x & old
            } else {
                x | old
            };
            wv.set_exec64(new);
            wv.swrite64(&ops[0], old)?;
            wv.scc = wv.exec64() != 0;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
