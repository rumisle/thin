//! The conversions with their tag rules.

use super::numbers::{self as num, f32_of, Small};
use super::wave::{Src, Wave, MAX_W};
use crate::tags::{src as dword_src, Lane};
use crate::{Base, Cause, EmuError, Field, Ins, Operand};

/// The scale exponent 1.0 has.
const ONE: u32 = 0x3F80_0000;

fn small(name: &str, fnuz: bool) -> Option<Small> {
    Some(match (name, fnuz) {
        ("fp8", false) => Small::E4M3,
        ("bf8", false) => Small::E5M2,
        ("fp8", true) => Small::E4M3Fnuz,
        ("bf8", true) => Small::E5M2Fnuz,
        ("fp6", _) => Small::E2M3,
        ("bf6", _) => Small::E3M2,
        ("fp4", _) => Small::E2M1,
        _ => return None,
    })
}

/// Reconstructs the 4-bit hardware VOP3 `OPSEL[3:0]` word from `ins.field("op_sel")`.
///
/// On both RDNA 4 and CDNA 3/4, a VOP3 `op_sel` list of length `len < 4` stores
/// source bits `0..len-1` followed by destination `OPSEL[3]` in its final slot;
/// a 4-element list stores `[OPSEL[0], OPSEL[1], OPSEL[2], OPSEL[3]]` directly.
pub fn hw_opsel(ins: &Ins) -> u32 {
    let Some(Field::List(items)) = ins.field("op_sel") else {
        return 0;
    };
    let len = items.len();
    let mut v = 0u32;
    for (j, item) in items.iter().enumerate() {
        let b = (*item as u32) & 1;
        let bit = if j + 1 == len && len < 4 { 3 } else { j };
        v |= b << bit;
    }
    v
}

pub(crate) fn put_byte(d: u32, k: usize, b: u32) -> u32 {
    (d & !(0xFF << (8 * k))) | ((b & 0xFF) << (8 * k))
}

pub(crate) fn put_half(d: u32, h: usize, v16: u32) -> u32 {
    (d & !(0xFFFF << (16 * h))) | ((v16 & 0xFFFF) << (16 * h))
}

fn is_int_or_hex_lit(text: &str) -> bool {
    text.is_empty() || text.parse::<i64>().is_ok() || text.trim_start_matches('-').starts_with("0x")
}

fn inline_16_bits(bits: u32, bf16: bool) -> u32 {
    if !bf16 && bits == 0x3E22_F983 {
        0x3118
    } else if bf16 {
        num::bf16_of_f32(bits)
    } else {
        num::f16_of_f32(bits)
    }
}

/// Reads a 16-bit half (`half` in `0..=1`) with 16-bit `abs`/`neg` applied on bit 15.
fn read_half16(wv: &Wave, o: &Operand, half: usize, bf16: bool) -> Result<Src, EmuError> {
    let w = wv.w;
    let mut x = match &o.base {
        Base::Literal(bits, text) if !is_int_or_hex_lit(text) => {
            let v16 = if half == 0 {
                inline_16_bits(*bits, bf16)
            } else {
                0
            };
            Src::splat(v16)
        }
        base => {
            let plain = Operand {
                base: base.clone(),
                neg: false,
                abs: false,
            };
            let mut s = wv.vread(&plain, 0)?;
            for v in &mut s.v[..w] {
                *v = (*v >> (16 * half)) & 0xFFFF;
            }
            if s.tagged {
                for t in s.tags_mut().iter_mut() {
                    *t = [t[2 * half], t[2 * half + 1], 0, 0];
                }
            }
            s
        }
    };
    for v in &mut x.v[..w] {
        if o.abs {
            *v &= 0x7FFF;
        }
        if o.neg {
            *v ^= 0x8000;
        }
    }
    Ok(x)
}

/// Reads a packed 2×16-bit (`PK2_F16` / `PK2_BF16`) source with `abs`/`neg` on bits 15 and 31.
fn read_pk2_16(wv: &Wave, o: &Operand, bf16: bool) -> Result<Src, EmuError> {
    let w = wv.w;
    let mut x = match &o.base {
        Base::Literal(bits, text) if !is_int_or_hex_lit(text) => {
            Src::splat(inline_16_bits(*bits, bf16))
        }
        base => {
            let plain = Operand {
                base: base.clone(),
                neg: false,
                abs: false,
            };
            wv.vread(&plain, 0)?
        }
    };
    for v in &mut x.v[..w] {
        if o.abs {
            *v &= 0x7FFF_7FFF;
        }
        if o.neg {
            *v ^= 0x8000_8000;
        }
    }
    Ok(x)
}

#[inline]
fn half16_to_f64(h: u32, bf16: bool) -> f64 {
    if bf16 {
        f64::from(f32_of((h & 0xFFFF) << 16))
    } else {
        num::f16_value(h & 0xFFFF)
    }
}

#[inline]
fn put_6bit(words: &mut [u32; 6], idx: usize, code: u32) {
    let bit_lo = 6 * idx;
    let (dw, sh) = (bit_lo / 32, bit_lo % 32);
    let c = code & 0x3F;
    words[dw] |= c << sh;
    if sh > 26 {
        words[dw + 1] |= c >> (32 - sh);
    }
}

#[inline]
fn get_6bit(words: &[u32; 6], idx: usize) -> u32 {
    let bit_lo = 6 * idx;
    let (dw, sh) = (bit_lo / 32, bit_lo % 32);
    let lo = words[dw] >> sh;
    let hi = if sh > 26 {
        words[dw + 1] << (32 - sh)
    } else {
        0
    };
    (lo | hi) & 0x3F
}

fn write_byte_into(
    wv: &mut Wave,
    dst: &Operand,
    byte_idx: usize,
    vals: &[u32; MAX_W],
    src_tags: Option<&[i64; MAX_W]>,
) -> Result<(), EmuError> {
    let w = wv.w;
    let old = wv.vread(dst, 0)?;
    let mut out = [0u32; MAX_W];
    let mut t: Vec<Lane> = old.tags(w).to_vec();
    for l in 0..w {
        out[l] = put_byte(old.v[l], byte_idx, vals[l]);
        if let Some(st) = src_tags {
            t[l][byte_idx] = st[l];
        } else {
            t[l][byte_idx] = 0;
        }
    }
    let tagged = old.tagged || src_tags.is_some();
    wv.vwrite(dst, 0, &out, tagged.then_some(t.as_slice()), wv.exec)
}

fn write_half_into(
    wv: &mut Wave,
    dst: &Operand,
    half_idx: usize,
    vals: &[u32; MAX_W],
    src_tags: Option<&[[i64; 2]; MAX_W]>,
) -> Result<(), EmuError> {
    let w = wv.w;
    let old = wv.vread(dst, 0)?;
    let mut out = [0u32; MAX_W];
    let mut t: Vec<Lane> = old.tags(w).to_vec();
    for l in 0..w {
        out[l] = put_half(old.v[l], half_idx, vals[l]);
        let pair = src_tags.map_or([0, 0], |st| st[l]);
        t[l][2 * half_idx] = pair[0];
        t[l][2 * half_idx + 1] = pair[1];
    }
    let tagged = old.tagged || src_tags.is_some();
    wv.vwrite(dst, 0, &out, tagged.then_some(t.as_slice()), wv.exec)
}

/// Runs a conversion; `Ok(false)` when `canon` is not one.
#[allow(clippy::too_many_lines)]
pub fn cvt(wv: &mut Wave, ins: &Ins, canon: &str, fnuz: bool) -> Result<bool, EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    let exec = wv.exec;
    let arg = |k: usize| {
        ops.get(k).ok_or_else(|| EmuError {
            cause: Some(Cause::Fault),
            message: format!("{}: operand {k} missing", ins.text),
        })
    };
    let mut out = [0u32; MAX_W];
    let name = canon.trim_start_matches("v.cvt.");
    if let Some(rest) = name.strip_prefix("scalef32.") {
        return cvt_scalef32(wv, ins, rest, fnuz);
    }
    let hw = hw_opsel(ins);
    match name {
        "i32.f32" | "u32.f32" => {
            let x = wv.vread(arg(1)?, 0)?;
            for l in 0..w {
                out[l] = if name == "i32.f32" {
                    num::i32_of_f32(x.v[l])
                } else {
                    num::u32_of_f32(x.v[l])
                };
            }
            wv.vwrite(&ops[0], 0, &out, None, exec)?;
        }
        "f32.i32" | "f32.u32" | "f32.f16" => {
            let x = wv.vread(arg(1)?, 0)?;
            for l in 0..w {
                out[l] = match name {
                    "f32.i32" => (x.v[l] as i32 as f32).to_bits(),
                    "f32.u32" => (x.v[l] as f32).to_bits(),
                    _ => num::f32_of_f16(x.v[l]),
                };
            }
            wv.vwrite(&ops[0], 0, &out, None, exec)?;
        }
        "f32.bf16" => {
            let x = read_half16(wv, arg(1)?, (hw & 1) as usize, true)?;
            let mut t = vec![[0; 4]; w];
            for l in 0..w {
                out[l] = x.v[l] << 16;
                let a = x.tag(l);
                t[l] = [a[0], a[1], a[0], a[1]];
            }
            wv.vwrite(&ops[0], 0, &out, x.tagged.then_some(t.as_slice()), exec)?;
        }
        "f16.f32" | "pk.rtz.f16.f32" | "pk.f16.f32" | "pk.bf16.f32" => {
            let x = wv.vread(arg(1)?, 0)?;
            let y = if name == "f16.f32" {
                Src::splat(0)
            } else {
                wv.vread(arg(2)?, 0)?
            };
            let mut t = vec![[0; 4]; w];
            for l in 0..w {
                let one = |v: u32| match name {
                    "f16.f32" | "pk.f16.f32" => num::f16_of_f32(v),
                    "pk.bf16.f32" => num::bf16_of_f32(v),
                    _ => num::f16_rtz(v),
                };
                out[l] = if name == "f16.f32" {
                    one(x.v[l])
                } else {
                    one(x.v[l]) | one(y.v[l]) << 16
                };
                let (sx, sy) = (dword_src(&x.tag(l)), dword_src(&y.tag(l)));
                t[l] = if name == "f16.f32" {
                    [sx, sx, 0, 0]
                } else {
                    [sx, sx, sy, sy]
                };
            }
            let mut tagged = x.tagged || y.tagged;
            // A 16-bit result goes to the low half; the high half keeps the
            // register's value and tags (RDNA4 ISA guide OPSEL, dest 0:
            // `{prev_dst_vgpr[31:16], result[15:0]}`; measured on gfx1201,
            // e32 and e64, tests/hw/test_emu_ops.py
            // test_a_16_bit_result_keeps_the_other_half).
            if name == "f16.f32" {
                let old = wv.vread(&ops[0], 0)?;
                for l in 0..w {
                    out[l] |= old.v[l] & 0xFFFF_0000;
                    let o = old.tag(l);
                    t[l][2..].copy_from_slice(&o[2..]);
                }
                tagged |= old.tagged;
            }
            wv.vwrite(&ops[0], 0, &out, tagged.then_some(t.as_slice()), exec)?;
        }
        "sr.f16.f32" | "sr.bf16.f32" => {
            let bf16 = name == "sr.bf16.f32";
            let (x, rand) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let mut st = [[0i64; 2]; MAX_W];
            for l in 0..w {
                out[l] = if bf16 {
                    num::sr_bf16_of_f32(x.v[l], rand.v[l])
                } else {
                    num::sr_f16_of_f32(x.v[l], rand.v[l])
                };
                let sx = dword_src(&x.tag(l));
                st[l] = [sx, sx];
            }
            let h = ((hw >> 3) & 1) as usize;
            write_half_into(wv, &ops[0], h, &out, x.tagged.then_some(&st))?;
        }
        "sr.fp8.f32" | "sr.bf8.f32" => {
            let fmt = small(&name[3..6], fnuz).unwrap_or(Small::E4M3);
            let (x, rand) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            let mut st = [0i64; MAX_W];
            for l in 0..w {
                out[l] = num::down_sr(f64::from(f32_of(x.v[l])), rand.v[l], ONE, fmt);
                st[l] = dword_src(&x.tag(l));
            }
            let byte_idx = ((hw >> 2) & 3) as usize;
            write_byte_into(wv, &ops[0], byte_idx, &out, x.tagged.then_some(&st))?;
        }
        "f32.fp8" | "f32.bf8" => {
            let fmt = small(&name[4..], fnuz).unwrap_or(Small::E4M3);
            let x = wv.vread(arg(1)?, 0)?;
            let mut t = vec![[0; 4]; w];
            for l in 0..w {
                out[l] = num::up_bits(num::up(x.v[l] & 0xFF, ONE, fmt)).0;
                t[l] = [x.tag(l)[0]; 4];
            }
            wv.vwrite(&ops[0], 0, &out, x.tagged.then_some(t.as_slice()), exec)?;
        }
        "pk.f32.fp8" | "pk.f32.bf8" => {
            let fmt = small(&name[7..], fnuz).unwrap_or(Small::E4M3);
            let x = wv.vread(arg(1)?, 0)?;
            for d in 0..2 {
                let mut t = vec![[0; 4]; w];
                for l in 0..w {
                    out[l] = num::up_bits(num::up(x.v[l] >> (8 * d) & 0xFF, ONE, fmt)).0;
                    t[l] = [x.tag(l)[d]; 4];
                }
                wv.vwrite(
                    &ops[0],
                    d as u32,
                    &out,
                    x.tagged.then_some(t.as_slice()),
                    exec,
                )?;
            }
        }
        "pk.fp8.f32" | "pk.bf8.f32" => {
            let fmt = small(&name[3..6], fnuz).unwrap_or(Small::E4M3);
            let (x, y, old) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(&ops[0], 0)?,
            );
            let h = ((hw >> 3) & 1) as usize;
            pack8(wv, ins, &x, &y, &old, fmt, h)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// CDNA 4 `v_cvt_scalef32_*` instructions (`rest` has `"v.cvt.scalef32."` stripped).
#[allow(clippy::too_many_lines)]
fn cvt_scalef32(wv: &mut Wave, ins: &Ins, rest: &str, fnuz: bool) -> Result<bool, EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    let exec = wv.exec;
    let arg = |k: usize| {
        ops.get(k).ok_or_else(|| EmuError {
            cause: Some(Cause::Fault),
            message: format!("{}: operand {k} missing", ins.text),
        })
    };
    let hw = hw_opsel(ins);
    let dst_half = ((hw >> 3) & 1) as usize;
    let dst_byte = ((hw >> 2) & 3) as usize;

    match rest {
        // 1. v_cvt_scalef32_pk_{fp8,bf8}_f32: 3 srcs (S0, S1, S2_scale), writes 16b half OPSEL[3].
        "pk.fp8.f32" | "pk.bf8.f32" => {
            let fmt = small(&rest[3..6], fnuz).expect("fp8/bf8");
            let (x, y, sc) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            let mut vals = [0u32; MAX_W];
            let mut st = [[0i64; 2]; MAX_W];
            for l in 0..w {
                let b0 = num::down(f64::from(f32_of(x.v[l])), sc.v[l], fmt);
                let b1 = num::down(f64::from(f32_of(y.v[l])), sc.v[l], fmt);
                vals[l] = (b0 & 0xFF) | ((b1 & 0xFF) << 8);
                st[l] = [dword_src(&x.tag(l)), dword_src(&y.tag(l))];
            }
            let tagged = x.tagged || y.tagged;
            write_half_into(wv, &ops[0], dst_half, &vals, tagged.then_some(&st))?;
        }
        // 2. v_cvt_scalef32_pk_{fp8,bf8}_{f16,bf16}: 2 srcs (S0_pk2, S1_scale), writes 16b half OPSEL[3].
        "pk.fp8.f16" | "pk.bf8.f16" | "pk.fp8.bf16" | "pk.bf8.bf16" => {
            let fmt = small(&rest[3..6], fnuz).expect("fp8/bf8");
            let bf16 = rest.ends_with(".bf16");
            let (x, sc) = (read_pk2_16(wv, arg(1)?, bf16)?, wv.vread(arg(2)?, 0)?);
            let mut vals = [0u32; MAX_W];
            let mut st = [[0i64; 2]; MAX_W];
            for l in 0..w {
                let v0 = half16_to_f64(x.v[l] & 0xFFFF, bf16);
                let v1 = half16_to_f64(x.v[l] >> 16, bf16);
                let b0 = num::down(v0, sc.v[l], fmt);
                let b1 = num::down(v1, sc.v[l], fmt);
                vals[l] = (b0 & 0xFF) | ((b1 & 0xFF) << 8);
                let xt = x.tag(l);
                st[l] = [dword_src(&xt[0..2]), dword_src(&xt[2..4])];
            }
            write_half_into(wv, &ops[0], dst_half, &vals, x.tagged.then_some(&st))?;
        }
        // 3. v_cvt_scalef32_sr_{fp8,bf8}_{f32,f16,bf16}: 3 srcs (S0, S1_rand, S2_scale), writes byte OPSEL[3:2].
        "sr.fp8.f32" | "sr.bf8.f32" | "sr.fp8.f16" | "sr.bf8.f16" | "sr.fp8.bf16"
        | "sr.bf8.bf16" => {
            let fmt = small(&rest[3..6], fnuz).expect("fp8/bf8");
            let src_ty = &rest[7..];
            let (rand, sc) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(3)?, 0)?);
            let mut vals = [0u32; MAX_W];
            let mut st = [0i64; MAX_W];
            let tagged = if src_ty == "f32" {
                let x = wv.vread(arg(1)?, 0)?;
                for l in 0..w {
                    vals[l] = num::down_sr(f64::from(f32_of(x.v[l])), rand.v[l], sc.v[l], fmt);
                    st[l] = dword_src(&x.tag(l));
                }
                x.tagged
            } else {
                let bf16 = src_ty == "bf16";
                let x = read_half16(wv, arg(1)?, (hw & 1) as usize, bf16)?;
                for l in 0..w {
                    let v = half16_to_f64(x.v[l], bf16);
                    vals[l] = num::down_sr(v, rand.v[l], sc.v[l], fmt);
                    st[l] = dword_src(&x.tag(l)[0..2]);
                }
                x.tagged
            };
            write_byte_into(wv, &ops[0], dst_byte, &vals, tagged.then_some(&st))?;
        }
        // 4. v_cvt_scalef32_pk_{f32,f16,bf16}_{fp8,bf8}: 2 srcs (S0, S1_scale), reads word OPSEL[0].
        "pk.f32.fp8" | "pk.f32.bf8" | "pk.f16.fp8" | "pk.f16.bf8" | "pk.bf16.fp8"
        | "pk.bf16.bf8" => {
            let (dst_ty, src_ty) = rest[3..].split_once('.').expect("dst.src");
            let fmt = small(src_ty, fnuz).expect("fp8/bf8");
            let src_word = (hw & 1) as usize;
            let (x, sc) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            if dst_ty == "f32" {
                let mut outs = [[0u32; MAX_W]; 2];
                let mut tags = [vec![[0; 4]; w], vec![[0; 4]; w]];
                for l in 0..w {
                    for d in 0..2 {
                        let byte_pos = 2 * src_word + d;
                        let code = (x.v[l] >> (8 * byte_pos)) & 0xFF;
                        outs[d][l] = num::up_bits(num::up(code, sc.v[l], fmt)).0;
                        tags[d][l] = [x.tag(l)[byte_pos]; 4];
                    }
                }
                for d in 0..2 {
                    wv.vwrite(
                        &ops[0],
                        d as u32,
                        &outs[d],
                        x.tagged.then_some(tags[d].as_slice()),
                        exec,
                    )?;
                }
            } else {
                let bf16 = dst_ty == "bf16";
                let mut out = [0u32; MAX_W];
                let mut t = vec![[0; 4]; w];
                for l in 0..w {
                    let b0 = (x.v[l] >> (16 * src_word)) & 0xFF;
                    let b1 = (x.v[l] >> (16 * src_word + 8)) & 0xFF;
                    let u0 = num::up_bits(num::up(b0, sc.v[l], fmt));
                    let u1 = num::up_bits(num::up(b1, sc.v[l], fmt));
                    let (h0, h1) = if bf16 { (u0.1, u1.1) } else { (u0.2, u1.2) };
                    out[l] = (h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16);
                    let xt = x.tag(l);
                    let (t0, t1) = (xt[2 * src_word], xt[2 * src_word + 1]);
                    t[l] = [t0, t0, t1, t1];
                }
                wv.vwrite(&ops[0], 0, &out, x.tagged.then_some(t.as_slice()), exec)?;
            }
        }
        // 5. v_cvt_scalef32_{f32,f16}_{fp8,bf8}: 2 srcs (S0, S1_scale), reads byte OPSEL[1:0].
        "f32.fp8" | "f32.bf8" | "f16.fp8" | "f16.bf8" => {
            let (dst_ty, src_ty) = rest.split_once('.').expect("dst.src");
            let fmt = small(src_ty, fnuz).expect("fp8/bf8");
            let src_byte = (hw & 3) as usize;
            let (x, sc) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            if dst_ty == "f32" {
                let mut out = [0u32; MAX_W];
                let mut t = vec![[0; 4]; w];
                for l in 0..w {
                    let code = (x.v[l] >> (8 * src_byte)) & 0xFF;
                    out[l] = num::up_bits(num::up(code, sc.v[l], fmt)).0;
                    t[l] = [x.tag(l)[src_byte]; 4];
                }
                wv.vwrite(&ops[0], 0, &out, x.tagged.then_some(t.as_slice()), exec)?;
            } else {
                let mut vals = [0u32; MAX_W];
                let mut st = [[0i64; 2]; MAX_W];
                for l in 0..w {
                    let code = (x.v[l] >> (8 * src_byte)) & 0xFF;
                    vals[l] = num::up_bits(num::up(code, sc.v[l], fmt)).2;
                    let tb = x.tag(l)[src_byte];
                    st[l] = [tb, tb];
                }
                write_half_into(wv, &ops[0], dst_half, &vals, x.tagged.then_some(&st))?;
            }
        }
        // 6. v_cvt_scalef32_pk_fp4_f32: 3 srcs (S0, S1, S2_scale), writes byte OPSEL[3:2].
        "pk.fp4.f32" => {
            let (x, y, sc) = (
                wv.vread(arg(1)?, 0)?,
                wv.vread(arg(2)?, 0)?,
                wv.vread(arg(3)?, 0)?,
            );
            let mut vals = [0u32; MAX_W];
            for l in 0..w {
                let n0 = num::down(f64::from(f32_of(x.v[l])), sc.v[l], Small::E2M1);
                let n1 = num::down(f64::from(f32_of(y.v[l])), sc.v[l], Small::E2M1);
                vals[l] = (n0 & 0xF) | ((n1 & 0xF) << 4);
            }
            write_byte_into(wv, &ops[0], dst_byte, &vals, None)?;
        }
        // 7. v_cvt_scalef32_pk_fp4_{f16,bf16}: 2 srcs (S0_pk2, S1_scale), writes byte OPSEL[3:2].
        "pk.fp4.f16" | "pk.fp4.bf16" => {
            let bf16 = rest.ends_with(".bf16");
            let (x, sc) = (read_pk2_16(wv, arg(1)?, bf16)?, wv.vread(arg(2)?, 0)?);
            let mut vals = [0u32; MAX_W];
            for l in 0..w {
                let v0 = half16_to_f64(x.v[l] & 0xFFFF, bf16);
                let v1 = half16_to_f64(x.v[l] >> 16, bf16);
                let n0 = num::down(v0, sc.v[l], Small::E2M1);
                let n1 = num::down(v1, sc.v[l], Small::E2M1);
                vals[l] = (n0 & 0xF) | ((n1 & 0xF) << 4);
            }
            write_byte_into(wv, &ops[0], dst_byte, &vals, None)?;
        }
        // 8. v_cvt_scalef32_sr_pk_fp4_f32: 3 srcs (S0_64b, S1_rand, S2_scale), writes byte OPSEL[3:2].
        "sr.pk.fp4.f32" => {
            let s0_lo = wv.vread(arg(1)?, 0)?;
            let s0_hi = wv.vread(arg(1)?, 1)?;
            let (rand, sc) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(3)?, 0)?);
            let mut vals = [0u32; MAX_W];
            for l in 0..w {
                let r0 = rand.v[l];
                let r1 = num::prng_b32(r0);
                let n0 = num::down_sr(f64::from(f32_of(s0_lo.v[l])), r0, sc.v[l], Small::E2M1);
                let n1 = num::down_sr(f64::from(f32_of(s0_hi.v[l])), r1, sc.v[l], Small::E2M1);
                vals[l] = (n0 & 0xF) | ((n1 & 0xF) << 4);
            }
            write_byte_into(wv, &ops[0], dst_byte, &vals, None)?;
        }
        // 9. v_cvt_scalef32_sr_pk_fp4_{f16,bf16}: 3 srcs (S0_pk2, S1_rand, S2_scale), writes byte OPSEL[3:2].
        "sr.pk.fp4.f16" | "sr.pk.fp4.bf16" => {
            let bf16 = rest.ends_with(".bf16");
            let x = read_pk2_16(wv, arg(1)?, bf16)?;
            let (rand, sc) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(3)?, 0)?);
            let mut vals = [0u32; MAX_W];
            for l in 0..w {
                let r0 = rand.v[l];
                let r1 = num::prng_b32(r0);
                let v0 = half16_to_f64(x.v[l] & 0xFFFF, bf16);
                let v1 = half16_to_f64(x.v[l] >> 16, bf16);
                let n0 = num::down_sr(v0, r0, sc.v[l], Small::E2M1);
                let n1 = num::down_sr(v1, r1, sc.v[l], Small::E2M1);
                vals[l] = (n0 & 0xF) | ((n1 & 0xF) << 4);
            }
            write_byte_into(wv, &ops[0], dst_byte, &vals, None)?;
        }
        // 10. v_cvt_scalef32_pk_{f32,f16,bf16}_fp4: 2 srcs (S0, S1_scale), reads byte OPSEL[1:0].
        "pk.f32.fp4" | "pk.f16.fp4" | "pk.bf16.fp4" => {
            let dst_ty = &rest[3..rest.len() - 4];
            let src_byte = (hw & 3) as usize;
            let (x, sc) = (wv.vread(arg(1)?, 0)?, wv.vread(arg(2)?, 0)?);
            if dst_ty == "f32" {
                let mut outs = [[0u32; MAX_W]; 2];
                for l in 0..w {
                    let byte = (x.v[l] >> (8 * src_byte)) & 0xFF;
                    outs[0][l] = num::up_bits(num::up(byte & 0xF, sc.v[l], Small::E2M1)).0;
                    outs[1][l] = num::up_bits(num::up((byte >> 4) & 0xF, sc.v[l], Small::E2M1)).0;
                }
                wv.vwrite(&ops[0], 0, &outs[0], None, exec)?;
                wv.vwrite(&ops[0], 1, &outs[1], None, exec)?;
            } else {
                let bf16 = dst_ty == "bf16";
                let mut out = [0u32; MAX_W];
                for l in 0..w {
                    let byte = (x.v[l] >> (8 * src_byte)) & 0xFF;
                    let u0 = num::up_bits(num::up(byte & 0xF, sc.v[l], Small::E2M1));
                    let u1 = num::up_bits(num::up((byte >> 4) & 0xF, sc.v[l], Small::E2M1));
                    let (h0, h1) = if bf16 { (u0.1, u1.1) } else { (u0.2, u1.2) };
                    out[l] = (h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16);
                }
                wv.vwrite(&ops[0], 0, &out, None, exec)?;
            }
        }
        // 11. v_cvt_scalef32_2xpk16_{fp6,bf6}_f32: interleaves S0[0..16] and S1[0..16] into D[0..6].
        "2xpk16.fp6.f32" | "2xpk16.bf6.f32" => {
            let fmt = small(&rest[7..10], fnuz).expect("fp6/bf6");
            let mut s0 = Vec::with_capacity(16);
            let mut s1 = Vec::with_capacity(16);
            for d in 0..16 {
                s0.push(wv.vread(arg(1)?, d)?);
                s1.push(wv.vread(arg(2)?, d)?);
            }
            let sc = wv.vread(arg(3)?, 0)?;
            let mut outs = [[0u32; MAX_W]; 6];
            for l in 0..w {
                let mut words = [0u32; 6];
                for pass in 0..16 {
                    let c0 = num::down(f64::from(f32_of(s0[pass].v[l])), sc.v[l], fmt);
                    let c1 = num::down(f64::from(f32_of(s1[pass].v[l])), sc.v[l], fmt);
                    put_6bit(&mut words, 2 * pass, c0);
                    put_6bit(&mut words, 2 * pass + 1, c1);
                }
                for d in 0..6 {
                    outs[d][l] = words[d];
                }
            }
            for (d, col) in outs.iter().enumerate() {
                wv.vwrite(&ops[0], d as u32, col, None, exec)?;
            }
        }
        // 12. v_cvt_scalef32_pk32_{fp6,bf6}_{f16,bf16}: 32×16b in S0[0..16] -> D[0..6].
        "pk32.fp6.f16" | "pk32.fp6.bf16" | "pk32.bf6.f16" | "pk32.bf6.bf16" => {
            let fmt = small(&rest[5..8], fnuz).expect("fp6/bf6");
            let bf16 = rest.ends_with(".bf16");
            let mut s0 = Vec::with_capacity(16);
            for d in 0..16 {
                s0.push(wv.vread(arg(1)?, d)?);
            }
            let sc = wv.vread(arg(2)?, 0)?;
            let mut outs = [[0u32; MAX_W]; 6];
            for l in 0..w {
                let mut words = [0u32; 6];
                for pass in 0..32 {
                    let raw16 = (s0[pass / 2].v[l] >> (16 * (pass % 2))) & 0xFFFF;
                    let code = num::down(half16_to_f64(raw16, bf16), sc.v[l], fmt);
                    put_6bit(&mut words, pass, code);
                }
                for d in 0..6 {
                    outs[d][l] = words[d];
                }
            }
            for (d, col) in outs.iter().enumerate() {
                wv.vwrite(&ops[0], d as u32, col, None, exec)?;
            }
        }
        // 13. v_cvt_scalef32_sr_pk32_{fp6,bf6}_{f32,f16,bf16}: 32 elements -> D[0..6], advancing PRNG per pass.
        "sr.pk32.fp6.f32" | "sr.pk32.bf6.f32" | "sr.pk32.fp6.f16" | "sr.pk32.bf6.f16"
        | "sr.pk32.fp6.bf16" | "sr.pk32.bf6.bf16" => {
            let fmt = small(&rest[8..11], fnuz).expect("fp6/bf6");
            let src_ty = &rest[12..];
            let ndw = if src_ty == "f32" { 32 } else { 16 };
            let mut s0 = Vec::with_capacity(ndw);
            for d in 0..ndw {
                s0.push(wv.vread(arg(1)?, d as u32)?);
            }
            let (rand, sc) = (wv.vread(arg(2)?, 0)?, wv.vread(arg(3)?, 0)?);
            let mut outs = [[0u32; MAX_W]; 6];
            let bf16 = src_ty == "bf16";
            for l in 0..w {
                let mut words = [0u32; 6];
                let mut rv = rand.v[l];
                for pass in 0..32 {
                    let val = if src_ty == "f32" {
                        f64::from(f32_of(s0[pass].v[l]))
                    } else {
                        let raw16 = (s0[pass / 2].v[l] >> (16 * (pass % 2))) & 0xFFFF;
                        half16_to_f64(raw16, bf16)
                    };
                    let code = num::down_sr(val, rv, sc.v[l], fmt);
                    put_6bit(&mut words, pass, code);
                    rv = num::prng_b32(rv);
                }
                for d in 0..6 {
                    outs[d][l] = words[d];
                }
            }
            for (d, col) in outs.iter().enumerate() {
                wv.vwrite(&ops[0], d as u32, col, None, exec)?;
            }
        }
        // 14. v_cvt_scalef32_pk32_{f32,f16,bf16}_{fp6,bf6}: S0[0..6] -> D[0..32] or D[0..16].
        "pk32.f32.fp6" | "pk32.f32.bf6" | "pk32.f16.fp6" | "pk32.f16.bf6" | "pk32.bf16.fp6"
        | "pk32.bf16.bf6" => {
            let (dst_ty, src_ty) = rest[5..].split_once('.').expect("dst.src");
            let fmt = small(src_ty, fnuz).expect("fp6/bf6");
            let mut s0 = Vec::with_capacity(6);
            for d in 0..6 {
                s0.push(wv.vread(arg(1)?, d)?);
            }
            let sc = wv.vread(arg(2)?, 0)?;
            if dst_ty == "f32" {
                let mut outs = vec![[0u32; MAX_W]; 32];
                for l in 0..w {
                    let words: [u32; 6] = std::array::from_fn(|d| s0[d].v[l]);
                    for pass in 0..32 {
                        let code = get_6bit(&words, pass);
                        outs[pass][l] = num::up_bits(num::up(code, sc.v[l], fmt)).0;
                    }
                }
                for (d, col) in outs.iter().enumerate() {
                    wv.vwrite(&ops[0], d as u32, col, None, exec)?;
                }
            } else {
                let bf16 = dst_ty == "bf16";
                let mut outs = [[0u32; MAX_W]; 16];
                for l in 0..w {
                    let words: [u32; 6] = std::array::from_fn(|d| s0[d].v[l]);
                    for d in 0..16 {
                        let u0 = num::up_bits(num::up(get_6bit(&words, 2 * d), sc.v[l], fmt));
                        let u1 = num::up_bits(num::up(get_6bit(&words, 2 * d + 1), sc.v[l], fmt));
                        let (h0, h1) = if bf16 { (u0.1, u1.1) } else { (u0.2, u1.2) };
                        outs[d][l] = (h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16);
                    }
                }
                for (d, col) in outs.iter().enumerate() {
                    wv.vwrite(&ops[0], d as u32, col, None, exec)?;
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Packs two down-converted f32 values into half `h` of d.
fn pack8(
    wv: &mut Wave,
    ins: &Ins,
    x: &Src,
    y: &Src,
    old: &Src,
    fmt: Small,
    h: usize,
) -> Result<(), EmuError> {
    let w = wv.w;
    let mut out = [0u32; MAX_W];
    let mut t = old.tags(w).to_vec();
    for l in 0..w {
        let a = num::down(f64::from(f32_of(x.v[l])), ONE, fmt);
        let b = num::down(f64::from(f32_of(y.v[l])), ONE, fmt);
        out[l] = put_byte(put_byte(old.v[l], 2 * h, a), 2 * h + 1, b);
        t[l][2 * h] = dword_src(&x.tag(l));
        t[l][2 * h + 1] = dword_src(&y.tag(l));
    }
    let tagged = x.tagged || y.tagged || old.tagged;
    wv.vwrite(
        &ins.operands[0],
        0,
        &out,
        tagged.then_some(t.as_slice()),
        wv.exec,
    )
}
