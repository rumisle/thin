//! The conversions with their tag rules.

use super::alu::list_at;
use super::numbers::{self as num, f32_of, Small};
use super::wave::{Src, Wave, MAX_W};
use crate::tags::src as dword_src;
use crate::{EmuError, Ins};

/// The scale exponent 1.0 has.
const ONE: u32 = 0x3F80_0000;

fn small(name: &str, fnuz: bool) -> Option<Small> {
    Some(match (name, fnuz) {
        ("fp8", false) => Small::E4M3,
        ("bf8", false) => Small::E5M2,
        ("fp8", true) => Small::E4M3Fnuz,
        ("bf8", true) => Small::E5M2Fnuz,
        _ => return None,
    })
}

fn sel(ins: &Ins, i: usize) -> usize {
    list_at(ins, "op_sel", i, 0) as usize & 1
}

fn put_byte(d: u32, k: usize, b: u32) -> u32 {
    (d & !(0xFF << (8 * k))) | ((b & 0xFF) << (8 * k))
}

/// Runs a conversion; `Ok(false)` when `canon` is not one.
#[allow(clippy::too_many_lines)]
pub fn cvt(wv: &mut Wave, ins: &Ins, canon: &str, fnuz: bool) -> Result<bool, EmuError> {
    let w = wv.w;
    let ops = &ins.operands;
    let exec = wv.exec;
    let arg = |k: usize| {
        ops.get(k).ok_or_else(|| EmuError {
            cause: Some(crate::Cause::Fault),
            message: format!("{}: operand {k} missing", ins.text),
        })
    };
    let mut out = [0u32; MAX_W];
    let name = canon.trim_start_matches("v.cvt.");
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
        "f16.f32" | "pk.rtz.f16.f32" => {
            let x = wv.vread(arg(1)?, 0)?;
            let y = if name == "f16.f32" {
                Src::splat(0)
            } else {
                wv.vread(arg(2)?, 0)?
            };
            let mut t = vec![[0; 4]; w];
            for l in 0..w {
                let one = |v: u32| match name {
                    "f16.f32" => num::f16_of_f32(v),
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
            let h = sel(ins, 2);
            pack8(wv, ins, &x, &y, &old, fmt, h)?;
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
