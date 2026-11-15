//! The tag transfer rules on lane arrays.
//!
//! An operand's tags are `[lanes][4]`; a rule returns the result's tags for
//! every lane, and the machine writes them in the EXEC lanes only.

use super::checker::TagError;
use super::combine::combine;
use super::encoding::{is_provenance_record, is_record, is_sum, src, Tables, Tag};

/// One lane's dword of tags.
pub type Lane = [Tag; 4];

/// The tag context of one rule: the tables and the sum mode.
pub struct Ctx<'a> {
    /// The process's tables.
    pub tables: &'a mut Tables,
    /// Sum mode.
    pub sum_mode: bool,
}

fn any_below_mark(t: &[Lane]) -> bool {
    t.iter().any(|l| l.iter().any(|b| *b < -1))
}

fn any_record(t: &[Lane]) -> bool {
    t.iter().any(|l| l.iter().any(|b| is_record(*b)))
}

/// The sum source of one operand lane.
fn sum_source(l: &Lane) -> Tag {
    let low = l.iter().copied().min().unwrap_or(0);
    if is_sum(low) {
        return low;
    }
    l.iter().copied().find(|b| *b > 0).unwrap_or(0)
}

/// The sum rule of one lane.
pub fn sum_lane(tables: &mut Tables, x: &Lane, y: &Lane) -> Lane {
    let (sx, sy) = (sum_source(x), sum_source(y));
    let out = match (sx, sy) {
        (0, 0) => 0,
        (0, s) | (s, 0) => s,
        _ => tables.sum(sx, sy),
    };
    [out; 4]
}

/// `float(x, y, stat, add)`.
pub fn float(
    ctx: &mut Ctx<'_>,
    x: &[Lane],
    y: &[Lane],
    stat: bool,
    add: bool,
) -> Result<Vec<Lane>, TagError> {
    if ctx.sum_mode && add {
        return Ok(x
            .iter()
            .zip(y)
            .map(|(a, b)| sum_lane(ctx.tables, a, b))
            .collect());
    }
    if !any_below_mark(x) && !any_below_mark(y) {
        return Ok(vec![[0; 4]; x.len()]);
    }
    let pairs: Vec<(Tag, Tag)> = x.iter().zip(y).map(|(a, b)| (src(a), src(b))).collect();
    if pairs
        .iter()
        .any(|(a, b)| is_sum(*a) && is_sum(*b) && a != b)
    {
        return Err(TagError::fault(
            "a product / max of two different sum-record values: unsupported multiplication of distinct sum records",
        ));
    }
    let mut out = Vec::with_capacity(pairs.len());
    for (rx, ry) in pairs {
        let both = rx < -1 && ry < -1;
        let t = if both && (rx != ry || add) {
            if is_sum(rx) || is_sum(ry) {
                rx.min(ry)
            } else {
                combine(ctx.tables, rx, ry, stat)
            }
        } else if rx < -1 {
            rx
        } else if ry < -1 {
            ry
        } else {
            0
        };
        out.push([t; 4]);
    }
    Ok(out)
}

/// `float(float(x, y), z, add = true)`: the fused forms.
pub fn fused(ctx: &mut Ctx<'_>, x: &[Lane], y: &[Lane], z: &[Lane]) -> Result<Vec<Lane>, TagError> {
    let p = float(ctx, x, y, false, false)?;
    float(ctx, &p, z, false, true)
}

/// The three-operand statistics: `float(float(x, y, stat), z, stat)`.
pub fn stat3(ctx: &mut Ctx<'_>, x: &[Lane], y: &[Lane], z: &[Lane]) -> Result<Vec<Lane>, TagError> {
    let p = float(ctx, x, y, true, false)?;
    float(ctx, &p, z, true, false)
}

/// and, per byte.
pub fn and_lane(xv: u32, xt: &Lane, yv: u32, yt: &Lane) -> Lane {
    let mut out = [0; 4];
    for b in 0..4 {
        let (xb, yb) = ((xv >> (8 * b)) & 0xFF, (yv >> (8 * b)) & 0xFF);
        out[b] = if yb == 0xFF && xt[b] != 0 {
            xt[b]
        } else if xb == 0xFF {
            yt[b]
        } else if yb == 0xFF {
            xt[b]
        } else {
            0
        };
    }
    out
}

/// or, per byte: tags decided before data.
pub fn or_lane(xv: u32, xt: &Lane, yv: u32, yt: &Lane) -> Lane {
    let mut out = [0; 4];
    for b in 0..4 {
        let (xb, yb) = ((xv >> (8 * b)) & 0xFF, (yv >> (8 * b)) & 0xFF);
        out[b] = if xt[b] == 0 {
            yt[b]
        } else if yt[b] == 0 {
            xt[b]
        } else if xb == 0 {
            yt[b]
        } else if yb == 0 {
            xt[b]
        } else {
            0
        };
    }
    out
}

/// shift of one lane by `amount` (mod 32): whole bytes move, else nothing.
pub fn shift_lane(t: &Lane, amount: u32, left: bool) -> Lane {
    let a = amount % 32;
    if a % 8 != 0 {
        return [0; 4];
    }
    let s = (a / 8) as usize;
    let mut out = [0; 4];
    for b in 0..4 {
        out[b] = if left {
            if b >= s {
                t[b - s]
            } else {
                0
            }
        } else if b + s < 4 {
            t[b + s]
        } else {
            0
        };
    }
    out
}

/// The bit-field extract's byte rule.
pub fn bfe_lane(t: &Lane, offset: u32, width: u32) -> Lane {
    let first = ((offset % 32) / 8) as usize;
    let nb = (width % 32).div_ceil(8) as usize;
    let mut out = [0; 4];
    for (j, o) in out.iter_mut().enumerate() {
        if j < nb && first + j < 4 {
            *o = t[first + j];
        }
    }
    out
}

/// R6b byte permute: selectors 0–3 pick y's bytes, 4–7 x's, ≥ 8 untagged.
pub fn perm_lane(xt: &Lane, yt: &Lane, sel: u32) -> Lane {
    let mut out = [0; 4];
    for (k, o) in out.iter_mut().enumerate() {
        let s = (sel >> (8 * k)) & 0xFF;
        *o = match s {
            0..=3 => yt[s as usize],
            4..=7 => xt[s as usize - 4],
            _ => 0,
        };
    }
    out
}

/// The integer rule: `bytes` is the byte rule's result for and, or
/// and shifts (none for the arithmetic ops).
pub fn integer(
    ctx: &mut Ctx<'_>,
    x: &[Lane],
    y: &[Lane],
    bytes: Option<Vec<Lane>>,
    shift: bool,
) -> Result<Vec<Lane>, TagError> {
    let records = any_record(x) || any_record(y);
    if (bytes.is_none() || shift) && records {
        let r = float(ctx, x, y, false, false)?;
        return Ok(match bytes {
            None => r,
            Some(t) => r
                .iter()
                .zip(t)
                .map(|(rl, tl)| if is_record(src(rl)) { *rl } else { tl })
                .collect(),
        });
    }
    Ok(bytes.unwrap_or_else(|| vec![[0; 4]; x.len()]))
}

/// Whether any byte of an operand is a record or a sum record.
pub fn has_records(t: &[Lane]) -> bool {
    t.iter().any(|l| l.iter().any(|b| is_provenance_record(*b)))
}
