//! Pending register writes: on a counter whose results land
//! at a wait, a returning memory op's register
//! write is kept until a wait retires it in issue order, or the wave ends.
//! Any other instruction touching one of its registers meanwhile is a
//! `RegisterInFlight` error: a missing wait.

use super::facts::{role, Facts};
use super::wave::{FileTags, Wave};
use crate::{Base, Cause, EmuError, Ins};

/// Registers' contents and tags, saved whole: (file, first, count).
#[derive(Clone, Debug)]
pub struct Saved {
    file: char,
    first: u32,
    count: u32,
    vals: Vec<u32>,
    vtags: FileTags,
}

/// A load's register write not yet landed.
#[derive(Clone, Debug)]
pub struct Pending {
    /// The load's instruction index.
    pub pc: usize,
    /// Its text.
    pub text: String,
    /// Its counter (0 load, 1 store, 2 ds, 3 km).
    pub counter: usize,
    /// Its sequence number on that counter.
    pub seq: u64,
    /// Retired only by a count of 0 (gfx1201's scalar loads).
    pub unordered: bool,
    /// The registers and values it writes when it lands.
    pub result: Saved,
}

fn save(w: &Wave, file: char, first: u32, count: u32) -> Saved {
    let (f, n) = (first as usize, count as usize);
    match file {
        's' => Saved {
            file,
            first,
            count,
            vals: (f..f + n)
                .map(|r| w.s.get(r).copied().unwrap_or(0))
                .collect(),
            vtags: Vec::new(),
        },
        _ => {
            let (vals, tags) = if file == 'a' {
                (&w.a, &w.at)
            } else {
                (&w.v, &w.vt)
            };
            let lanes = w.w;
            let end = ((f + n) * lanes).min(vals.len());
            Saved {
                file,
                first,
                count,
                vals: vals[(f * lanes).min(end)..end].to_vec(),
                vtags: (f..f + n).map(|r| tags.get(r).cloned().flatten()).collect(),
            }
        }
    }
}

fn put(w: &mut Wave, s: &Saved) {
    let (f, n) = (s.first as usize, s.count as usize);
    match s.file {
        's' => {
            for (k, r) in (f..f + n).enumerate() {
                if r < w.s.len() {
                    w.s[r] = s.vals[k];
                }
            }
        }
        file => {
            let lanes = w.w;
            let (vals, tags) = if file == 'a' {
                (&mut w.a, &mut w.at)
            } else {
                (&mut w.v, &mut w.vt)
            };
            let start = f * lanes;
            vals[start..start + s.vals.len()].copy_from_slice(&s.vals);
            for (k, r) in (f..f + n).enumerate() {
                if r < tags.len() {
                    tags[r] = s.vtags[k].clone();
                }
            }
        }
    }
}

/// The (file, first, count) of a register operand, v/s/a only.
fn reg_of(base: &Base) -> Option<(char, u32, u32)> {
    match base {
        Base::Reg(f, r, n) if matches!(f.as_str(), "v" | "s" | "a") => {
            Some((f.chars().next().unwrap_or('v'), *r, (*n).max(1)))
        }
        _ => None,
    }
}

/// The counter, unordered flag and destination of a returning memory op on
/// a counter that lands at a wait; none otherwise.
pub fn landing(facts: &Facts, ins: &Ins) -> Option<(usize, bool, (char, u32, u32))> {
    let c = ins.canon.as_str();
    let (counter, unordered) = match ins.kind.as_str() {
        "smem"
            if c.starts_with("s.load")
                || c.starts_with("s.buffer.load")
                || c == "s.memtime"
                || c == "s.memrealtime" =>
        {
            (role::SMEM, true)
        }
        // A returning message counts with scalar memory, out of order.
        "salu" if c.starts_with("s.sendmsg.rtn") => (role::SMEM, true),
        "ds" if c.starts_with("ds.load")
            || c.starts_with("ds.bpermute")
            || c.starts_with("ds.permute")
            || c.starts_with("ds.swizzle")
            || c.contains(".rtn") =>
        {
            (role::DS, false)
        }
        "vmem" => {
            if c.contains(".load.lds.") || ins.field("lds").is_some() {
                return None;
            }
            let load = c.contains(".load");
            let atomic = c.contains(".atomic")
                && ins.operands.len() >= 4
                && facts.scope.returns.is_none_or(|f| ins.field(f).is_some());
            if !(load || atomic) {
                return None;
            }
            (role::LOAD, false)
        }
        _ => return None,
    };
    if !facts.lands.get(counter).copied().unwrap_or(false) {
        return None;
    }
    let dest = reg_of(&ins.operands.first()?.base)?;
    Some((counter, unordered, dest))
}

/// RegisterInFlight if an instruction touches a pending register.
pub fn check(w: &Wave, ins: &Ins) -> Result<(), EmuError> {
    if w.pending.is_empty() {
        return Ok(());
    }
    let mut hits: Vec<(char, u32, &Pending)> = Vec::new();
    for o in &ins.operands {
        let Some((file, r, n)) = reg_of(&o.base) else {
            continue;
        };
        for p in &w.pending {
            let (pf, pr, pn) = (p.result.file, p.result.first, p.result.count);
            if pf != file {
                continue;
            }
            let (lo, hi) = (r.max(pr), (r + n).min(pr + pn));
            if lo < hi {
                hits.push((file, lo, p));
            }
        }
    }
    let Some((file, reg, p)) = hits.into_iter().min_by_key(|h| (h.0, h.1)) else {
        return Ok(());
    };
    Err(EmuError {
        cause: Some(Cause::InFlight),
        message: format!(
            "{}: {file}{reg} is still being loaded by `{}` (instruction {}): a wait must retire it first",
            ins.text, p.text, p.pc
        ),
    })
}

/// Records a pending write: the op has run (its registers written), so
/// they are put back and the result kept.
pub fn record(
    w: &mut Wave,
    before: Saved,
    pc: usize,
    text: &str,
    counter: usize,
    seq: u64,
    unordered: bool,
) {
    let after = save(w, before.file, before.first, before.count);
    put(w, &before);
    w.pending.push(Pending {
        pc,
        text: text.to_owned(),
        counter,
        seq,
        unordered,
        result: after,
    });
}

/// The registers a landing op will write, as they are before it.
pub fn before(w: &Wave, dest: (char, u32, u32)) -> Saved {
    save(w, dest.0, dest.1, dest.2)
}

/// A wait with count n on a counter, `issued` its ops so far: lands the
/// pending writes it retires, in issue order.
pub fn retire(w: &mut Wave, counter: usize, n: u64, issued: u64) {
    let done = issued.saturating_sub(n);
    let (land, keep): (Vec<Pending>, Vec<Pending>) = std::mem::take(&mut w.pending)
        .into_iter()
        .partition(|p| p.counter == counter && (if p.unordered { n == 0 } else { p.seq < done }));
    w.pending = keep;
    for p in land {
        put(w, &p.result);
    }
}

/// The wave's end: every pending write lands, in issue order.
pub fn retire_all(w: &mut Wave) {
    for p in std::mem::take(&mut w.pending) {
        put(w, &p.result);
    }
}
