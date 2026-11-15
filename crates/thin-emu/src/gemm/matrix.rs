//! The matrix-op check: decoding the operands, pairing and
//! accumulating the records.

use std::collections::BTreeSet;

use crate::tags::{
    is_record, py_list, tag_buf, tag_off, AccState, Cols, MatrixOp, MatrixOutcome, OperandFormat,
    Tables, Tag, TagError,
};

use super::layout::Kind;
use super::GemmHooks;

/// A decoded operand: its kind, the row of each slot, and per lane
/// group the memory k at each position (−1: none).
#[derive(Clone, Debug)]
struct Operand {
    kind: Kind,
    rows: Vec<Option<i64>>,
    kmem: Vec<Vec<i64>>,
}

/// One operand byte decoded: (row, k, sub), or none for a zero byte.
type Byte = Option<(i64, i64, i64)>;

/// Per lane and position, (row, k) or none: the decode's working form.
type Grid = Vec<Vec<Option<(i64, i64)>>>;

fn report(h: &mut GemmHooks, pi: usize, msg: String) -> Result<(), TagError> {
    h.probs[pi].err(msg)
}

fn distinct(v: impl IntoIterator<Item = i64>, most: usize) -> String {
    let s: BTreeSet<i64> = v.into_iter().collect();
    py_list(s.into_iter().take(most))
}

/// Validate per-byte row and K coordinates and accumulate coverage.
/// Where a decode reports and what shapes it: the problem, the message
/// prefix, the form's R and the operand's format.
struct Site<'a> {
    pi: usize,
    pre: &'a str,
    r: usize,
    fmt: &'a OperandFormat,
}

fn shape(
    h: &mut GemmHooks,
    site: &Site<'_>,
    bytes: &[Vec<Byte>],
    sub_check: bool,
    kind: Kind,
) -> Result<Operand, TagError> {
    let Site { pi, pre, r, fmt } = *site;
    let w = bytes.len();
    let g = w / r;
    let mut grid: Grid = Vec::with_capacity(w);
    let mut sub_error = None;
    for (l, lane) in bytes.iter().enumerate() {
        let mut pos = Vec::new();
        for (b, byte) in lane.iter().enumerate() {
            if let (true, Some((_, _, sub))) = (sub_check && fmt.elem_bytes > 1, byte) {
                let want = (b % fmt.elem_bytes) as i64;
                if *sub != want && sub_error.is_none() {
                    sub_error = Some((l, b, *sub, want));
                }
            }
            pos.push(byte.map(|(row, k, _)| (row, k)));
        }
        grid.push(pos);
    }
    if let Some((l, b, s, e)) = sub_error {
        report(
            h,
            pi,
            format!("{pre} lane {l} byte {b} is byte {s} of its element (expected {e})"),
        )?;
    }
    let rowzero: Vec<bool> = grid.iter().map(|p| p.iter().all(Option::is_none)).collect();
    let mut lane_row = vec![None; w];
    let mut reported = false;
    for (l, pos) in grid.iter().enumerate() {
        let rows: Vec<i64> = pos.iter().flatten().map(|(row, _)| *row).collect();
        let Some(first) = rows.first().copied() else {
            continue;
        };
        lane_row[l] = Some(first);
        if !reported && rows.iter().any(|x| *x != first) {
            reported = true;
            report(
                h,
                pi,
                format!(
                    "{pre} lane {l} holds rows {} (one row expected)",
                    distinct(rows, 4)
                ),
            )?;
        }
    }
    let mut slots = vec![None; r];
    for (s, slot) in slots.iter_mut().enumerate() {
        let held: Vec<i64> = (0..g).filter_map(|q| lane_row[s + r * q]).collect();
        let set: BTreeSet<i64> = held.iter().copied().collect();
        if set.len() > 1 {
            report(
                h,
                pi,
                format!(
                    "{pre} row slot {s}: lane groups hold different rows {}",
                    distinct(held.clone(), 4)
                ),
            )?;
        }
        *slot = held.first().copied();
    }
    let p = grid.first().map_or(0, Vec::len);
    let mut kmem = vec![vec![-1; p]; g];
    for (q, kq) in kmem.iter_mut().enumerate() {
        let lanes: Vec<usize> = (q * r..q * r + r).filter(|l| !rowzero[*l]).collect();
        let Some(&first) = lanes.first() else {
            continue;
        };
        let ks = |l: usize| {
            grid[l]
                .iter()
                .map(|x| x.map_or(-1, |(_, k)| k))
                .collect::<Vec<i64>>()
        };
        let base = ks(first);
        for (pos, want) in base.iter().enumerate() {
            let vals: Vec<i64> = lanes.iter().map(|l| ks(*l)[pos]).collect();
            if vals.iter().any(|v| v != want) {
                report(
                    h,
                    pi,
                    format!(
                        "{pre} lane group {q} position {pos}: rows hold different k {}",
                        distinct(vals, 6)
                    ),
                )?;
                break;
            }
        }
        *kq = base;
    }
    Ok(Operand {
        kind,
        rows: slots,
        kmem,
    })
}

/// decodes one operand's raw tags; none when absent.
fn decode(
    h: &mut GemmHooks,
    t: &mut Tables,
    op: &MatrixOp<'_>,
    s: usize,
) -> Result<Option<Operand>, TagError> {
    let pi = h.current;
    let pre = format!("{}: src{s}", op.text);
    let o = &op.src[s];
    let tags = o.tags;
    if tags.iter().flatten().any(|x| is_record(*x)) {
        return record_operand(h, t, op, s, &pre);
    }
    let mut bufs: BTreeSet<i64> = BTreeSet::new();
    for x in tags.iter().flatten() {
        if *x > 0 {
            bufs.insert(tag_buf(*x));
        }
    }
    if bufs.len() > 1 {
        let names: Vec<String> = bufs.iter().map(|b| format!("'{}'", h.name(*b))).collect();
        report(
            h,
            pi,
            format!("{pre} reads from several buffers: [{}]", names.join(", ")),
        )?;
    }
    let Some(&buf) = bufs.iter().next() else {
        return Ok(None);
    };
    let layout = usize::try_from(buf).ok().and_then(|b| h.layout_of(b));
    let name = h.name(buf);
    let Some(layout) = layout.filter(|l| matches!(l.kind, Kind::A | Kind::B)) else {
        report(
            h,
            pi,
            format!("{pre} holds bytes of {name}, not an operand"),
        )?;
        return Ok(None);
    };
    let k_lim = h.probs[h.current].k;
    let mut bytes = Vec::with_capacity(tags.len());
    for lane in tags {
        let mut out = Vec::with_capacity(lane.len());
        for x in lane {
            if *x <= 0 {
                out.push(None);
                continue;
            }
            let d = layout.decode(tag_off(*x));
            let k = d.k;
            out.push(if k >= k_lim {
                None
            } else {
                Some((d.row, k, d.sub))
            });
        }
        bytes.push(out);
    }
    let site = Site {
        pi,
        pre: &pre,
        r: op.form.r,
        fmt: o.format,
    };
    Ok(Some(shape(h, &site, &bytes, true, layout.kind)?))
}

/// an operand of a previous problem's D records.
fn record_operand(
    h: &mut GemmHooks,
    t: &mut Tables,
    op: &MatrixOp<'_>,
    s: usize,
    pre: &str,
) -> Result<Option<Operand>, TagError> {
    let start = h.current;
    let tags = op.src[s].tags;
    let mixes = tags
        .iter()
        .position(|lane| lane.contains(&0) && lane.iter().any(|x| *x != 0));
    if let Some(l) = mixes {
        report(
            h,
            start,
            format!("{pre} lane {l} mixes provenance and zero bytes"),
        )?;
    }
    let distinct: BTreeSet<Tag> = tags
        .iter()
        .flatten()
        .copied()
        .filter(|x| is_record(*x))
        .collect();
    let pids: BTreeSet<i64> = distinct.iter().map(|x| t.record_of(*x).pid).collect();
    let linked = h.probs.iter().position(|p| {
        p.record_operand
            .as_ref()
            .is_some_and(|r| pids.contains(&r.source))
    });
    match linked {
        Some(pi) => h.current = pi,
        None if h.probs[h.current].record_operand.is_none() => {
            report(
                h,
                start,
                format!(
                    "{pre} holds accumulator records, but no problem declares them as an operand"
                ),
            )?;
            return Ok(None);
        }
        None => {}
    }
    let link = h.probs[h.current]
        .record_operand
        .clone()
        .unwrap_or(super::RecordOperand {
            side: 0,
            source: -1,
            map: None,
        });
    if tags.iter().flatten().any(|x| *x > 0) {
        report(h, start, format!("{pre} mixes memory bytes and records"))?;
    }
    let source = t.problem(link.source);
    let full: BTreeSet<i64> = if source.k >= 32 {
        (0..(source.k + 31) / 32).collect()
    } else {
        [0].into()
    };
    let mut mapped = std::collections::HashMap::new();
    for x in &distinct {
        let r = t.record_of(*x);
        let cell = r.cell();
        let fault = if r.is_poison() {
            Some(format!("holds a value that {}", r.reason()))
        } else if r.pid != link.source {
            Some(format!("holds results of another problem {cell}"))
        } else if r.stat || matches!(r.n, Cols::Many(_)) {
            Some(format!(
                "holds a statistic over columns {}, not a value",
                r.n.text()
            ))
        } else if r.blocks() != full {
            let missing: Vec<i64> = full.difference(&r.blocks()).copied().collect();
            Some(format!(
                "holds {cell} with k-blocks {} missing: not the source's complete result",
                py_list(missing)
            ))
        } else {
            None
        };
        if let Some(f) = fault {
            report(h, start, format!("{pre} {f}"))?;
        }
        let at = if r.is_poison() {
            (0, 0)
        } else {
            let (m, n) = (r.m, r.n.first());
            match &link.map {
                None => (m, n),
                Some((rows, ks)) => {
                    let get = |t: &Vec<Vec<i64>>| {
                        usize::try_from(m)
                            .ok()
                            .and_then(|i| t.get(i))
                            .and_then(|row| usize::try_from(n).ok().and_then(|j| row.get(j)))
                            .copied()
                            .unwrap_or(-1)
                    };
                    (get(rows), get(ks))
                }
            }
        };
        mapped.insert(*x, at);
    }
    let bytes: Vec<Vec<Byte>> = tags
        .iter()
        .map(|lane| {
            lane.iter()
                .map(|x| mapped.get(x).map(|(row, k)| (*row, *k, 0)))
                .collect()
        })
        .collect();
    let kind = if link.side == 0 { Kind::A } else { Kind::B };
    let site = Site {
        pi: start,
        pre,
        r: op.form.r,
        fmt: op.src[s].format,
    };
    Ok(Some(shape(h, &site, &bytes, false, kind)?))
}

/// The hardware-k array of an operand: m[hwk[q][e]] = its memory k.
fn by_hwk(o: &Operand, fmt: &OperandFormat, k_hw: i64) -> Vec<i64> {
    let mut m = vec![-1; k_hw.max(0) as usize];
    let step = fmt.elem_bytes;
    for (q, ks) in o.kmem.iter().enumerate() {
        for (e, hk) in fmt.hwk.get(q).into_iter().flatten().enumerate() {
            if let (Some(k), Some(slot)) = (ks.get(e * step), usize::try_from(*hk).ok()) {
                if slot < m.len() {
                    m[slot] = *k;
                }
            }
        }
    }
    m
}

/// The matrix-op check of one op.
pub fn check(
    h: &mut GemmHooks,
    t: &mut Tables,
    op: &MatrixOp<'_>,
    acc: &mut AccState,
) -> Result<MatrixOutcome, TagError> {
    let start = h.current;
    let ops = [decode(h, t, op, 0)?, decode(h, t, op, 1)?];
    if let (Some(a), Some(b)) = (&ops[0], &ops[1]) {
        if a.kind == b.kind {
            report(
                h,
                start,
                format!(
                    "{}: both sources are {:?} operands (one A and one B)",
                    op.text, a.kind
                ),
            )?;
        }
    }
    let k_hw = op.form.k_hw;
    let ms = [0, 1].map(|s| {
        ops[s]
            .as_ref()
            .map_or_else(Vec::new, |o| by_hwk(o, op.src[s].format, k_hw))
    });
    let k_lim = h.probs[h.current].k;
    let ka = match (&ops[0], &ops[1]) {
        (Some(_), Some(_)) => {
            if let Some(hk) = (0..ms[0].len()).find(|i| ms[0][*i] != ms[1][*i]) {
                report(
                    h,
                    start,
                    format!(
                        "{}: at hardware k {hk} src0 holds memory k {} and src1 k {}",
                        op.text, ms[0][hk], ms[1][hk]
                    ),
                )?;
            }
            let held: Vec<i64> = ms[0]
                .iter()
                .copied()
                .filter(|k| *k >= 0)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            if let (Some(&lo), Some(&hi)) = (held.first(), held.last()) {
                let consecutive = hi - lo + 1 == held.len() as i64;
                if !consecutive || !(held.len() as i64 == k_hw || hi == k_lim - 1) {
                    report(
                        h,
                        start,
                        format!(
                            "{}: the k-block is not contiguous: k {lo}..{hi} ({} distinct)",
                            op.text,
                            held.len()
                        ),
                    )?;
                }
            }
            held.first().copied()
        }
        (Some(_), None) => ms[0].iter().copied().filter(|k| *k >= 0).min(),
        (None, Some(_)) => ms[1].iter().copied().filter(|k| *k >= 0).min(),
        (None, None) => None,
    };
    let k_lim = h.probs[h.current].k;
    let blocks: BTreeSet<i64> = match ka {
        Some(ka) => (ka / 32..((ka + k_hw).min(k_lim) + 31) / 32).collect(),
        None => BTreeSet::new(),
    };
    let a_first = ops[0].as_ref().is_some_and(|o| o.kind == Kind::A)
        || (ops[0].is_none() && ops[1].as_ref().is_some_and(|o| o.kind == Kind::B));
    let row = |s: usize, i: usize| {
        ops[s]
            .as_ref()
            .and_then(|o| o.rows.get(i).copied().flatten())
            .unwrap_or(-1)
    };
    let pid = h.probs[h.current].pid;
    for (e, lanes) in op.form.d_map.iter().enumerate() {
        let new: Vec<(i64, i64)> = lanes
            .iter()
            .map(|(i0, i1)| {
                let (ra, rb) = (row(0, *i0), row(1, *i1));
                let (m, n) = if a_first { (ra, rb) } else { (rb, ra) };
                if m < 0 || n < 0 {
                    (-1, -1)
                } else {
                    (m, n)
                }
            })
            .collect();
        let entry = acc.entry(op.entry + e);
        let fresh = op.c_zero || !op.marked.get(e).copied().unwrap_or(false);
        if fresh {
            entry.m = new.iter().map(|x| x.0).collect();
            entry.n = new.iter().map(|x| x.1).collect();
            entry.k = blocks.clone();
            entry.pid = pid;
            continue;
        }
        let clash = (0..new.len())
            .find(|l| entry.m[*l] >= 0 && new[*l].0 >= 0 && (entry.m[*l], entry.n[*l]) != new[*l]);
        let shared: Vec<i64> = entry.k.intersection(&blocks).copied().collect();
        let old = clash.map(|l| (l, entry.m[l], entry.n[l]));
        for (l, (m, n)) in new.iter().enumerate() {
            if *m >= 0 {
                entry.m[l] = *m;
            }
            if *n >= 0 {
                entry.n[l] = *n;
            }
        }
        entry.k.extend(blocks.iter().copied());
        if let Some((l, om, on)) = old {
            let (m, n) = new[l];
            report(
                h,
                start,
                format!(
                    "{}: accumulator {} dword {e} lane {l} held ({om}, {on}), now adds ({m}, {n})",
                    op.text, op.dst
                ),
            )?;
        }
        if !shared.is_empty() {
            report(
                h,
                start,
                format!(
                    "{}: accumulator {} adds k-blocks {} a second time",
                    op.text,
                    op.dst,
                    py_list(shared)
                ),
            )?;
        }
    }
    Ok(MatrixOutcome::Checked)
}
