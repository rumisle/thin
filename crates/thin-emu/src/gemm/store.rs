//! The store checks: D stores, scatter atomics and the
//! coverage log.

use crate::tags::{
    blocks_text, is_record, py_list, AtomicAnswer, Cols, StoreCall, Tables, Tag, TagError,
};

use super::layout::Kind;
use super::problem::Cell;
use super::GemmHooks;

fn bytes(call: &StoreCall<'_>) -> Vec<(u64, Tag)> {
    let mut out = Vec::new();
    for (offs, tags) in call.offsets.iter().zip(call.tags) {
        for (o, t) in offs.iter().zip(tags) {
            out.push((*o, *t));
        }
    }
    out
}

fn cols_text(c: &Cols) -> String {
    c.text()
}

/// `store(buffer, offsets, data, tags, wave)`.
pub fn store(h: &mut GemmHooks, t: &mut Tables, call: &StoreCall<'_>) -> Result<(), TagError> {
    let Some(pi) = h.owner_of(call.buffer) else {
        return Ok(());
    };
    h.current = pi;
    let Some(layout) = h.probs[pi].layout(call.name).cloned() else {
        return Ok(());
    };
    if layout.kind != Kind::D {
        return Ok(());
    }
    let w = call.wave;
    let mut kept = Vec::new();
    for (off, tag) in bytes(call) {
        let d = layout.decode(off);
        let want_m = h.probs[pi].want_m(d.row).map_err(TagError::fault)?;
        if want_m >= 0 {
            kept.push((d, want_m, tag));
        }
    }
    let p = &mut h.probs[pi];
    if let Some((d, _, _)) = kept.iter().find(|(_, _, tag)| *tag == 0) {
        p.err(format!(
            "wave {w}: D[{}][{}] stored from bytes with no provenance",
            d.row, d.k
        ))?;
    }
    if let Some((d, _, _)) = kept
        .iter()
        .find(|(_, _, tag)| *tag != 0 && !is_record(*tag))
    {
        p.err(format!(
            "wave {w}: D[{}][{}] stored from a plain memory byte, not an accumulator",
            d.row, d.k
        ))?;
    }
    let mut cells: Vec<Cell> = Vec::new();
    for (d, want_m, tag) in kept {
        if !is_record(tag) {
            continue;
        }
        let (m, n) = (d.row, d.k);
        let r = t.record_of(tag);
        let p = &mut h.probs[pi];
        if r.is_poison() {
            p.err(format!(
                "wave {w}: D[{m}][{n}] stored from a value that {}",
                r.reason()
            ))?;
            continue;
        }
        if r.m != want_m {
            p.err(format!(
                "wave {w}: D[{m}][{n}] receives the result of row {}, expected {want_m}",
                r.m
            ))?;
        }
        let want = p.want_n(m, n).map_err(TagError::fault)?;
        if r.n != want {
            p.err(format!(
                "wave {w}: D[{m}][{n}] receives the result of column(s) {}, expected {}",
                cols_text(&r.n),
                cols_text(&want)
            ))?;
        }
        let full = p
            .d_kblocks
            .as_ref()
            .and_then(|k| k.get(&m).cloned())
            .unwrap_or_else(|| p.kblocks());
        let have = r.blocks();
        if have != full {
            // Blocks missing, or blocks the problem does not have.
            let missing: Vec<i64> = full.difference(&have).copied().collect();
            let extra: Vec<i64> = have.difference(&full).copied().collect();
            let what = match (missing.is_empty(), extra.is_empty()) {
                (false, true) => format!("k-blocks {} missing", blocks_text(missing)),
                (true, false) => format!("k-blocks {} past K", blocks_text(extra)),
                _ => format!(
                    "k-blocks {} missing and {} past K",
                    blocks_text(missing),
                    blocks_text(extra)
                ),
            };
            p.err(format!("wave {w}: D[{m}][{n}] stored with {what}"))?;
        }
        if d.sub == 0 {
            cells.push((m, n, None));
        }
    }
    h.probs[pi].log_stores(&cells);
    Ok(())
}

/// `atomic(buffer, offsets, tags, wave)`: a scatter into D.
pub fn atomic(
    h: &mut GemmHooks,
    t: &mut Tables,
    call: &StoreCall<'_>,
) -> Result<AtomicAnswer, TagError> {
    let Some(pi) = h.owner_of(call.buffer) else {
        return Ok(AtomicAnswer::Scratch);
    };
    let w = call.wave;
    let layout = h.probs[pi].layout(call.name).cloned();
    let scatter = h.probs[pi].scatter.clone();
    let (Some(layout), Some(scatter)) = (layout.filter(|l| l.kind == Kind::D), scatter) else {
        return Err(TagError::fault(format!(
            "wave {w}: an atomic add into {}",
            call.name
        )));
    };
    h.current = pi;
    let mut cells = Vec::new();
    for (off, tag) in bytes(call) {
        let d = layout.decode(off);
        if d.sub != 0 {
            continue;
        }
        let (tr, n) = (d.row, d.k);
        let p = &mut h.probs[pi];
        if !is_record(tag) {
            p.err(format!(
                "wave {w}: D[{tr}][{n}] += a value with no accumulator provenance"
            ))?;
            continue;
        }
        let r = t.record_of(tag);
        if r.is_poison() {
            p.err(format!(
                "wave {w}: D[{tr}][{n}] += a value that {}",
                r.reason()
            ))?;
            continue;
        }
        let rows = scatter.get(&tr).cloned().unwrap_or_default();
        let want = p.want_n(r.m, n).map_err(TagError::fault)?;
        let all = p.kblocks();
        if !(rows.contains(&r.m) && r.n == want && r.blocks() == all) {
            let shown: Vec<i64> = rows.iter().copied().take(4).collect();
            p.err(format!(
                "wave {w}: D[{tr}][{n}] += row {} column {} over {} of {} k-blocks; expected one of \
                 rows {}, column {}",
                r.m,
                r.n.text(),
                r.blocks().len(),
                all.len(),
                py_list(shown),
                want.text()
            ))?;
        }
        cells.push((tr, n, Some(r.m)));
    }
    h.probs[pi].log_stores(&cells);
    Ok(AtomicAnswer::Checked)
}
