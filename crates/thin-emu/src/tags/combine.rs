//! Combining two records of one lane.

use std::collections::BTreeSet;

use super::encoding::{blocks_text, py_list, Cols, KSet, Record, Tables, Tag};

fn set_text(s: &BTreeSet<i64>) -> String {
    py_list(s.iter().copied())
}

/// `combine(x, y, stat)` of two record tags.
pub fn combine(t: &mut Tables, x: Tag, y: Tag, stat: bool) -> Tag {
    let rx = t.record_of(x);
    let ry = t.record_of(y);
    if x == y {
        if rx.is_poison() || stat {
            return x;
        }
        return t.poison(format!("adds the same partial sum twice at {}", rx.cell()));
    }
    if rx.is_poison() {
        return x;
    }
    if ry.is_poison() {
        return y;
    }
    if rx.pid != ry.pid {
        return normalizer(t, x, &rx, y, &ry);
    }
    if rx.stat != ry.stat && !stat {
        let (v, s, vt) = if rx.stat {
            (&ry, &rx, y)
        } else {
            (&rx, &ry, x)
        };
        return value_and_statistic(t, v, s, vt);
    }
    if rx.m != ry.m {
        return t.poison(format!(
            "adds results of different rows: {} + {}",
            rx.cell(),
            ry.cell()
        ));
    }
    let (kx, ky) = (rx.blocks(), ry.blocks());
    if rx.n == ry.n {
        let shared: BTreeSet<i64> = kx.intersection(&ky).copied().collect();
        if !shared.is_empty() && !stat {
            return t.poison(format!(
                "adds partial sums that share k-blocks {} at {}",
                set_text(&shared),
                rx.cell()
            ));
        }
        let r = Record {
            m: rx.m,
            n: rx.n.clone(),
            k: KSet::Set(kx.union(&ky).copied().collect()),
            stat: stat || rx.stat,
            pid: rx.pid,
        };
        return t.record(&r);
    }
    if kx != ky {
        return t.poison(format!(
            "combines columns {} and {} of row {} with different k coverage",
            rx.n.text(),
            ry.n.text(),
            rx.m
        ));
    }
    let cols: BTreeSet<i64> = rx.n.list().into_iter().chain(ry.n.list()).collect();
    let r = Record {
        m: rx.m,
        n: Cols::Many(cols.into_iter().collect()),
        k: KSet::Set(kx),
        stat: stat || (rx.stat && ry.stat),
        pid: rx.pid,
    };
    t.record(&r)
}

fn value_and_statistic(t: &mut Tables, v: &Record, s: &Record, vt: Tag) -> Tag {
    if v.m != s.m {
        return t.poison(format!(
            "row {}'s value scaled by a statistic of row {}",
            v.m, s.m
        ));
    }
    let scols: BTreeSet<i64> = s.n.list().into_iter().collect();
    if !v.n.list().iter().all(|c| scols.contains(c)) {
        return t.poison(format!(
            "column(s) {} of row {} scaled by a statistic of columns not including them",
            v.n.text(),
            v.m
        ));
    }
    if v.blocks() != s.blocks() {
        return t.poison(format!(
            "column(s) {} of row {} scaled by a statistic with a different k coverage",
            v.n.text(),
            v.m
        ));
    }
    vt
}

fn normalizer(t: &mut Tables, x: Tag, rx: &Record, y: Tag, ry: &Record) -> Tag {
    let (px, py) = (t.problem(rx.pid), t.problem(ry.pid));
    let (v, vt, s) = if px.source == Some(ry.pid) {
        (rx, x, ry)
    } else if py.source == Some(rx.pid) {
        (ry, y, rx)
    } else {
        return t.poison(format!(
            "combines results of two unrelated problems: {} and {}",
            rx.cell(),
            ry.cell()
        ));
    };
    if v.m != s.m {
        return t.poison(format!(
            "row {}'s value normalized by a reduction of row {}",
            v.m, s.m
        ));
    }
    let cols: BTreeSet<i64> = s.n.list().into_iter().collect();
    let n_limit = t.problem(s.pid).n;
    let uncovered: Vec<i64> = v
        .blocks()
        .into_iter()
        .filter(|b| {
            let end = n_limit.map_or(32 * b + 32, |n| (32 * b + 32).min(n));
            !(32 * b..end).all(|c| cols.contains(&c))
        })
        .collect();
    if !uncovered.is_empty() {
        return t.poison(format!(
            "row {}'s value normalized by a reduction that does not cover its k-blocks {} (the \
             other problem's columns)",
            v.m,
            blocks_text(uncovered)
        ));
    }
    vt
}
