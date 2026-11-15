//! Redundant M0 writes.

use thin_target::Kind;

use crate::solve::solve_forward_all;
use crate::{Instr, Snapshot};

/// What M0 holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// No path reaches here yet.
    Unreached,
    /// This constant, on every path.
    Set(i128),
    /// Anything.
    Unknown,
}

/// Unreached joins as nothing; equal states as themselves; else Unknown.
fn merge(a: &State, b: &State) -> State {
    match (a, b) {
        (State::Unreached, x) | (x, State::Unreached) => *x,
        (x, y) if x == y => *x,
        _ => State::Unknown,
    }
}

/// One instruction's effect on M0.
fn step(i: &Instr, st: &mut State) {
    if matches!(i.form().kind, Kind::Label | Kind::Comment) || !i.m0_write() {
        return;
    }
    *st = i.attrs().m0_set.map_or(State::Unknown, State::Set);
}

/// The indices, in program order, of the M0Set writes whose constant M0
/// already holds on every path into them. Code no path reaches is no
/// path's, so nothing there is redundant, and it holds nothing for the
/// code after it.
pub fn redundant(s: &Snapshot) -> Vec<usize> {
    let instrs = s.instrs();
    if !instrs.iter().any(|i| i.attrs().m0_set.is_some()) {
        return Vec::new();
    }
    let g = s.cfg(None);
    let reached = g.reached();
    let live = |j: usize| g.block(j).next().is_some_and(|k| reached[k]);
    let through = |st: &State, j: usize| {
        if !live(j) {
            return State::Unreached;
        }
        let mut st = *st;
        g.block(j).for_each(|k| step(&instrs[k], &mut st));
        st
    };
    let entries = solve_forward_all(
        &g,
        State::Unreached,
        vec![(0, State::Unknown)],
        through,
        merge,
    );
    let mut out = Vec::new();
    for (j, &entry) in entries.iter().enumerate().filter(|&(j, _)| live(j)) {
        let mut st = entry;
        for k in g.block(j) {
            if let (Some(c), State::Set(held)) = (instrs[k].attrs().m0_set, st) {
                if c == held {
                    out.push(k);
                }
            }
            step(&instrs[k], &mut st);
        }
    }
    out
}
