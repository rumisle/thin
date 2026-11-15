//! The dataflow solvers. Their visit orders are normative.
//!
//! An analysis gives a state type, a transfer `through(state, block)` (the
//! state leaving the block, or for the backward solver the state before
//! it) and a merge.

use std::collections::BTreeSet;

use crate::cfg::Cfg;

/// A widening and the number of changes of a block's entry state after
/// which it applies.
pub struct Widen<'a, S> {
    /// `widen(old, new)`: a state above both that makes chains finite.
    pub widen: &'a dyn Fn(&S, &S) -> S,
    /// Changes of a block's state before widening applies.
    pub patience: u32,
}

/// The forward solver with no bottom state: per block its entry state, or
/// none for a block no start reaches.
pub fn solve_forward<S: Clone + PartialEq>(
    g: &Cfg,
    starts: Vec<(usize, S)>,
    through: impl FnMut(&S, usize) -> S,
    merge: impl Fn(&S, &S) -> S,
    widen: Option<Widen<'_, S>>,
) -> Vec<Option<S>> {
    solve_forward_on(g, starts, through, merge, widen, |_, _| true)
}

/// [`solve_forward`] over only the edges `follow(from, to)` accepts.
pub fn solve_forward_on<S: Clone + PartialEq>(
    g: &Cfg,
    starts: Vec<(usize, S)>,
    mut through: impl FnMut(&S, usize) -> S,
    merge: impl Fn(&S, &S) -> S,
    widen: Option<Widen<'_, S>>,
    follow: impl Fn(usize, usize) -> bool,
) -> Vec<Option<S>> {
    let mut state: Vec<Option<S>> = vec![None; g.nblocks()];
    if g.nblocks() == 0 {
        return state; // an empty stream (every instruction deleted)
    }
    let mut changes = vec![0u32; g.nblocks()];
    let mut queue = BTreeSet::new();
    for (j, s) in starts {
        state[j] = Some(s);
        queue.insert(j);
    }
    while let Some(j) = queue.pop_first() {
        let out = through(state[j].as_ref().expect("a queued block has a state"), j);
        for &t in g.block_succ[j].iter().filter(|&&t| follow(j, t)) {
            let mut new = match &state[t] {
                None => out.clone(),
                Some(old) => merge(old, &out),
            };
            if state[t].as_ref() == Some(&new) {
                continue;
            }
            changes[t] += 1;
            if let (Some(w), Some(old)) = (&widen, &state[t]) {
                if changes[t] > w.patience {
                    new = (w.widen)(old, &new);
                    // A widening that gives back the old state changes
                    // nothing: re-queueing would loop for ever.
                    if new == *old {
                        continue;
                    }
                }
            }
            state[t] = Some(new);
            queue.insert(t);
        }
    }
    state
}

/// The forward solver with a bottom state, visiting every block, reached
/// or not.
pub fn solve_forward_all<S: Clone + PartialEq>(
    g: &Cfg,
    bottom: S,
    starts: Vec<(usize, S)>,
    mut through: impl FnMut(&S, usize) -> S,
    merge: impl Fn(&S, &S) -> S,
) -> Vec<S> {
    let mut state = vec![bottom.clone(); g.nblocks()];
    if g.nblocks() == 0 {
        return state;
    }
    for (j, s) in starts {
        state[j] = merge(&bottom, &s);
    }
    let mut queue: BTreeSet<usize> = (0..g.nblocks()).collect();
    while let Some(j) = queue.pop_first() {
        let out = through(&state[j], j);
        for &t in &g.block_succ[j] {
            let new = merge(&state[t], &out);
            if new != state[t] {
                state[t] = new;
                queue.insert(t);
            }
        }
    }
    state
}

/// The backward solver: per block the state before it, from every block
/// in reverse program order; `merge_list` of no states is the exit state
///
pub fn solve_backward<S: Clone + PartialEq>(
    g: &Cfg,
    bottom: S,
    merge_list: impl Fn(&[&S]) -> S,
    mut through: impl FnMut(&S, usize) -> S,
) -> Vec<S> {
    let mut before = vec![bottom; g.nblocks()];
    let mut queue: BTreeSet<usize> = (0..g.nblocks()).collect();
    while let Some(j) = queue.pop_last() {
        let after: Vec<&S> = g.block_succ[j].iter().map(|&s| &before[s]).collect();
        let state = through(&merge_list(&after), j);
        if state != before[j] {
            before[j] = state;
            queue.extend(g.block_preds[j].iter().copied());
        }
    }
    before
}

/// Per instruction, `result` of the state before it: each block with an
/// entry state is walked in order, `step` applied after each instruction's
/// result is taken; instructions of other blocks get `unknown`
///
pub fn per_instr<S: Clone, T: Clone>(
    g: &Cfg,
    entry: &[Option<S>],
    mut step: impl FnMut(&mut S, usize),
    result: impl Fn(&S) -> T,
    unknown: T,
) -> Vec<T> {
    let mut out = vec![unknown; g.succ.len()];
    for (j, s) in entry.iter().enumerate() {
        let Some(s) = s else { continue };
        let mut s = s.clone();
        for k in g.block(j) {
            out[k] = result(&s);
            step(&mut s, k);
        }
    }
    out
}
