//! Memory ops in flight on entry to each instruction.

use std::collections::HashMap;

use thin_target::Flags;

use crate::cfg::Cfg;
use crate::solve::{per_instr, solve_forward};
use crate::{FieldId, FieldValue, Imm, ImmValue, Instr, Operand, Snapshot};

/// The cap on how many younger ops the state counts.
const CAP: u32 = 64;

/// (tracked op, the fewest younger ops of its counter on any path), in
/// insertion order.
type State = Vec<(usize, u32)>;

/// An op's counter, and whether it completes out of order on it.
fn counter_of(i: &Instr) -> (Option<usize>, bool) {
    let f = i.form();
    (f.counter, f.counter.is_some() && f.unordered)
}

/// Per instruction, the positions in `loads` of the tracked ops that may be
/// outstanding on entry on some path, in the state's order.
pub fn in_flight(s: &Snapshot, g: &Cfg, loads: &[usize]) -> Vec<Vec<usize>> {
    let instrs = s.instrs();
    let mut tracked = HashMap::new();
    for (j, &k) in loads.iter().enumerate() {
        tracked.entry(k).or_insert(j);
    }
    let fields: Vec<FieldId> = (s.target().definition().counters.iter())
        .map(|c| FieldId::named(c.field).expect("a registered count field"))
        .collect();
    let step = |st: &mut State, k: usize| {
        let i = &instrs[k];
        let of = |j: usize| counter_of(&instrs[loads[j]]);
        if i.form().flags.has(Flags::WAITCNT) {
            // A count is a field, or the one operand of a single-count wait.
            let operand = match i.ops() {
                [Operand::Imm(Imm {
                    value: ImmValue::Int(m),
                    ..
                })] => s
                    .target()
                    .operand_wait_counter(&i.form().name)
                    .map(|c| (c, *m)),
                _ => None,
            };
            for (c, &field) in fields.iter().enumerate() {
                let m = match i.field(field) {
                    Some(FieldValue::Int(m)) => *m,
                    _ => match operand {
                        Some((oc, m)) if oc == c => m,
                        _ => continue,
                    },
                };
                st.retain(|&(j, y)| {
                    let (counter, unordered) = of(j);
                    !(counter == Some(c) && (m == 0 || y as i128 >= m && !unordered))
                });
            }
            return;
        }
        if let (Some(c), false) = counter_of(i) {
            for (j, y) in st.iter_mut() {
                if of(*j).0 == Some(c) {
                    *y = (*y + 1).min(CAP);
                }
            }
        }
        if let Some(&j) = tracked.get(&k) {
            match st.iter_mut().find(|(t, _)| *t == j) {
                Some(entry) => entry.1 = 0,
                None => st.push((j, 0)),
            }
        }
    };
    let through = |st: &State, j: usize| {
        let mut st = st.clone();
        g.block(j).for_each(|k| step(&mut st, k));
        st
    };
    let entries = if g.nblocks() == 0 {
        Vec::new()
    } else {
        solve_forward(g, vec![(0, State::new())], through, merge, None)
    };
    per_instr(
        g,
        &entries,
        step,
        |st| st.iter().map(|&(j, _)| j).collect(),
        Vec::new(),
    )
}

/// a's entries, then b's new ones; the fewer younger ops where both have
/// an op.
fn merge(a: &State, b: &State) -> State {
    let mut out = a.clone();
    for &(j, y) in b {
        match out.iter_mut().find(|(t, _)| *t == j) {
            Some(entry) => entry.1 = entry.1.min(y),
            None => out.push((j, y)),
        }
    }
    out
}
