//! Path states, forks and joins.

use std::collections::BTreeMap;
use std::sync::Arc;

use thin_ir::Operand;
use thin_target::{Flags, Kind};

use super::Builder;
use crate::error::{schedule, Result};

/// A path state: per counter the ops issued on the path, the region-op
/// record, and the author's records, which the host keeps by the
/// `records` number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    /// Per counter, the ops issued.
    pub counts: Vec<u32>,
    /// Per (counter, region name), the count at the region's last op.
    pub mem: BTreeMap<(usize, Arc<str>), u32>,
    /// The number of the author's records dict.
    pub records: u64,
}

/// What happened to the author's records, for the host to do on its dicts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateOp {
    /// `records[to] = deepcopy(records[from])`.
    Copy {
        /// The source.
        from: u64,
        /// The copy.
        to: u64,
    },
    /// Join each of `from`'s records into `into`.
    Join {
        /// The current state's records.
        into: u64,
        /// The joined states' records.
        from: Vec<u64>,
    },
}

/// A state the author holds: a fork, or the current state as `b.cnt` read
/// it.
#[derive(Clone, Debug)]
pub(crate) struct Slot {
    state: State,
    fork: Option<(Option<u64>, Option<u64>)>,
    checked: bool,
}

impl Builder {
    /// A copy of a state: its counts, its region record, a deep copy of its
    /// records.
    pub(crate) fn copy_state(&mut self, s: &State) -> State {
        let to = self.next_records;
        self.next_records += 1;
        self.state_ops.push(StateOp::Copy {
            from: s.records,
            to,
        });
        State {
            records: to,
            ..s.clone()
        }
    }

    /// A copy of the current state.
    pub(crate) fn copy_current(&mut self) -> State {
        let s = self.state.clone();
        self.copy_state(&s)
    }

    /// Joins states into the current one: the region record by the
    /// larger count per key, the author's records by their `join`.
    pub(crate) fn join(&mut self, others: &[State]) {
        for o in others {
            for (k, &v) in &o.mem {
                let e = self.state.mem.entry(k.clone()).or_insert(v);
                *e = (*e).max(v);
            }
        }
        let from = others.iter().map(|o| o.records).collect();
        self.state_ops.push(StateOp::Join {
            into: self.state.records,
            from,
        });
    }

    /// `b.fork()`: a copy of the current state that remembers the
    /// instruction before it and, once emitted, the one after.
    pub fn fork(&mut self) -> Result<usize> {
        self.settle()?;
        let state = self.copy_current();
        let before = self.last_real();
        self.states.push(Slot {
            state,
            fork: Some((before, None)),
            checked: false,
        });
        self.pending_forks.push(self.states.len() - 1);
        Ok(self.states.len() - 1)
    }

    /// `b.cnt`: the current state, as a handle.
    pub fn cnt(&mut self) -> Result<usize> {
        self.settle()?;
        let state = self.state.clone();
        self.states.push(Slot {
            state,
            fork: None,
            checked: true,
        });
        Ok(self.states.len() - 1)
    }

    /// The author's records number of a held state.
    pub fn records_of(&self, slot: usize) -> u64 {
        self.states[slot].state.records
    }

    /// `b.cnt = state`.
    pub fn set_cnt(&mut self, slot: usize) -> Result<()> {
        self.settle()?;
        let s = &self.states[slot];
        if let (Some((a, b)), false) = (s.fork, s.checked) {
            let branch = |id: Option<u64>| {
                id.and_then(|id| self.row_index(id))
                    .is_some_and(|k| self.rows[k].ins.form().kind == Kind::Branch)
            };
            if !branch(a) && !branch(b) {
                let text = |id: Option<u64>| {
                    id.and_then(|id| self.row_index(id))
                        .map_or("(none)".to_owned(), |k| self.text(&self.rows[k].ins))
                };
                return schedule(format!(
                    "a path starts from a fork not taken at a branch (the instructions around it: `{} | {}`); fork \
                     right before or right after the branch instruction",
                    text(a),
                    text(b)
                ));
            }
            self.states[slot].checked = true;
        }
        self.state = self.states[slot].state.clone();
        Ok(())
    }

    /// the instruction is the next one of every fork taken
    /// since the last such instruction.
    pub(crate) fn close_forks(&mut self, id: u64) {
        for k in std::mem::take(&mut self.pending_forks) {
            if let Some((_, after)) = &mut self.states[k].fork {
                *after = Some(id);
            }
        }
    }

    /// branch rule, for an emitted branch.
    pub(crate) fn branch_rule(&mut self, i: &thin_ir::Instr) {
        let t = i.ops().iter().find_map(|o| {
            if let Operand::Block(b) = o {
                Some(b.to_string())
            } else {
                None
            }
        });
        let uncond = i.form().flags.has(Flags::UNCOND);
        match t {
            Some(t) if !uncond && !t.is_empty() && !self.placed(&t) => {
                let st = self.copy_current();
                self.forward.push((t, st, false));
            }
            _ if i.form().op == "s_branch" => self.reachable = false,
            _ => {}
        }
    }

    /// Marks the forward branches to `label` recorded from index k on as
    /// other waves' skips.
    pub(crate) fn mark_other_waves(&mut self, label: &str, from: usize) {
        self.forward
            .iter_mut()
            .skip(from)
            .filter(|f| f.0 == label)
            .for_each(|f| f.2 = true);
    }

    /// The forward branches recorded so far.
    pub(crate) fn forward_len(&self) -> usize {
        self.forward.len()
    }

    /// Makes the next point unreachable.
    pub(crate) fn unreachable(&mut self) {
        self.reachable = false;
    }
}
