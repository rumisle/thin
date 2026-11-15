//! The control-flow graph at two levels, instructions and basic blocks
//!

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use thin_target::{Flags, Kind};

use crate::{FieldId, FieldValue, Instr};

/// A wave class's resolved branches: (instruction index, taken).
pub type Taken = [(usize, bool)];

/// The control-flow graph of an instruction list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cfg {
    /// Per instruction, its distinct successors in order (a branch's
    /// target first, then the next instruction).
    pub succ: Vec<Vec<usize>>,
    /// Per instruction, its predecessors in order.
    pub preds: Vec<Vec<usize>>,
    /// The blocks' first instructions, ascending.
    pub starts: Vec<usize>,
    /// Per instruction, the block holding it.
    pub block_of: Vec<usize>,
    /// Per block, the blocks of its last instruction's successors, in
    /// order, first occurrence kept.
    pub block_succ: Vec<Vec<usize>>,
    /// Per block, the blocks of its first instruction's predecessors,
    /// ascending.
    pub block_preds: Vec<Vec<usize>>,
}

/// The graph of `instrs`: branches resolved by `waves` where it lists them,
/// and with `label_blocks`, every label and every instruction after a
/// branch or an end starting a block.
pub fn cfg_of(instrs: &[Arc<Instr>], waves: Option<&Taken>, label_blocks: bool) -> Cfg {
    let n = instrs.len();
    let label = FieldId::of("name");
    let mut labels = HashMap::new();
    for (k, i) in instrs.iter().enumerate() {
        if i.form().kind == Kind::Label {
            if let Some(FieldValue::Text(name)) = i.field(label) {
                labels.insert(name.clone(), k);
            }
        }
    }
    let target = |i: &Instr| {
        (i.ops().iter()).find_map(|op| match op {
            crate::Operand::Block(name) => labels.get(name).copied(),
            _ => None,
        })
    };
    let mut succ: Vec<Vec<usize>> = (instrs.iter().enumerate())
        .map(|(k, i)| {
            let flags = i.form().flags;
            let mut s = Vec::new();
            if flags.has(Flags::BRANCH) {
                s.extend(target(i));
            }
            // A conditional branch to the next instruction has one
            // successor, not the same one twice.
            if !flags.has(Flags::UNCOND) && k + 1 < n && !s.contains(&(k + 1)) {
                s.push(k + 1);
            }
            s
        })
        .collect();
    for &(k, taken) in waves.unwrap_or(&[]) {
        if k < n {
            succ[k] = match taken {
                true => target(&instrs[k]).into_iter().collect(),
                false => (k + 1 < n).then_some(k + 1).into_iter().collect(),
            };
        }
    }
    // A branch always ends its block, even one whose only successor is
    // the next instruction: what follows it is a join point of its own.
    let extra = |k: usize| {
        let i = &instrs[k];
        let branch = i.form().flags.has(Flags::BRANCH);
        let ends = branch || label_blocks && i.form().flags.has(Flags::ENDPGM);
        Some((label_blocks && i.form().kind == Kind::Label, ends))
    };
    Cfg::build(succ, extra)
}

impl Cfg {
    /// The block-level graph of a given per-instruction successor list,
    /// with no extra leaders.
    pub fn from_succ_only(succ: Vec<Vec<usize>>) -> Cfg {
        Cfg::build(succ, |_| None)
    }

    /// The graph of `succ`, where `extra(k)` says whether k starts a block
    /// of its own and whether k+1 does (none: no extra leaders).
    fn build(succ: Vec<Vec<usize>>, extra: impl Fn(usize) -> Option<(bool, bool)>) -> Cfg {
        let n = succ.len();
        let mut preds = vec![Vec::new(); n];
        let mut leader = vec![false; n];
        if n > 0 {
            leader[0] = true;
        }
        for (k, s) in succ.iter().enumerate() {
            for &t in s {
                preds[t].push(k);
            }
            if s[..] != [k + 1] {
                s.iter().for_each(|&t| leader[t] = true);
                if k + 1 < n {
                    leader[k + 1] = true;
                }
            }
            if let Some((own, next)) = extra(k) {
                leader[k] |= own;
                if next && k + 1 < n {
                    leader[k + 1] = true;
                }
            }
        }
        let starts: Vec<usize> = (0..n).filter(|&k| leader[k]).collect();
        let mut block_of = vec![0; n];
        for (j, &s) in starts.iter().enumerate() {
            let end = starts.get(j + 1).copied().unwrap_or(n);
            block_of[s..end].fill(j);
        }
        let ends = |j: usize| starts.get(j + 1).copied().unwrap_or(n);
        let block_succ = (0..starts.len())
            .map(|j| {
                let mut out = Vec::new();
                for &t in &succ[ends(j) - 1] {
                    if !out.contains(&block_of[t]) {
                        out.push(block_of[t]);
                    }
                }
                out
            })
            .collect();
        let block_preds = (starts.iter())
            .map(|&s| {
                let set: BTreeSet<usize> = preds[s].iter().map(|&p| block_of[p]).collect();
                set.into_iter().collect()
            })
            .collect();
        Cfg {
            succ,
            preds,
            starts,
            block_of,
            block_succ,
            block_preds,
        }
    }

    /// The number of blocks.
    pub fn nblocks(&self) -> usize {
        self.starts.len()
    }

    /// Block j's end: the next block's start, or the instruction count.
    pub fn end(&self, j: usize) -> usize {
        self.starts.get(j + 1).copied().unwrap_or(self.succ.len())
    }

    /// Block j's instructions.
    pub fn block(&self, j: usize) -> Range<usize> {
        self.starts[j]..self.end(j)
    }

    /// Per instruction, whether a path from instruction 0 reaches it.
    pub fn reached(&self) -> Vec<bool> {
        let mut seen = vec![false; self.succ.len()];
        let mut queue = VecDeque::new();
        if !seen.is_empty() {
            seen[0] = true;
            queue.push_back(0);
        }
        while let Some(k) = queue.pop_front() {
            for &t in &self.succ[k] {
                if !seen[t] {
                    seen[t] = true;
                    queue.push_back(t);
                }
            }
        }
        seen
    }

    /// The blocks reachable from block 0, in reverse postorder of a
    /// depth-first search that takes successors in list order and marks a
    /// block seen when it pushes it.
    pub fn rpo(&self) -> Vec<usize> {
        let mut post = Vec::new();
        if self.nblocks() == 0 {
            return post;
        }
        let mut seen = vec![false; self.nblocks()];
        let mut stack = vec![(0, 0)];
        seen[0] = true;
        while let Some(&mut (b, ref mut next)) = stack.last_mut() {
            if let Some(&t) = self.block_succ[b].get(*next) {
                *next += 1;
                if !seen[t] {
                    seen[t] = true;
                    stack.push((t, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        post
    }

    /// The block edges (q, t) with t not later than q in `rpo`, ascending.
    pub fn back_edges(&self, rpo: &[usize]) -> BTreeSet<(usize, usize)> {
        let mut pos = vec![usize::MAX; self.nblocks()];
        rpo.iter().enumerate().for_each(|(i, &b)| pos[b] = i);
        let mut out = BTreeSet::new();
        for &q in rpo {
            for &t in &self.block_succ[q] {
                if pos[t] <= pos[q] {
                    out.insert((q, t));
                }
            }
        }
        out
    }
}
