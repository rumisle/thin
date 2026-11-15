//! Wave-dependent branches and barrier pairing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use thin_ir::cfg::Cfg;
use thin_ir::solve::solve_forward;
use thin_ir::{Diag, Instr, Operand, Part, Reg};
use thin_target::{phys_named, Flags, Kind, Pseudo};

use super::{refusal, Lds};

/// The SGPR dwords (value, dword) that may differ between waves, and
/// whether SCC may.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Taint {
    sgprs: BTreeSet<(i32, u32)>,
    scc: bool,
}

impl Taint {
    fn join(&mut self, o: &Taint) {
        self.sgprs.extend(o.sgprs.iter().copied());
        self.scc |= o.scc;
    }
}

/// An S register's dwords as keys.
fn dwords(r: &Reg) -> impl Iterator<Item = (i32, u32)> + '_ {
    (r.file == 's')
        .then_some(r)
        .into_iter()
        .flat_map(|r| (r.lo..r.lo + r.n).map(move |d| (r.value, d)))
}

/// What finds: per divergent branch its join (none for no path to
/// the end, `n` for the virtual end) and region, and the barriers whose
/// epochs pair.
#[derive(Clone, Debug, Default)]
pub(super) struct Branches {
    /// The divergent branches by index, with their joins and regions.
    pub divergent: Vec<(usize, Option<usize>, BTreeSet<usize>)>,
    /// Groups of barriers whose epochs merge.
    pub pairs: Vec<Vec<usize>>,
}

impl Lds<'_> {
    fn scc(&self) -> usize {
        phys_named("scc").expect("scc").number
    }

    /// Whether an operand's whole register is vcc or exec.
    fn is_mask(&self, number: usize) -> bool {
        let t = self.s.target();
        ["vcc", "exec"]
            .iter()
            .any(|n| t.mask_reg(phys_named(n).expect("a mask")).number == number)
    }

    fn reads_scc(&self, i: &Instr) -> bool {
        i.form().pseudo.is_some() || i.reads().contains(&self.scc())
    }

    fn writes_scc(&self, i: &Instr) -> bool {
        i.form().pseudo.is_some() || i.writes().contains(&self.scc())
    }

    /// One instruction's taint step.
    fn taint_step(&self, t: &mut Taint, i: &Instr, seed: &BTreeSet<(i32, u32)>) {
        let f = i.form();
        if matches!(f.kind, Kind::Label | Kind::Comment) {
            if f.pseudo == Some(Pseudo::Uniform) {
                i.ops()
                    .iter()
                    .filter_map(Operand::reg)
                    .flat_map(dwords)
                    .for_each(|d| _ = t.sgprs.remove(&d));
                t.scc &= !i.ops().iter().any(|o| matches!(o, Operand::Phys(_)));
            }
            return;
        }
        let var = matches!(f.kind, Kind::Valu | Kind::Matrix)
            || f.flags.has(Flags::WAVE_VARYING)
            || i.ops()
                .iter()
                .any(|o| o.phys().is_some_and(|p| self.is_mask(p.number)))
            || i.uses()
                .any(|r| r.file != 's' || dwords(r).any(|d| t.sgprs.contains(&d)))
            || self.reads_scc(i) && t.scc;
        for d in i.defs().flat_map(dwords) {
            if var {
                t.sgprs.insert(d);
            } else if !seed.contains(&d) {
                t.sgprs.remove(&d);
            }
        }
        if self.writes_scc(i) {
            t.scc = if f.pseudo.is_some() { t.scc | var } else { var };
        }
    }

    /// The taint before each instruction, with the forced taints.
    fn taints(&self, g: &Cfg, forced: &BTreeMap<usize, Taint>) -> Vec<Option<Taint>> {
        let s = self.s;
        let seed: BTreeSet<(i32, u32)> = s.wave_reg().iter().flat_map(dwords).collect();
        let written = s
            .instrs()
            .iter()
            .any(|i| i.defs().flat_map(dwords).any(|d| seed.contains(&d)));
        let entry = Taint {
            sgprs: if written {
                BTreeSet::new()
            } else {
                seed.clone()
            },
            scc: false,
        };
        let step = |t: &mut Taint, k: usize| {
            if let Some(f) = forced.get(&k) {
                t.join(f);
            }
            self.taint_step(t, &s.instrs()[k], &seed);
        };
        let through = |t: &Taint, j: usize| {
            let mut t = t.clone();
            g.block(j).for_each(|k| step(&mut t, k));
            t
        };
        let merge = |a: &Taint, b: &Taint| {
            let mut a = a.clone();
            a.join(b);
            a
        };
        if g.nblocks() == 0 {
            return Vec::new();
        }
        let entries = solve_forward(g, vec![(0, entry)], through, merge, None);
        let mut before = vec![None; s.instrs().len()];
        for (j, t) in entries.into_iter().enumerate() {
            let Some(mut t) = t else { continue };
            for k in g.block(j) {
                before[k] = Some(t.clone());
                step(&mut t, k);
            }
        }
        before
    }

    /// Whether branch k is divergent on the taint before it.
    fn divergent(&self, k: usize, t: &Taint) -> bool {
        let i = &self.s.instrs()[k];
        let f = i.form();
        if !f.flags.has(Flags::BRANCH) || f.flags.has(Flags::UNCOND) {
            return false;
        }
        match i.reads().contains(&self.scc()) {
            true => t.scc,
            false => i.reads().iter().any(|&r| self.is_mask(r)),
        }
    }

    /// The successors with the virtual end `n`.
    pub(super) fn succ_end(&self, k: usize) -> Vec<usize> {
        let n = self.s.instrs().len();
        let mut out = self.g.succ[k].clone();
        let i = &self.s.instrs()[k];
        let cond = i.form().flags.has(Flags::BRANCH) && !i.form().flags.has(Flags::UNCOND);
        if out.is_empty() || k + 1 == n && cond {
            out.push(n);
        }
        out
    }

    /// Each instruction's immediate post-dominator over the whole CFG with
    /// the virtual end (`n`); none without a path to the end. The
    /// iterative dominator algorithm on the reversed CFG, rooted at the end.
    fn ipdoms(&self) -> Vec<Option<usize>> {
        let n = self.s.instrs().len();
        let succ: Vec<Vec<usize>> = (0..n).map(|k| self.succ_end(k)).collect();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (k, ts) in succ.iter().enumerate() {
            ts.iter().for_each(|&t| preds[t].push(k));
        }
        let mut order = Vec::with_capacity(n + 1);
        let mut seen = vec![false; n + 1];
        let mut stack = vec![(n, 0)];
        seen[n] = true;
        while let Some((x, i)) = stack.pop() {
            match preds[x].get(i) {
                Some(&p) => {
                    stack.push((x, i + 1));
                    if !seen[p] {
                        seen[p] = true;
                        stack.push((p, 0));
                    }
                }
                None => order.push(x),
            }
        }
        let mut number = vec![usize::MAX; n + 1];
        order.iter().enumerate().for_each(|(i, &x)| number[x] = i);
        let mut idom: Vec<Option<usize>> = vec![None; n + 1];
        idom[n] = Some(n);
        let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
            while a != b {
                while number[a] < number[b] {
                    a = idom[a].expect("a processed node");
                }
                while number[b] < number[a] {
                    b = idom[b].expect("a processed node");
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &x in order.iter().rev().filter(|&&x| x != n) {
                let mut done = succ[x].iter().copied().filter(|&t| idom[t].is_some());
                let Some(first) = done.next() else { continue };
                let new = done.fold(first, |a, t| intersect(&idom, a, t));
                if idom[x] != Some(new) {
                    idom[x] = Some(new);
                    changed = true;
                }
            }
        }
        idom.truncate(n);
        idom
    }

    /// The instructions on paths from branch b to its join, the join
    /// excluded (and b itself).
    fn region(&self, b: usize, join: usize) -> BTreeSet<usize> {
        let n = self.s.instrs().len();
        let mut seen = BTreeSet::new();
        let mut queue: VecDeque<usize> = self.succ_end(b).into_iter().collect();
        while let Some(x) = queue.pop_front() {
            if x == join || x == n || !seen.insert(x) {
                continue;
            }
            queue.extend(self.succ_end(x));
        }
        seen.remove(&b);
        seen
    }

    /// the divergent branches, their joins and regions, recomputed
    /// until the taints forced at the joins stop changing.
    fn find_branches(&self) -> Vec<(usize, Option<usize>, BTreeSet<usize>)> {
        let s = self.s;
        let n = s.instrs().len();
        let ipdom = self.ipdoms();
        let mut forced: BTreeMap<usize, Taint> = BTreeMap::new();
        loop {
            let before = self.taints(&self.g, &forced);
            let divergent: Vec<(usize, Option<usize>, BTreeSet<usize>)> = (0..n)
                .filter(|&k| before[k].as_ref().is_some_and(|t| self.divergent(k, t)))
                .map(|k| {
                    (
                        k,
                        ipdom[k],
                        ipdom[k].map(|j| self.region(k, j)).unwrap_or_default(),
                    )
                })
                .collect();
            let mut next: BTreeMap<usize, Taint> = BTreeMap::new();
            for (_, join, region) in &divergent {
                let Some(j) = join.filter(|&j| j < n) else {
                    continue;
                };
                let t = next.entry(j).or_default();
                for &x in region {
                    let i = &s.instrs()[x];
                    t.sgprs.extend(i.defs().flat_map(dwords));
                    t.scc |=
                        self.writes_scc(i) && !matches!(i.form().kind, Kind::Label | Kind::Comment);
                }
            }
            if next == forced {
                return divergent;
            }
            forced = next;
        }
    }

    fn is_barrier(&self, k: usize) -> bool {
        self.s.instrs()[k].form().flags.has(Flags::BARRIER)
    }

    /// in full: the branches, their refusals, and the barriers that
    /// pair.
    pub(super) fn branches(&self) -> Result<Branches, Diag> {
        let mut out = Branches {
            divergent: self.find_branches(),
            pairs: Vec::new(),
        };
        for (b, join, region) in &out.divergent {
            let Some(j) = *join else {
                return Err(refusal(vec![
                    Part::Text("`".into()),
                    Part::Instr(*b),
                    Part::Text(format!(
                        "` (instruction {b}): a wave-dependent branch on a path that never ends"
                    )),
                ]));
            };
            self.loop_check(*b, j, region)?;
            let groups = self.counts(*b, j, region)?;
            out.pairs.extend(groups);
        }
        Ok(out)
    }

    /// a loop holding a barrier inside the region.
    fn loop_check(&self, b: usize, j: usize, region: &BTreeSet<usize>) -> Result<(), Diag> {
        if !region.iter().any(|&x| self.is_barrier(x)) {
            return Ok(());
        }
        let rb = |x: usize| x == b || region.contains(&x);
        let looped = (region.iter().copied().chain([b])).any(|x| {
            self.g.succ[x]
                .iter()
                .any(|&t| t <= x && rb(t) && (t..=x).any(|y| self.is_barrier(y)))
        });
        if !looped {
            return Ok(());
        }
        Err(refusal(vec![
            Part::Text("the wave-dependent branch `".into()),
            Part::Instr(b),
            Part::Text(format!(
                "` (instruction {b}) leads into a loop holding s_barrier before its join (instruction {j}): waves \
                 would pass different numbers of barriers (barriers pair by count)"
            )),
        ]))
    }

    /// the barrier counts, the groups that pair, and the
    /// refusal of a join reached over different counts.
    fn counts(
        &self,
        b: usize,
        j: usize,
        region: &BTreeSet<usize>,
    ) -> Result<Vec<Vec<usize>>, Diag> {
        let n = self.s.instrs().len();
        let rb = |x: usize| x == b || region.contains(&x);
        let mut count: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::from([(b, BTreeSet::from([0]))]);
        let from = |count: &BTreeMap<usize, BTreeSet<u32>>, preds: Vec<usize>| -> BTreeSet<u32> {
            (preds.into_iter())
                .filter(|&y| rb(y))
                .filter_map(|y| count.get(&y).map(|c| (y, c)))
                .flat_map(|(y, c)| c.iter().map(move |v| v + u32::from(self.is_barrier(y))))
                .collect()
        };
        for &x in region.iter().chain((j < n).then_some(&j)) {
            let c = from(&count, self.g.preds[x].clone());
            if !c.is_empty() {
                count.insert(x, c);
            }
        }
        let at_join = match j < n {
            true => count.get(&j).cloned().unwrap_or_default(),
            false => from(
                &count,
                (0..n).filter(|&y| self.succ_end(y).contains(&n)).collect(),
            ),
        };
        let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &x in region.iter().filter(|&&x| self.is_barrier(x)) {
            count
                .get(&x)
                .into_iter()
                .flatten()
                .for_each(|&c| groups.entry(c).or_default().push(x));
        }
        if at_join.len() > 1 && j < n && self.reaches_barrier(j) {
            let c: Vec<String> = at_join.iter().map(u32::to_string).collect();
            let (last, rest) = c.split_last().expect("two counts");
            let label = self.label(j);
            let mut msg = vec![
                Part::Text("the wave-dependent branch `".into()),
                Part::Instr(b),
            ];
            msg.push(Part::Text(format!("` (instruction {b}) rejoins at ")));
            msg.extend(label);
            msg.push(Part::Text(format!(
                " (instruction {j}) over paths passing {} and {last} s_barrier: barriers pair by count, so the epochs \
                 between them would not be every wave's - give every path the same barriers",
                rest.join(", ")
            )));
            return Err(refusal(msg));
        }
        Ok(groups.into_values().filter(|g| g.len() >= 2).collect())
    }

    /// A label's name, else the instruction's line.
    fn label(&self, k: usize) -> Vec<Part> {
        let i = &self.s.instrs()[k];
        match i.form().kind {
            Kind::Label => vec![Part::Text(
                i.field_named("name")
                    .map(|v| v.py_str())
                    .unwrap_or_default(),
            )],
            _ => vec![Part::Instr(k)],
        }
    }

    fn reaches_barrier(&self, from: usize) -> bool {
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([from]);
        while let Some(x) = queue.pop_front() {
            if !seen.insert(x) {
                continue;
            }
            if self.is_barrier(x) {
                return true;
            }
            queue.extend(self.g.succ[x].iter().copied());
        }
        false
    }
}
