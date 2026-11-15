//! The LDS layout's proof and races: regions that share
//! bytes are never live together, barriers order the accesses of
//! different waves, and waves that run apart touch their own bytes.

mod taint;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use thin_ir::cfg::Cfg;
use thin_ir::flight::in_flight;
use thin_ir::{Diag, DiagClass, Part, Snapshot};
use thin_target::{Flags, Kind};

/// A leaf of the layout: its name, first byte and bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf {
    /// Its name.
    pub name: String,
    /// Its first byte.
    pub base: u64,
    /// Its bytes.
    pub size: u64,
}

/// A union of the layout: its bytes and its arms, each a name and the
/// leaves (by position in the layout's leaves) under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Union {
    /// Its first byte.
    pub base: u64,
    /// Its end.
    pub end: u64,
    /// Its arms.
    pub arms: Vec<(String, Vec<usize>)>,
}

/// What the proof reads of the layout: its leaves in layout order and its
/// unions, innermost first in layout order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    /// The leaves.
    pub leaves: Vec<Leaf>,
    /// The unions.
    pub unions: Vec<Union>,
}

/// The proof's options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// An ablation: skip the barrier protocol.
    pub ablate: bool,
    /// Leaves one wave uses apart in its program order do not conflict.
    pub wave_order: bool,
}

/// The proof's result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Proof {
    /// The epochs' names, by number.
    pub epochs: Vec<String>,
    /// Per instruction, its epochs.
    pub of_instr: Vec<BTreeSet<usize>>,
    /// Per leaf, its epochs.
    pub of_leaf: Vec<BTreeSet<usize>>,
    /// The report's lines.
    pub report: Vec<String>,
}

/// The analysis of one snapshot.
struct Lds<'a> {
    s: &'a Snapshot,
    g: Arc<Cfg>,
}

fn refusal(msg: Vec<Part>) -> Diag {
    Diag {
        class: DiagClass::Mem,
        loc: 0,
        msg,
        notes: Vec::new(),
    }
}

/// The atomics that return nothing and commute.
const COMMUTING: [&str; 14] = [
    "ds.add.u32",
    "ds.add.u64",
    "ds.max.u32",
    "ds.max.i32",
    "ds.min.u32",
    "ds.min.i32",
    "ds.and.b32",
    "ds.or.b32",
    "ds.xor.b32",
    "ds.max.u64",
    "ds.min.u64",
    "ds.and.b64",
    "ds.or.b64",
    "ds.xor.b64",
];

/// The epochs: each one's original members, names, and the merged
/// numbering.
struct Epochs {
    /// Per original epoch, its start's barrier (none for the entry).
    barrier: Vec<Option<usize>>,
    /// Per original epoch, its members.
    members: Vec<BTreeSet<usize>>,
    /// Per original epoch, its merged number.
    merged: Vec<usize>,
    /// Per merged epoch, its name and whether it merges several.
    names: Vec<(String, bool)>,
}

impl<'a> Lds<'a> {
    fn site(&self, k: usize) -> String {
        let i = &self.s.instrs()[k];
        match i.attrs().lds_access.as_ref().filter(|a| !a.what.is_empty()) {
            Some(a) => a.what.to_string(),
            None => i.form().op.clone(),
        }
    }

    fn is_fence(&self, k: usize) -> bool {
        self.is_barrier_k(k) && self.s.instrs()[k].attrs().fence
    }

    fn is_barrier_k(&self, k: usize) -> bool {
        self.s.instrs()[k].form().flags.has(Flags::BARRIER)
    }

    /// The accesses: instructions whose LDS access's region is on the
    /// layout, with their leaf names.
    fn accesses(&self) -> Vec<(usize, Arc<str>)> {
        (0..self.s.instrs().len())
            .filter_map(|k| {
                self.s
                    .region_of(k)
                    .filter(|r| r.on_layout)
                    .map(|r| (k, r.leaf.clone()))
            })
            .collect()
    }

    fn writes(&self, k: usize) -> bool {
        let f = self.s.instrs()[k].form();
        !(f.kind == Kind::Ds && f.flags.has(Flags::LOAD))
    }

    fn ordered(&self, k: usize) -> bool {
        self.s.instrs()[k]
            .attrs()
            .lds_access
            .as_ref()
            .is_some_and(|a| !a.ordered.is_empty())
    }

    fn commute(&self, a: usize, b: usize) -> bool {
        let (ca, cb) = (
            &self.s.instrs()[a].form().canon,
            &self.s.instrs()[b].form().canon,
        );
        ca == cb && COMMUTING.contains(&ca.as_str())
    }

    /// The bytes of an access's region.
    fn bytes(&self, k: usize) -> (u64, u64) {
        self.s
            .region_of(k)
            .map_or((0, 0), |r| (r.base, r.base + r.size))
    }

    fn comment(&self, k: usize) -> Option<String> {
        self.s.instrs()[k]
            .attrs()
            .comment
            .as_ref()
            .map(|c| c.to_string())
    }

    /// What is reached from `from` over g without passing a barrier (one is
    /// included, not passed) and without entering `stop`.
    fn reach(
        &self,
        g: &Cfg,
        from: impl IntoIterator<Item = usize>,
        back: bool,
        stop: &dyn Fn(usize) -> bool,
    ) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut queue: VecDeque<usize> = from.into_iter().collect();
        while let Some(x) = queue.pop_front() {
            if !seen.insert(x) {
                continue;
            }
            let next = if back { &g.preds[x] } else { &g.succ[x] };
            queue.extend(next.iter().copied().filter(|&t| !stop(t)));
        }
        seen
    }

    /// The forward closure without passing a barrier.
    fn epoch_reach(&self, from: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut queue: VecDeque<usize> = from.into_iter().collect();
        while let Some(x) = queue.pop_front() {
            if seen.insert(x) && !self.is_barrier_k(x) {
                queue.extend(self.g.succ[x].iter().copied());
            }
        }
        seen
    }

    /// Partition accesses into barrier-delimited synchronization epochs.
    fn epochs(&self, pairs: &[Vec<usize>]) -> Epochs {
        let n = self.s.instrs().len();
        let mut barrier = vec![None];
        barrier.extend(
            (0..n)
                .filter(|&k| self.is_barrier_k(k) && k + 1 < n)
                .map(Some),
        );
        let members: Vec<BTreeSet<usize>> = barrier
            .iter()
            .map(|b| {
                if n == 0 {
                    BTreeSet::new()
                } else {
                    self.epoch_reach([b.map_or(0, |k| k + 1)])
                }
            })
            .collect();
        let names: Vec<String> = (barrier.iter())
            .map(|b| match b {
                None => "entry".into(),
                Some(k) => self
                    .comment(*k)
                    .map_or(format!("the barrier at {k}"), |c| format!("after {c}")),
            })
            .collect();
        let mut root: Vec<usize> = (0..barrier.len()).collect();
        let find = |root: &Vec<usize>, mut e: usize| {
            while root[e] != e {
                e = root[e];
            }
            e
        };
        for group in pairs {
            let es: Vec<usize> = group
                .iter()
                .filter_map(|&k| barrier.iter().position(|&b| b == Some(k)))
                .collect();
            for w in es.windows(2) {
                let (a, b) = (find(&root, w[0]), find(&root, w[1]));
                root[a.max(b)] = a.min(b);
            }
        }
        let roots: Vec<usize> = (0..barrier.len()).map(|e| find(&root, e)).collect();
        let order: Vec<usize> = roots
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let merged = roots
            .iter()
            .map(|r| order.iter().position(|o| o == r).expect("a root"))
            .collect();
        let named = (order.iter())
            .map(|&r| {
                let parts: Vec<&str> = (0..barrier.len())
                    .filter(|&e| roots[e] == r)
                    .map(|e| names[e].as_str())
                    .collect();
                (parts.join(" | "), parts.len() > 1)
            })
            .collect();
        Epochs {
            barrier,
            members,
            merged,
            names: named,
        }
    }
}

impl Epochs {
    /// Per instruction, its merged epochs.
    fn of_instr(&self, n: usize) -> Vec<BTreeSet<usize>> {
        let mut out = vec![BTreeSet::new(); n];
        for (e, m) in self.members.iter().enumerate() {
            m.iter().for_each(|&k| _ = out[k].insert(self.merged[e]));
        }
        out
    }
}

/// The lives of the leaves on one CFG.
struct Lives {
    flight: Vec<Vec<usize>>,
    access: Vec<usize>,
    life: Vec<BTreeSet<usize>>,
}

impl Lds<'_> {
    /// spans and lives on g, the accesses' leaves by layout position.
    fn lives(&self, g: &Cfg, access: &[(usize, usize)], nleaves: usize) -> Lives {
        let ks: Vec<usize> = access.iter().map(|a| a.0).collect();
        let flight = in_flight(self.s, g, &ks);
        let reached = g.reached();
        let mut span = vec![BTreeSet::new(); nleaves];
        for &(k, leaf) in access.iter().filter(|a| reached[a.0]) {
            span[leaf].insert(k);
        }
        for (q, flying) in flight.iter().enumerate().filter(|(q, _)| reached[*q]) {
            flying.iter().for_each(|&j| _ = span[access[j].1].insert(q));
        }
        let fence = |k: usize| self.is_fence(k);
        let life = (span.iter())
            .map(|sp| {
                let fwd = self.reach(g, sp.iter().copied(), false, &fence);
                let bwd = self.reach(g, sp.iter().copied(), true, &fence);
                fwd.intersection(&bwd)
                    .copied()
                    .filter(|&k| reached[k])
                    .collect()
            })
            .collect();
        Lives {
            flight,
            access: ks,
            life,
        }
    }
}

/// Proves the layout: the epochs, or the first MemError.
pub fn prove(s: &Snapshot, layout: &Layout, o: Options) -> Result<Proof, Diag> {
    let a = Lds { s, g: s.cfg(None) };
    let n = s.instrs().len();
    let branches = a.branches()?;
    let ep = a.epochs(&branches.pairs);
    let of_instr = ep.of_instr(n);
    let leaf_of = |name: &str| layout.leaves.iter().position(|l| l.name == name);
    let access: Vec<(usize, usize)> = a
        .accesses()
        .into_iter()
        .filter_map(|(k, l)| leaf_of(&l).map(|p| (k, p)))
        .collect();
    let lives = a.lives(&a.g, &access, layout.leaves.len());
    for k in (0..n).filter(|&k| a.is_fence(k)) {
        if let Some(&j) = lives.flight[k].first() {
            let first = lives.access[j];
            let text = format!(
                "the LDS fence {k} `{}` with `{}` (instruction {first}) in flight: an LDS fence needs every wave's LDS \
                 ops and DMAs done (a wait) before it",
                a.site(k),
                a.site(first)
            );
            return Err(refusal(vec![Part::Text(text)]));
        }
    }
    let of_leaf: Vec<BTreeSet<usize>> = lives
        .life
        .iter()
        .map(|life| {
            life.iter()
                .flat_map(|&k| of_instr[k].iter().copied())
                .collect()
        })
        .collect();
    let found = Found {
        access: &access,
        lives: &lives,
        of_leaf: &of_leaf,
        of_instr: &of_instr,
        ep: &ep,
    };
    a.pairs(layout, o, &found, &branches.divergent)?;
    if !o.ablate {
        a.protocol(&lives, &ep)?;
    }
    let epochs: Vec<String> = ep.names.iter().map(|(name, _)| name.clone()).collect();
    let report = report(layout, &of_leaf, &epochs);
    Ok(Proof {
        epochs,
        of_instr,
        of_leaf,
        report,
    })
}

/// What the pairs of leaves read: the accesses with their leaves, the
/// lives, and the epochs of the leaves and instructions.
struct Found<'p> {
    access: &'p [(usize, usize)],
    lives: &'p Lives,
    of_leaf: &'p [BTreeSet<usize>],
    of_instr: &'p [BTreeSet<usize>],
    ep: &'p Epochs,
}

impl Lds<'_> {
    /// pairs of leaves on the same bytes.
    fn pairs(
        &self,
        layout: &Layout,
        o: Options,
        p: &Found<'_>,
        divergent: &[(usize, Option<usize>, BTreeSet<usize>)],
    ) -> Result<(), Diag> {
        let leaves = &layout.leaves;
        let Found {
            access,
            lives,
            of_leaf,
            of_instr,
            ep,
        } = *p;
        for a in 0..leaves.len() {
            for b in a + 1..leaves.len() {
                let (la, lb) = (&leaves[a], &leaves[b]);
                if !(la.base < lb.base + lb.size && lb.base < la.base + la.size) {
                    continue;
                }
                if o.wave_order && self.one_wave_apart(access, a, b, leaves.len()) {
                    continue;
                }
                let together = |e: usize| {
                    let in_e = |life: &BTreeSet<usize>| -> BTreeSet<usize> {
                        life.iter()
                            .copied()
                            .filter(|&k| of_instr[k].contains(&e))
                            .collect()
                    };
                    let (ae, be) = (in_e(&lives.life[a]), in_e(&lives.life[b]));
                    let meets = |x: &BTreeSet<usize>, y: &BTreeSet<usize>| {
                        !self.epoch_reach(x.iter().copied()).is_disjoint(y)
                    };
                    meets(&ae, &be)
                        || meets(&be, &ae)
                        || divergent.iter().any(|(k, _, _)| {
                            of_instr[*k].contains(&e) && {
                                let r = self.epoch_reach([*k]);
                                !r.is_disjoint(&ae) && !r.is_disjoint(&be)
                            }
                        })
                };
                let common = of_leaf[a].intersection(&of_leaf[b]).copied();
                if let Some(e) = common.into_iter().find(|&e| ep.names[e].1 || together(e)) {
                    return Err(self.shared(p, layout, (a, b), e, &ep.names[e].0));
                }
            }
        }
        Ok(())
    }

    /// Whether two leaves are used by one wave alone, apart in its order.
    fn one_wave_apart(
        &self,
        access: &[(usize, usize)],
        a: usize,
        b: usize,
        nleaves: usize,
    ) -> bool {
        let groups = self.s.wave_groups();
        if groups.is_empty() {
            return false;
        }
        let mine: Vec<usize> = access
            .iter()
            .filter(|x| x.1 == a || x.1 == b)
            .map(|x| x.0)
            .collect();
        let classes: Vec<(Vec<u32>, Arc<Cfg>)> = (groups.into_iter())
            .map(|(waves, taken)| (waves, self.s.cfg(Some(&taken))))
            .filter(|(_, g)| {
                let reached = g.reached();
                mine.iter().any(|&k| reached[k])
            })
            .collect();
        let [(waves, g)] = classes.as_slice() else {
            return false;
        };
        if waves.len() != 1 {
            return false;
        }
        let lives = self.lives(g, access, nleaves);
        lives.life[a].is_disjoint(&lives.life[b])
    }

    /// The shared-bytes refusal.
    fn shared(
        &self,
        p: &Found<'_>,
        layout: &Layout,
        (a, b): (usize, usize),
        e: usize,
        name: &str,
    ) -> Diag {
        let side = |l: usize| {
            let leaf = &layout.leaves[l];
            let ks: Vec<usize> = p.access.iter().filter(|x| x.1 == l).map(|x| x.0).collect();
            let (first, last) = (
                ks.first().copied().unwrap_or(0),
                ks.last().copied().unwrap_or(0),
            );
            let at = ks
                .iter()
                .find(|&&k| p.of_instr[k].contains(&e))
                .map_or("in flight or live across it".into(), |&k| {
                    format!("in it at {k} `{}`", self.site(k))
                });
            (
                format!("{} [{}, {})", leaf.name, leaf.base, leaf.base + leaf.size),
                format!(
                    "{} accessed {first} `{}` .. {last} `{}`, {at}",
                    leaf.name,
                    self.site(first),
                    self.site(last)
                ),
            )
        };
        let ((ha, ta), (hb, tb)) = (side(a), side(b));
        refusal(vec![Part::Text(format!(
            "{ha} and {hb} share bytes and are both live in the epoch '{name}': {ta}; {tb}"
        ))])
    }

    /// an access in flight at a barrier against the accesses after
    /// it.
    fn protocol(&self, lives: &Lives, ep: &Epochs) -> Result<(), Diag> {
        let lds_op = |k: usize| {
            let f = self.s.instrs()[k].form();
            f.kind == Kind::Ds || f.flags.has(Flags::LDS_MEM)
        };
        let overlap = |x: usize, y: usize| {
            let ((a, b), (c, d)) = (self.bytes(x), self.bytes(y));
            a < d && c < b
        };
        for (e, barrier) in ep.barrier.iter().enumerate() {
            let Some(k) = *barrier else { continue };
            for &j in &lives.flight[k] {
                let k1 = lives.access[j];
                if self.ordered(k1) {
                    continue;
                }
                let after = lives
                    .access
                    .iter()
                    .copied()
                    .filter(|k2| ep.members[e].contains(k2));
                for k2 in after {
                    let skip =
                        !(lds_op(k1) || lds_op(k2)) || self.ordered(k2) || self.commute(k1, k2);
                    if skip || !(self.writes(k1) || self.writes(k2)) || !overlap(k1, k2) {
                        continue;
                    }
                    let at = self.comment(k).map_or(String::new(), |c| format!(" '{c}'"));
                    let back = if k1 > k {
                        " (the previous iteration's, over the loop's back edge)"
                    } else {
                        ""
                    };
                    let region = self
                        .s
                        .region_of(k1)
                        .map_or(String::new(), |r| r.name.to_string());
                    let text = format!(
                        "`{}` (instruction {k1}) is in flight at the barrier{at} (instruction {k}){back}, and `{}` \
                         (instruction {k2}) accesses its bytes after it: another wave may run it before the op lands \
                         (wait before the barrier: b.wait({region}))",
                        self.site(k1),
                        self.site(k2)
                    );
                    return Err(refusal(vec![Part::Text(text)]));
                }
            }
        }
        Ok(())
    }
}

/// report lines.
fn report(layout: &Layout, of_leaf: &[BTreeSet<usize>], epochs: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for u in layout.unions.iter().filter(|u| u.arms.len() > 1) {
        let arms: Vec<String> = (u.arms.iter())
            .map(|(name, leaves)| {
                let es: BTreeSet<usize> = leaves
                    .iter()
                    .flat_map(|&l| of_leaf[l].iter().copied())
                    .collect();
                let names: Vec<String> = es.iter().map(|&e| format!("'{}'", epochs[e])).collect();
                let shown = if names.is_empty() {
                    "never accessed".into()
                } else {
                    names.join(", ")
                };
                format!("{name} in {shown}")
            })
            .collect();
        out.push(format!(
            "lds: union [{}, {}) proven by the lives: {}",
            u.base,
            u.end,
            arms.join("; ")
        ));
    }
    let never: Vec<&str> = (layout.leaves.iter().zip(of_leaf))
        .filter(|(_, e)| e.is_empty())
        .map(|(l, _)| l.name.as_str())
        .collect();
    if !never.is_empty() {
        out.push(format!("lds: never accessed: {}", never.join(", ")));
    }
    out
}

/// The bytes [lo, hi) per access, one map per wave.
pub type WaveBytes = BTreeMap<usize, (u64, u64)>;

/// Races between waves, on the proof made of the same stream.
pub fn races(s: &Snapshot, proof: &Proof, per_wave: &[WaveBytes]) -> Result<(), Diag> {
    let a = Lds { s, g: s.cfg(None) };
    let accesses: Vec<usize> = a.accesses().into_iter().map(|x| x.0).collect();
    let groups = s.wave_groups();
    let reached: Vec<Vec<bool>> = (0..per_wave.len() as u32)
        .map(|w| match groups.iter().find(|g| g.0.contains(&w)) {
            Some((_, taken)) => s.cfg(Some(taken)).reached(),
            None => a.g.reached(),
        })
        .collect();
    let bytes = |w: usize, k: usize| {
        reached[w][k].then(|| per_wave[w].get(&k).copied().unwrap_or_else(|| a.bytes(k)))
    };
    for e in 0..proof.epochs.len() {
        let inside: Vec<usize> = accesses
            .iter()
            .copied()
            .filter(|&k| proof.of_instr[k].contains(&e))
            .collect();
        for (x, &k1) in inside.iter().enumerate() {
            for &k2 in &inside[x..] {
                if !(a.writes(k1) || a.writes(k2))
                    || a.ordered(k1)
                    || a.ordered(k2)
                    || a.commute(k1, k2)
                {
                    continue;
                }
                for w1 in 0..per_wave.len() {
                    for w2 in (0..per_wave.len()).filter(|&w2| w2 != w1) {
                        let (Some(b1), Some(b2)) = (bytes(w1, k1), bytes(w2, k2)) else {
                            continue;
                        };
                        if b1.0 < b2.1 && b2.0 < b1.1 {
                            return Err(race(&a, (k1, w1, b1), (k2, w2, b2), &proof.epochs[e]));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn race(
    a: &Lds<'_>,
    (k1, w1, b1): (usize, usize, (u64, u64)),
    (k2, w2, b2): (usize, usize, (u64, u64)),
    name: &str,
) -> Diag {
    refusal(vec![Part::Text(format!(
        "`{}` (instruction {k1}) in wave {w1} and `{}` (instruction {k2}) in wave {w2} may touch the same bytes \
         ([{}, {}) and [{}, {})) in the epoch '{name}', a write among them: the waves run apart there, either may go \
         first (a barrier between them, or each wave's bytes its own; a block only some waves run: b.waves([...]) - a \
         branch on the wave index is not read as one)",
        a.site(k1),
        a.site(k2),
        b1.0,
        b1.1,
        b2.0,
        b2.1
    ))])
}
